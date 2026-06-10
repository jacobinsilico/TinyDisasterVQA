#!/usr/bin/env python3
"""
06_train_student.py

Train TinyDisasterVQA TDM student models.

Current student formulation:
  - single-head edge_global classifier
  - cap5 / 14 classes by default
  - template question encoder: one-hot template vector + Linear
  - no multihead routing
  - no LSTM / attention in student

Supported student variants:
  - tdm_s
  - tdm_m
  - tdm_l
  - tdm_fast

Supported modes:
  1. CE:
     student learns from hard edge_global labels.

  2. weighted CE:
     student learns from hard labels with edge_global class weights.

  3. KD:
     student learns from hard labels + soft teacher logits.

Example CE:

PYTHONPATH=src python scripts/06_train_student.py \
  --mode ce \
  --student-variant tdm_m \
  --epochs 50 \
  --batch-size 64 \
  --run-name S2_tdm_m_ce

Example weighted CE:

PYTHONPATH=src python scripts/06_train_student.py \
  --mode weighted_ce \
  --student-variant tdm_m \
  --epochs 50 \
  --batch-size 64 \
  --run-name S4_tdm_m_weighted_ce

Example KD:

PYTHONPATH=src python scripts/06_train_student.py \
  --mode kd \
  --student-variant tdm_m \
  --teacher-checkpoint /path/to/teacher/checkpoints/best.pt \
  --epochs 50 \
  --batch-size 64 \
  --run-name S5_tdm_m_kd_T5
"""

from __future__ import annotations

import argparse
import inspect
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tinydisastervqa.data import FloodNetVQADataset, get_image_transform, load_json  # noqa: E402
from tinydisastervqa.metrics import ClassificationMetrics, format_metrics  # noqa: E402
from tinydisastervqa.models import (  # noqa: E402
    build_tdm_from_metadata,
    build_teacher_from_metadata,
    describe_model,
)
from tinydisastervqa.utils import (  # noqa: E402
    AverageMeter,
    Timer,
    append_jsonl,
    make_run_dir,
    save_checkpoint,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Paths.
    parser.add_argument("--train-csv", type=Path, default=Path("outputs/training_data_cap5/train.csv"))
    parser.add_argument("--valid-csv", type=Path, default=Path("outputs/training_data_cap5/valid.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("outputs/training_data_cap5/test.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("outputs/training_data_cap5/metadata.json"))
    parser.add_argument(
        "--class-weights",
        type=Path,
        default=Path("outputs/answer_space_cap5/class_weights_edge_global_by_label.json"),
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))

    # Run.
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    # Mode.
    parser.add_argument(
        "--mode",
        type=str,
        default="ce",
        choices=["ce", "weighted_ce", "kd"],
        help="Training objective.",
    )
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        default=False,
        help="Backward-compatible alias for --mode weighted_ce.",
    )

    # Data.
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Optional separate valid/test batch size. Defaults to --batch-size.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--augment-train", action="store_true", default=True)
    parser.add_argument("--no-augment-train", action="store_false", dest="augment_train")
    parser.add_argument("--overfit-samples", type=int, default=0)

    # Student model.
    parser.add_argument(
        "--student-variant",
        type=str,
        default="tdm_m",
        choices=["tdm_s", "tdm_m", "tdm_l", "tdm_fast"],
        help="Student model variant.",
    )
    parser.add_argument(
        "--student-size",
        type=str,
        default=None,
        choices=[
            "s",
            "m",
            "l",
            "fast",
            "tdm_s",
            "tdm_m",
            "tdm_l",
            "tdm_fast",
            # Deprecated aliases, mapped to new names.
            "xxs",
            "xs",
            "tdm_xxs",
            "tdm_xs",
        ],
        help="Deprecated alias for --student-variant.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Usually inferred from metadata. cap5 should be 14.",
    )
    parser.add_argument(
        "--num-question-templates",
        type=int,
        default=None,
        help="Usually inferred from metadata.",
    )
    parser.add_argument("--template-embed-dim", type=int, default=None)
    parser.add_argument("--fusion-hidden-dim", type=int, default=None)
    parser.add_argument("--fusion-dropout", type=float, default=None)
    parser.add_argument("--fusion-layers", type=int, default=None)

    # Teacher / KD.
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--teacher-backbone",
        type=str,
        default=None,
        choices=[None, "convnext_tiny", "swin_tiny", "efficientnet_b0", "efficientnet_b1", "resnet18", "resnet50"],
    )
    parser.add_argument(
        "--teacher-question-encoder",
        type=str,
        default=None,
        choices=[None, "lstm", "template"],
    )
    parser.add_argument("--teacher-pretrained", action="store_true", default=None)
    parser.add_argument("--teacher-no-pretrained", action="store_false", dest="teacher_pretrained")
    parser.add_argument("--teacher-question-embed-dim", type=int, default=None)
    parser.add_argument("--teacher-question-hidden-dim", type=int, default=None)
    parser.add_argument("--teacher-template-embed-dim", type=int, default=None)
    parser.add_argument("--teacher-fusion-hidden-dim", type=int, default=None)
    parser.add_argument("--teacher-fusion-dropout", type=float, default=None)
    parser.add_argument(
        "--teacher-use-count-aux",
        action="store_true",
        default=None,
        help="Usually inferred from checkpoint config. Needed for loading T6 teacher.",
    )

    # Optimization.
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")

    # KD.
    parser.add_argument("--kd-alpha", type=float, default=0.7)
    parser.add_argument("--kd-temperature", type=float, default=4.0)

    # Early stopping.
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=0.0)

    # Logging/checkpointing.
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-every-epoch", action="store_true", default=False)

    return parser.parse_args()


def normalize_student_variant(args: argparse.Namespace) -> None:
    alias = {
        "s": "tdm_s",
        "m": "tdm_m",
        "l": "tdm_l",
        "fast": "tdm_fast",
        "tdm_s": "tdm_s",
        "tdm_m": "tdm_m",
        "tdm_l": "tdm_l",
        "tdm_fast": "tdm_fast",
        # Deprecated old naming. These map to the nearest new scale.
        "xxs": "tdm_s",
        "xs": "tdm_m",
        "tdm_xxs": "tdm_s",
        "tdm_xs": "tdm_m",
    }

    if args.student_size is not None:
        args.student_variant = alias[args.student_size]


def get_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")

    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer_num_classes(metadata: dict[str, Any], fallback: int | None = None) -> int:
    for key in ["num_classes", "num_edge_global_classes"]:
        if key in metadata:
            return int(metadata[key])

    try:
        return int(metadata["answer_space"]["target_modes"]["edge_global"]["num_classes"])
    except KeyError:
        pass

    if fallback is not None:
        return int(fallback)

    raise KeyError(
        "Could not infer num_classes from metadata. Expected num_classes, "
        "num_edge_global_classes, or answer_space.target_modes.edge_global.num_classes."
    )


def infer_num_question_templates(metadata: dict[str, Any], fallback: int | None = None) -> int:
    if "num_question_templates" in metadata:
        return int(metadata["num_question_templates"])

    if fallback is not None:
        return int(fallback)

    return 31


def infer_label_to_class(metadata: dict[str, Any]) -> dict[str, str] | None:
    if "edge_label_to_class" in metadata:
        return {
            str(k): str(v)
            for k, v in metadata["edge_label_to_class"].items()
        }

    try:
        return {
            str(k): str(v)
            for k, v in metadata["answer_space"]["target_modes"]["edge_global"]["label_to_class"].items()
        }
    except KeyError:
        return None


def infer_count_cap(metadata: dict[str, Any]) -> int | None:
    if "count_cap" in metadata and metadata["count_cap"] is not None:
        return int(metadata["count_cap"])

    try:
        count_cap = metadata["answer_space"]["count_cap"]
        if count_cap is not None:
            return int(count_cap)
    except KeyError:
        pass

    return None


def build_loaders(args: argparse.Namespace) -> dict[str, DataLoader]:
    train_transform = get_image_transform(
        image_size=args.image_size,
        train=True,
        augment=args.augment_train,
    )

    eval_transform = get_image_transform(
        image_size=args.image_size,
        train=False,
        augment=False,
    )

    train_dataset = FloodNetVQADataset(
        csv_path=args.train_csv,
        target_mode="edge_global",
        transform=train_transform,
        dataset_root=args.dataset_root,
        verify_images=False,
    )

    valid_dataset = FloodNetVQADataset(
        csv_path=args.valid_csv,
        target_mode="edge_global",
        transform=eval_transform,
        dataset_root=args.dataset_root,
        verify_images=False,
    )

    test_dataset = FloodNetVQADataset(
        csv_path=args.test_csv,
        target_mode="edge_global",
        transform=eval_transform,
        dataset_root=args.dataset_root,
        verify_images=False,
    )

    if args.overfit_samples > 0:
        n = min(args.overfit_samples, len(train_dataset))
        indices = list(range(n))

        train_dataset = Subset(train_dataset, indices)
        valid_dataset = Subset(
            FloodNetVQADataset(
                csv_path=args.train_csv,
                target_mode="edge_global",
                transform=eval_transform,
                dataset_root=args.dataset_root,
                verify_images=False,
            ),
            indices,
        )
        test_dataset = valid_dataset

        print(f"Overfit mode enabled: using first {n} training samples for train/valid/test.")

    pin_memory = torch.cuda.is_available()
    eval_batch_size = args.eval_batch_size or args.batch_size

    train_loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }

    eval_loader_kwargs = {
        "batch_size": eval_batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }

    if args.num_workers > 0:
        train_loader_kwargs["persistent_workers"] = True
        train_loader_kwargs["prefetch_factor"] = 2
        eval_loader_kwargs["persistent_workers"] = True
        eval_loader_kwargs["prefetch_factor"] = 2

    return {
        "train": DataLoader(train_dataset, **train_loader_kwargs),
        "valid": DataLoader(valid_dataset, **eval_loader_kwargs),
        "test": DataLoader(test_dataset, **eval_loader_kwargs),
    }


def build_ce_criterion(args: argparse.Namespace, device: torch.device) -> nn.Module:
    use_weights = args.mode == "weighted_ce"

    if not use_weights:
        return nn.CrossEntropyLoss()

    if not args.class_weights.exists():
        raise FileNotFoundError(f"Class weights file not found: {args.class_weights}")

    weights_dict = load_json(args.class_weights)
    weights = torch.ones(args.num_classes, dtype=torch.float32)

    for label_str, weight in weights_dict.items():
        label = int(label_str)

        if 0 <= label < args.num_classes:
            weights[label] = float(weight)

    print("Using edge_global class weights:")
    print(weights.detach().cpu().tolist())

    return nn.CrossEntropyLoss(weight=weights.to(device))


def read_checkpoint_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = checkpoint.get("config", {})
    if config is None:
        return {}
    return dict(config)


def _config_get(
    ckpt_config: dict[str, Any],
    args_value: Any,
    key: str,
    default: Any,
) -> Any:
    if args_value is not None:
        return args_value

    if key in ckpt_config and ckpt_config[key] is not None:
        return ckpt_config[key]

    return default


def infer_teacher_use_count_aux(
    ckpt_config: dict[str, Any],
    args_value: bool | None,
) -> bool:
    if args_value is not None:
        return bool(args_value)

    if "use_count_aux" in ckpt_config:
        return bool(ckpt_config["use_count_aux"])

    if ckpt_config.get("loss_mode") == "count_aux":
        return True

    return False


def adapt_legacy_template_embedding_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Converts old teacher checkpoints:

        question_encoder.embedding.weight  [num_templates, embed_dim]

    into new one-hot + Linear format:

        question_encoder.linear.weight     [embed_dim, num_templates]
        question_encoder.linear.bias       [embed_dim]

    This is mathematically equivalent to embedding lookup with zero bias.
    """
    old_key = "question_encoder.embedding.weight"
    new_weight_key = "question_encoder.linear.weight"
    new_bias_key = "question_encoder.linear.bias"

    if old_key in state_dict and new_weight_key not in state_dict:
        embedding_weight = state_dict.pop(old_key)

        # nn.Linear computes y = x @ W.T + b.
        # one_hot @ embedding_weight == Linear(one_hot) with W = embedding_weight.T
        state_dict[new_weight_key] = embedding_weight.T.contiguous()
        state_dict[new_bias_key] = torch.zeros(
            embedding_weight.shape[1],
            dtype=embedding_weight.dtype,
            device=embedding_weight.device,
        )

        print("Adapted legacy teacher checkpoint: embedding.weight -> linear.weight/bias")

    return state_dict

def build_teacher_for_kd(
    args: argparse.Namespace,
    metadata: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    if args.teacher_checkpoint is None:
        raise ValueError("--teacher-checkpoint is required when --mode kd.")

    if not args.teacher_checkpoint.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {args.teacher_checkpoint}")

    checkpoint = torch.load(args.teacher_checkpoint, map_location=device)
    ckpt_config = read_checkpoint_config(checkpoint)

    teacher_backbone = str(
        _config_get(
            ckpt_config=ckpt_config,
            args_value=args.teacher_backbone,
            key="backbone",
            default="convnext_tiny",
        )
    )

    teacher_question_encoder = str(
        _config_get(
            ckpt_config=ckpt_config,
            args_value=args.teacher_question_encoder,
            key="question_encoder",
            default="template",
        )
    )

    teacher_pretrained = bool(
        _config_get(
            ckpt_config=ckpt_config,
            args_value=args.teacher_pretrained,
            key="pretrained",
            default=True,
        )
    )

    teacher_question_embed_dim = int(
        _config_get(
            ckpt_config=ckpt_config,
            args_value=args.teacher_question_embed_dim,
            key="question_embed_dim",
            default=128,
        )
    )

    teacher_question_hidden_dim = int(
        _config_get(
            ckpt_config=ckpt_config,
            args_value=args.teacher_question_hidden_dim,
            key="question_hidden_dim",
            default=256,
        )
    )

    teacher_template_embed_dim = int(
        _config_get(
            ckpt_config=ckpt_config,
            args_value=args.teacher_template_embed_dim,
            key="template_embed_dim",
            default=128,
        )
    )

    teacher_fusion_hidden_dim = int(
        _config_get(
            ckpt_config=ckpt_config,
            args_value=args.teacher_fusion_hidden_dim,
            key="fusion_hidden_dim",
            default=512,
        )
    )

    teacher_fusion_dropout = float(
        _config_get(
            ckpt_config=ckpt_config,
            args_value=args.teacher_fusion_dropout,
            key="fusion_dropout",
            default=0.3,
        )
    )

    teacher_use_count_aux = infer_teacher_use_count_aux(
        ckpt_config=ckpt_config,
        args_value=args.teacher_use_count_aux,
    )

    num_count_classes = ckpt_config.get("num_count_classes", None)
    if num_count_classes is not None:
        num_count_classes = int(num_count_classes)

    teacher = build_teacher_from_metadata(
        metadata=metadata,
        image_backbone=teacher_backbone,
        pretrained=teacher_pretrained,
        num_classes=args.num_classes,
        freeze_image_encoder=False,
        question_encoder=teacher_question_encoder,
        question_embed_dim=teacher_question_embed_dim,
        question_hidden_dim=teacher_question_hidden_dim,
        template_embed_dim=teacher_template_embed_dim,
        fusion_hidden_dim=teacher_fusion_hidden_dim,
        fusion_dropout=teacher_fusion_dropout,
        use_count_aux=teacher_use_count_aux,
        num_count_classes=num_count_classes,
    ).to(device)

    state_dict = adapt_legacy_template_embedding_state_dict(
    checkpoint["model_state_dict"]
    )
    teacher.load_state_dict(state_dict, strict=True)
    teacher.eval()

    for param in teacher.parameters():
        param.requires_grad = False

    print("Loaded KD teacher:")
    print(f"  checkpoint:        {args.teacher_checkpoint}")
    print(f"  backbone:          {teacher_backbone}")
    print(f"  question_encoder:  {teacher_question_encoder}")
    print(f"  pretrained:        {teacher_pretrained}")
    print(f"  use_count_aux:     {teacher_use_count_aux}")

    return teacher


def autocast_context(device: torch.device, enabled: bool):
    return torch.amp.autocast(
        device_type=device.type,
        enabled=(enabled and device.type == "cuda"),
    )


def extract_logits(outputs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        return outputs

    if isinstance(outputs, dict):
        if "logits" not in outputs:
            raise KeyError("Model output dict must contain key 'logits'.")
        return outputs["logits"]

    raise TypeError(f"Expected tensor or dict output, got {type(outputs)}.")


def kd_loss_fn(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    KL(student || teacher) with softened distributions.
    Multiplied by T^2 following standard KD practice.
    """
    t = float(temperature)

    student_log_probs = F.log_softmax(student_logits / t, dim=1)
    teacher_probs = F.softmax(teacher_logits / t, dim=1)

    return F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    ) * (t * t)


def forward_student(
    student: nn.Module,
    batch: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    images = batch["image"].to(device, non_blocking=True)
    question_tokens = batch["question_tokens"].to(device, non_blocking=True)
    question_lengths = batch["question_length"].to(device, non_blocking=True)
    question_template_ids = batch["question_template_id"].to(device, non_blocking=True)

    return student(
        images=images,
        question_tokens=question_tokens,
        question_lengths=question_lengths,
        question_template_ids=question_template_ids,
    )


def forward_teacher(
    teacher: nn.Module,
    batch: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    images = batch["image"].to(device, non_blocking=True)
    question_tokens = batch["question_tokens"].to(device, non_blocking=True)
    question_lengths = batch["question_length"].to(device, non_blocking=True)
    question_template_ids = batch["question_template_id"].to(device, non_blocking=True)

    forward_sig = inspect.signature(teacher.forward)
    params = forward_sig.parameters

    kwargs: dict[str, Any] = {
        "images": images,
    }

    if "question_tokens" in params:
        kwargs["question_tokens"] = question_tokens

    if "question_lengths" in params:
        kwargs["question_lengths"] = question_lengths

    if "question_template_ids" in params:
        kwargs["question_template_ids"] = question_template_ids

    if "return_aux" in params:
        kwargs["return_aux"] = False

    outputs = teacher(**kwargs)
    return extract_logits(outputs)


@torch.no_grad()
def evaluate_student(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    criterion: nn.Module | None = None,
    label_to_class: dict[str, str] | None = None,
) -> dict[str, Any]:
    model.eval()

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    meter = ClassificationMetrics(
        num_classes=num_classes,
        label_to_class=label_to_class,
    )

    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        targets = batch["target"].to(device, non_blocking=True)

        logits = forward_student(
            student=model,
            batch=batch,
            device=device,
        )

        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        meter.update(
            logits=logits,
            targets=targets,
            edge_heads=batch.get("edge_head"),
            question_types=batch.get("question_type"),
        )

    result = meter.compute()
    result["loss"] = total_loss / max(total_samples, 1)

    return result


def train_one_epoch(
    student: nn.Module,
    teacher: nn.Module | None,
    loader: DataLoader,
    ce_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    student.train()

    loss_meter = AverageMeter("loss")
    ce_loss_meter = AverageMeter("ce_loss")
    kd_loss_meter = AverageMeter("kd_loss")

    metrics = ClassificationMetrics(
        num_classes=args.num_classes,
        label_to_class=args.label_to_class,
    )

    timer = Timer()

    for step, batch in enumerate(loader, start=1):
        targets = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, args.amp):
            student_logits = forward_student(
                student=student,
                batch=batch,
                device=device,
            )

            ce_loss = ce_criterion(student_logits, targets)

            if args.mode == "kd":
                assert teacher is not None

                with torch.no_grad():
                    teacher_logits = forward_teacher(
                        teacher=teacher,
                        batch=batch,
                        device=device,
                    )

                kd_loss = kd_loss_fn(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    temperature=args.kd_temperature,
                )

                loss = (1.0 - float(args.kd_alpha)) * ce_loss + float(args.kd_alpha) * kd_loss

            else:
                kd_loss = student_logits.new_tensor(0.0)
                loss = ce_loss

        scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)

        loss_meter.update(float(loss.item()), n=batch_size)
        ce_loss_meter.update(float(ce_loss.item()), n=batch_size)
        kd_loss_meter.update(float(kd_loss.item()), n=batch_size)

        metrics.update(
            logits=student_logits.detach(),
            targets=targets.detach(),
            edge_heads=batch.get("edge_head"),
            question_types=batch.get("question_type"),
        )

        if step % args.log_interval == 0 or step == 1 or step == len(loader):
            current_metrics = metrics.compute()
            overall = current_metrics["overall"]
            by_head = current_metrics["by_head"]

            head_str = " | ".join(
                f"{head}={values['accuracy']:.3f}"
                for head, values in sorted(by_head.items())
            )

            print(
                f"Epoch {epoch:03d} | "
                f"step {step:04d}/{len(loader):04d} | "
                f"loss={loss_meter.avg:.4f} | "
                f"ce={ce_loss_meter.avg:.4f} | "
                f"kd={kd_loss_meter.avg:.4f} | "
                f"acc={overall['accuracy']:.4f} | "
                f"{head_str} | "
                f"time={timer.elapsed_str()}"
            )

    result = metrics.compute()
    result["loss"] = loss_meter.avg
    result["ce_loss"] = ce_loss_meter.avg
    result["kd_loss"] = kd_loss_meter.avg

    return result


def build_run_prefix(args: argparse.Namespace) -> str:
    cap_tag = f"cap{args.count_cap}" if args.count_cap is not None else "capNA"
    return f"student_{cap_tag}_{args.student_variant}_{args.mode}"


def main() -> None:
    args = parse_args()

    normalize_student_variant(args)

    if args.use_class_weights:
        args.mode = "weighted_ce"

    if args.mode == "kd" and args.teacher_checkpoint is None:
        raise ValueError("--teacher-checkpoint is required for --mode kd.")

    if args.kd_alpha < 0 or args.kd_alpha > 1:
        raise ValueError("--kd-alpha must be in [0, 1].")

    if args.kd_temperature <= 0:
        raise ValueError("--kd-temperature must be positive.")

    set_seed(args.seed)

    device = get_device(args.device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    metadata = load_json(args.metadata)

    args.num_classes = infer_num_classes(metadata, fallback=args.num_classes)
    args.num_question_templates = infer_num_question_templates(
        metadata,
        fallback=args.num_question_templates,
    )
    args.label_to_class = infer_label_to_class(metadata)
    args.count_cap = infer_count_cap(metadata)

    run_prefix = build_run_prefix(args)

    run_dir = make_run_dir(
        base_dir=args.runs_dir,
        run_name=args.run_name,
        prefix=run_prefix,
    )

    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config.items()
        if key != "label_to_class"
    }
    config["run_dir"] = str(run_dir)
    config["device"] = str(device)

    save_json(config, run_dir / "config.json")

    print("=" * 80)
    print("TinyDisasterVQA / Train TDM Student")
    print("=" * 80)
    print(f"Run dir:              {run_dir}")
    print(f"Device:               {device}")
    print(f"Student variant:      {args.student_variant}")
    print(f"Mode:                 {args.mode}")
    print(f"Count cap:            {args.count_cap}")
    print(f"Num classes:          {args.num_classes}")
    print(f"Question templates:   {args.num_question_templates}")
    print(f"AMP:                  {args.amp}")
    print(f"Image size:           {args.image_size}")
    print(f"Batch size:           {args.batch_size}")
    print(f"Eval batch:           {args.eval_batch_size or args.batch_size}")
    print(f"Epochs:               {args.epochs}")
    print(f"LR:                   {args.lr}")
    print(f"Weight decay:         {args.weight_decay}")
    print(f"Patience:             {args.patience}")
    print(f"Min delta:            {args.min_delta}")
    print(f"KD alpha:             {args.kd_alpha if args.mode == 'kd' else 0.0}")
    print(f"KD temp:              {args.kd_temperature if args.mode == 'kd' else 0.0}")
    print()

    loaders = build_loaders(args)

    student = build_tdm_from_metadata(
        metadata=metadata,
        variant=args.student_variant,
        num_classes=args.num_classes,
        num_question_templates=args.num_question_templates,
        question_template_embed_dim=args.template_embed_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        fusion_dropout=args.fusion_dropout,
        fusion_layers=args.fusion_layers,
    ).to(device)

    print(describe_model(student))
    print()

    teacher = None

    if args.mode == "kd":
        print("Loading teacher...")
        teacher = build_teacher_for_kd(args, metadata, device)
        print()
        print(describe_model(teacher))
        print()

    ce_criterion = build_ce_criterion(args, device)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(args.amp and device.type == "cuda"),
    )

    best_valid_acc = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    completed_epoch = 0

    metrics_path = run_dir / "metrics.jsonl"
    total_timer = Timer()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        print()
        print("=" * 80)
        print(f"Epoch {epoch}/{args.epochs}")
        print("=" * 80)

        train_metrics = train_one_epoch(
            student=student,
            teacher=teacher,
            loader=loaders["train"],
            ce_criterion=ce_criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            args=args,
        )

        valid_metrics = evaluate_student(
            model=student,
            dataloader=loaders["valid"],
            device=device,
            num_classes=args.num_classes,
            criterion=ce_criterion,
            label_to_class=args.label_to_class,
        )

        scheduler.step()

        train_acc = float(train_metrics["overall"]["accuracy"])
        valid_acc = float(valid_metrics["overall"]["accuracy"])

        completed_epoch = epoch

        improved = valid_acc > (best_valid_acc + args.min_delta)

        if improved:
            best_valid_acc = valid_acc
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                path=checkpoints_dir / "best.pt",
                model=student,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics={
                    "train": train_metrics,
                    "valid": valid_metrics,
                    "best_valid_acc": best_valid_acc,
                    "best_epoch": best_epoch,
                },
                config=config,
            )
        else:
            epochs_without_improvement += 1

        if args.save_every_epoch:
            save_checkpoint(
                path=checkpoints_dir / f"epoch_{epoch:03d}.pt",
                model=student,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics={
                    "train": train_metrics,
                    "valid": valid_metrics,
                    "best_valid_acc": best_valid_acc,
                    "best_epoch": best_epoch,
                },
                config=config,
            )

        epoch_record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_ce_loss": train_metrics["ce_loss"],
            "train_kd_loss": train_metrics["kd_loss"],
            "train_acc": train_acc,
            "valid_loss": valid_metrics["loss"],
            "valid_acc": valid_acc,
            "valid_macro_accuracy": valid_metrics.get("macro_accuracy"),
            "best_valid_acc": best_valid_acc,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "epoch_time_sec": time.time() - epoch_start,
        }

        if "count_exact" in valid_metrics:
            epoch_record["valid_count_exact"] = valid_metrics["count_exact"]["accuracy"]

        if "count_pm1" in valid_metrics:
            epoch_record["valid_count_pm1"] = valid_metrics["count_pm1"]["accuracy"]

        append_jsonl(epoch_record, metrics_path)

        print()
        print(format_metrics(train_metrics, prefix="train"))
        print()
        print(format_metrics(valid_metrics, prefix="valid"))
        print()
        print(
            f"Epoch {epoch} done | "
            f"train_acc={train_acc:.4f} | "
            f"valid_acc={valid_acc:.4f} | "
            f"best_valid_acc={best_valid_acc:.4f} at epoch {best_epoch} | "
            f"no_improve={epochs_without_improvement}/{args.patience} | "
            f"{'IMPROVED' if improved else 'no improvement'}"
        )

        if (
            args.patience > 0
            and epochs_without_improvement >= args.patience
        ):
            print()
            print("=" * 80)
            print(
                f"Early stopping triggered after {args.patience} epochs "
                f"without improvement."
            )
            print("=" * 80)
            break

    print()
    print("=" * 80)
    print("Evaluating best checkpoint on test set")
    print("=" * 80)

    best_path = checkpoints_dir / "best.pt"

    if not best_path.exists():
        raise FileNotFoundError(f"No best checkpoint found at {best_path}")

    best_ckpt = torch.load(best_path, map_location=device)
    student.load_state_dict(best_ckpt["model_state_dict"], strict=True)

    test_metrics = evaluate_student(
        model=student,
        dataloader=loaders["test"],
        device=device,
        num_classes=args.num_classes,
        criterion=ce_criterion,
        label_to_class=args.label_to_class,
    )

    save_json(test_metrics, run_dir / "test_metrics.json")

    save_checkpoint(
        path=checkpoints_dir / "last.pt",
        model=student,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=completed_epoch,
        metrics={
            "test": test_metrics,
            "best_valid_acc": best_valid_acc,
            "best_epoch": best_epoch,
        },
        config=config,
    )

    final_summary = {
        "run_dir": str(run_dir),
        "student_variant": args.student_variant,
        "mode": args.mode,
        "count_cap": args.count_cap,
        "num_classes": args.num_classes,
        "best_valid_acc": best_valid_acc,
        "best_epoch": best_epoch,
        "test_acc": float(test_metrics["overall"]["accuracy"]),
        "test_macro_accuracy": test_metrics.get("macro_accuracy"),
        "test_count_exact": (
            test_metrics["count_exact"]["accuracy"]
            if "count_exact" in test_metrics
            else None
        ),
        "test_count_pm1": (
            test_metrics["count_pm1"]["accuracy"]
            if "count_pm1" in test_metrics
            else None
        ),
        "total_params": sum(p.numel() for p in student.parameters()),
        "trainable_params": sum(p.numel() for p in student.parameters() if p.requires_grad),
        "teacher_checkpoint": str(args.teacher_checkpoint) if args.teacher_checkpoint else None,
        "kd_alpha": args.kd_alpha if args.mode == "kd" else None,
        "kd_temperature": args.kd_temperature if args.mode == "kd" else None,
        "image_size": args.image_size,
    }

    if "by_head" in test_metrics:
        for head, values in test_metrics["by_head"].items():
            final_summary[f"test_{head}_acc"] = values["accuracy"]

    save_json(final_summary, run_dir / "final_summary.json")

    print()
    print(format_metrics(test_metrics, prefix="test"))

    print()
    print("=" * 80)
    print("Student training complete")
    print("=" * 80)
    print(f"Run dir:          {run_dir}")
    print(f"Student:          {args.student_variant}")
    print(f"Mode:             {args.mode}")
    print(f"Best valid acc:   {best_valid_acc:.4f}")
    print(f"Best epoch:       {best_epoch}")
    print(f"Test acc:         {test_metrics['overall']['accuracy']:.4f}")

    if "macro_accuracy" in test_metrics:
        print(f"Test macro acc:   {test_metrics['macro_accuracy']:.4f}")

    if "count_exact" in test_metrics:
        print(f"Test count exact: {test_metrics['count_exact']['accuracy']:.4f}")

    print(f"Total time:       {total_timer.elapsed_str()}")


if __name__ == "__main__":
    main()