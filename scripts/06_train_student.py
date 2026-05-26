#!/usr/bin/env python3
"""
06_train_student.py

Train the TDM-S student model.

Modes:
  1. CE only:
     student learns from hard edge_global labels.

  2. KD:
     student learns from hard labels + soft teacher logits.

Run CE-only:

PYTHONPATH=src python scripts/06_train_student.py \
  --mode ce \
  --epochs 50 \
  --batch-size 64 \
  --run-name tdm_s_ce

Run KD:

PYTHONPATH=src python scripts/06_train_student.py \
  --mode kd \
  --teacher-checkpoint /content/drive/MyDrive/TinyDisasterVQA/runs/convnext_tiny_teacher_edge_global/checkpoints/best.pt \
  --epochs 50 \
  --batch-size 64 \
  --run-name tdm_s_kd

Default early stopping:
  stop after 5 epochs without valid accuracy improvement.
"""

from __future__ import annotations

import argparse
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
    build_tdm_s_from_metadata,
    build_tdm_m_from_metadata,
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
    parser.add_argument("--train-csv", type=Path, default=Path("outputs/training_data/train.csv"))
    parser.add_argument("--valid-csv", type=Path, default=Path("outputs/training_data/valid.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("outputs/training_data/test.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("outputs/training_data/metadata.json"))
    parser.add_argument(
        "--class-weights",
        type=Path,
        default=Path("outputs/answer_space/class_weights_edge_global_by_label.json"),
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))

    # Run.
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    # Mode.
    parser.add_argument("--mode", type=str, default="ce", choices=["ce", "kd"])
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)

    # Data.
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--augment-train", action="store_true", default=True)
    parser.add_argument("--no-augment-train", action="store_false", dest="augment_train")
    parser.add_argument("--overfit-samples", type=int, default=0)

    # Student model.
    parser.add_argument("--student-size", type=str, default="s", choices=["s", "m"])
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--num-question-templates", type=int, default=31)
    parser.add_argument("--template-embed-dim", type=int, default=32)
    parser.add_argument("--fusion-hidden-dim", type=int, default=192)
    parser.add_argument("--fusion-dropout", type=float, default=0.1)

    # Teacher model config, must match saved teacher checkpoint.
    parser.add_argument(
        "--teacher-backbone",
        type=str,
        default="convnext_tiny",
        choices=["convnext_tiny", "efficientnet_b0", "efficientnet_b1", "resnet18", "resnet50"],
    )
    parser.add_argument("--teacher-pretrained", action="store_true", default=True)
    parser.add_argument("--teacher-question-embed-dim", type=int, default=128)
    parser.add_argument("--teacher-question-hidden-dim", type=int, default=256)
    parser.add_argument("--teacher-fusion-hidden-dim", type=int, default=512)
    parser.add_argument("--teacher-fusion-dropout", type=float, default=0.3)

    # Optimization.
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--use-class-weights", action="store_true", default=False)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")

    # KD.
    parser.add_argument("--kd-alpha", type=float, default=0.7)
    parser.add_argument("--kd-temperature", type=float, default=4.0)

    # Early stopping.
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)

    # Logging/checkpointing.
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-every-epoch", action="store_true", default=False)

    return parser.parse_args()

def apply_student_defaults(args: argparse.Namespace) -> None:
    """
    Applies default architecture hyperparameters for each student size
    unless explicitly overridden from command line.
    """
    if args.student_size == "s":
        if args.template_embed_dim is None:
            args.template_embed_dim = 32
        if args.fusion_hidden_dim is None:
            args.fusion_hidden_dim = 192
        if args.fusion_dropout is None:
            args.fusion_dropout = 0.1

    elif args.student_size == "m":
        if args.template_embed_dim is None:
            args.template_embed_dim = 64
        if args.fusion_hidden_dim is None:
            args.fusion_hidden_dim = 384
        if args.fusion_dropout is None:
            args.fusion_dropout = 0.15

    else:
        raise ValueError(f"Unknown student size: {args.student_size}")

def get_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    return {
        "train": DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ),
        "valid": DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ),
    }


def build_ce_criterion(args: argparse.Namespace, device: torch.device) -> nn.Module:
    if not args.use_class_weights:
        return nn.CrossEntropyLoss()

    if not args.class_weights.exists():
        raise FileNotFoundError(f"Class weights file not found: {args.class_weights}")

    weights_dict = load_json(args.class_weights)
    weights = torch.ones(args.num_classes, dtype=torch.float32)

    for label_str, weight in weights_dict.items():
        label = int(label_str)
        if 0 <= label < args.num_classes:
            weights[label] = float(weight)

    print("Using class weights:")
    print(weights.tolist())

    return nn.CrossEntropyLoss(weight=weights.to(device))


def build_teacher(args: argparse.Namespace, metadata: dict[str, Any], device: torch.device) -> nn.Module:
    if args.teacher_checkpoint is None:
        raise ValueError("--teacher-checkpoint is required when --mode kd")

    if not args.teacher_checkpoint.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {args.teacher_checkpoint}")

    teacher = build_teacher_from_metadata(
        metadata=metadata,
        image_backbone=args.teacher_backbone,
        pretrained=args.teacher_pretrained,
        num_classes=args.num_classes,
        freeze_image_encoder=False,
        question_embed_dim=args.teacher_question_embed_dim,
        question_hidden_dim=args.teacher_question_hidden_dim,
        fusion_hidden_dim=args.teacher_fusion_hidden_dim,
        fusion_dropout=args.teacher_fusion_dropout,
    ).to(device)

    checkpoint = torch.load(args.teacher_checkpoint, map_location=device)
    teacher.load_state_dict(checkpoint["model_state_dict"], strict=True)
    teacher.eval()

    for param in teacher.parameters():
        param.requires_grad = False

    return teacher


def autocast_context(device: torch.device, enabled: bool):
    return torch.amp.autocast(
        device_type=device.type,
        enabled=(enabled and device.type == "cuda"),
    )


def kd_loss_fn(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    KL(student || teacher) with softened distributions.
    Multiplied by T^2 following standard KD practice.
    """
    t = temperature

    student_log_probs = F.log_softmax(student_logits / t, dim=1)
    teacher_probs = F.softmax(teacher_logits / t, dim=1)

    return F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    ) * (t * t)


@torch.no_grad()
def evaluate_student(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    criterion: nn.Module | None = None,
) -> dict[str, Any]:
    model.eval()

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    meter = ClassificationMetrics(num_classes=num_classes)
    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        question_tokens = batch["question_tokens"].to(device, non_blocking=True)
        question_lengths = batch["question_length"].to(device, non_blocking=True)
        question_template_ids = batch["question_template_id"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        logits = model(
            images=images,
            question_tokens=question_tokens,
            question_lengths=question_lengths,
            question_template_ids=question_template_ids,
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

    metrics = ClassificationMetrics(num_classes=args.num_classes)
    timer = Timer()

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        question_tokens = batch["question_tokens"].to(device, non_blocking=True)
        question_lengths = batch["question_length"].to(device, non_blocking=True)
        question_template_ids = batch["question_template_id"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, args.amp):
            student_logits = student(
                images=images,
                question_tokens=question_tokens,
                question_lengths=question_lengths,
                question_template_ids=question_template_ids,
            )

            ce_loss = ce_criterion(student_logits, targets)

            if args.mode == "kd":
                assert teacher is not None

                with torch.no_grad():
                    teacher_logits = teacher(
                        images=images,
                        question_tokens=question_tokens,
                        question_lengths=question_lengths,
                    )

                kd_loss = kd_loss_fn(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    temperature=args.kd_temperature,
                )

                loss = (1.0 - args.kd_alpha) * ce_loss + args.kd_alpha * kd_loss
            else:
                kd_loss = torch.zeros((), device=device)
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
            current = current_metrics["overall"]
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
                f"acc={current['accuracy']:.4f} | "
                f"{head_str} | "
                f"time={timer.elapsed_str()}"
            )

    result = metrics.compute()
    result["loss"] = loss_meter.avg
    result["ce_loss"] = ce_loss_meter.avg
    result["kd_loss"] = kd_loss_meter.avg

    return result


def main() -> None:
    args = parse_args()
    apply_student_defaults(args)
    set_seed(args.seed)

    device = get_device(args.device)

    if args.mode == "kd" and args.teacher_checkpoint is None:
        raise ValueError("--teacher-checkpoint is required for --mode kd")

    run_prefix = f"tdm_{args.student_size}_{args.mode}"
    run_dir = make_run_dir(
        base_dir=args.runs_dir,
        run_name=args.run_name,
        prefix=run_prefix,
    )

    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    config["run_dir"] = str(run_dir)
    config["device"] = str(device)

    save_json(config, run_dir / "config.json")

    print("=" * 80)
    print("TinyDisasterVQA / Train TDM-S Student")
    print("=" * 80)
    print(f"Run dir:       {run_dir}")
    print(f"Device:        {device}")
    print(f"Student size:  TDM-{args.student_size.upper()}")
    print(f"Mode:          {args.mode}")
    print(f"AMP:           {args.amp}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Epochs:        {args.epochs}")
    print(f"LR:            {args.lr}")
    print(f"Patience:      {args.patience}")
    print(f"KD alpha:      {args.kd_alpha}")
    print(f"KD temp:       {args.kd_temperature}")
    print()

    metadata = load_json(args.metadata)
    loaders = build_loaders(args)

    if args.student_size == "s":
        student = build_tdm_s_from_metadata(
            metadata=metadata,
            num_classes=args.num_classes,
            num_question_templates=args.num_question_templates,
            question_template_embed_dim=args.template_embed_dim,
            fusion_hidden_dim=args.fusion_hidden_dim,
            fusion_dropout=args.fusion_dropout,
        ).to(device)

    elif args.student_size == "m":
        student = build_tdm_m_from_metadata(
            metadata=metadata,
            num_classes=args.num_classes,
            num_question_templates=args.num_question_templates,
            question_template_embed_dim=args.template_embed_dim,
            fusion_hidden_dim=args.fusion_hidden_dim,
            fusion_dropout=args.fusion_dropout,
        ).to(device)

    else:
        raise ValueError(f"Unknown student size: {args.student_size}")

    print(describe_model(student))
    print()

    teacher = None
    if args.mode == "kd":
        print("Loading teacher...")
        teacher = build_teacher(args, metadata, device)
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
        )

        scheduler.step()

        train_acc = float(train_metrics["overall"]["accuracy"])
        valid_acc = float(valid_metrics["overall"]["accuracy"])

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
            "best_valid_acc": best_valid_acc,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "epoch_time_sec": time.time() - epoch_start,
        }

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

        if epochs_without_improvement >= args.patience:
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
    )

    save_json(test_metrics, run_dir / "test_metrics.json")

    save_checkpoint(
        path=checkpoints_dir / "last.pt",
        model=student,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=best_epoch,
        metrics={
            "test": test_metrics,
            "best_valid_acc": best_valid_acc,
            "best_epoch": best_epoch,
        },
        config=config,
    )

    print()
    print(format_metrics(test_metrics, prefix="test"))

    print()
    print("=" * 80)
    print("Student training complete")
    print("=" * 80)
    print(f"Run dir:          {run_dir}")
    print(f"Mode:             {args.mode}")
    print(f"Best valid acc:   {best_valid_acc:.4f}")
    print(f"Best epoch:       {best_epoch}")
    print(f"Test acc:         {test_metrics['overall']['accuracy']:.4f}")
    print(f"Total time:       {total_timer.elapsed_str()}")


if __name__ == "__main__":
    main()