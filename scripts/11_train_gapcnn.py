#!/usr/bin/env python3
"""
Train GAPCNN-S student for object+color COCO-QA.

Task:
  image + question type -> object/color answer prediction

Main modes:

1) Supervised GAPCNN-S:
  python scripts/11_train_gapcnn.py \
    --epochs 10 \
    --batch-size 128 \
    --lr 1e-3 \
    --checkpoint-dir checkpoints/gapcnn_s_supervised

2) KD GAPCNN-S from compatible object+color MobileNetV2 teacher:
  python scripts/11_train_gapcnn.py \
    --epochs 10 \
    --batch-size 128 \
    --lr 1e-3 \
    --teacher-checkpoint checkpoints/object_color_mobilenet_teacher/best.pt \
    --kd-alpha 0.4 \
    --kd-temperature 3 \
    --checkpoint-dir checkpoints/gapcnn_s_kd

Important:
  The teacher checkpoint must be trained on the SAME object+color answer_vocab.json.
  Do not use the old object+color+number 70-class teacher unless you explicitly build
  a mapping layer first.
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset import CocoQADataset
from src.metrics import (
    AccuracyTracker,
    AverageMeter,
    ConfusionTracker,
    format_metrics,
    print_top_confusions,
)
from src.model import (
    build_baseline_vqa_model,
    build_gapcnn_s_vqa_model,
    count_parameters,
    estimate_int8_weight_size_bytes,
)
from src.text import QuestionVocab


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_answer_vocab(path: Path) -> dict:
    data = read_json(path)

    required_keys = ["answer_to_id", "id_to_answer", "object_answers", "color_answers"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"answer_vocab.json missing required key: {key}")

    return data


def build_answer_ids_from_vocab(answer_vocab: dict) -> dict[str, list[int]]:
    answer_to_id = answer_vocab["answer_to_id"]

    object_answer_ids = [
        int(answer_to_id[ans])
        for ans in answer_vocab["object_answers"]
        if ans in answer_to_id
    ]

    color_answer_ids = [
        int(answer_to_id[ans])
        for ans in answer_vocab["color_answers"]
        if ans in answer_to_id
    ]

    if not object_answer_ids:
        raise ValueError("No object answer IDs found in answer_vocab.json.")
    if not color_answer_ids:
        raise ValueError("No color answer IDs found in answer_vocab.json.")

    return {
        "object": sorted(object_answer_ids),
        "color": sorted(color_answer_ids),
    }


def id_to_answer_from_vocab(answer_vocab: dict) -> dict[int, str]:
    return {
        int(k): v
        for k, v in answer_vocab["id_to_answer"].items()
    }


def make_json_serializable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [make_json_serializable(v) for v in obj]

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    return obj


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {
        "images": batch["image"].to(device, non_blocking=True),
        "question_ids": batch["question_ids"].to(device, non_blocking=True),
        "question_len": batch["question_len"].to(device, non_blocking=True),
        "answer_id": batch["answer_id"].to(device, non_blocking=True),
        "type_id": batch["type_id"].to(device, non_blocking=True),
        "type_onehot": batch["type_onehot"].to(device, non_blocking=True),
    }

    if "teacher_image" in batch:
        out["teacher_images"] = batch["teacher_image"].to(device, non_blocking=True)
    else:
        out["teacher_images"] = out["images"]

    return out


def build_teacher_model(
    teacher_checkpoint_path: Path,
    question_vocab: QuestionVocab,
    num_answers: int,
    answer_ids_by_type: dict[str, list[int]],
    device: torch.device,
) -> nn.Module:
    if not teacher_checkpoint_path.exists():
        raise FileNotFoundError(f"Missing teacher checkpoint: {teacher_checkpoint_path}")

    print(f"Loading teacher checkpoint: {teacher_checkpoint_path}")
    ckpt = torch.load(teacher_checkpoint_path, map_location="cpu")

    config = ckpt.get("config", {})
    teacher_model_name = config.get("model_name", "mobilenet_v2")
    teacher_head_type = config.get("head_type", "separate")

    checkpoint_num_answers = config.get("num_answer_classes", num_answers)
    if int(checkpoint_num_answers) != int(num_answers):
        raise ValueError(
            "Teacher checkpoint answer vocab size does not match current dataset.\n"
            f"  teacher num answers: {checkpoint_num_answers}\n"
            f"  current num answers: {num_answers}\n"
            "Train a new object+color teacher first, or implement explicit vocab mapping."
        )

    print("Teacher config:")
    print(f"  model_name: {teacher_model_name}")
    print(f"  head_type:  {teacher_head_type}")

    teacher = build_baseline_vqa_model(
        vocab_size=question_vocab.size,
        num_answers=num_answers,
        pad_id=question_vocab.pad_id,
        model_name=teacher_model_name,
        pretrained=False,
        freeze_image_encoder=False,
        head_type=teacher_head_type,
        object_answer_ids=answer_ids_by_type["object"],
        color_answer_ids=answer_ids_by_type["color"],
        number_answer_ids=None,
    )

    missing, unexpected = teacher.load_state_dict(
        ckpt["model_state_dict"],
        strict=False,
    )

    if missing:
        print("WARNING: Missing teacher keys:")
        for key in missing[:20]:
            print(f"  {key}")
        if len(missing) > 20:
            print(f"  ... {len(missing) - 20} more")

    if unexpected:
        print("WARNING: Unexpected teacher keys:")
        for key in unexpected[:20]:
            print(f"  {key}")
        if len(unexpected) > 20:
            print(f"  ... {len(unexpected) - 20} more")

    teacher.to(device)
    teacher.eval()

    for param in teacher.parameters():
        param.requires_grad = False

    print(
        f"Teacher parameters: {count_parameters(teacher, trainable_only=False):,}"
    )

    return teacher


def compute_supervised_kd_loss(
    student_logits: torch.Tensor,
    targets: torch.Tensor,
    ce_criterion: nn.Module,
    teacher_logits: torch.Tensor | None = None,
    kd_alpha: float = 0.0,
    kd_temperature: float = 3.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    ce_loss = ce_criterion(student_logits, targets)

    if teacher_logits is None or kd_alpha <= 0.0:
        return ce_loss, {
            "loss_ce": float(ce_loss.detach().item()),
            "loss_kd": 0.0,
            "loss_total": float(ce_loss.detach().item()),
        }

    if kd_temperature <= 0.0:
        raise ValueError("kd_temperature must be > 0.")

    temperature = float(kd_temperature)

    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)

    kd_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    ) * (temperature * temperature)

    total_loss = (1.0 - kd_alpha) * ce_loss + kd_alpha * kd_loss

    return total_loss, {
        "loss_ce": float(ce_loss.detach().item()),
        "loss_kd": float(kd_loss.detach().item()),
        "loss_total": float(total_loss.detach().item()),
    }


def train_one_epoch(
    student: nn.Module,
    teacher: nn.Module | None,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ce_criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_interval: int,
    use_amp: bool,
    kd_alpha: float,
    kd_temperature: float,
) -> dict:
    student.train()
    if teacher is not None:
        teacher.eval()

    loss_meter = AverageMeter()
    ce_meter = AverageMeter()
    kd_meter = AverageMeter()
    acc_tracker = AccuracyTracker()

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_time = time.time()

    for step, batch in enumerate(loader, start=1):
        batch_dev = move_batch_to_device(batch, device)

        images = batch_dev["images"]
        question_ids = batch_dev["question_ids"]
        question_len = batch_dev["question_len"]
        answer_id = batch_dev["answer_id"]
        type_id = batch_dev["type_id"]
        type_onehot = batch_dev["type_onehot"]

        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            teacher_logits = None
            if teacher is not None:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    teacher_logits = teacher(
                        images=batch_dev["teacher_images"],
                        question_ids=question_ids,
                        question_len=question_len,
                        type_id=type_id,
                    )

        with torch.amp.autocast("cuda", enabled=use_amp):
            student_logits = student(
                images=images,
                type_onehot=type_onehot,
                type_id=type_id,
            )

            loss, loss_parts = compute_supervised_kd_loss(
                student_logits=student_logits,
                targets=answer_id,
                ce_criterion=ce_criterion,
                teacher_logits=teacher_logits,
                kd_alpha=kd_alpha,
                kd_temperature=kd_temperature,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.shape[0]

        loss_meter.update(loss_parts["loss_total"], n=batch_size)
        ce_meter.update(loss_parts["loss_ce"], n=batch_size)
        kd_meter.update(loss_parts["loss_kd"], n=batch_size)

        acc_tracker.update(student_logits.detach(), answer_id, type_id)

        if log_interval > 0 and step % log_interval == 0:
            partial_metrics = acc_tracker.compute()
            partial_metrics["loss"] = loss_meter.avg
            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch:03d} | step {step:05d}/{len(loader):05d} | "
                f"{format_metrics(partial_metrics, prefix='train')} | "
                f"ce={ce_meter.avg:.4f} | kd={kd_meter.avg:.4f} | "
                f"time={elapsed:.1f}s"
            )

    metrics = acc_tracker.compute()
    metrics["loss"] = loss_meter.avg
    metrics["loss_ce"] = ce_meter.avg
    metrics["loss_kd"] = kd_meter.avg
    metrics["time_sec"] = time.time() - start_time

    return metrics


@torch.no_grad()
def evaluate(
    student: nn.Module,
    loader: DataLoader,
    ce_criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    id_to_answer: dict[int, str] | None = None,
) -> tuple[dict, list[tuple[str, str, int]]]:
    student.eval()

    loss_meter = AverageMeter()
    acc_tracker = AccuracyTracker()
    confusion_tracker = ConfusionTracker(id_to_answer) if id_to_answer is not None else None

    start_time = time.time()

    for batch in loader:
        batch_dev = move_batch_to_device(batch, device)

        images = batch_dev["images"]
        answer_id = batch_dev["answer_id"]
        type_id = batch_dev["type_id"]
        type_onehot = batch_dev["type_onehot"]

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = student(
                images=images,
                type_onehot=type_onehot,
                type_id=type_id,
            )
            loss = ce_criterion(logits, answer_id)

        batch_size = images.shape[0]

        loss_meter.update(loss.item(), n=batch_size)
        acc_tracker.update(logits, answer_id, type_id)

        if confusion_tracker is not None:
            confusion_tracker.update(logits, answer_id)

    metrics = acc_tracker.compute()
    metrics["loss"] = loss_meter.avg
    metrics["time_sec"] = time.time() - start_time

    confusions = confusion_tracker.topk(20) if confusion_tracker is not None else []

    return metrics, confusions


def save_checkpoint(
    path: Path,
    student: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_acc: float,
    config: dict,
    train_metrics: dict,
    val_metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "config": config,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        },
        path,
    )


def append_log_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def build_config(
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
    student: nn.Module,
    question_vocab: QuestionVocab,
    answer_vocab: dict,
    answer_ids_by_type: dict[str, list[int]],
    teacher: nn.Module | None,
) -> dict:
    config = make_json_serializable(vars(args))

    config["device"] = str(device)
    config["use_amp"] = bool(use_amp)
    config["model_family"] = "gapcnn_s"
    config["num_model_parameters_trainable"] = int(
        count_parameters(student, trainable_only=True)
    )
    config["num_model_parameters_total"] = int(
        count_parameters(student, trainable_only=False)
    )
    config["estimated_int8_weight_size_bytes"] = int(
        estimate_int8_weight_size_bytes(student, trainable_only=False)
    )
    config["estimated_int8_weight_size_mb"] = (
        config["estimated_int8_weight_size_bytes"] / (1024 * 1024)
    )
    config["question_vocab_size"] = int(question_vocab.size)
    config["num_answer_classes"] = int(len(answer_vocab["id_to_answer"]))
    config["num_object_answers"] = int(len(answer_vocab["object_answers"]))
    config["num_color_answers"] = int(len(answer_vocab["color_answers"]))
    config["answer_ids_by_type"] = make_json_serializable(answer_ids_by_type)
    config["uses_teacher"] = teacher is not None

    return config


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/gapcnn_s"),
    )

    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)

    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-augment", action="store_true")

    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Early stopping patience based on val accuracy. Use 0 to disable.",
    )

    # GAPCNN-S architecture knobs.
    parser.add_argument("--image-feature-dim", type=int, default=160)
    parser.add_argument("--type-feature-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.1)

    # KD options.
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=None,
        help="Optional compatible object+color teacher checkpoint.",
    )
    parser.add_argument(
        "--kd-alpha",
        type=float,
        default=0.0,
        help="Weight on KD loss. 0 disables KD even if teacher is provided.",
    )
    parser.add_argument(
        "--kd-temperature",
        type=float,
        default=3.0,
        help="KD softmax temperature.",
    )

    parser.add_argument(
        "--teacher-image-size",
        type=int,
        default=0,
        help="If >0 and teacher is used, provide teacher_image at this size.",
    )

    args = parser.parse_args()

    if args.kd_alpha < 0.0 or args.kd_alpha > 1.0:
        raise ValueError("--kd-alpha must be in [0, 1].")

    if args.teacher_checkpoint is None and args.kd_alpha > 0.0:
        raise ValueError("--kd-alpha > 0 requires --teacher-checkpoint.")

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)

    print(f"Device: {device}")
    print(f"AMP: {use_amp}")
    print("Model: GAPCNN-S object+color")
    print(f"KD alpha: {args.kd_alpha}")
    print(f"KD temperature: {args.kd_temperature}")

    train_manifest = args.processed_dir / "cocoqa_train_resolved.jsonl"
    val_manifest = args.processed_dir / "cocoqa_val_resolved.jsonl"
    question_vocab_path = args.processed_dir / "question_vocab.json"
    answer_vocab_path = args.processed_dir / "answer_vocab.json"

    question_vocab = QuestionVocab.load(question_vocab_path)
    answer_vocab = load_answer_vocab(answer_vocab_path)
    id_to_answer = id_to_answer_from_vocab(answer_vocab)
    answer_ids_by_type = build_answer_ids_from_vocab(answer_vocab)

    num_answers = len(id_to_answer)

    print()
    print("Answer IDs by type:")
    print(f"  object: {len(answer_ids_by_type['object'])}")
    print(f"  color:  {len(answer_ids_by_type['color'])}")
    print(f"  total:  {num_answers}")

    print()
    print("Building datasets...")

    teacher_image_size = args.teacher_image_size if args.teacher_image_size > 0 else None

    train_dataset = CocoQADataset(
        manifest_path=train_manifest,
        question_vocab_path=question_vocab_path,
        answer_vocab_path=answer_vocab_path,
        image_size=args.image_size,
        train=(not args.no_augment),
        repo_root=REPO_ROOT,
        limit=args.train_limit,
        teacher_image_size=teacher_image_size,
    )

    val_dataset = CocoQADataset(
        manifest_path=val_manifest,
        question_vocab_path=question_vocab_path,
        answer_vocab_path=answer_vocab_path,
        image_size=args.image_size,
        train=False,
        repo_root=REPO_ROOT,
        limit=args.val_limit,
        teacher_image_size=teacher_image_size,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Question vocab size: {question_vocab.size}")
    print(f"Answer classes: {num_answers}")

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    print()
    print("Building student model...")

    student = build_gapcnn_s_vqa_model(
        num_answers=num_answers,
        object_answer_ids=answer_ids_by_type["object"],
        color_answer_ids=answer_ids_by_type["color"],
        image_feature_dim=args.image_feature_dim,
        type_feature_dim=args.type_feature_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    trainable_params = count_parameters(student, trainable_only=True)
    total_params = count_parameters(student, trainable_only=False)
    int8_bytes = estimate_int8_weight_size_bytes(student, trainable_only=False)

    print(f"Trainable parameters:       {trainable_params:,}")
    print(f"Total parameters:           {total_params:,}")
    print(f"Estimated INT8 weight size: {int8_bytes / (1024 * 1024):.3f} MB")

    teacher = None
    if args.teacher_checkpoint is not None:
        teacher = build_teacher_model(
            teacher_checkpoint_path=args.teacher_checkpoint,
            question_vocab=question_vocab,
            num_answers=num_answers,
            answer_ids_by_type=answer_ids_by_type,
            device=device,
        )

    ce_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = args.checkpoint_dir / "best.pt"
    last_ckpt_path = args.checkpoint_dir / "last.pt"
    log_csv_path = args.checkpoint_dir / "train_log.csv"
    config_path = args.checkpoint_dir / "config.json"

    config = build_config(
        args=args,
        device=device,
        use_amp=use_amp,
        student=student,
        question_vocab=question_vocab,
        answer_vocab=answer_vocab,
        answer_ids_by_type=answer_ids_by_type,
        teacher=teacher,
    )

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print()
    print("Starting training...")
    print(f"Checkpoints: {args.checkpoint_dir}")

    best_val_acc = 0.0
    best_epoch = 0
    best_confusions = []
    epochs_without_improvement = 0

    total_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            student=student,
            teacher=teacher,
            loader=train_loader,
            optimizer=optimizer,
            ce_criterion=ce_criterion,
            device=device,
            epoch=epoch,
            log_interval=args.log_interval,
            use_amp=use_amp,
            kd_alpha=args.kd_alpha,
            kd_temperature=args.kd_temperature,
        )

        val_metrics, val_confusions = evaluate(
            student=student,
            loader=val_loader,
            ce_criterion=ce_criterion,
            device=device,
            use_amp=use_amp,
            id_to_answer=id_to_answer,
        )

        val_acc = val_metrics["accuracy"]
        scheduler.step(val_acc)

        improved = val_acc > best_val_acc

        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            best_confusions = val_confusions
            epochs_without_improvement = 0

            save_checkpoint(
                path=best_ckpt_path,
                student=student,
                optimizer=optimizer,
                epoch=epoch,
                best_val_acc=best_val_acc,
                config=config,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            path=last_ckpt_path,
            student=student,
            optimizer=optimizer,
            epoch=epoch,
            best_val_acc=best_val_acc,
            config=config,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_metrics["loss"],
            "train_loss_ce": train_metrics["loss_ce"],
            "train_loss_kd": train_metrics["loss_kd"],
            "train_acc": train_metrics["accuracy"],
            "train_acc_object": train_metrics["accuracy_object"],
            "train_acc_color": train_metrics["accuracy_color"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_acc_object": val_metrics["accuracy_object"],
            "val_acc_color": val_metrics["accuracy_color"],
            "train_time_sec": train_metrics["time_sec"],
            "val_time_sec": val_metrics["time_sec"],
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "kd_alpha": args.kd_alpha,
            "kd_temperature": args.kd_temperature,
            "uses_teacher": teacher is not None,
            "params_total": total_params,
            "int8_weight_size_mb": int8_bytes / (1024 * 1024),
        }

        append_log_csv(log_csv_path, row)

        print()
        print("=" * 100)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | lr={current_lr:.2e} | "
            f"{'BEST' if improved else 'no improvement'}"
        )
        print(format_metrics(train_metrics, prefix="train"))
        print(
            f"train ce={train_metrics['loss_ce']:.4f} | "
            f"kd={train_metrics['loss_kd']:.4f}"
        )
        print(format_metrics(val_metrics, prefix="val"))
        print(f"best_val_acc={best_val_acc:.4f} at epoch {best_epoch}")
        print_top_confusions(val_confusions[:10], title="Val top confusions:")
        print("=" * 100)
        print()

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(
                f"Early stopping: no val improvement for "
                f"{epochs_without_improvement} epochs."
            )
            break

    total_time = time.time() - total_start_time

    print()
    print("Training finished.")
    print(f"Total time: {total_time / 60.0:.1f} min")
    print(f"Best val acc: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"Last checkpoint: {last_ckpt_path}")
    print(f"Log CSV: {log_csv_path}")
    print()
    print_top_confusions(best_confusions, title="Best val confusions:")


if __name__ == "__main__":
    main()