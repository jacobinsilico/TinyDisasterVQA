#!/usr/bin/env python3
"""
Train MobileNet teacher for object+color COCO-QA.

Task:
  image + question + question type -> object/color answer prediction

Recommended run:
  python scripts/11_train_mobilenet_teacher.py \
    --model-name mobilenet_v3_large \
    --head-type separate \
    --epochs 10 \
    --batch-size 128 \
    --lr 3e-4 \
    --num-workers 2 \
    --checkpoint-dir /content/drive/MyDrive/edge-vlm-gap9-runs/mobilenet_v3_large_teacher_oc

MobileNetV2 fallback:
  python scripts/11_train_mobilenet_teacher.py \
    --model-name mobilenet_v2 \
    --head-type separate \
    --epochs 10 \
    --batch-size 128 \
    --lr 3e-4 \
    --num-workers 2 \
    --checkpoint-dir /content/drive/MyDrive/edge-vlm-gap9-runs/mobilenet_v2_teacher_oc
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
from src.model import build_baseline_vqa_model, count_parameters
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


def id_to_answer_from_vocab(answer_vocab: dict) -> dict[int, str]:
    return {int(k): v for k, v in answer_vocab["id_to_answer"].items()}


def build_answer_ids_from_vocab(answer_vocab: dict) -> dict[str, list[int]]:
    answer_to_id = answer_vocab["answer_to_id"]

    object_ids = [
        int(answer_to_id[ans])
        for ans in answer_vocab["object_answers"]
        if ans in answer_to_id
    ]

    color_ids = [
        int(answer_to_id[ans])
        for ans in answer_vocab["color_answers"]
        if ans in answer_to_id
    ]

    if not object_ids:
        raise ValueError("No object answer IDs found.")
    if not color_ids:
        raise ValueError("No color answer IDs found.")

    return {
        "object": sorted(object_ids),
        "color": sorted(color_ids),
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


def move_batch_to_device(batch: dict, device: torch.device) -> tuple:
    images = batch["image"].to(device, non_blocking=True)
    question_ids = batch["question_ids"].to(device, non_blocking=True)
    question_len = batch["question_len"].to(device, non_blocking=True)
    answer_id = batch["answer_id"].to(device, non_blocking=True)
    type_id = batch["type_id"].to(device, non_blocking=True)

    return images, question_ids, question_len, answer_id, type_id


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_interval: int,
    use_amp: bool,
) -> dict:
    model.train()

    loss_meter = AverageMeter()
    acc_tracker = AccuracyTracker()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_time = time.time()

    for step, batch in enumerate(loader, start=1):
        images, question_ids, question_len, answer_id, type_id = move_batch_to_device(
            batch, device
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(
                images=images,
                question_ids=question_ids,
                question_len=question_len,
                type_id=type_id,
            )
            loss = criterion(logits, answer_id)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.shape[0]

        loss_meter.update(loss.item(), n=batch_size)
        acc_tracker.update(logits.detach(), answer_id, type_id)

        if log_interval > 0 and step % log_interval == 0:
            partial = acc_tracker.compute()
            partial["loss"] = loss_meter.avg
            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch:03d} | step {step:05d}/{len(loader):05d} | "
                f"{format_metrics(partial, prefix='train')} | "
                f"time={elapsed:.1f}s"
            )

    metrics = acc_tracker.compute()
    metrics["loss"] = loss_meter.avg
    metrics["time_sec"] = time.time() - start_time

    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    id_to_answer: dict[int, str] | None = None,
) -> tuple[dict, list[tuple[str, str, int]]]:
    model.eval()

    loss_meter = AverageMeter()
    acc_tracker = AccuracyTracker()
    confusion_tracker = ConfusionTracker(id_to_answer) if id_to_answer is not None else None

    start_time = time.time()

    for batch in loader:
        images, question_ids, question_len, answer_id, type_id = move_batch_to_device(
            batch, device
        )

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(
                images=images,
                question_ids=question_ids,
                question_len=question_len,
                type_id=type_id,
            )
            loss = criterion(logits, answer_id)

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
    model: nn.Module,
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
            "model_state_dict": model.state_dict(),
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
    model: nn.Module,
    question_vocab: QuestionVocab,
    answer_vocab: dict,
    answer_ids_by_type: dict[str, list[int]],
) -> dict:
    config = make_json_serializable(vars(args))

    config["device"] = str(device)
    config["use_amp"] = bool(use_amp)
    config["model_family"] = "mobilenet_teacher"
    config["num_model_parameters_trainable"] = int(
        count_parameters(model, trainable_only=True)
    )
    config["num_model_parameters_total"] = int(
        count_parameters(model, trainable_only=False)
    )
    config["question_vocab_size"] = int(question_vocab.size)
    config["num_answer_classes"] = int(len(answer_vocab["id_to_answer"]))
    config["num_object_answers"] = int(len(answer_vocab["object_answers"]))
    config["num_color_answers"] = int(len(answer_vocab["color_answers"]))
    config["answer_ids_by_type"] = make_json_serializable(answer_ids_by_type)

    return config


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/mobilenet_teacher_oc"),
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="mobilenet_v3_large",
        choices=["mobilenet_v2", "mobilenet_v3_large", "convnext_tiny"],
    )
    parser.add_argument(
        "--head-type",
        type=str,
        default="separate",
        choices=["shared", "separate"],
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained weights.",
    )
    parser.add_argument(
        "--freeze-image-encoder",
        action="store_true",
        help="Freeze image backbone.",
    )

    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
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
        default=5,
        help="Early stopping patience based on val accuracy. Use 0 to disable.",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)
    pretrained = not args.no_pretrained

    print(f"Device: {device}")
    print(f"AMP: {use_amp}")
    print(f"Teacher model: {args.model_name}")
    print(f"Head type: {args.head_type}")
    print(f"Pretrained: {pretrained}")
    print(f"Freeze image encoder: {args.freeze_image_encoder}")

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

    train_dataset = CocoQADataset(
        manifest_path=train_manifest,
        question_vocab_path=question_vocab_path,
        answer_vocab_path=answer_vocab_path,
        image_size=args.image_size,
        train=(not args.no_augment),
        repo_root=REPO_ROOT,
        limit=args.train_limit,
    )

    val_dataset = CocoQADataset(
        manifest_path=val_manifest,
        question_vocab_path=question_vocab_path,
        answer_vocab_path=answer_vocab_path,
        image_size=args.image_size,
        train=False,
        repo_root=REPO_ROOT,
        limit=args.val_limit,
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
    print("Building teacher model...")

    model = build_baseline_vqa_model(
        vocab_size=question_vocab.size,
        num_answers=num_answers,
        pad_id=question_vocab.pad_id,
        model_name=args.model_name,
        pretrained=pretrained,
        freeze_image_encoder=args.freeze_image_encoder,
        head_type=args.head_type,
        object_answer_ids=answer_ids_by_type["object"],
        color_answer_ids=answer_ids_by_type["color"],
        number_answer_ids=None,
    ).to(device)

    print(f"Trainable parameters: {count_parameters(model, trainable_only=True):,}")
    print(f"Total parameters:     {count_parameters(model, trainable_only=False):,}")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
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
        model=model,
        question_vocab=question_vocab,
        answer_vocab=answer_vocab,
        answer_ids_by_type=answer_ids_by_type,
    )

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print()
    print("Starting teacher training...")
    print(f"Checkpoints: {args.checkpoint_dir}")

    best_val_acc = 0.0
    best_epoch = 0
    best_confusions = []
    epochs_without_improvement = 0

    total_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            log_interval=args.log_interval,
            use_amp=use_amp,
        )

        val_metrics, val_confusions = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
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
                model=model,
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
            model=model,
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
            "model_name": args.model_name,
            "head_type": args.head_type,
            "pretrained": pretrained,
            "freeze_image_encoder": args.freeze_image_encoder,
            "train_loss": train_metrics["loss"],
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
        }

        append_log_csv(log_csv_path, row)

        print()
        print("=" * 100)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | lr={current_lr:.2e} | "
            f"{'BEST' if improved else 'no improvement'}"
        )
        print(format_metrics(train_metrics, prefix="train"))
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
    print("Teacher training finished.")
    print(f"Total time: {total_time / 60.0:.1f} min")
    print(f"Best val acc: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"Last checkpoint: {last_ckpt_path}")
    print(f"Log CSV: {log_csv_path}")
    print()
    print_top_confusions(best_confusions, title="Best val confusions:")


if __name__ == "__main__":
    main()