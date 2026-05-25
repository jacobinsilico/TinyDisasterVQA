#!/usr/bin/env python3
"""
Train MobileNet teacher for object+color COCO-QA.

Task:
  image + question + question type -> object/color answer prediction

Recommended run:
  python scripts/12_train_mobilenet_teacher.py \
    --model-name mobilenet_v3_large \
    --head-type separate \
    --epochs 10 \
    --batch-size 128 \
    --lr 3e-4 \
    --num-workers 2 \
    --checkpoint-dir /content/drive/MyDrive/edge-vlm-gap9-runs/mobilenet_v3_large_teacher_oc

MobileNetV2 fallback:
  python scripts/12_train_mobilenet_teacher.py \
    --model-name mobilenet_v2 \
    --head-type separate \
    --epochs 10 \
    --batch-size 128 \
    --lr 3e-4 \
    --num-workers 2 \
    --checkpoint-dir /content/drive/MyDrive/edge-vlm-gap9-runs/mobilenet_v2_teacher_oc
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import (
    CocoQADataset,
    QuestionVocab,
    load_answer_vocab,
    build_answer_ids_from_vocab,
    id_to_answer_from_vocab,
)
from src.utils import set_seed, append_log_csv, format_metrics, count_parameters
from src.models import build_baseline_vqa_model
from src.evaluation import print_top_confusions
from src.training import build_config, train_one_epoch, evaluate_epoch, save_checkpoint


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
        model_family="mobilenet_teacher",
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
            teacher=None,
            loader=train_loader,
            optimizer=optimizer,
            ce_criterion=criterion,
            device=device,
            epoch=epoch,
            log_interval=args.log_interval,
            use_amp=use_amp,
        )

        val_metrics, val_confusions = evaluate_epoch(
            model=model,
            loader=val_loader,
            ce_criterion=criterion,
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