#!/usr/bin/env python3
"""
05_train_teacher.py

Train the TinyDisasterVQA teacher model.

Default:
  ConvNeXt-Tiny or Swin-Tiny image encoder
  LSTM question encoder
  19-class edge_global classifier

Run from repo root:

PYTHONPATH=src python scripts/05_train_teacher.py

Useful smoke test:

PYTHONPATH=src python scripts/05_train_teacher.py \
  --epochs 1 \
  --batch-size 8 \
  --num-workers 0 \
  --run-name smoke_teacher

Useful overfit test:

PYTHONPATH=src python scripts/05_train_teacher.py \
  --epochs 30 \
  --batch-size 8 \
  --num-workers 0 \
  --overfit-samples 32 \
  --run-name overfit_teacher_32
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Allow running without manually setting PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tinydisastervqa.data import (  # noqa: E402
    FloodNetVQADataset,
    get_image_transform,
    load_json,
)
from tinydisastervqa.metrics import (  # noqa: E402
    ClassificationMetrics,
    evaluate_classifier,
    format_metrics,
)
from tinydisastervqa.models import (  # noqa: E402
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

    # Data.
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Optional separate batch size for valid/test. Defaults to --batch-size.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--augment-train", action="store_true", default=True)
    parser.add_argument("--no-augment-train", action="store_false", dest="augment_train")
    parser.add_argument("--overfit-samples", type=int, default=0)

    # Model.
    parser.add_argument(
        "--backbone",
        type=str,
        default="convnext_tiny",
        choices=[
            "convnext_tiny",
            "swin_tiny",
            "efficientnet_b0",
            "efficientnet_b1",
            "resnet18",
            "resnet50",
        ],
    )
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    parser.add_argument("--freeze-image-encoder", action="store_true", default=False)
    parser.add_argument("--question-embed-dim", type=int, default=128)
    parser.add_argument("--question-hidden-dim", type=int, default=256)
    parser.add_argument("--fusion-hidden-dim", type=int, default=512)
    parser.add_argument("--fusion-dropout", type=float, default=0.3)
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)

    # Optimization.
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--use-class-weights", action="store_true", default=False)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")

    # Logging/checkpointing.
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-every-epoch", action="store_true", default=False)

    return parser.parse_args()


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

    # Safe speedup for Colab/WSL when using worker processes.
    # DataLoader rejects persistent_workers/prefetch_factor when num_workers == 0.
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


def build_criterion(args: argparse.Namespace, device: torch.device) -> nn.Module:
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


def autocast_context(device: torch.device, enabled: bool):
    return torch.amp.autocast(
        device_type=device.type,
        enabled=(enabled and device.type == "cuda"),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model.train()

    loss_meter = AverageMeter("train_loss")
    metrics = ClassificationMetrics(num_classes=args.num_classes)
    timer = Timer()

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        question_tokens = batch["question_tokens"].to(device, non_blocking=True)
        question_lengths = batch["question_length"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, args.amp):
            logits = model(
                images=images,
                question_tokens=question_tokens,
                question_lengths=question_lengths,
            )
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        loss_meter.update(float(loss.item()), n=batch_size)

        metrics.update(
            logits=logits.detach(),
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
                f"acc={current['accuracy']:.4f} | "
                f"{head_str} | "
                f"time={timer.elapsed_str()}"
            )

    result = metrics.compute()
    result["loss"] = loss_meter.avg

    return result


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    device = get_device(args.device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    class_weight_tag = "_cw" if args.use_class_weights else ""
    aug_tag = "" if args.augment_train else "_noaug"
    pretrained_tag = "" if args.pretrained else "_scratch"
    run_prefix = (
        f"teacher_{args.backbone}_{args.image_size}"
        f"{class_weight_tag}{aug_tag}{pretrained_tag}"
    )

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
    print("TinyDisasterVQA / Train Teacher")
    print("=" * 80)
    print(f"Run dir:       {run_dir}")
    print(f"Device:        {device}")
    print(f"Backbone:      {args.backbone}")
    print(f"Pretrained:    {args.pretrained}")
    print(f"AMP:           {args.amp}")
    print(f"Image size:    {args.image_size}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Eval batch:    {args.eval_batch_size or args.batch_size}")
    print(f"Augment train: {args.augment_train}")
    print(f"Epochs:        {args.epochs}")
    print(f"LR:            {args.lr}")
    print(f"Class weights: {args.use_class_weights}")
    print(f"Freeze image:  {args.freeze_image_encoder}")
    print()

    metadata = load_json(args.metadata)

    loaders = build_loaders(args)

    model = build_teacher_from_metadata(
        metadata=metadata,
        image_backbone=args.backbone,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        freeze_image_encoder=args.freeze_image_encoder,
        question_embed_dim=args.question_embed_dim,
        question_hidden_dim=args.question_hidden_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        fusion_dropout=args.fusion_dropout,
    ).to(device)

    print(describe_model(model))
    print()

    criterion = build_criterion(args, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
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
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            args=args,
        )

        valid_metrics = evaluate_classifier(
            model=model,
            dataloader=loaders["valid"],
            device=device,
            num_classes=args.num_classes,
        )

        scheduler.step()

        valid_acc = float(valid_metrics["overall"]["accuracy"])
        train_acc = float(train_metrics["overall"]["accuracy"])

        completed_epoch = epoch

        improved = valid_acc > (best_valid_acc + args.early_stopping_min_delta)

        if improved:
            best_valid_acc = valid_acc
            best_epoch = epoch

            save_checkpoint(
                path=checkpoints_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics={
                    "train": train_metrics,
                    "valid": valid_metrics,
                    "best_valid_acc": best_valid_acc,
                },
                config=config,
            )

        if improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if args.save_every_epoch:
            save_checkpoint(
                path=checkpoints_dir / f"epoch_{epoch:03d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics={
                    "train": train_metrics,
                    "valid": valid_metrics,
                    "best_valid_acc": best_valid_acc,
                },
                config=config,
            )

        epoch_record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_acc": train_acc,
            "valid_loss": valid_metrics["loss"],
            "valid_acc": valid_acc,
            "best_valid_acc": best_valid_acc,
            "best_epoch": best_epoch,
            "epoch_time_sec": time.time() - epoch_start,
            "epochs_without_improvement": epochs_without_improvement,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
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
            f"{'IMPROVED' if improved else 'no improvement'}"
        )

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print()
            print("=" * 80)
            print(
                f"Early stopping triggered after {epochs_without_improvement} "
                f"epochs without validation improvement."
            )
            print(f"Best valid acc: {best_valid_acc:.4f} at epoch {best_epoch}")
            print("=" * 80)
            break

    print()
    print("=" * 80)
    print("Evaluating best checkpoint on test set")
    print("=" * 80)

    best_ckpt = torch.load(checkpoints_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_metrics = evaluate_classifier(
        model=model,
        dataloader=loaders["test"],
        device=device,
        num_classes=args.num_classes,
    )

    save_json(test_metrics, run_dir / "test_metrics.json")

    save_checkpoint(
        path=checkpoints_dir / "last.pt",
        model=model,
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

    print()
    print(format_metrics(test_metrics, prefix="test"))

    print()
    print("=" * 80)
    print("Training complete")
    print("=" * 80)
    print(f"Run dir:          {run_dir}")
    print(f"Best valid acc:   {best_valid_acc:.4f}")
    print(f"Best epoch:       {best_epoch}")
    print(f"Test acc:         {test_metrics['overall']['accuracy']:.4f}")
    print(f"Total time:       {total_timer.elapsed_str()}")


if __name__ == "__main__":
    main()