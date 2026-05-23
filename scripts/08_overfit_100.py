#!/usr/bin/env python3
"""
Overfit a tiny subset of COCO-QA to verify that the training stack works.

Goal:
  Train on ~100 samples and reach very high training accuracy.

If this fails, do not proceed to full training yet.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset import CocoQADataset
from src.metrics import AccuracyTracker, AverageMeter, format_metrics
from src.model import build_baseline_vqa_model, count_parameters
from src.text import QuestionVocab


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_answer_vocab(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data["id_to_answer"].items()}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.train()

    loss_meter = AverageMeter()
    acc_tracker = AccuracyTracker()

    for batch in loader:
        images = batch["image"].to(device)
        question_ids = batch["question_ids"].to(device)
        question_len = batch["question_len"].to(device)
        answer_id = batch["answer_id"].to(device)
        type_id = batch["type_id"].to(device)

        logits = model(
            images=images,
            question_ids=question_ids,
            question_len=question_len,
            type_id=type_id,
        )

        loss = criterion(logits, answer_id)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        loss_meter.update(loss.item(), n=batch_size)
        acc_tracker.update(logits.detach(), answer_id, type_id)

    metrics = acc_tracker.compute()
    metrics["loss"] = loss_meter.avg
    return metrics


@torch.no_grad()
def evaluate_on_training_subset(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()

    loss_meter = AverageMeter()
    acc_tracker = AccuracyTracker()

    for batch in loader:
        images = batch["image"].to(device)
        question_ids = batch["question_ids"].to(device)
        question_len = batch["question_len"].to(device)
        answer_id = batch["answer_id"].to(device)
        type_id = batch["type_id"].to(device)

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

    metrics = acc_tracker.compute()
    metrics["loss"] = loss_meter.avg
    return metrics


@torch.no_grad()
def print_predictions(
    model: nn.Module,
    loader: DataLoader,
    id_to_answer: dict[int, str],
    device: torch.device,
    max_items: int = 8,
) -> None:
    model.eval()

    batch = next(iter(loader))

    images = batch["image"].to(device)
    question_ids = batch["question_ids"].to(device)
    question_len = batch["question_len"].to(device)
    answer_id = batch["answer_id"].to(device)
    type_id = batch["type_id"].to(device)

    logits = model(
        images=images,
        question_ids=question_ids,
        question_len=question_len,
        type_id=type_id,
    )

    preds = logits.argmax(dim=1).cpu().tolist()
    targets = answer_id.cpu().tolist()

    print()
    print("Prediction examples:")
    for i in range(min(max_items, len(preds))):
        question = batch["metadata"]["question"][i]
        target_answer = id_to_answer[targets[i]]
        pred_answer = id_to_answer[preds[i]]
        ok = "OK" if pred_answer == target_answer else "WRONG"

        print("-" * 80)
        print(f"Q:      {question}")
        print(f"Target: {target_answer}")
        print(f"Pred:   {pred_answer}")
        print(f"Result: {ok}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--target-acc",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        default=Path("checkpoints/overfit_100.pt"),
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Subset size: {args.subset_size}")

    train_manifest = args.processed_dir / "cocoqa_train_resolved.jsonl"
    question_vocab_path = args.processed_dir / "question_vocab.json"
    answer_vocab_path = args.processed_dir / "answer_vocab.json"

    question_vocab = QuestionVocab.load(question_vocab_path)
    id_to_answer = load_answer_vocab(answer_vocab_path)

    # For overfitting, disable train augmentation.
    # We want to prove the model can memorize exact samples.
    dataset = CocoQADataset(
        manifest_path=train_manifest,
        question_vocab_path=question_vocab_path,
        image_size=args.image_size,
        train=False,
        repo_root=REPO_ROOT,
        limit=args.subset_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    eval_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    model = build_baseline_vqa_model(
        vocab_size=question_vocab.size,
        num_answers=len(id_to_answer),
        pad_id=question_vocab.pad_id,
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_acc = 0.0
    best_epoch = 0

    print()
    print("Starting overfit test...")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        eval_metrics = evaluate_on_training_subset(
            model=model,
            loader=eval_loader,
            criterion=criterion,
            device=device,
        )

        acc = eval_metrics["accuracy"]

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch

            args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "config": vars(args),
                },
                args.checkpoint_out,
            )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"{format_metrics(train_metrics, prefix='train')} | "
            f"{format_metrics(eval_metrics, prefix='memorize')}"
        )

        if acc >= args.target_acc:
            print()
            print(
                f"Target accuracy reached: {acc:.4f} >= {args.target_acc:.4f} "
                f"at epoch {epoch}"
            )
            break

    print()
    print(f"Best memorization accuracy: {best_acc:.4f} at epoch {best_epoch}")
    print(f"Checkpoint saved to: {args.checkpoint_out}")

    print_predictions(
        model=model,
        loader=eval_loader,
        id_to_answer=id_to_answer,
        device=device,
        max_items=8,
    )

    if best_acc < args.target_acc:
        print()
        print("WARNING: target overfit accuracy was not reached.")
        print("This does not automatically mean failure, but we should inspect before full training.")
    else:
        print()
        print("Overfit test passed.")


if __name__ == "__main__":
    main()