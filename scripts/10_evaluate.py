#!/usr/bin/env python3
"""
Evaluate a trained baseline VQA checkpoint on train/val/test.

Example Colab usage:

python scripts/10_evaluate.py \
  --checkpoint /content/drive/MyDrive/edge-vlm-gap9-runs/baseline_cnn_128/best.pt \
  --split val \
  --batch-size 128 \
  --num-workers 2 \
  --out-dir /content/drive/MyDrive/edge-vlm-gap9-runs/baseline_cnn_128/eval

python scripts/10_evaluate.py \
  --checkpoint /content/drive/MyDrive/edge-vlm-gap9-runs/baseline_cnn_128/best.pt \
  --split test \
  --batch-size 128 \
  --num-workers 2 \
  --out-dir /content/drive/MyDrive/edge-vlm-gap9-runs/baseline_cnn_128/eval
"""

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset import CocoQADataset, ID_TO_TYPE
from src.metrics import AccuracyTracker, AverageMeter, format_metrics
from src.model import build_baseline_vqa_model, count_parameters
from src.text import QuestionVocab


def load_answer_vocab(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return {int(k): v for k, v in data["id_to_answer"].items()}


def move_batch_to_device(batch: dict, device: torch.device) -> tuple:
    images = batch["image"].to(device, non_blocking=True)
    question_ids = batch["question_ids"].to(device, non_blocking=True)
    question_len = batch["question_len"].to(device, non_blocking=True)
    answer_id = batch["answer_id"].to(device, non_blocking=True)
    type_id = batch["type_id"].to(device, non_blocking=True)

    return images, question_ids, question_len, answer_id, type_id


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    id_to_answer: dict[int, str],
    save_predictions: bool,
    use_amp: bool,
) -> tuple[dict, list[dict]]:
    model.eval()

    loss_meter = AverageMeter()
    acc_tracker = AccuracyTracker()

    pred_rows = []
    confusion_by_answer = defaultdict(Counter)

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

        preds = logits.argmax(dim=1)

        batch_size = images.shape[0]
        loss_meter.update(loss.item(), n=batch_size)
        acc_tracker.update(logits, answer_id, type_id)

        for i in range(batch_size):
            target_id = int(answer_id[i].cpu())
            pred_id = int(preds[i].cpu())
            t_id = int(type_id[i].cpu())

            target_answer = id_to_answer[target_id]
            pred_answer = id_to_answer[pred_id]
            qtype = ID_TO_TYPE[t_id]

            confusion_by_answer[target_answer][pred_answer] += 1

            if save_predictions:
                pred_rows.append(
                    {
                        "sample_id": batch["metadata"]["sample_id"][i],
                        "image_id": batch["metadata"]["image_id"][i],
                        "question": batch["metadata"]["question"][i],
                        "type": qtype,
                        "target_answer": target_answer,
                        "pred_answer": pred_answer,
                        "correct": int(target_answer == pred_answer),
                        "image_path": batch["metadata"]["image_path"][i],
                    }
                )

    metrics = acc_tracker.compute()
    metrics["loss"] = loss_meter.avg
    metrics["time_sec"] = time.time() - start_time

    # Per-answer accuracy.
    answer_total = Counter()
    answer_correct = Counter()

    for row in pred_rows:
        ans = row["target_answer"]
        answer_total[ans] += 1
        answer_correct[ans] += int(row["correct"])

    per_answer_accuracy = {}
    for ans in sorted(answer_total.keys()):
        per_answer_accuracy[ans] = {
            "correct": answer_correct[ans],
            "total": answer_total[ans],
            "accuracy": answer_correct[ans] / answer_total[ans]
            if answer_total[ans] > 0
            else 0.0,
        }

    metrics["per_answer_accuracy"] = per_answer_accuracy

    # Top confusions.
    top_confusions = []
    for target_answer, pred_counter in confusion_by_answer.items():
        for pred_answer, count in pred_counter.most_common():
            if pred_answer != target_answer:
                top_confusions.append(
                    {
                        "target": target_answer,
                        "pred": pred_answer,
                        "count": count,
                    }
                )

    top_confusions = sorted(
        top_confusions,
        key=lambda x: x["count"],
        reverse=True,
    )[:50]

    metrics["top_confusions"] = top_confusions

    return metrics, pred_rows


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_predictions_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_prediction_examples(rows: list[dict], max_items: int = 12) -> None:
    if not rows:
        return

    print()
    print("Prediction examples:")

    for row in rows[:max_items]:
        result = "OK" if row["correct"] else "WRONG"

        print("-" * 80)
        print(f"Q:      {row['question']}")
        print(f"Type:   {row['type']}")
        print(f"Target: {row['target_answer']}")
        print(f"Pred:   {row['pred_answer']}")
        print(f"Result: {result}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to best.pt or last.pt checkpoint.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("eval/baseline"),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)

    print(f"Device: {device}")
    print(f"AMP: {use_amp}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")

    manifest_path = args.processed_dir / f"cocoqa_{args.split}_resolved.jsonl"
    question_vocab_path = args.processed_dir / "question_vocab.json"
    answer_vocab_path = args.processed_dir / "answer_vocab.json"

    question_vocab = QuestionVocab.load(question_vocab_path)
    id_to_answer = load_answer_vocab(answer_vocab_path)

    dataset = CocoQADataset(
        manifest_path=manifest_path,
        question_vocab_path=question_vocab_path,
        image_size=args.image_size,
        train=False,
        repo_root=REPO_ROOT,
        limit=args.limit,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = build_baseline_vqa_model(
        vocab_size=question_vocab.size,
        num_answers=len(id_to_answer),
        pad_id=question_vocab.pad_id,
    ).to(device)

    print(f"Dataset samples: {len(dataset)}")
    print(f"Question vocab size: {question_vocab.size}")
    print(f"Answer classes: {len(id_to_answer)}")
    print(f"Model parameters: {count_parameters(model):,}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"Checkpoint best_val_acc: {checkpoint.get('best_val_acc')}")

    criterion = nn.CrossEntropyLoss()

    metrics, pred_rows = evaluate(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        id_to_answer=id_to_answer,
        save_predictions=True,
        use_amp=use_amp,
    )

    print()
    print("Evaluation metrics:")
    print(format_metrics(metrics, prefix=args.split))
    print(f"time_sec={metrics['time_sec']:.1f}")

    print()
    print("Top confusions:")
    for item in metrics["top_confusions"][:20]:
        print(f"{item['target']} -> {item['pred']}: {item['count']}")

    print_prediction_examples(pred_rows, max_items=12)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    metrics_out = args.out_dir / f"{args.split}_metrics.json"
    preds_out = args.out_dir / f"{args.split}_predictions.csv"

    save_json(metrics_out, metrics)

    if args.save_predictions:
        save_predictions_csv(preds_out, pred_rows)

    print()
    print(f"Saved metrics: {metrics_out}")

    if args.save_predictions:
        print(f"Saved predictions: {preds_out}")


if __name__ == "__main__":
    main()