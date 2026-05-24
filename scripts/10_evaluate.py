#!/usr/bin/env python3
"""
Evaluate a trained VQA checkpoint on train/val/test.

Works for:
  - cnn
  - mobilenet_v2

Example Colab usage:

python scripts/10_evaluate.py \
  --checkpoint /content/drive/MyDrive/edge-vlm-gap9-runs/baseline_cnn_128/best.pt \
  --split test \
  --batch-size 128 \
  --num-workers 2 \
  --out-dir /content/drive/MyDrive/edge-vlm-gap9-runs/baseline_cnn_128/eval \
  --save-predictions

python scripts/10_evaluate.py \
  --checkpoint /content/drive/MyDrive/edge-vlm-gap9-runs/mobilenet_v2_frozen_128/best.pt \
  --split test \
  --batch-size 128 \
  --num-workers 2 \
  --out-dir /content/drive/MyDrive/edge-vlm-gap9-runs/mobilenet_v2_frozen_128/eval \
  --save-predictions
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


def infer_model_settings_from_checkpoint(
    checkpoint: dict,
    cli_model_name: str | None,
    cli_freeze_image_encoder: bool,
    cli_use_pretrained_init: bool,
) -> dict:
    """
    Reconstruct model settings.

    For evaluation, pretrained initialization is not needed because the full
    checkpoint state_dict is loaded. Keeping it False avoids unnecessary downloads.
    """
    config = checkpoint.get("config", {})

    model_name = cli_model_name or config.get("model_name", "cnn")
    freeze_image_encoder = cli_freeze_image_encoder or bool(
        config.get("freeze_image_encoder", False)
    )

    return {
        "model_name": model_name,
        "pretrained": bool(cli_use_pretrained_init),
        "freeze_image_encoder": freeze_image_encoder,
        "checkpoint_config": config,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    id_to_answer: dict[int, str],
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

            correct = int(target_answer == pred_answer)

            confusion_by_answer[target_answer][pred_answer] += 1

            pred_rows.append(
                {
                    "sample_id": batch["metadata"]["sample_id"][i],
                    "image_id": batch["metadata"]["image_id"][i],
                    "question": batch["metadata"]["question"][i],
                    "type": qtype,
                    "target_answer": target_answer,
                    "pred_answer": pred_answer,
                    "correct": correct,
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
        total = answer_total[ans]
        correct = answer_correct[ans]
        per_answer_accuracy[ans] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total > 0 else 0.0,
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

    metrics["top_confusions"] = sorted(
        top_confusions,
        key=lambda x: x["count"],
        reverse=True,
    )[:50]

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


def print_per_answer_summary(metrics: dict, max_items: int = 15) -> None:
    per_answer = metrics.get("per_answer_accuracy", {})

    if not per_answer:
        return

    sorted_low = sorted(
        per_answer.items(),
        key=lambda kv: (kv[1]["accuracy"], -kv[1]["total"]),
    )

    sorted_high = sorted(
        per_answer.items(),
        key=lambda kv: (kv[1]["accuracy"], kv[1]["total"]),
        reverse=True,
    )

    print()
    print("Lowest per-answer accuracies:")
    for ans, item in sorted_low[:max_items]:
        print(
            f"{ans:>15s}: acc={item['accuracy']:.3f} "
            f"({item['correct']}/{item['total']})"
        )

    print()
    print("Highest per-answer accuracies:")
    for ans, item in sorted_high[:max_items]:
        print(
            f"{ans:>15s}: acc={item['accuracy']:.3f} "
            f"({item['correct']}/{item['total']})"
        )


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
    parser.add_argument(
        "--model-name",
        choices=["cnn", "mobilenet_v2"],
        default=None,
        help="Override model architecture. By default inferred from checkpoint config.",
    )
    parser.add_argument(
        "--freeze-image-encoder",
        action="store_true",
        help="Override/freeze image encoder when rebuilding model.",
    )
    parser.add_argument(
        "--use-pretrained-init",
        action="store_true",
        help=(
            "Initialize backbone with pretrained weights before loading checkpoint. "
            "Usually unnecessary for evaluation."
        ),
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)

    print(f"Device: {device}")
    print(f"AMP: {use_amp}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")

    print("Loading checkpoint metadata...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model_settings = infer_model_settings_from_checkpoint(
        checkpoint=checkpoint,
        cli_model_name=args.model_name,
        cli_freeze_image_encoder=args.freeze_image_encoder,
        cli_use_pretrained_init=args.use_pretrained_init,
    )

    print(f"Model name: {model_settings['model_name']}")
    print(f"Pretrained init: {model_settings['pretrained']}")
    print(f"Freeze image encoder: {model_settings['freeze_image_encoder']}")

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
        model_name=model_settings["model_name"],
        pretrained=model_settings["pretrained"],
        freeze_image_encoder=model_settings["freeze_image_encoder"],
    ).to(device)

    print(f"Dataset samples: {len(dataset)}")
    print(f"Question vocab size: {question_vocab.size}")
    print(f"Answer classes: {len(id_to_answer)}")
    print(f"Trainable parameters: {count_parameters(model, trainable_only=True):,}")
    print(f"Total parameters:     {count_parameters(model, trainable_only=False):,}")

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

    print_per_answer_summary(metrics, max_items=15)
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