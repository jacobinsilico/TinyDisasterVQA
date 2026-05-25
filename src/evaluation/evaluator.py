import time
from collections import Counter, defaultdict
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.helpers import build_answer_ids_by_type
from src.evaluation.metrics import AccuracyTracker, AverageMeter
from src.training.helpers import forward_model

# Constants
ID_TO_TYPE = {
    0: "object",
    1: "color",
    2: "number",
}


def infer_model_settings_from_checkpoint(
    checkpoint: dict,
    processed_dir: Path,
    cli_model_name: str | None,
    cli_head_type: str | None,
    cli_freeze_image_encoder: bool,
    cli_use_pretrained_init: bool,
) -> dict:
    """
    Reconstruct model settings from checkpoint config.
    """
    config = checkpoint.get("config", {})

    model_name = cli_model_name or config.get("model_name", "cnn")
    head_type = cli_head_type or config.get("head_type", "shared")

    freeze_image_encoder = cli_freeze_image_encoder or bool(
        config.get("freeze_image_encoder", False)
    )

    answer_ids_by_type = config.get("answer_ids_by_type")

    if answer_ids_by_type is None:
        answer_ids_by_type = build_answer_ids_by_type(processed_dir)

    # Normalize answer IDs
    answer_ids_by_type = {
        "object": [int(x) for x in answer_ids_by_type["object"]],
        "color": [int(x) for x in answer_ids_by_type["color"]],
        "number": [int(x) for x in answer_ids_by_type.get("number", [])],
    }

    return {
        "model_name": model_name,
        "head_type": head_type,
        "pretrained": bool(cli_use_pretrained_init),
        "freeze_image_encoder": freeze_image_encoder,
        "answer_ids_by_type": answer_ids_by_type,
        "checkpoint_config": config,
    }


@torch.no_grad()
def run_full_evaluation(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    id_to_answer: dict[int, str],
    use_amp: bool,
) -> tuple[dict, list[dict]]:
    """
    Run evaluation, collect metrics, prediction dictionaries, per-class stats, and top confusions.
    """
    model.eval()

    loss_meter = AverageMeter()
    acc_tracker = AccuracyTracker()

    pred_rows = []
    confusion_by_answer = defaultdict(Counter)

    start_time = time.time()

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        question_ids = batch["question_ids"].to(device, non_blocking=True)
        question_len = batch["question_len"].to(device, non_blocking=True)
        answer_id = batch["answer_id"].to(device, non_blocking=True)
        type_id = batch["type_id"].to(device, non_blocking=True)

        batch_dev = {
            "images": images,
            "question_ids": question_ids,
            "question_len": question_len,
            "answer_id": answer_id,
            "type_id": type_id,
        }
        if "type_onehot" in batch:
            batch_dev["type_onehot"] = batch["type_onehot"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = forward_model(model, batch_dev)
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
            qtype = ID_TO_TYPE.get(t_id, "unknown")

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
