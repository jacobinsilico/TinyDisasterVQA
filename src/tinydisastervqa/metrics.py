"""
metrics.py

Evaluation metrics for TinyDisasterVQA.

Supports:
  - overall accuracy
  - per-edge-head accuracy
  - per-question-type accuracy
  - class accuracy
  - confusion matrix
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class AccuracyMeter:
    correct: int = 0
    total: int = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds = preds.detach().cpu()
        targets = targets.detach().cpu()

        self.correct += int((preds == targets).sum().item())
        self.total += int(targets.numel())

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    def to_dict(self) -> dict[str, float | int]:
        return {
            "correct": self.correct,
            "total": self.total,
            "accuracy": self.accuracy,
        }


@dataclass
class ClassificationMetrics:
    """
    Accumulates classification metrics over an epoch.

    Works for edge_global teacher training and later student training.
    """

    num_classes: int
    overall: AccuracyMeter = field(default_factory=AccuracyMeter)
    by_head: dict[str, AccuracyMeter] = field(default_factory=lambda: defaultdict(AccuracyMeter))
    by_question_type: dict[str, AccuracyMeter] = field(default_factory=lambda: defaultdict(AccuracyMeter))
    by_class: dict[int, AccuracyMeter] = field(default_factory=lambda: defaultdict(AccuracyMeter))
    confusion: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.confusion = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.long,
        )

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        edge_heads: list[str] | tuple[str, ...] | None = None,
        question_types: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        preds = logits.argmax(dim=1)

        preds_cpu = preds.detach().cpu()
        targets_cpu = targets.detach().cpu()

        self.overall.update(preds_cpu, targets_cpu)

        for pred, target in zip(preds_cpu, targets_cpu):
            pred_i = int(pred.item())
            target_i = int(target.item())

            if 0 <= target_i < self.num_classes and 0 <= pred_i < self.num_classes:
                self.confusion[target_i, pred_i] += 1

            self.by_class[target_i].update(
                torch.tensor([pred_i]),
                torch.tensor([target_i]),
            )

        if edge_heads is not None:
            for i, head in enumerate(edge_heads):
                self.by_head[str(head)].update(
                    preds_cpu[i : i + 1],
                    targets_cpu[i : i + 1],
                )

        if question_types is not None:
            for i, question_type in enumerate(question_types):
                self.by_question_type[str(question_type)].update(
                    preds_cpu[i : i + 1],
                    targets_cpu[i : i + 1],
                )

    def compute(self) -> dict[str, Any]:
        return {
            "overall": self.overall.to_dict(),
            "by_head": {
                key: meter.to_dict()
                for key, meter in sorted(self.by_head.items())
            },
            "by_question_type": {
                key: meter.to_dict()
                for key, meter in sorted(self.by_question_type.items())
            },
            "by_class": {
                str(key): meter.to_dict()
                for key, meter in sorted(self.by_class.items())
            },
            "confusion_matrix": self.confusion.tolist()
            if self.confusion is not None
            else None,
        }


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == targets).float().mean().item())


def topk_accuracy_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: tuple[int, ...] = (1, 3),
) -> dict[str, float]:
    """
    Computes top-k accuracy for classification.
    """
    max_k = max(topk)

    _, pred = logits.topk(max_k, dim=1)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    results = {}

    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results[f"top{k}"] = float((correct_k / targets.numel()).item())

    return results


def confusion_matrix_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    preds = logits.argmax(dim=1).detach().cpu()
    targets = targets.detach().cpu()

    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for target, pred in zip(targets, preds):
        target_i = int(target.item())
        pred_i = int(pred.item())

        if 0 <= target_i < num_classes and 0 <= pred_i < num_classes:
            matrix[target_i, pred_i] += 1

    return matrix


def per_group_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    groups: list[str] | tuple[str, ...],
) -> dict[str, dict[str, float | int]]:
    """
    Computes accuracy per arbitrary string group.

    Example groups:
      edge_head
      question_type
    """
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()

    meters: dict[str, AccuracyMeter] = defaultdict(AccuracyMeter)

    for i, group in enumerate(groups):
        meters[str(group)].update(preds[i : i + 1], targets[i : i + 1])

    return {
        key: meter.to_dict()
        for key, meter in sorted(meters.items())
    }


def format_metrics(metrics: dict[str, Any], prefix: str = "") -> str:
    """
    Pretty formatting for console logs.
    """
    lines = []

    overall = metrics["overall"]
    name = f"{prefix} " if prefix else ""

    lines.append(
        f"{name}overall: "
        f"acc={overall['accuracy']:.4f} "
        f"({overall['correct']}/{overall['total']})"
    )

    if "by_head" in metrics and metrics["by_head"]:
        lines.append(f"{name}by edge head:")
        for head, values in metrics["by_head"].items():
            lines.append(
                f"  {head:<12} "
                f"acc={values['accuracy']:.4f} "
                f"({values['correct']}/{values['total']})"
            )

    if "by_question_type" in metrics and metrics["by_question_type"]:
        lines.append(f"{name}by question type:")
        for question_type, values in metrics["by_question_type"].items():
            lines.append(
                f"  {question_type:<40} "
                f"acc={values['accuracy']:.4f} "
                f"({values['correct']}/{values['total']})"
            )

    return "\n".join(lines)


@torch.no_grad()
def evaluate_classifier(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    num_classes: int,
) -> dict[str, Any]:
    """
    Generic evaluator for models returning logits.

    Assumes batch contains:
      image
      question_tokens
      question_length
      target
      edge_head
      question_type
    """
    model.eval()

    meter = ClassificationMetrics(num_classes=num_classes)

    total_loss = 0.0
    total_samples = 0
    criterion = torch.nn.CrossEntropyLoss()

    for batch in dataloader:
        images = batch["image"].to(device)
        question_tokens = batch["question_tokens"].to(device)
        question_lengths = batch["question_length"].to(device)
        targets = batch["target"].to(device)

        logits = model(
            images=images,
            question_tokens=question_tokens,
            question_lengths=question_lengths,
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