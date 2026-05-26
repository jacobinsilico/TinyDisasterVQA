"""
metrics.py

Evaluation metrics for TinyDisasterVQA.

Supports:
  - overall accuracy
  - per-edge-head accuracy
  - per-question-type accuracy
  - class accuracy
  - confusion matrix

Compatible with:
  - teacher models using question_tokens/question_lengths
  - student models using question_template_id
  - single-head students returning global logits [B, num_classes]
  - multi-head students returning global logits [B, num_classes]
"""

from __future__ import annotations

import inspect
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class AccuracyMeter:
    correct: int = 0
    total: int = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds = preds.detach().cpu().view(-1)
        targets = targets.detach().cpu().view(-1)

        if preds.numel() != targets.numel():
            raise ValueError(
                f"preds and targets must have same number of elements, "
                f"got {preds.numel()} and {targets.numel()}."
            )

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


def _normalize_group_sequence(groups: Any, expected_len: int | None = None) -> list[str]:
    """
    Convert DataLoader-collated metadata into a clean list[str].

    Handles common cases:
      - list/tuple of strings
      - list/tuple of bytes
      - tensor of integer ids
      - scalar strings / scalars
    """
    if groups is None:
        return []

    if isinstance(groups, torch.Tensor):
        groups_cpu = groups.detach().cpu()
        if groups_cpu.ndim == 0:
            values = [groups_cpu.item()]
        else:
            values = groups_cpu.view(-1).tolist()
    elif isinstance(groups, (list, tuple)):
        values = list(groups)
    else:
        values = [groups]

    normalized: list[str] = []
    for value in values:
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                value = value.detach().cpu().item()
            else:
                value = value.detach().cpu().view(-1).tolist()

        if isinstance(value, list):
            # Flatten one level if a collate function produced nested tensors/lists.
            for inner in value:
                if isinstance(inner, bytes):
                    inner = inner.decode("utf-8")
                normalized.append(str(inner))
            continue

        if isinstance(value, bytes):
            value = value.decode("utf-8")

        normalized.append(str(value))

    if expected_len is not None and len(normalized) != expected_len:
        raise ValueError(
            f"Group metadata length mismatch: expected {expected_len}, "
            f"got {len(normalized)}."
        )

    return normalized


def _validate_logits_and_targets(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"Expected logits tensor, got {type(logits)}.")

    if logits.ndim != 2:
        raise ValueError(f"Expected logits with shape [B, C], got {tuple(logits.shape)}.")

    if logits.size(1) != num_classes:
        raise ValueError(
            f"Expected logits with {num_classes} classes, got {logits.size(1)}."
        )

    targets = targets.view(-1)
    if logits.size(0) != targets.size(0):
        raise ValueError(
            f"Batch size mismatch: logits batch={logits.size(0)}, "
            f"targets batch={targets.size(0)}."
        )

    return logits, targets


@dataclass
class ClassificationMetrics:
    """
    Accumulates classification metrics over an epoch.

    Works for:
      - edge_global teacher training
      - single-head student training
      - multi-head student training, as long as the model returns global logits
        with shape [B, num_classes]
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
        edge_heads: Any | None = None,
        question_types: Any | None = None,
    ) -> None:
        logits, targets = _validate_logits_and_targets(
            logits=logits,
            targets=targets,
            num_classes=self.num_classes,
        )

        preds = logits.argmax(dim=1)

        preds_cpu = preds.detach().cpu().view(-1)
        targets_cpu = targets.detach().cpu().view(-1)

        batch_size = int(targets_cpu.numel())

        self.overall.update(preds_cpu, targets_cpu)

        for pred, target in zip(preds_cpu, targets_cpu):
            pred_i = int(pred.item())
            target_i = int(target.item())

            if 0 <= target_i < self.num_classes and 0 <= pred_i < self.num_classes:
                assert self.confusion is not None
                self.confusion[target_i, pred_i] += 1

            self.by_class[target_i].update(
                torch.tensor([pred_i]),
                torch.tensor([target_i]),
            )

        if edge_heads is not None:
            edge_heads_list = _normalize_group_sequence(edge_heads, expected_len=batch_size)
            for i, head in enumerate(edge_heads_list):
                self.by_head[head].update(
                    preds_cpu[i : i + 1],
                    targets_cpu[i : i + 1],
                )

        if question_types is not None:
            question_types_list = _normalize_group_sequence(question_types, expected_len=batch_size)
            for i, question_type in enumerate(question_types_list):
                self.by_question_type[question_type].update(
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
    logits, targets = _validate_logits_and_targets(
        logits=logits,
        targets=targets,
        num_classes=num_classes,
    )

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
    groups: Any,
) -> dict[str, dict[str, float | int]]:
    """
    Computes accuracy per arbitrary string/integer group.

    Example groups:
      edge_head
      question_type
    """
    preds = preds.detach().cpu().view(-1)
    targets = targets.detach().cpu().view(-1)
    groups_list = _normalize_group_sequence(groups, expected_len=int(targets.numel()))

    meters: dict[str, AccuracyMeter] = defaultdict(AccuracyMeter)

    for i, group in enumerate(groups_list):
        meters[group].update(preds[i : i + 1], targets[i : i + 1])

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


def _forward_model_for_eval(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """
    Flexible eval forward pass.

    Teacher forward usually accepts:
      images, question_tokens, question_lengths

    Student forward usually accepts:
      images, question_tokens, question_lengths, question_template_ids

    Multi-head student additionally accepts:
      edge_heads or edge_head_ids
    """
    forward_sig = inspect.signature(model.forward)
    params = forward_sig.parameters

    kwargs: dict[str, Any] = {}

    kwargs["images"] = batch["image"].to(device, non_blocking=True)

    if "question_tokens" in params and "question_tokens" in batch:
        kwargs["question_tokens"] = batch["question_tokens"].to(device, non_blocking=True)

    if "question_lengths" in params and "question_length" in batch:
        kwargs["question_lengths"] = batch["question_length"].to(device, non_blocking=True)

    if "question_template_ids" in params and "question_template_id" in batch:
        kwargs["question_template_ids"] = batch["question_template_id"].to(device, non_blocking=True)

    if "edge_heads" in params and "edge_head" in batch:
        # Keep string edge_head metadata on CPU/list form; the model handles it.
        kwargs["edge_heads"] = batch["edge_head"]

    if "edge_head_ids" in params and "edge_head_id" in batch:
        kwargs["edge_head_ids"] = batch["edge_head_id"].to(device, non_blocking=True)

    return model(**kwargs)


@torch.no_grad()
def evaluate_classifier(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    num_classes: int,
    criterion: torch.nn.Module | None = None,
) -> dict[str, Any]:
    """
    Generic evaluator for models returning global logits [B, num_classes].

    Compatible with teacher, single-head student, and multi-head student models.
    For multi-head students, the dataloader batch must contain edge_head or
    edge_head_id so the model can select the correct task-specific head.
    """
    model.eval()

    meter = ClassificationMetrics(num_classes=num_classes)

    total_loss = 0.0
    total_samples = 0
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    for batch in dataloader:
        targets = batch["target"].to(device, non_blocking=True)

        logits = _forward_model_for_eval(
            model=model,
            batch=batch,
            device=device,
        )

        logits, targets = _validate_logits_and_targets(
            logits=logits,
            targets=targets,
            num_classes=num_classes,
        )

        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        meter.update(
            logits=logits,
            targets=targets,
            edge_heads=batch.get("edge_head", batch.get("edge_head_id")),
            question_types=batch.get("question_type"),
        )

    result = meter.compute()
    result["loss"] = total_loss / max(total_samples, 1)

    return result
