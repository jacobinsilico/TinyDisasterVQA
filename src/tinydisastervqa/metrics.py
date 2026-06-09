"""
metrics.py

Evaluation metrics for TinyDisasterVQA.

Supports:
  - overall accuracy
  - macro / balanced class accuracy over observed classes
  - per-edge-head accuracy
  - per-question-type accuracy
  - per-class accuracy
  - confusion matrix
  - optional count exact and count ±1 accuracy when label_to_class is provided

Compatible with:
  - teacher models using question_tokens/question_lengths
  - teacher models using question_template_id
  - teacher models returning either logits tensor or {"logits": ..., ...}
  - student models using question_template_id
  - single-head students returning global logits [B, num_classes]
  - backward-compatible multi-head students returning global logits [B, num_classes]
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

    def update_bool(self, correct: bool) -> None:
        self.correct += int(bool(correct))
        self.total += 1

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


def _extract_logits(outputs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Accept either raw logits or model output dictionaries.
    """
    if isinstance(outputs, torch.Tensor):
        return outputs

    if isinstance(outputs, dict):
        if "logits" not in outputs:
            raise KeyError("Model output dict must contain key 'logits'.")
        return outputs["logits"]

    raise TypeError(f"Expected tensor or dict output, got {type(outputs)}.")


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


def _parse_count_class(edge_class: str | None) -> int | None:
    """
    Parse labels like:
      count:1  -> 1
      count:5+ -> None

    Bucket classes like 5+ / 10+ do not have an exact scalar value.
    """
    if edge_class is None:
        return None

    edge_class = str(edge_class)

    if not edge_class.startswith("count:"):
        return None

    answer = edge_class.split(":", maxsplit=1)[1]

    if answer.isdigit():
        return int(answer)

    return None


def _is_count_class(edge_class: str | None) -> bool:
    return edge_class is not None and str(edge_class).startswith("count:")


@dataclass
class ClassificationMetrics:
    """
    Accumulates classification metrics over an epoch.

    Works for:
      - edge_global teacher training
      - single-head student training
      - backward-compatible multi-head student training, as long as the model
        returns global logits with shape [B, num_classes]
    """

    num_classes: int
    label_to_class: dict[str, str] | None = None

    overall: AccuracyMeter = field(default_factory=AccuracyMeter)
    by_head: dict[str, AccuracyMeter] = field(default_factory=lambda: defaultdict(AccuracyMeter))
    by_question_type: dict[str, AccuracyMeter] = field(default_factory=lambda: defaultdict(AccuracyMeter))
    by_class: dict[int, AccuracyMeter] = field(default_factory=lambda: defaultdict(AccuracyMeter))

    count_exact: AccuracyMeter = field(default_factory=AccuracyMeter)
    count_pm1: AccuracyMeter = field(default_factory=AccuracyMeter)

    confusion: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.confusion = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.long,
        )

        if self.label_to_class is not None:
            self.label_to_class = {
                str(k): str(v)
                for k, v in self.label_to_class.items()
            }

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

            if self.label_to_class is not None:
                target_class = self.label_to_class.get(str(target_i))
                pred_class = self.label_to_class.get(str(pred_i))

                if _is_count_class(target_class):
                    exact_correct = pred_i == target_i
                    self.count_exact.update_bool(exact_correct)

                    target_count = _parse_count_class(target_class)
                    pred_count = _parse_count_class(pred_class)

                    if target_count is not None and pred_count is not None:
                        pm1_correct = abs(pred_count - target_count) <= 1
                    else:
                        # For bucket labels like count:5+, require exact bucket match.
                        pm1_correct = exact_correct

                    self.count_pm1.update_bool(pm1_correct)

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
        by_class = {
            str(key): meter.to_dict()
            for key, meter in sorted(self.by_class.items())
        }

        observed_class_accuracies = [
            values["accuracy"]
            for values in by_class.values()
            if values["total"] > 0
        ]

        macro_accuracy = (
            sum(observed_class_accuracies) / len(observed_class_accuracies)
            if observed_class_accuracies
            else 0.0
        )

        result = {
            "overall": self.overall.to_dict(),
            "macro_accuracy": macro_accuracy,
            "by_head": {
                key: meter.to_dict()
                for key, meter in sorted(self.by_head.items())
            },
            "by_question_type": {
                key: meter.to_dict()
                for key, meter in sorted(self.by_question_type.items())
            },
            "by_class": by_class,
            "confusion_matrix": self.confusion.tolist()
            if self.confusion is not None
            else None,
        }

        if self.label_to_class is not None:
            result["count_exact"] = self.count_exact.to_dict()
            result["count_pm1"] = self.count_pm1.to_dict()

        return result


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    logits, targets = _validate_logits_and_targets(
        logits=logits,
        targets=targets,
        num_classes=logits.size(1),
    )

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
    logits, targets = _validate_logits_and_targets(
        logits=logits,
        targets=targets,
        num_classes=logits.size(1),
    )

    max_k = min(max(topk), logits.size(1))

    _, pred = logits.topk(max_k, dim=1)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    results = {}

    for k in topk:
        k_eff = min(k, logits.size(1))
        correct_k = correct[:k_eff].reshape(-1).float().sum(0)
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

    if "macro_accuracy" in metrics:
        lines.append(f"{name}macro acc: {metrics['macro_accuracy']:.4f}")

    if "count_exact" in metrics:
        count_exact = metrics["count_exact"]
        lines.append(
            f"{name}count exact: "
            f"acc={count_exact['accuracy']:.4f} "
            f"({count_exact['correct']}/{count_exact['total']})"
        )

    if "count_pm1" in metrics:
        count_pm1 = metrics["count_pm1"]
        lines.append(
            f"{name}count ±1: "
            f"acc={count_pm1['accuracy']:.4f} "
            f"({count_pm1['correct']}/{count_pm1['total']})"
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
      images, question_tokens, question_lengths, question_template_ids, return_aux

    Student forward usually accepts:
      images, question_tokens, question_lengths, question_template_ids

    Backward-compatible multi-head student additionally accepts:
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
        kwargs["edge_heads"] = batch["edge_head"]

    if "edge_head_ids" in params:
        if "edge_head_id" in batch:
            kwargs["edge_head_ids"] = batch["edge_head_id"].to(device, non_blocking=True)
        elif "head_id" in batch:
            kwargs["edge_head_ids"] = batch["head_id"].to(device, non_blocking=True)

    if "return_aux" in params:
        kwargs["return_aux"] = False

    outputs = model(**kwargs)
    return _extract_logits(outputs)


def _get_targets_from_batch(batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    if "target" in batch:
        return batch["target"].to(device, non_blocking=True)

    if "target_edge_global" in batch:
        return batch["target_edge_global"].to(device, non_blocking=True)

    raise KeyError("Batch must contain either 'target' or 'target_edge_global'.")


@torch.no_grad()
def evaluate_classifier(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    num_classes: int,
    criterion: torch.nn.Module | None = None,
    label_to_class: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Generic evaluator for models returning global logits [B, num_classes].

    Compatible with teacher, single-head student, and backward-compatible
    multi-head student models.
    """
    model.eval()

    meter = ClassificationMetrics(
        num_classes=num_classes,
        label_to_class=label_to_class,
    )

    total_loss = 0.0
    total_samples = 0

    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    for batch in dataloader:
        targets = _get_targets_from_batch(batch, device=device)

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
            edge_heads=batch.get("edge_head", batch.get("edge_head_id", batch.get("head_id"))),
            question_types=batch.get("question_type"),
        )

    result = meter.compute()
    result["loss"] = total_loss / max(total_samples, 1)

    return result