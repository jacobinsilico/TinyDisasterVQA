"""
Metrics for COCO-QA object+color training/evaluation.

Main metrics:
  - overall accuracy
  - object accuracy
  - color accuracy
  - average loss tracking
  - top confusion pairs

Question type IDs:
  0 = object
  1 = color
"""

from collections import Counter, defaultdict

import torch


ID_TO_TYPE = {
    0: "object",
    1: "color",
}

TYPE_TO_ID = {
    "object": 0,
    "color": 1,
}


class AverageMeter:
    """
    Tracks running average of a scalar quantity.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


@torch.no_grad()
def accuracy_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Compute top-1 accuracy from logits.

    Args:
      logits:  [B, C]
      targets: [B]

    Returns:
      accuracy as float in [0, 1]
    """
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()

    if total == 0:
        return 0.0

    return correct / total


class AccuracyTracker:
    """
    Tracks overall and per-type accuracy across batches.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.correct_total = 0
        self.count_total = 0

        self.correct_by_type = defaultdict(int)
        self.count_by_type = defaultdict(int)

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        type_ids: torch.Tensor,
    ) -> None:
        """
        Args:
          logits:
            Global logits [B, num_answers].
          targets:
            Global answer IDs [B].
          type_ids:
            0=object, 1=color.
        """
        preds = logits.argmax(dim=1)
        correct = preds == targets

        self.correct_total += correct.sum().item()
        self.count_total += targets.numel()

        for type_id, type_name in ID_TO_TYPE.items():
            mask = type_ids == type_id
            count = mask.sum().item()

            if count == 0:
                continue

            self.correct_by_type[type_name] += correct[mask].sum().item()
            self.count_by_type[type_name] += count

    def compute(self) -> dict:
        if self.count_total == 0:
            overall = 0.0
        else:
            overall = self.correct_total / self.count_total

        out = {
            "accuracy": overall,
            "num_samples": self.count_total,
        }

        for type_id, type_name in ID_TO_TYPE.items():
            count = self.count_by_type[type_name]
            correct = self.correct_by_type[type_name]

            if count == 0:
                acc = 0.0
            else:
                acc = correct / count

            out[f"accuracy_{type_name}"] = acc
            out[f"num_{type_name}"] = count

        return out


class ConfusionTracker:
    """
    Tracks top prediction confusions.

    Stores pairs:
      ground_truth_answer -> predicted_answer

    Useful for qualitative analysis.
    """

    def __init__(
        self,
        id_to_answer: dict[int, str] | dict[str, str],
    ) -> None:
        self.id_to_answer = {
            int(k): v for k, v in id_to_answer.items()
        }
        self.reset()

    def reset(self) -> None:
        self.counter = Counter()

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        preds = logits.argmax(dim=1)

        preds_cpu = preds.detach().cpu().tolist()
        targets_cpu = targets.detach().cpu().tolist()

        for pred_id, target_id in zip(preds_cpu, targets_cpu):
            if pred_id == target_id:
                continue

            pred_answer = self.id_to_answer.get(int(pred_id), f"<unk:{pred_id}>")
            target_answer = self.id_to_answer.get(int(target_id), f"<unk:{target_id}>")

            self.counter[(target_answer, pred_answer)] += 1

    def topk(self, k: int = 20) -> list[tuple[str, str, int]]:
        return [
            (target, pred, count)
            for (target, pred), count in self.counter.most_common(k)
        ]


def format_metrics(metrics: dict, prefix: str = "") -> str:
    """
    Format metrics dict into a compact readable string.
    """
    if prefix:
        prefix = prefix.rstrip() + " "

    parts = []

    if "loss" in metrics:
        parts.append(f"loss={metrics['loss']:.4f}")

    if "accuracy" in metrics:
        parts.append(f"acc={metrics['accuracy']:.4f}")

    for type_name in ["object", "color"]:
        key = f"accuracy_{type_name}"
        if key in metrics:
            parts.append(f"{type_name}_acc={metrics[key]:.4f}")

    return prefix + " | ".join(parts)


def print_top_confusions(
    confusions: list[tuple[str, str, int]],
    title: str = "Top confusions:",
) -> None:
    """
    Pretty-print top confusions.
    """
    print(title)

    if not confusions:
        print("  None")
        return

    for target, pred, count in confusions:
        print(f"  {target} -> {pred}: {count}")