from collections import defaultdict
import torch

ID_TO_TYPE = {
    0: "object",
    1: "color",
    2: "number",  # Expanded to support number for full compatibility!
}

TYPE_TO_ID = {
    "object": 0,
    "color": 1,
    "number": 2,
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
            0=object, 1=color, 2=number.
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
                continue

            acc = correct / count
            out[f"accuracy_{type_name}"] = acc
            out[f"num_{type_name}"] = count

        return out
