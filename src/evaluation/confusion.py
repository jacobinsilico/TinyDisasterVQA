from collections import Counter
import torch


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
