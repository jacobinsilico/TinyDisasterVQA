from pathlib import Path
import torch
import torch.nn as nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_acc: float,
    config: dict,
    train_metrics: dict,
    val_metrics: dict,
) -> None:
    """
    Standardize saving PyTorch VQA training checkpoints.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "config": config,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        },
        path,
    )
