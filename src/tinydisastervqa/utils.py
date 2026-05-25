"""
utils.py

General utilities for TinyDisasterVQA:
  - reproducibility
  - device helpers
  - checkpoint saving/loading
  - JSON/CSV helpers
  - run directory management
"""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducibility.

    deterministic=True can make CUDA slower but more reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def estimate_model_size_mb(model: torch.nn.Module, bytes_per_param: int = 4) -> float:
    return count_parameters(model, trainable_only=False) * bytes_per_param / (1024**2)


def now_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_dir(
    base_dir: str | Path = "runs",
    run_name: str | None = None,
    prefix: str = "run",
) -> Path:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        run_name = f"{prefix}_{now_string()}"

    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    return run_dir


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_json(path: str | Path) -> Any:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_text(text: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_jsonl(record: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """
    Moves tensor values in a batch dictionary to device.
    Non-tensor fields stay untouched.
    """
    moved = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value

    return moved


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int | None = None,
    metrics: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    Save a training checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
        "config": config or {},
        "extra": extra or {},
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """
    Load checkpoint into model and optionally optimizer/scheduler.
    Returns full checkpoint dict.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")

    checkpoint = torch.load(path, map_location=map_location)

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


class AverageMeter:
    """
    Tracks running average of a scalar.
    """

    def __init__(self, name: str = "meter") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.value = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value: float, n: int = 1) -> None:
        self.value = float(value)
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


class Timer:
    """
    Simple wall-clock timer.
    """

    def __init__(self) -> None:
        self.start_time = time.time()

    def reset(self) -> None:
        self.start_time = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def elapsed_str(self) -> str:
        seconds = self.elapsed()

        if seconds < 60:
            return f"{seconds:.1f}s"

        minutes = seconds / 60

        if minutes < 60:
            return f"{minutes:.1f}min"

        hours = minutes / 60
        return f"{hours:.2f}h"


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}min"

    hours = minutes / 60
    return f"{hours:.2f}h"


def copy_file(src: str | Path, dst: str | Path) -> None:
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_run_config(
    run_dir: str | Path,
    config: dict[str, Any],
    filename: str = "config.json",
) -> None:
    run_dir = Path(run_dir)
    save_json(config, run_dir / filename)


def log_metrics(
    metrics: dict[str, Any],
    path: str | Path,
) -> None:
    """
    Append one metrics record to JSONL.
    """
    append_jsonl(metrics, path)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def print_header(title: str) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_subheader(title: str) -> None:
    print("-" * 80)
    print(title)
    print("-" * 80)