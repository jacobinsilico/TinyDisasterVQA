import csv
from pathlib import Path


def append_log_csv(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


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

    for type_name in ["object", "color", "number"]:
        key = f"accuracy_{type_name}"
        if key in metrics:
            parts.append(f"{type_name}_acc={metrics[key]:.4f}")

    return prefix + " | ".join(parts)
