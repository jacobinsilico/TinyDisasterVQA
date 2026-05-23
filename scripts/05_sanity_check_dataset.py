#!/usr/bin/env python3
"""
Sanity check resolved COCO-QA manifests and downloaded images.

Input:
  data/processed/cocoqa_train_resolved.jsonl
  data/processed/cocoqa_val_resolved.jsonl
  data/processed/cocoqa_test_resolved.jsonl

Output:
  data/processed/sanity_check_<split>.png
"""

import argparse
import json
import random
import textwrap
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    return samples


def check_missing_images(samples: list[dict]) -> list[dict]:
    missing = []
    for sample in samples:
        image_path = Path(sample["image_path"])
        if not image_path.exists():
            missing.append(sample)
    return missing


def make_grid(samples: list[dict], out_path: Path, seed: int) -> None:
    rng = random.Random(seed)
    chosen = rng.sample(samples, k=min(len(samples), args.num_samples))

    n = len(chosen)
    cols = args.cols
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))

    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for ax in [a for row in axes for a in row]:
        ax.axis("off")

    for i, sample in enumerate(chosen):
        r = i // cols
        c = i % cols
        ax = axes[r][c]

        image_path = Path(sample["image_path"])
        img = Image.open(image_path).convert("RGB")

        ax.imshow(img)
        ax.axis("off")

        question = textwrap.fill(sample["question"], width=38)
        title = (
            f"Q: {question}\n"
            f"A: {sample['answer']} | type: {sample['type']} | id: {sample['answer_id']}"
        )
        ax.set_title(title, fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    global args
    args = parser.parse_args()

    manifest_path = args.processed_dir / f"cocoqa_{args.split}_resolved.jsonl"
    out_path = args.processed_dir / f"sanity_check_{args.split}.png"

    samples = read_jsonl(manifest_path)
    missing = check_missing_images(samples)

    print(f"Split: {args.split}")
    print(f"Manifest: {manifest_path}")
    print(f"Samples: {len(samples)}")
    print(f"Missing image rows: {len(missing)}")

    if missing:
        print("First missing examples:")
        for sample in missing[:10]:
            print(sample["image_path"])
        raise SystemExit("Missing images detected. Dataset is not ready.")

    make_grid(samples, out_path, seed=args.seed)

    print(f"Saved sanity check grid: {out_path}")


if __name__ == "__main__":
    main()