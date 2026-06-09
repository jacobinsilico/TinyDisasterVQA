#!/usr/bin/env python3
"""
01_test_dataloader.py

Sanity check for TinyDisasterVQA dataloaders.

Default:
  tests the new cap5 training data:
    outputs/training_data_cap5/

Run from repo root:

python testing/01_test_dataloader.py

Optional:

python testing/01_test_dataloader.py --data-dir outputs/training_data_cap10
python testing/01_test_dataloader.py --target-modes edge_global
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tinydisastervqa.data import build_dataloaders, describe_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("outputs/training_data_cap5"),
        help="Directory containing train.csv, valid.csv, test.csv.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
    )
    parser.add_argument(
        "--target-modes",
        nargs="+",
        default=["edge_global", "edge_multihead", "original"],
        choices=["edge_global", "edge_multihead", "original"],
        help="Target modes to sanity-check.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)

    return parser.parse_args()


def check_paths(data_dir: Path) -> tuple[Path, Path, Path]:
    train_csv = data_dir / "train.csv"
    valid_csv = data_dir / "valid.csv"
    test_csv = data_dir / "test.csv"

    missing = [
        path
        for path in [train_csv, valid_csv, test_csv]
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing prepared training CSV(s):\n"
            + "\n".join(f"  {path}" for path in missing)
            + "\n\nRun scripts 02-04 first, or pass --data-dir outputs/training_data_cap5."
        )

    return train_csv, valid_csv, test_csv


def check_batch(batch: dict, target_mode: str) -> None:
    print()
    print("=" * 80)
    print(f"Checking target_mode={target_mode}")
    print("=" * 80)
    print(describe_batch(batch))

    required_keys = [
        "image",
        "question_tokens",
        "question_length",
        "question_template_id",
        "target",
        "target_edge_global",
        "target_edge_head",
        "target_original",
        "head_id",
        "edge_head",
        "question_type",
        "answer_norm",
    ]

    for key in required_keys:
        assert key in batch, f"Missing key in batch: {key}"

    assert batch["image"].ndim == 4
    assert batch["image"].shape[1] == 3
    assert batch["question_tokens"].ndim == 2
    assert batch["question_length"].ndim == 1
    assert batch["question_template_id"].ndim == 1
    assert batch["target"].ndim == 1
    assert batch["head_id"].ndim == 1

    assert torch.is_floating_point(batch["image"])
    assert batch["question_tokens"].dtype == torch.long
    assert batch["question_length"].dtype == torch.long
    assert batch["question_template_id"].dtype == torch.long
    assert batch["target"].dtype == torch.long
    assert batch["head_id"].dtype == torch.long

    print()
    print("Tensor checks:")
    print(f"  image dtype:                 {batch['image'].dtype}")
    print(f"  image shape:                 {tuple(batch['image'].shape)}")
    print(f"  image min/max:               {batch['image'].min().item():.3f} / {batch['image'].max().item():.3f}")
    print(f"  question_tokens dtype:       {batch['question_tokens'].dtype}")
    print(f"  question_tokens shape:       {tuple(batch['question_tokens'].shape)}")
    print(f"  question_template_id dtype:  {batch['question_template_id'].dtype}")
    print(f"  target dtype:                {batch['target'].dtype}")
    print(f"  unique head_ids:             {torch.unique(batch['head_id']).tolist()}")
    print(f"  unique targets:              {torch.unique(batch['target']).tolist()[:20]}")

    if target_mode.startswith("edge_global"):
        assert torch.equal(batch["target"], batch["target_edge_global"])
    elif target_mode.startswith("edge_multihead"):
        assert torch.equal(batch["target"], batch["target_edge_head"])
    elif target_mode.startswith("original"):
        assert torch.equal(batch["target"], batch["target_original"])

    print()
    print("First 5 samples:")
    batch_size = min(5, len(batch["target"]))

    for i in range(batch_size):
        print(
            f"  [{i}] image_id={batch['image_id'][i]} | "
            f"head={batch['edge_head'][i]} | "
            f"target={int(batch['target'][i])} | "
            f"edge_global={int(batch['target_edge_global'][i])} | "
            f"edge_head={int(batch['target_edge_head'][i])} | "
            f"original={int(batch['target_original'][i])} | "
            f"answer={batch['answer_norm'][i]} | "
            f"q={batch['question'][i]}"
        )


def main() -> None:
    args = parse_args()

    train_csv, valid_csv, test_csv = check_paths(args.data_dir)

    print("=" * 80)
    print("TinyDisasterVQA / Dataloader sanity check")
    print("=" * 80)
    print(f"Data dir:      {args.data_dir.resolve()}")
    print(f"Train CSV:     {train_csv}")
    print(f"Valid CSV:     {valid_csv}")
    print(f"Test CSV:      {test_csv}")
    print(f"Dataset root:  {args.dataset_root.resolve()}")
    print(f"Target modes:  {args.target_modes}")
    print()

    for target_mode in args.target_modes:
        loaders = build_dataloaders(
            train_csv=train_csv,
            valid_csv=valid_csv,
            test_csv=test_csv,
            dataset_root=args.dataset_root,
            target_mode=target_mode,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augment_train=False,
            pin_memory=False,
            verify_images=False,
        )

        train_batch = next(iter(loaders["train"]))
        valid_batch = next(iter(loaders["valid"]))
        test_batch = next(iter(loaders["test"]))

        check_batch(train_batch, target_mode=f"{target_mode} / train")
        check_batch(valid_batch, target_mode=f"{target_mode} / valid")
        check_batch(test_batch, target_mode=f"{target_mode} / test")

    print()
    print("=" * 80)
    print("Dataloader sanity check passed.")
    print("=" * 80)


if __name__ == "__main__":
    main()