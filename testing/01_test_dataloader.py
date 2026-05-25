#!/usr/bin/env python3
"""
Sanity check for TinyDisasterVQA dataloaders.

Run from repo root:

PYTHONPATH=src python testing/test_dataloader.py
"""

from __future__ import annotations

import torch

from tinydisastervqa.data import build_dataloaders, describe_batch


def check_batch(batch: dict, target_mode: str) -> None:
    print()
    print("=" * 80)
    print(f"Checking target_mode={target_mode}")
    print("=" * 80)
    print(describe_batch(batch))

    assert "image" in batch
    assert "question_tokens" in batch
    assert "target" in batch
    assert "head_id" in batch

    assert batch["image"].ndim == 4
    assert batch["image"].shape[1] == 3
    assert batch["question_tokens"].ndim == 2
    assert batch["target"].ndim == 1

    print()
    print("Tensor checks:")
    print(f"  image dtype:           {batch['image'].dtype}")
    print(f"  image min/max:         {batch['image'].min().item():.3f} / {batch['image'].max().item():.3f}")
    print(f"  question_tokens dtype: {batch['question_tokens'].dtype}")
    print(f"  target dtype:          {batch['target'].dtype}")
    print(f"  unique head_ids:       {torch.unique(batch['head_id']).tolist()}")
    print(f"  unique targets:        {torch.unique(batch['target']).tolist()[:20]}")

    print()
    print("First 5 samples:")
    batch_size = min(5, len(batch["target"]))

    for i in range(batch_size):
        print(
            f"  [{i}] image_id={batch['image_id'][i]} | "
            f"head={batch['edge_head'][i]} | "
            f"target={int(batch['target'][i])} | "
            f"answer={batch['answer_norm'][i]} | "
            f"q={batch['question'][i]}"
        )


def main() -> None:
    for target_mode in ["edge_global", "edge_multihead", "original"]:
        loaders = build_dataloaders(
            train_csv="outputs/training_data/train.csv",
            valid_csv="outputs/training_data/valid.csv",
            test_csv="outputs/training_data/test.csv",
            dataset_root="dataset",
            target_mode=target_mode,
            image_size=224,
            batch_size=8,
            num_workers=0,          # keep sanity test simple/reliable
            augment_train=False,    # deterministic for test
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