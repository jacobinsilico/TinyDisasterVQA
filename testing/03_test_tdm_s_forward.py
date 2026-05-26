#!/usr/bin/env python3
"""
Sanity check for TDM-S forward pass.

Run from repo root:

PYTHONPATH=src python testing/03_test_tdm_s_forward.py
"""

from __future__ import annotations

import torch

from tinydisastervqa.data import build_dataloaders, load_json
from tinydisastervqa.models import build_tdm_s_from_metadata, describe_model


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("TinyDisasterVQA / TDM-S Forward Sanity Check")
    print("=" * 80)
    print(f"Device: {device}")

    metadata = load_json("outputs/training_data/metadata.json")

    loaders = build_dataloaders(
        train_csv="outputs/training_data/train.csv",
        valid_csv="outputs/training_data/valid.csv",
        test_csv="outputs/training_data/test.csv",
        dataset_root="dataset",
        target_mode="edge_global",
        image_size=224,
        batch_size=8,
        num_workers=0,
        augment_train=False,
        pin_memory=False,
        verify_images=False,
    )

    batch = next(iter(loaders["train"]))

    model = build_tdm_s_from_metadata(
        metadata=metadata,
        num_classes=19,
        num_question_templates=31,
        question_template_embed_dim=32,
        fusion_hidden_dim=192,
        fusion_dropout=0.1,
    ).to(device)

    model.eval()

    print()
    print(describe_model(model))

    images = batch["image"].to(device)
    question_tokens = batch["question_tokens"].to(device)
    question_lengths = batch["question_length"].to(device)
    question_template_ids = batch["question_template_id"].to(device)
    targets = batch["target"].to(device)

    print()
    print("Input shapes:")
    print(f"  images:                {tuple(images.shape)}")
    print(f"  question_tokens:       {tuple(question_tokens.shape)}")
    print(f"  question_lengths:      {tuple(question_lengths.shape)}")
    print(f"  question_template_ids: {tuple(question_template_ids.shape)}")
    print(f"  targets:               {tuple(targets.shape)}")

    with torch.no_grad():
        logits = model(
            images=images,
            question_tokens=question_tokens,
            question_lengths=question_lengths,
            question_template_ids=question_template_ids,
        )

    print()
    print("Output:")
    print(f"  logits shape:  {tuple(logits.shape)}")
    print(f"  logits dtype:   {logits.dtype}")
    print(f"  logits min/max: {logits.min().item():.4f} / {logits.max().item():.4f}")

    assert logits.ndim == 2
    assert logits.shape[0] == images.shape[0]
    assert logits.shape[1] == 19

    preds = logits.argmax(dim=1)

    print()
    print("First batch predictions:")
    for i in range(images.shape[0]):
        print(
            f"  [{i}] pred={int(preds[i])} | "
            f"target={int(targets[i])} | "
            f"template_id={int(question_template_ids[i])} | "
            f"head={batch['edge_head'][i]} | "
            f"answer={batch['answer_norm'][i]} | "
            f"q={batch['question'][i]}"
        )

    print()
    print("=" * 80)
    print("TDM-S forward sanity check passed.")
    print("=" * 80)


if __name__ == "__main__":
    main()