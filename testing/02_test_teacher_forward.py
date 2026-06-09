#!/usr/bin/env python3
"""
02_test_teacher_forward.py

Sanity check for TeacherVQA forward pass.

Tests:
  - LSTM teacher
  - template teacher
  - optional count-aux teacher

Default:
  outputs/training_data_cap5 -> 14-class cap5 task

Run from repo root:

python testing/02_test_teacher_forward.py

Optional:

python testing/02_test_teacher_forward.py --data-dir outputs/training_data_cap10
python testing/02_test_teacher_forward.py --pretrained
python testing/02_test_teacher_forward.py --modes lstm template count_aux
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from tinydisastervqa.data import build_dataloaders, load_json
from tinydisastervqa.models import build_teacher_from_metadata, describe_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("outputs/training_data_cap5"),
        help="Directory containing train.csv, valid.csv, test.csv, metadata.json.",
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument(
        "--backbone",
        type=str,
        default="convnext_tiny",
        choices=[
            "convnext_tiny",
            "swin_tiny",
            "efficientnet_b0",
            "efficientnet_b1",
            "resnet18",
            "resnet50",
        ],
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=False,
        help="Use pretrained torchvision weights. Off by default to avoid downloads in smoke tests.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["lstm", "template", "count_aux"],
        choices=["lstm", "template", "count_aux"],
        help="Teacher variants to test.",
    )

    return parser.parse_args()


def infer_num_classes(metadata: dict[str, Any]) -> int:
    if "num_classes" in metadata:
        return int(metadata["num_classes"])

    if "num_edge_global_classes" in metadata:
        return int(metadata["num_edge_global_classes"])

    return int(metadata["answer_space"]["target_modes"]["edge_global"]["num_classes"])


def infer_num_count_classes(metadata: dict[str, Any]) -> int:
    if "head_label_maps" in metadata and "count" in metadata["head_label_maps"]:
        return len(metadata["head_label_maps"]["count"])

    return len(
        metadata["answer_space"]["target_modes"]["edge_head_local"]["head_label_maps"]["count"]
    )


def check_paths(data_dir: Path) -> tuple[Path, Path, Path, Path]:
    train_csv = data_dir / "train.csv"
    valid_csv = data_dir / "valid.csv"
    test_csv = data_dir / "test.csv"
    metadata_path = data_dir / "metadata.json"

    missing = [
        path
        for path in [train_csv, valid_csv, test_csv, metadata_path]
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing prepared training-data file(s):\n"
            + "\n".join(f"  {path}" for path in missing)
            + "\n\nRun scripts 02-04 first, or pass --data-dir outputs/training_data_cap5."
        )

    return train_csv, valid_csv, test_csv, metadata_path


def run_teacher_check(
    mode: str,
    batch: dict,
    metadata: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    num_classes = infer_num_classes(metadata)
    num_count_classes = infer_num_count_classes(metadata)

    question_encoder = "template" if mode in {"template", "count_aux"} else "lstm"
    use_count_aux = mode == "count_aux"

    print()
    print("=" * 80)
    print(f"Testing teacher mode: {mode}")
    print("=" * 80)

    model = build_teacher_from_metadata(
        metadata=metadata,
        image_backbone=args.backbone,
        pretrained=args.pretrained,
        num_classes=num_classes,
        freeze_image_encoder=False,
        question_encoder=question_encoder,
        question_embed_dim=128,
        question_hidden_dim=256,
        template_embed_dim=128,
        fusion_hidden_dim=512,
        fusion_dropout=0.3,
        use_count_aux=use_count_aux,
        num_count_classes=num_count_classes,
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
        outputs = model(
            images=images,
            question_tokens=question_tokens,
            question_lengths=question_lengths,
            question_template_ids=question_template_ids,
            return_aux=use_count_aux,
        )

    if isinstance(outputs, dict):
        logits = outputs["logits"]
    else:
        logits = outputs

    print()
    print("Main output:")
    print(f"  logits shape:   {tuple(logits.shape)}")
    print(f"  logits dtype:    {logits.dtype}")
    print(f"  logits min/max:  {logits.min().item():.4f} / {logits.max().item():.4f}")

    assert logits.ndim == 2
    assert logits.shape[0] == images.shape[0]
    assert logits.shape[1] == num_classes

    if use_count_aux:
        assert isinstance(outputs, dict)
        assert "count_logits" in outputs

        count_logits = outputs["count_logits"]

        print()
        print("Count auxiliary output:")
        print(f"  count_logits shape:  {tuple(count_logits.shape)}")
        print(f"  count_logits dtype:   {count_logits.dtype}")
        print(f"  count_logits min/max: {count_logits.min().item():.4f} / {count_logits.max().item():.4f}")

        assert count_logits.ndim == 2
        assert count_logits.shape[0] == images.shape[0]
        assert count_logits.shape[1] == num_count_classes

    preds = logits.argmax(dim=1)

    print()
    print("First batch predictions:")
    for i in range(images.shape[0]):
        print(
            f"  [{i}] pred={int(preds[i])} | "
            f"target={int(targets[i])} | "
            f"head={batch['edge_head'][i]} | "
            f"answer={batch['answer_norm'][i]} | "
            f"q={batch['question'][i]}"
        )

    print()
    print(f"Teacher mode '{mode}' forward check passed.")


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_csv, valid_csv, test_csv, metadata_path = check_paths(args.data_dir)
    metadata = load_json(metadata_path)

    num_classes = infer_num_classes(metadata)
    num_count_classes = infer_num_count_classes(metadata)

    print("=" * 80)
    print("TinyDisasterVQA / Teacher Forward Sanity Check")
    print("=" * 80)
    print(f"Device:             {device}")
    print(f"Data dir:           {args.data_dir.resolve()}")
    print(f"Metadata:           {metadata_path}")
    print(f"Backbone:           {args.backbone}")
    print(f"Pretrained:         {args.pretrained}")
    print(f"Num classes:        {num_classes}")
    print(f"Num count classes:  {num_count_classes}")
    print(f"Modes:              {args.modes}")

    loaders = build_dataloaders(
        train_csv=train_csv,
        valid_csv=valid_csv,
        test_csv=test_csv,
        dataset_root=args.dataset_root,
        target_mode="edge_global",
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment_train=False,
        pin_memory=False,
        verify_images=False,
    )

    batch = next(iter(loaders["train"]))

    for mode in args.modes:
        run_teacher_check(
            mode=mode,
            batch=batch,
            metadata=metadata,
            device=device,
            args=args,
        )

    print()
    print("=" * 80)
    print("Teacher forward sanity check passed.")
    print("=" * 80)


if __name__ == "__main__":
    main()