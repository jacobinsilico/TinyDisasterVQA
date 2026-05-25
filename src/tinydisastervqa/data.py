"""
data.py

PyTorch dataset and dataloader utilities for TinyDisasterVQA.

Expected prepared files:

outputs/training_data/
  train.csv
  valid.csv
  test.csv
  metadata.json

Each CSV is produced by scripts/04_prepare_training_data.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


TargetMode = Literal["edge_global", "edge_multihead", "original"]


def load_json(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_token_ids(token_string: str) -> list[int]:
    """
    Parses question_token_ids stored as strings like:
      "3 2 23 8 12 4 0 0 0 0 0"
    """
    return [int(x) for x in str(token_string).strip().split()]


def get_image_transform(
    image_size: int = 224,
    train: bool = False,
    augment: bool = False,
) -> transforms.Compose:
    """
    Image transform for FloodNet.

    Keep this conservative. FloodNet is aerial imagery, so horizontal/vertical
    flips are reasonable, but aggressive crops can hurt counting.
    """
    transform_list = []

    if train and augment:
        transform_list.extend(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )
    else:
        transform_list.append(transforms.Resize((image_size, image_size)))

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return transforms.Compose(transform_list)


class FloodNetVQADataset(Dataset):
    """
    Dataset for TinyDisasterVQA.

    Returns a dictionary containing:
      image: Tensor [3, H, W]
      question_tokens: LongTensor [max_question_len]
      question_length: LongTensor scalar
      question_template_id: LongTensor scalar
      head_id: LongTensor scalar
      target: LongTensor scalar, depending on target_mode
      target_edge_global
      target_edge_head
      target_original
      metadata fields for debugging/evaluation
    """

    def __init__(
        self,
        csv_path: str | Path,
        target_mode: TargetMode = "edge_global",
        transform: transforms.Compose | None = None,
        dataset_root: str | Path | None = None,
        verify_images: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Missing CSV file: {self.csv_path}")

        if target_mode not in {"edge_global", "edge_multihead", "original"}:
            raise ValueError(f"Invalid target_mode: {target_mode}")

        self.target_mode = target_mode
        self.transform = transform
        self.dataset_root = Path(dataset_root) if dataset_root is not None else None

        self.df = pd.read_csv(self.csv_path)

        self._check_required_columns()

        if verify_images:
            self._verify_images()

    def _check_required_columns(self) -> None:
        required = [
            "image_path",
            "image_rel_path",
            "question",
            "question_token_ids",
            "question_length",
            "question_template_id",
            "head_id",
            "target_original",
            "target_edge_global",
            "target_edge_head",
            "edge_head",
            "question_type",
            "answer_norm",
        ]

        missing = [col for col in required if col not in self.df.columns]

        if missing:
            raise ValueError(
                f"{self.csv_path} is missing required columns: {missing}"
            )

    def _resolve_image_path(self, row: pd.Series) -> Path:
        image_path = Path(str(row["image_path"]))

        if image_path.exists():
            return image_path

        if self.dataset_root is not None:
            fallback = self.dataset_root / str(row["image_rel_path"])
            if fallback.exists():
                return fallback

        return image_path

    def _verify_images(self) -> None:
        missing = []

        for _, row in self.df.iterrows():
            path = self._resolve_image_path(row)
            if not path.exists():
                missing.append(str(path))

        if missing:
            raise FileNotFoundError(
                f"Found {len(missing)} missing images. "
                f"First examples: {missing[:5]}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def _get_target(self, row: pd.Series) -> int:
        if self.target_mode == "edge_global":
            return int(row["target_edge_global"])

        if self.target_mode == "edge_multihead":
            return int(row["target_edge_head"])

        if self.target_mode == "original":
            return int(row["target_original"])

        raise RuntimeError(f"Unexpected target_mode: {self.target_mode}")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]

        image_path = self._resolve_image_path(row)

        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        question_tokens = torch.tensor(
            parse_token_ids(row["question_token_ids"]),
            dtype=torch.long,
        )

        question_length = torch.tensor(
            int(row["question_length"]),
            dtype=torch.long,
        )

        question_template_id = torch.tensor(
            int(row["question_template_id"]),
            dtype=torch.long,
        )

        head_id = torch.tensor(
            int(row["head_id"]),
            dtype=torch.long,
        )

        target = torch.tensor(
            self._get_target(row),
            dtype=torch.long,
        )

        target_original = torch.tensor(
            int(row["target_original"]),
            dtype=torch.long,
        )

        target_edge_global = torch.tensor(
            int(row["target_edge_global"]),
            dtype=torch.long,
        )

        target_edge_head = torch.tensor(
            int(row["target_edge_head"]),
            dtype=torch.long,
        )

        return {
            "image": image,
            "question_tokens": question_tokens,
            "question_length": question_length,
            "question_template_id": question_template_id,
            "head_id": head_id,
            "target": target,
            "target_original": target_original,
            "target_edge_global": target_edge_global,
            "target_edge_head": target_edge_head,
            "image_id": str(row.get("image_id", "")),
            "image_path": str(image_path),
            "question_id": int(row["question_id"]) if "question_id" in row else -1,
            "question": str(row["question"]),
            "question_type": str(row["question_type"]),
            "edge_head": str(row["edge_head"]),
            "edge_answer": str(row.get("edge_answer", "")),
            "answer_norm": str(row["answer_norm"]),
            "split": str(row.get("split", "")),
        }


def build_dataloaders(
    train_csv: str | Path = "outputs/training_data/train.csv",
    valid_csv: str | Path = "outputs/training_data/valid.csv",
    test_csv: str | Path | None = "outputs/training_data/test.csv",
    dataset_root: str | Path = "dataset",
    target_mode: TargetMode = "edge_global",
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    augment_train: bool = True,
    pin_memory: bool = True,
    verify_images: bool = False,
) -> dict[str, DataLoader]:
    """
    Builds train/valid/test dataloaders.

    For teacher training:
      target_mode="edge_global"

    For final student multi-head training:
      target_mode="edge_multihead"

    For TinyVQA-style original-label comparison:
      target_mode="original"
    """
    train_transform = get_image_transform(
        image_size=image_size,
        train=True,
        augment=augment_train,
    )

    eval_transform = get_image_transform(
        image_size=image_size,
        train=False,
        augment=False,
    )

    train_dataset = FloodNetVQADataset(
        csv_path=train_csv,
        target_mode=target_mode,
        transform=train_transform,
        dataset_root=dataset_root,
        verify_images=verify_images,
    )

    valid_dataset = FloodNetVQADataset(
        csv_path=valid_csv,
        target_mode=target_mode,
        transform=eval_transform,
        dataset_root=dataset_root,
        verify_images=verify_images,
    )

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ),
        "valid": DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ),
    }

    if test_csv is not None:
        test_dataset = FloodNetVQADataset(
            csv_path=test_csv,
            target_mode=target_mode,
            transform=eval_transform,
            dataset_root=dataset_root,
            verify_images=verify_images,
        )

        dataloaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    return dataloaders


def describe_batch(batch: dict[str, Any]) -> str:
    """
    Small helper for sanity checks.
    """
    lines = []
    lines.append("Batch summary")
    lines.append("=" * 60)
    lines.append(f"image:                {tuple(batch['image'].shape)}")
    lines.append(f"question_tokens:      {tuple(batch['question_tokens'].shape)}")
    lines.append(f"question_length:      {tuple(batch['question_length'].shape)}")
    lines.append(f"question_template_id: {tuple(batch['question_template_id'].shape)}")
    lines.append(f"head_id:              {tuple(batch['head_id'].shape)}")
    lines.append(f"target:               {tuple(batch['target'].shape)}")
    lines.append("")
    lines.append(f"example image_id:     {batch['image_id'][0]}")
    lines.append(f"example question:     {batch['question'][0]}")
    lines.append(f"example question type:{batch['question_type'][0]}")
    lines.append(f"example edge head:    {batch['edge_head'][0]}")
    lines.append(f"example answer:       {batch['answer_norm'][0]}")
    lines.append(f"example target:       {int(batch['target'][0])}")
    return "\n".join(lines)