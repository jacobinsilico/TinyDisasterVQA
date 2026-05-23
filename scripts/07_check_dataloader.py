#!/usr/bin/env python3
"""
Check that the COCO-QA PyTorch Dataset/DataLoader works.

Checks:
  - dataset sizes
  - one batch loads correctly
  - tensor shapes are correct
  - questions decode correctly
  - metadata is aligned with labels
  - optional preview grid is saved
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset import CocoQADataset, ID_TO_TYPE
from src.text import QuestionVocab


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_answer_vocab(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return {int(k): v for k, v in data["id_to_answer"].items()}


def unnormalize_image(image: torch.Tensor) -> torch.Tensor:
    image = image.cpu() * IMAGENET_STD + IMAGENET_MEAN
    return image.clamp(0.0, 1.0)


def save_preview_grid(batch: dict, id_to_answer: dict[int, str], out_path: Path, max_items: int = 8) -> None:
    images = batch["image"]
    answer_ids = batch["answer_id"]
    type_ids = batch["type_id"]
    metadata = batch["metadata"]

    n = min(max_items, images.shape[0])
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.5))

    if rows == 1:
        axes = [axes]

    axes_flat = []
    for row in axes:
        if isinstance(row, (list, tuple)):
            axes_flat.extend(row)
        else:
            try:
                axes_flat.extend(list(row))
            except TypeError:
                axes_flat.append(row)

    for ax in axes_flat:
        ax.axis("off")

    for i in range(n):
        ax = axes_flat[i]
        img = unnormalize_image(images[i]).permute(1, 2, 0).numpy()

        answer_id = int(answer_ids[i])
        type_id = int(type_ids[i])

        question = metadata["question"][i]
        answer = id_to_answer[answer_id]
        qtype = ID_TO_TYPE[type_id]

        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f"Q: {question[:55]}\nA: {answer} | type: {qtype}",
            fontsize=8,
        )

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
        "--image-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Use only first N samples. 0 means full split.",
    )
    parser.add_argument(
        "--save-preview",
        action="store_true",
    )

    args = parser.parse_args()

    manifest_path = args.processed_dir / f"cocoqa_{args.split}_resolved.jsonl"
    question_vocab_path = args.processed_dir / "question_vocab.json"
    answer_vocab_path = args.processed_dir / "answer_vocab.json"

    print(f"Split: {args.split}")
    print(f"Manifest: {manifest_path}")
    print(f"Question vocab: {question_vocab_path}")
    print(f"Answer vocab: {answer_vocab_path}")

    dataset = CocoQADataset(
        manifest_path=manifest_path,
        question_vocab_path=question_vocab_path,
        image_size=args.image_size,
        train=(args.split == "train"),
        repo_root=REPO_ROOT,
        limit=args.limit,
    )

    question_vocab = QuestionVocab.load(question_vocab_path)
    id_to_answer = load_answer_vocab(answer_vocab_path)

    print()
    print("Dataset info:")
    print(f"  num_samples: {len(dataset)}")
    print(f"  question_vocab_size: {question_vocab.size}")
    print(f"  max_question_length: {question_vocab.max_length}")
    print(f"  num_answer_classes: {len(id_to_answer)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(args.split == "train"),
        num_workers=args.num_workers,
        pin_memory=False,
    )

    print()
    print("Loading one batch...")
    batch = next(iter(loader))

    print()
    print("Batch tensor shapes:")
    print(f"  image:        {tuple(batch['image'].shape)}")
    print(f"  question_ids: {tuple(batch['question_ids'].shape)}")
    print(f"  question_len: {tuple(batch['question_len'].shape)}")
    print(f"  answer_id:    {tuple(batch['answer_id'].shape)}")
    print(f"  type_id:      {tuple(batch['type_id'].shape)}")

    assert batch["image"].ndim == 4
    assert batch["image"].shape[1] == 3
    assert batch["image"].shape[2] == args.image_size
    assert batch["image"].shape[3] == args.image_size
    assert batch["question_ids"].ndim == 2
    assert batch["question_ids"].shape[1] == question_vocab.max_length
    assert batch["answer_id"].ndim == 1
    assert batch["type_id"].ndim == 1

    print()
    print("First examples:")
    n_show = min(5, args.batch_size)
    for i in range(n_show):
        question_ids = batch["question_ids"][i].tolist()
        decoded_question = question_vocab.decode(question_ids)

        answer_id = int(batch["answer_id"][i])
        type_id = int(batch["type_id"][i])

        print("-" * 80)
        print(f"metadata question: {batch['metadata']['question'][i]}")
        print(f"decoded question:  {decoded_question}")
        print(f"answer_id:         {answer_id}")
        print(f"answer:            {id_to_answer[answer_id]}")
        print(f"type_id:           {type_id}")
        print(f"type:              {ID_TO_TYPE[type_id]}")
        print(f"image_path:        {batch['metadata']['image_path'][i]}")

    if args.save_preview:
        out_path = args.processed_dir / f"dataloader_preview_{args.split}.png"
        save_preview_grid(batch, id_to_answer, out_path)
        print()
        print(f"Saved preview grid: {out_path}")

    print()
    print("Dataloader check passed.")


if __name__ == "__main__":
    main()