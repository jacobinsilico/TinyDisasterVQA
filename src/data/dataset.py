from pathlib import Path
from typing import Callable, Optional
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from src.data.vocab import QuestionVocab
from src.data.transforms import default_image_transform
from src.data.helpers import read_json, read_jsonl

ImageFile.LOAD_TRUNCATED_IMAGES = True

TYPE_TO_ID = {
    "object": 0,
    "color": 1,
}

ID_TO_TYPE = {
    0: "object",
    1: "color",
}


class CocoQADataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path | None = None,
        question_vocab_path: str | Path | None = None,
        image_transform: Optional[Callable] = None,
        image_size: int = 128,
        train: bool = False,
        repo_root: str | Path | None = None,
        limit: int = 0,
        answer_vocab_path: str | Path | None = None,
        teacher_image_transform: Optional[Callable] = None,
        teacher_image_size: int | None = None,
        padding_fill_rgb: tuple[int, int, int] = (123, 116, 103),
    ) -> None:
        """
        Args:
          manifest_path:
            Path to cocoqa_{split}_resolved.jsonl.
          question_vocab_path:
            Path to question_vocab.json.
          image_transform:
            Optional torchvision/PIL transform.
          image_size:
            Used only if image_transform is None.
          train:
            Whether to use train-time augmentation in default transform.
          repo_root:
            Base directory for resolving relative image paths.
            If None, uses current working directory.
          limit:
            If >0, keep only first N samples. Useful for debugging/overfit tests.
          answer_vocab_path:
            Path to answer_vocab.json. If None, defaults to manifest parent.
        """
        if manifest_path is None:
            manifest_path = "data/processed/cocoqa_train_resolved.jsonl"
        if question_vocab_path is None:
            question_vocab_path = "data/processed/question_vocab.json"

        self.manifest_path = Path(manifest_path)
        self.question_vocab_path = Path(question_vocab_path)
        self.repo_root = Path(repo_root) if repo_root is not None else Path.cwd()
        self.image_size = image_size
        self.teacher_image_size = teacher_image_size

        if answer_vocab_path is None:
            answer_vocab_path = self.manifest_path.parent / "answer_vocab.json"
            if not answer_vocab_path.exists():
                answer_vocab_path = Path("data/processed/answer_vocab.json")
        self.answer_vocab_path = Path(answer_vocab_path)

        self.samples = read_jsonl(self.manifest_path)

        if limit > 0:
            self.samples = self.samples[:limit]

        self.question_vocab = QuestionVocab.load(self.question_vocab_path)
        self.answer_vocab = read_json(self.answer_vocab_path)

        self.object_answers = list(self.answer_vocab.get("object_answers", []))
        self.color_answers = list(self.answer_vocab.get("color_answers", []))

        if not self.object_answers:
            raise ValueError("answer_vocab.json has no object_answers list.")
        if not self.color_answers:
            raise ValueError("answer_vocab.json has no color_answers list.")

        self.head_answer_to_id = {
            "object": {ans: idx for idx, ans in enumerate(self.object_answers)},
            "color": {ans: idx for idx, ans in enumerate(self.color_answers)},
        }

        self.num_object_answers = len(self.object_answers)
        self.num_color_answers = len(self.color_answers)
        self.num_answer_types = len(TYPE_TO_ID)

        self.image_transform = image_transform
        if self.image_transform is None:
            self.image_transform = default_image_transform(
                image_size=image_size,
                train=train,
                padding_fill_rgb=padding_fill_rgb,
            )

        self.teacher_image_transform = teacher_image_transform

        if self.teacher_image_transform is None and teacher_image_size is not None:
            # Teacher image should be deterministic, even during student training.
            self.teacher_image_transform = default_image_transform(
                image_size=teacher_image_size,
                train=False,
                padding_fill_rgb=padding_fill_rgb,
            )

        self._validate_samples()

    def _validate_samples(self) -> None:
        bad_types = sorted(set(s["type"] for s in self.samples) - set(TYPE_TO_ID))
        if bad_types:
            raise ValueError(
                f"Dataset contains unsupported question types: {bad_types}. "
                f"Expected only: {sorted(TYPE_TO_ID)}"
            )

        missing_head_answers = []
        for sample in self.samples:
            qtype = sample["type"]
            answer = sample["answer"]
            if answer not in self.head_answer_to_id[qtype]:
                missing_head_answers.append((qtype, answer))

        if missing_head_answers:
            preview = missing_head_answers[:20]
            raise ValueError(
                "Some answers are missing from their type-specific head vocab. "
                f"Preview: {preview}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_image_path(self, image_path: str | Path) -> Path:
        image_path = Path(image_path)

        if image_path.is_absolute():
            return image_path

        return self.repo_root / image_path

    def _type_onehot(self, type_id: int) -> torch.Tensor:
        out = torch.zeros(len(TYPE_TO_ID), dtype=torch.float32)
        out[type_id] = 1.0
        return out

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        image_path = self._resolve_image_path(sample["image_path"])

        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")

        pil_image = Image.open(image_path).convert("RGB")

        image = self.image_transform(pil_image)

        teacher_image = None
        if self.teacher_image_transform is not None:
            teacher_image = self.teacher_image_transform(pil_image)

        question_ids, question_len = self.question_vocab.encode(sample["question"])

        answer_id = int(sample["answer_id"])

        qtype = sample["type"]
        if qtype not in TYPE_TO_ID:
            raise ValueError(f"Unexpected question type: {qtype}")

        type_id = TYPE_TO_ID[qtype]
        head_answer_id = self.head_answer_to_id[qtype][sample["answer"]]

        if qtype == "object":
            object_answer_id = head_answer_id
            color_answer_id = -1
        elif qtype == "color":
            object_answer_id = -1
            color_answer_id = head_answer_id
        else:
            object_answer_id = -1
            color_answer_id = -1

        metadata = {
            "sample_id": str(sample.get("sample_id", "")),
            "image_id": int(sample["image_id"]),
            "question": str(sample["question"]),
            "answer": str(sample["answer"]),
            "type": str(sample["type"]),
            "image_path": str(image_path),
        }

        if "answer_original" in sample:
            metadata["answer_original"] = str(sample["answer_original"])
        else:
            metadata["answer_original"] = ""

        out = {
            "image": image,
            "question": str(sample["question"]),
            "question_ids": torch.tensor(question_ids, dtype=torch.long),
            "question_len": torch.tensor(question_len, dtype=torch.long),
            "type": qtype,
            "answer": str(sample["answer"]),
            "answer_id": torch.tensor(answer_id, dtype=torch.long),
            "object_answer_id": torch.tensor(object_answer_id, dtype=torch.long),
            "color_answer_id": torch.tensor(color_answer_id, dtype=torch.long),
            "image_id": torch.tensor(int(sample["image_id"]), dtype=torch.long),
            "type_id": torch.tensor(type_id, dtype=torch.long),
            "type_onehot": self._type_onehot(type_id),
            "head_answer_id": torch.tensor(head_answer_id, dtype=torch.long),
            "metadata": metadata,
        }

        if teacher_image is not None:
            out["teacher_image"] = teacher_image

        # Smoke test shape assertions
        if image is not None:
            assert image.shape == (3, self.image_size, self.image_size), \
                f"Student image shape mismatch: expected {(3, self.image_size, self.image_size)}, got {image.shape}"

        if teacher_image is not None:
            assert teacher_image.shape == (3, self.teacher_image_size, self.teacher_image_size), \
                f"Teacher image shape mismatch: expected {(3, self.teacher_image_size, self.teacher_image_size)}, got {teacher_image.shape}"

        return out


def build_cocoqa_datasets(
    processed_dir: str | Path = "data/processed",
    image_size: int = 128,
    repo_root: str | Path | None = None,
) -> tuple[CocoQADataset, CocoQADataset, CocoQADataset]:
    """
    Convenience function for building train/val/test datasets.
    """
    processed_dir = Path(processed_dir)
    question_vocab_path = processed_dir / "question_vocab.json"
    answer_vocab_path = processed_dir / "answer_vocab.json"

    train_dataset = CocoQADataset(
        manifest_path=processed_dir / "cocoqa_train_resolved.jsonl",
        question_vocab_path=question_vocab_path,
        answer_vocab_path=answer_vocab_path,
        image_size=image_size,
        train=True,
        repo_root=repo_root,
    )

    val_dataset = CocoQADataset(
        manifest_path=processed_dir / "cocoqa_val_resolved.jsonl",
        question_vocab_path=question_vocab_path,
        answer_vocab_path=answer_vocab_path,
        image_size=image_size,
        train=False,
        repo_root=repo_root,
    )

    test_dataset = CocoQADataset(
        manifest_path=processed_dir / "cocoqa_test_resolved.jsonl",
        question_vocab_path=question_vocab_path,
        answer_vocab_path=answer_vocab_path,
        image_size=image_size,
        train=False,
        repo_root=repo_root,
    )

    return train_dataset, val_dataset, test_dataset
