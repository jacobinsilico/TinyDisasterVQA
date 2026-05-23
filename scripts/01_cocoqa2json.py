#!/usr/bin/env python3
"""
Convert raw COCO-QA txt files into JSONL manifests.

Input:
  data/coco-qa/train/{questions,answers,img_ids,types}.txt
  data/coco-qa/test/{questions,answers,img_ids,types}.txt

Output:
  data/processed/cocoqa_train_full.jsonl
  data/processed/cocoqa_test_full.jsonl
  data/processed/cocoqa_full_stats.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path


TYPE_MAP = {
    0: "object",
    1: "number",
    2: "color",
    3: "location",
}


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def load_cocoqa_split(cocoqa_root: Path, split: str) -> list[dict]:
    split_dir = cocoqa_root / split

    questions = read_lines(split_dir / "questions.txt")
    answers = read_lines(split_dir / "answers.txt")
    img_ids = read_lines(split_dir / "img_ids.txt")
    types = read_lines(split_dir / "types.txt")

    lengths = {
        "questions": len(questions),
        "answers": len(answers),
        "img_ids": len(img_ids),
        "types": len(types),
    }

    if len(set(lengths.values())) != 1:
        raise ValueError(f"File length mismatch for split={split}: {lengths}")

    samples = []

    for idx, (question, answer, img_id, type_id) in enumerate(
        zip(questions, answers, img_ids, types)
    ):
        try:
            img_id_int = int(img_id)
            type_id_int = int(type_id)
        except ValueError as e:
            raise ValueError(f"Bad int at split={split}, row={idx}: {e}")

        if type_id_int not in TYPE_MAP:
            raise ValueError(
                f"Unknown type_id={type_id_int} at split={split}, row={idx}"
            )

        sample = {
            "sample_id": f"{split}_{idx:06d}",
            "split": split,
            "image_id": img_id_int,
            "question": question,
            "answer": answer,
            "type_id": type_id_int,
            "type": TYPE_MAP[type_id_int],
        }

        samples.append(sample)

    return samples


def write_jsonl(samples: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def build_stats(samples: list[dict]) -> dict:
    type_counter = Counter(sample["type"] for sample in samples)
    answer_counter = Counter(sample["answer"] for sample in samples)
    image_counter = Counter(sample["image_id"] for sample in samples)

    return {
        "num_samples": len(samples),
        "num_unique_images": len(image_counter),
        "num_unique_answers": len(answer_counter),
        "type_counts": dict(type_counter),
        "top_30_answers": answer_counter.most_common(30),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cocoqa-root",
        type=Path,
        default=Path("data/coco-qa"),
        help="Path to raw COCO-QA folder containing train/ and test/",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for JSONL manifests",
    )
    args = parser.parse_args()

    train_samples = load_cocoqa_split(args.cocoqa_root, "train")
    test_samples = load_cocoqa_split(args.cocoqa_root, "test")

    train_out = args.out_dir / "cocoqa_train_full.jsonl"
    test_out = args.out_dir / "cocoqa_test_full.jsonl"
    stats_out = args.out_dir / "cocoqa_full_stats.json"

    write_jsonl(train_samples, train_out)
    write_jsonl(test_samples, test_out)

    stats = {
        "train": build_stats(train_samples),
        "test": build_stats(test_samples),
    }

    with stats_out.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("Done.")
    print(f"Train manifest: {train_out}")
    print(f"Test manifest:  {test_out}")
    print(f"Stats:          {stats_out}")
    print()
    print("Train stats:")
    print(json.dumps(stats["train"], indent=2, ensure_ascii=False))
    print()
    print("Test stats:")
    print(json.dumps(stats["test"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()