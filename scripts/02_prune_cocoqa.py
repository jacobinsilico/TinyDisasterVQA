#!/usr/bin/env python3
"""
Prune COCO-QA full JSONL manifests and create train/val/test splits.

Input:
  data/processed/cocoqa_train_full.jsonl
  data/processed/cocoqa_test_full.jsonl

Output:
  data/processed/cocoqa_train_pruned.jsonl
  data/processed/cocoqa_val_pruned.jsonl
  data/processed/cocoqa_test_pruned.jsonl
  data/processed/answer_vocab.json
  data/processed/cocoqa_pruned_stats.json

Strategy:
  - Drop location questions
  - Keep object/color/number questions
  - Build answer vocab from TRAIN ONLY
  - Keep top-K object answers
  - Keep all color answers
  - Keep all number answers
  - Filter original test split to train answer vocab
  - Split original test split into val/test by image_id
  - Cap train/val/test samples per answer separately
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


QUESTION_TYPES_TO_KEEP = {"object", "color", "number"}


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


def write_jsonl(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def filter_types(samples: list[dict]) -> list[dict]:
    return [s for s in samples if s["type"] in QUESTION_TYPES_TO_KEEP]


def build_answer_vocab(train_samples: list[dict], object_top_k: int) -> dict:
    object_answers = Counter(
        s["answer"] for s in train_samples if s["type"] == "object"
    )

    color_answers = sorted(
        set(s["answer"] for s in train_samples if s["type"] == "color")
    )

    number_answers = sorted(
        set(s["answer"] for s in train_samples if s["type"] == "number")
    )

    top_object_answers = [ans for ans, _ in object_answers.most_common(object_top_k)]

    vocab_answers = []
    vocab_answers.extend(top_object_answers)
    vocab_answers.extend(color_answers)
    vocab_answers.extend(number_answers)

    # Remove duplicates while preserving order.
    seen = set()
    vocab_answers_unique = []
    for ans in vocab_answers:
        if ans not in seen:
            vocab_answers_unique.append(ans)
            seen.add(ans)

    answer_to_id = {ans: idx for idx, ans in enumerate(vocab_answers_unique)}

    return {
        "answer_to_id": answer_to_id,
        "id_to_answer": {str(idx): ans for ans, idx in answer_to_id.items()},
        "object_top_k": object_top_k,
        "num_answers": len(answer_to_id),
        "object_answers": top_object_answers,
        "color_answers": color_answers,
        "number_answers": number_answers,
    }


def add_answer_ids(samples: list[dict], answer_to_id: dict[str, int]) -> list[dict]:
    out = []

    for sample in samples:
        answer = sample["answer"]

        if answer not in answer_to_id:
            continue

        sample = dict(sample)
        sample["answer_id"] = answer_to_id[answer]
        out.append(sample)

    return out


def split_by_image_id(
    samples: list[dict],
    val_fraction: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")

    rng = random.Random(seed)

    image_ids = sorted(set(s["image_id"] for s in samples))
    rng.shuffle(image_ids)

    num_val_images = int(round(len(image_ids) * val_fraction))

    val_image_ids = set(image_ids[:num_val_images])
    test_image_ids = set(image_ids[num_val_images:])

    val_samples = []
    test_samples = []

    for sample in samples:
        sample = dict(sample)

        if sample["image_id"] in val_image_ids:
            sample["split"] = "val"
            sample["sample_id"] = sample["sample_id"].replace("test_", "val_")
            val_samples.append(sample)
        elif sample["image_id"] in test_image_ids:
            sample["split"] = "test"
            test_samples.append(sample)
        else:
            raise RuntimeError(f"Image ID not assigned: {sample['image_id']}")

    # Safety check: no image overlap.
    val_ids = set(s["image_id"] for s in val_samples)
    test_ids = set(s["image_id"] for s in test_samples)
    overlap = val_ids & test_ids

    if overlap:
        raise RuntimeError(f"Val/test image overlap detected: {len(overlap)} images")

    return val_samples, test_samples


def cap_samples_per_answer(
    samples: list[dict],
    max_per_answer: int,
    seed: int,
) -> list[dict]:
    if max_per_answer <= 0:
        return samples

    rng = random.Random(seed)

    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["answer"]].append(sample)

    capped = []
    for answer, group in grouped.items():
        if len(group) > max_per_answer:
            group = rng.sample(group, max_per_answer)
        capped.extend(group)

    rng.shuffle(capped)
    return capped


def build_stats(samples: list[dict]) -> dict:
    answer_counter = Counter(s["answer"] for s in samples)
    type_counter = Counter(s["type"] for s in samples)
    image_counter = Counter(s["image_id"] for s in samples)

    type_answer_counts = {}
    for qtype in sorted(set(s["type"] for s in samples)):
        type_answer_counts[qtype] = Counter(
            s["answer"] for s in samples if s["type"] == qtype
        ).most_common(20)

    return {
        "num_samples": len(samples),
        "num_unique_images": len(image_counter),
        "num_unique_answers": len(answer_counter),
        "type_counts": dict(type_counter),
        "top_30_answers": answer_counter.most_common(30),
        "top_20_answers_by_type": type_answer_counts,
    }


def assert_no_image_overlap(val_samples: list[dict], test_samples: list[dict]) -> None:
    val_ids = set(s["image_id"] for s in val_samples)
    test_ids = set(s["image_id"] for s in test_samples)
    overlap = val_ids & test_ids

    if overlap:
        raise RuntimeError(f"Val/test image overlap after capping: {len(overlap)}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-full",
        type=Path,
        default=Path("data/processed/cocoqa_train_full.jsonl"),
    )
    parser.add_argument(
        "--test-full",
        type=Path,
        default=Path("data/processed/cocoqa_test_full.jsonl"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--object-top-k",
        type=int,
        default=50,
        help="Keep top-K object answers from train split.",
    )
    parser.add_argument(
        "--max-train-per-answer",
        type=int,
        default=500,
        help="Cap train samples per answer. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-val-per-answer",
        type=int,
        default=200,
        help="Cap val samples per answer. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-test-per-answer",
        type=int,
        default=100,
        help="Cap test samples per answer. Use 0 to disable.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.5,
        help="Fraction of original COCO-QA test image IDs assigned to val.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    print("Loading full manifests...")
    train_full = read_jsonl(args.train_full)
    test_full = read_jsonl(args.test_full)

    print("Dropping location questions...")
    train_type_filtered = filter_types(train_full)
    test_type_filtered = filter_types(test_full)

    print("Building answer vocabulary from train only...")
    vocab = build_answer_vocab(
        train_samples=train_type_filtered,
        object_top_k=args.object_top_k,
    )
    answer_to_id = vocab["answer_to_id"]

    print(f"Answer vocab size: {vocab['num_answers']}")

    print("Filtering train/test pool to answer vocabulary...")
    train_pruned = add_answer_ids(train_type_filtered, answer_to_id)
    test_pool_pruned = add_answer_ids(test_type_filtered, answer_to_id)

    print("Splitting original COCO-QA test pool into val/test by image_id...")
    val_pruned, test_pruned = split_by_image_id(
        test_pool_pruned,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    print("Capping samples per answer...")
    train_pruned = cap_samples_per_answer(
        train_pruned,
        max_per_answer=args.max_train_per_answer,
        seed=args.seed,
    )

    val_pruned = cap_samples_per_answer(
        val_pruned,
        max_per_answer=args.max_val_per_answer,
        seed=args.seed + 1,
    )

    test_pruned = cap_samples_per_answer(
        test_pruned,
        max_per_answer=args.max_test_per_answer,
        seed=args.seed + 2,
    )

    assert_no_image_overlap(val_pruned, test_pruned)

    train_out = args.out_dir / "cocoqa_train_pruned.jsonl"
    val_out = args.out_dir / "cocoqa_val_pruned.jsonl"
    test_out = args.out_dir / "cocoqa_test_pruned.jsonl"
    vocab_out = args.out_dir / "answer_vocab.json"
    stats_out = args.out_dir / "cocoqa_pruned_stats.json"

    print("Writing outputs...")
    write_jsonl(train_pruned, train_out)
    write_jsonl(val_pruned, val_out)
    write_jsonl(test_pruned, test_out)

    with vocab_out.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)

    stats = {
        "config": {
            "types_kept": sorted(list(QUESTION_TYPES_TO_KEEP)),
            "dropped_type": "location",
            "object_top_k": args.object_top_k,
            "max_train_per_answer": args.max_train_per_answer,
            "max_val_per_answer": args.max_val_per_answer,
            "max_test_per_answer": args.max_test_per_answer,
            "val_fraction_by_image_id": args.val_fraction,
            "seed": args.seed,
        },
        "train": build_stats(train_pruned),
        "val": build_stats(val_pruned),
        "test": build_stats(test_pruned),
    }

    with stats_out.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("Done.")
    print(f"Train pruned: {train_out}")
    print(f"Val pruned:   {val_out}")
    print(f"Test pruned:  {test_out}")
    print(f"Vocab:        {vocab_out}")
    print(f"Stats:        {stats_out}")
    print()
    print("Pruned train stats:")
    print(json.dumps(stats["train"], indent=2, ensure_ascii=False))
    print()
    print("Pruned val stats:")
    print(json.dumps(stats["val"], indent=2, ensure_ascii=False))
    print()
    print("Pruned test stats:")
    print(json.dumps(stats["test"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()