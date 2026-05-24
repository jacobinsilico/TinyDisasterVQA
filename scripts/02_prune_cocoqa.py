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
  - Keep object/color questions only
  - Drop location questions
  - Drop number/counting questions
  - Normalize conservative singular/plural answer variants
  - Build answer vocab from TRAIN ONLY
  - Keep top-K object answers
  - Keep all color answers
  - Filter original test split to train answer vocab
  - Split original test split into val/test by image_id
  - Cap train/val/test samples per answer separately
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


QUESTION_TYPES_TO_KEEP = {"object", "color"}


# Conservative singular/plural normalization only.
# No semantic merges: e.g. jet != airplane, bicycle != motorcycle.
PLURAL_TO_SINGULAR = {
    # common COCO / COCO-QA object plurals
    "airplanes": "airplane",
    "apples": "apple",
    "backpacks": "backpack",
    "bananas": "banana",
    "bears": "bear",
    "beds": "bed",
    "benches": "bench",
    "bicycles": "bicycle",
    "birds": "bird",
    "boats": "boat",
    "books": "book",
    "bottles": "bottle",
    "bowls": "bowl",
    "buses": "bus",
    "cakes": "cake",
    "cars": "car",
    "carrots": "carrot",
    "cats": "cat",
    "chairs": "chair",
    "clocks": "clock",
    "couches": "couch",
    "cows": "cow",
    "cups": "cup",
    "dogs": "dog",
    "donuts": "donut",
    "elephants": "elephant",
    "forks": "fork",
    "frisbees": "frisbee",
    "giraffes": "giraffe",
    "handbags": "handbag",
    "horses": "horse",
    "jets": "jet",
    "keyboards": "keyboard",
    "kites": "kite",
    "knives": "knife",
    "laptops": "laptop",
    "microwaves": "microwave",
    "motorcycles": "motorcycle",
    "oranges": "orange",
    "ovens": "oven",
    "parking meters": "parking meter",
    "persons": "person",
    "people": "person",
    "pizzas": "pizza",
    "plates": "plate",
    "remotes": "remote",
    "refrigerators": "refrigerator",
    "sandwiches": "sandwich",
    "sheep": "sheep",
    "skateboards": "skateboard",
    "snowboards": "snowboard",
    "spoons": "spoon",
    "suitcases": "suitcase",
    "surfboards": "surfboard",
    "tables": "table",
    "ties": "tie",
    "toilets": "toilet",
    "toothbrushes": "toothbrush",
    "towers": "tower",
    "trains": "train",
    "trucks": "truck",
    "umbrellas": "umbrella",
    "vases": "vase",
    "zebras": "zebra",

    # common human plurals
    "men": "man",
    "women": "woman",
    "children": "child",
    "boys": "boy",
    "girls": "girl",

    # multi-word variants
    "baseball bats": "baseball bat",
    "baseball gloves": "baseball glove",
    "cell phones": "cell phone",
    "dining tables": "dining table",
    "fire hydrants": "fire hydrant",
    "hot dogs": "hot dog",
    "potted plants": "potted plant",
    "sports balls": "sports ball",
    "stop signs": "stop sign",
    "teddy bears": "teddy bear",
    "tennis rackets": "tennis racket",
    "traffic lights": "traffic light",
    "tv monitors": "tv monitor",
    "wine glasses": "wine glass",
}

SEMANTIC_MERGES = {
    # conservative semantic / near-synonym merges
    "jet": "airplane",
    "jets": "airplane",

    "laptop": "computer",
    "laptops": "computer",

    "cell phone": "phone",
    "cell phones": "phone",
    "cellphone": "phone",
    "cellphones": "phone",

    "tv": "television",
    "tv monitor": "television",
    "tv monitors": "television",

    "sofa": "couch",
    "sofas": "couch",
}


def normalize_answer(answer: str) -> str:
    """Normalize answer strings.

    Order:
      1. formatting
      2. singular/plural normalization
      3. conservative semantic merges

    We still avoid broad superclass merges like car/truck/bus -> vehicle.
    """
    ans = answer.strip().lower()
    ans = " ".join(ans.split())

    if ans in PLURAL_TO_SINGULAR:
        ans = PLURAL_TO_SINGULAR[ans]

    if ans in SEMANTIC_MERGES:
        ans = SEMANTIC_MERGES[ans]

    return ans


def normalize_samples(samples: list[dict]) -> list[dict]:
    out = []

    for sample in samples:
        sample = dict(sample)
        original_answer = sample["answer"]
        normalized_answer = normalize_answer(original_answer)

        sample["answer"] = normalized_answer

        if normalized_answer != original_answer:
            sample["answer_original"] = original_answer

        out.append(sample)

    return out


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

    top_object_answers = [ans for ans, _ in object_answers.most_common(object_top_k)]

    vocab_answers = []
    vocab_answers.extend(top_object_answers)
    vocab_answers.extend(color_answers)

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
        "types_kept": sorted(list(QUESTION_TYPES_TO_KEEP)),
        "normalization": {
            "mode": "conservative_singular_plural_only",
            "plural_to_singular": PLURAL_TO_SINGULAR,
            "semantic_merges": SEMANTIC_MERGES,
        },
    }


def add_answer_ids(samples: list[dict], vocab: dict) -> list[dict]:
    """
    Add global answer IDs, but filter using type-specific vocabularies.

    Important:
      A sample should only survive if its answer is valid for its own type.
      Example:
        "orange" may be a color answer.
        But object samples with answer "orange" should only survive if
        "orange" is also explicitly included in object_answers.

    This prevents color answers from accidentally keeping object samples,
    and vice versa.
    """
    answer_to_id = vocab["answer_to_id"]

    valid_answers_by_type = {
        "object": set(vocab["object_answers"]),
        "color": set(vocab["color_answers"]),
    }

    out = []

    for sample in samples:
        qtype = sample["type"]
        answer = sample["answer"]

        if qtype not in valid_answers_by_type:
            continue

        if answer not in valid_answers_by_type[qtype]:
            continue

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

    normalized_counter = Counter(
        (s.get("answer_original"), s["answer"])
        for s in samples
        if "answer_original" in s
    )

    return {
        "num_samples": len(samples),
        "num_unique_images": len(image_counter),
        "num_unique_answers": len(answer_counter),
        "type_counts": dict(type_counter),
        "top_30_answers": answer_counter.most_common(30),
        "top_20_answers_by_type": type_answer_counts,
        "top_30_normalized_answer_pairs": [
            {
                "original": original,
                "normalized": normalized,
                "count": count,
            }
            for (original, normalized), count in normalized_counter.most_common(30)
        ],
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
        default=52,
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

    print("Keeping object/color questions only...")
    train_type_filtered = filter_types(train_full)
    test_type_filtered = filter_types(test_full)

    print("Normalizing conservative singular/plural answer variants...")
    train_type_filtered = normalize_samples(train_type_filtered)
    test_type_filtered = normalize_samples(test_type_filtered)

    print("Building answer vocabulary from train only...")
    vocab = build_answer_vocab(
        train_samples=train_type_filtered,
        object_top_k=args.object_top_k,
    )
    answer_to_id = vocab["answer_to_id"]

    print(f"Answer vocab size: {vocab['num_answers']}")
    print(f"Object answers:    {len(vocab['object_answers'])}")
    print(f"Color answers:     {len(vocab['color_answers'])}")

    print("Filtering train/test pool to answer vocabulary...")
    train_pruned = add_answer_ids(train_type_filtered, vocab)
    test_pool_pruned = add_answer_ids(test_type_filtered, vocab)

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
            "dropped_types": ["location", "number"],
            "object_top_k": args.object_top_k,
            "max_train_per_answer": args.max_train_per_answer,
            "max_val_per_answer": args.max_val_per_answer,
            "max_test_per_answer": args.max_test_per_answer,
            "val_fraction_by_image_id": args.val_fraction,
            "seed": args.seed,
            "answer_normalization": {
                "mode": "conservative_singular_plural_only",
                "semantic_merges": SEMANTIC_MERGES,
            },
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