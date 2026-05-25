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
    "airliner": "airplane",
    "airliners": "airplane",

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

OBJECT_WHITELIST = {
    "airplane",
    "cat",
    "giraffe",
    "dog",
    "bus",
    "train",
    "horse",
    "elephant",
    "bear",
    "zebra",
    "bird",
    "motorcycle",
    "car",
    "truck",
    "boat",
    "computer",
    "cow",
    "pizza",
    "kite",
    "bicycle",
    "plate",
    "umbrella",
    "clock",
    "bed",
    "cake",
    "phone",
    "vase",
    "sandwich",
    "hydrant",
    "ball",
    "donut",
    "bat",
    "sheep",
    "bench",
    "toilet",
    "bowl",
    "banana",
    "skateboard",
    "skis",
    "frisbee",
}

COLOR_WHITELIST = {
    "black",
    "blue",
    "brown",
    "gray",
    "green",
    "orange",
    "purple",
    "red",
    "white",
    "yellow",
}

HUMAN_MERGES = {
    "man": "person",
    "men": "person",
    "woman": "person",
    "women": "person",
    "boy": "person",
    "boys": "person",
    "girl": "person",
    "girls": "person",
    "child": "person",
    "children": "person",
    "people": "person",
    "persons": "person",
}



def run_vehicle_safety_assertions() -> None:
    vehicles = ["bicycle", "motorcycle", "car", "bus", "truck"]
    for v in vehicles:
        norm, mtype = normalize_answer(v)
        assert norm == v, f"Safety violation: vehicle class '{v}' was normalized to '{norm}'"

        # Test plural forms
        plural = "buses" if v == "bus" else (v + "s")
        plural_norm, plural_mtype = normalize_answer(plural)
        assert plural_norm == v, f"Safety violation: plural vehicle class '{plural}' was normalized to '{plural_norm}'"


def normalize_answer(answer: str) -> tuple[str, str]:
    """Normalize answer strings.

    Order:
      1. formatting
      2. singular/plural normalization
      3. conservative semantic merges

    We still avoid broad superclass merges like car/truck/bus -> vehicle.
    Returns (normalized_answer, merge_type).
    """
    ans = answer.strip().lower()
    ans = " ".join(ans.split())

    # grey -> gray normalization
    if ans == "grey" or ans == "greys":
        return "gray", "grey_to_gray"

    if ans in PLURAL_TO_SINGULAR:
        ans_sing = PLURAL_TO_SINGULAR[ans]
        if ans_sing in HUMAN_MERGES:
            return HUMAN_MERGES[ans_sing], "human_merge"
        if ans_sing in SEMANTIC_MERGES:
            return SEMANTIC_MERGES[ans_sing], "semantic_merge"
        return ans_sing, "plural_to_singular"

    if ans in HUMAN_MERGES:
        return HUMAN_MERGES[ans], "human_merge"

    if ans in SEMANTIC_MERGES:
        return SEMANTIC_MERGES[ans], "semantic_merge"

    return ans, "none"


def normalize_samples(samples: list[dict]) -> list[dict]:
    out = []

    for sample in samples:
        sample = dict(sample)
        original_answer = sample["answer"]
        normalized_answer, merge_type = normalize_answer(original_answer)

        sample["answer"] = normalized_answer

        if normalized_answer != original_answer:
            sample["answer_original"] = original_answer
            sample["merge_type"] = merge_type

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


def filter_by_whitelists(samples: list[dict]) -> tuple[list[dict], int, int]:
    out = []
    dropped_obj = 0
    dropped_col = 0
    for sample in samples:
        qtype = sample["type"]
        ans = sample["answer"]
        if qtype == "object":
            if ans not in OBJECT_WHITELIST:
                dropped_obj += 1
                continue
        elif qtype == "color":
            if ans not in COLOR_WHITELIST:
                dropped_col += 1
                continue
        out.append(sample)
    return out, dropped_obj, dropped_col




def build_answer_vocab(train_samples: list[dict] = None, object_top_k: int = 0) -> dict:
    object_answers = sorted(list(OBJECT_WHITELIST))
    color_answers = sorted(list(COLOR_WHITELIST))

    vocab_answers = []
    vocab_answers.extend(object_answers)
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
        "object_answers": object_answers,
        "color_answers": color_answers,
        "types_kept": sorted(list(QUESTION_TYPES_TO_KEEP)),
        "normalization": {
            "mode": "conservative_and_semantic",
            "plural_to_singular": PLURAL_TO_SINGULAR,
            "semantic_merges": SEMANTIC_MERGES,
            "object_whitelist": list(OBJECT_WHITELIST),
            "color_whitelist": list(COLOR_WHITELIST),
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


def assert_no_image_overlap(
    train_samples: list[dict],
    val_samples: list[dict],
    test_samples: list[dict],
) -> None:
    train_ids = set(s["image_id"] for s in train_samples)
    val_ids = set(s["image_id"] for s in val_samples)
    test_ids = set(s["image_id"] for s in test_samples)

    val_test_overlap = val_ids & test_ids
    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids

    if val_test_overlap:
        raise RuntimeError(f"Val/test image overlap after capping: {len(val_test_overlap)}")
    if train_val_overlap:
        raise RuntimeError(f"Train/val image overlap after capping: {len(train_val_overlap)}")
    if train_test_overlap:
        raise RuntimeError(f"Train/test image overlap after capping: {len(train_test_overlap)}")



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

    print("Running programmatic vehicle safety assertions...")
    run_vehicle_safety_assertions()

    print("Loading full manifests...")
    train_full = read_jsonl(args.train_full)
    test_full = read_jsonl(args.test_full)

    print("Keeping object/color questions only...")
    train_type_filtered = filter_types(train_full)
    test_type_filtered = filter_types(test_full)

    print("Normalizing conservative singular/plural, human merges, and grey->gray...")
    train_normalized = normalize_samples(train_type_filtered)
    test_normalized = normalize_samples(test_type_filtered)

    print("Filtering against manual whitelists...")
    train_whitelist_filtered, train_dropped_obj, train_dropped_col = filter_by_whitelists(train_normalized)
    test_whitelist_filtered, test_dropped_obj, test_dropped_col = filter_by_whitelists(test_normalized)

    print("Building answer vocabulary from whitelists...")
    vocab = build_answer_vocab()
    answer_to_id = vocab["answer_to_id"]

    print(f"Answer vocab size: {vocab['num_answers']}")
    print(f"Object answers:    {len(vocab['object_answers'])}")
    print(f"Color answers:     {len(vocab['color_answers'])}")

    print("Filtering train/test pool to answer vocabulary...")
    train_pruned_before_capping = add_answer_ids(train_whitelist_filtered, vocab)
    test_pool_pruned = add_answer_ids(test_whitelist_filtered, vocab)

    print("Splitting original COCO-QA test pool into val/test by image_id...")
    val_pruned_before_capping, test_pruned_before_capping = split_by_image_id(
        test_pool_pruned,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    print("Capping samples per answer...")
    train_pruned = cap_samples_per_answer(
        train_pruned_before_capping,
        max_per_answer=args.max_train_per_answer,
        seed=args.seed,
    )

    val_pruned = cap_samples_per_answer(
        val_pruned_before_capping,
        max_per_answer=args.max_val_per_answer,
        seed=args.seed + 1,
    )

    test_pruned = cap_samples_per_answer(
        test_pruned_before_capping,
        max_per_answer=args.max_test_per_answer,
        seed=args.seed + 2,
    )

    print("Asserting absolute zero image overlap across train, val, and test splits...")
    assert_no_image_overlap(train_pruned, val_pruned, test_pruned)

    # Double check disjoint set assertion
    train_ids = set(s["image_id"] for s in train_pruned)
    val_ids = set(s["image_id"] for s in val_pruned)
    test_ids = set(s["image_id"] for s in test_pruned)
    assert len(train_ids & val_ids) == 0, "Train and Val image overlap detected!"
    assert len(train_ids & test_ids) == 0, "Train and Test image overlap detected!"
    assert len(val_ids & test_ids) == 0, "Val and Test image overlap detected!"

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

    # Construct Diagnostics JSON
    object_set = set(vocab["object_answers"])
    color_set = set(vocab["color_answers"])
    overlap_vocab = sorted(list(object_set & color_set))

    # Add split key dynamically for statistics
    for s in train_pruned: s["split"] = "train"
    for s in val_pruned: s["split"] = "val"
    for s in test_pruned: s["split"] = "test"

    all_pruned_samples = train_pruned + val_pruned + test_pruned
    pair_counts = Counter()
    pair_types = {}
    for s in all_pruned_samples:
        if "answer_original" in s:
            orig = s["answer_original"]
            norm = s["answer"]
            mtype = s.get("merge_type", "none")
            pair_counts[(orig, norm)] += 1
            pair_types[(orig, norm)] = mtype

    top_pairs = [
        {
            "original": orig,
            "normalized": norm,
            "count": count,
            "type": pair_types[(orig, norm)]
        }
        for (orig, norm), count in pair_counts.most_common()
    ]

    def get_counts_by_type(samples: list[dict]) -> dict:
        counts = {}
        for qtype in ["object", "color"]:
            counts[qtype] = dict(Counter(s["answer"] for s in samples if s["type"] == qtype))
        return counts

    diagnostics = {
        "image_size_config": {
            "teacher_image_size": 224,
            "student_image_size": 128,
            "padding_fill_rgb": [123, 116, 103]
        },
        "answer_counts_by_split_type": {
            "train": get_counts_by_type(train_pruned),
            "val": get_counts_by_type(val_pruned),
            "test": get_counts_by_type(test_pruned),
        },
        "object_color_vocab_overlap": {
            "overlap_answers": overlap_vocab,
            "overlap_count": len(overlap_vocab)
        },
        "dropped_answer_counts": {
            "train": {
                "dropped_type_location_or_number": len(train_full) - len(train_type_filtered),
                "dropped_not_in_object_whitelist": train_dropped_obj,
                "dropped_not_in_color_whitelist": train_dropped_col,
                "dropped_not_in_vocab": len(train_whitelist_filtered) - len(train_pruned_before_capping),
                "dropped_capping": len(train_pruned_before_capping) - len(train_pruned),
                "total_dropped": len(train_full) - len(train_pruned)
            },
            "val_and_test_pool": {
                "dropped_type_location_or_number": len(test_full) - len(test_type_filtered),
                "dropped_not_in_object_whitelist": test_dropped_obj,
                "dropped_not_in_color_whitelist": test_dropped_col,
                "dropped_not_in_vocab": len(test_whitelist_filtered) - len(test_pool_pruned),
                "dropped_capping_val": len(val_pruned_before_capping) - len(val_pruned),
                "dropped_capping_test": len(test_pruned_before_capping) - len(test_pruned),
                "total_dropped": len(test_full) - (len(val_pruned) + len(test_pruned))
            }
        },
        "top_normalized_answer_pairs": top_pairs
    }

    stats = {
        "config": {
            "types_kept": sorted(list(QUESTION_TYPES_TO_KEEP)),
            "dropped_types": ["location", "number"],
            "teacher_image_size": 224,
            "student_image_size": 128,
            "padding_fill_rgb": [123, 116, 103],
            "max_train_per_answer": args.max_train_per_answer,
            "max_val_per_answer": args.max_val_per_answer,
            "max_test_per_answer": args.max_test_per_answer,
            "val_fraction_by_image_id": args.val_fraction,
            "seed": args.seed,
            "answer_normalization": {
                "mode": "conservative_and_semantic",
                "semantic_merges": SEMANTIC_MERGES,
                "color_whitelist": list(COLOR_WHITELIST),
                "object_whitelist": list(OBJECT_WHITELIST),
                "human_merges": HUMAN_MERGES
            },
        },
        "train": build_stats(train_pruned),
        "val": build_stats(val_pruned),
        "test": build_stats(test_pruned),
        "diagnostics": diagnostics
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

    # 10. Print pruned statistics report
    print("==================================================")
    print("PRUNING SUMMARY REPORT")
    print("==================================================")
    print(f"Total samples per split:")
    print(f"  train: {len(train_pruned)}")
    print(f"  val:   {len(val_pruned)}")
    print(f"  test:  {len(test_pruned)}")
    print()

    def count_by_type(samples: list[dict]) -> tuple[int, int]:
        obj_cnt = sum(1 for s in samples if s["type"] == "object")
        col_cnt = sum(1 for s in samples if s["type"] == "color")
        return obj_cnt, col_cnt

    tr_obj, tr_col = count_by_type(train_pruned)
    va_obj, va_col = count_by_type(val_pruned)
    te_obj, te_col = count_by_type(test_pruned)

    print(f"Object/color samples per split:")
    print(f"  train: object={tr_obj}, color={tr_col}")
    print(f"  val:   object={va_obj}, color={va_col}")
    print(f"  test:  object={te_obj}, color={te_col}")
    print()

    print(f"Number of object classes: {len(vocab['object_answers'])}")
    print(f"Number of color classes:  {len(vocab['color_answers'])}")
    print()

    def get_counts(samples: list[dict], cls_list: list[str]) -> dict:
        counts = {cls: {"train": 0, "val": 0, "test": 0} for cls in cls_list}
        for s in samples:
            ans = s["answer"]
            split = s.get("split", "train")
            if ans in counts:
                counts[ans][split] += 1
        return counts

    obj_counts = get_counts(all_pruned_samples, vocab["object_answers"])
    col_counts = get_counts(all_pruned_samples, vocab["color_answers"])

    print("Full Object Class List with Train/Val/Test counts:")
    print(f"  {'Class':<20} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Total':<8}")
    print("  " + "-" * 57)
    for cls in sorted(vocab["object_answers"]):
        t = obj_counts[cls]["train"]
        v = obj_counts[cls]["val"]
        te = obj_counts[cls]["test"]
        tot = t + v + te
        print(f"  {cls:<20} | {t:<8} | {v:<8} | {te:<8} | {tot:<8}")
    print()

    print("Full Color Class List with Train/Val/Test counts:")
    print(f"  {'Class':<20} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Total':<8}")
    print("  " + "-" * 57)
    for cls in sorted(vocab["color_answers"]):
        t = col_counts[cls]["train"]
        v = col_counts[cls]["val"]
        te = col_counts[cls]["test"]
        tot = t + v + te
        print(f"  {cls:<20} | {t:<8} | {v:<8} | {te:<8} | {tot:<8}")
    print("==================================================")




if __name__ == "__main__":
    main()