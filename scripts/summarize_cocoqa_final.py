#!/usr/bin/env python3
"""
Summarize pruned COCO-QA splits and assert dataset integrity constraints.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path


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


def main() -> None:
    manifest_dir = Path("data/processed")
    train_path = manifest_dir / "cocoqa_train_pruned.jsonl"
    val_path = manifest_dir / "cocoqa_val_pruned.jsonl"
    test_path = manifest_dir / "cocoqa_test_pruned.jsonl"
    vocab_path = manifest_dir / "answer_vocab.json"
    stats_path = manifest_dir / "cocoqa_pruned_stats.json"

    print("Loading datasets...")
    train_samples = read_jsonl(train_path)
    val_samples = read_jsonl(val_path)
    test_samples = read_jsonl(test_path)

    all_samples = train_samples + val_samples + test_samples

    print("Loading vocabulary...")
    with vocab_path.open("r", encoding="utf-8") as f:
        vocab = json.load(f)

    print("Loading configuration...")
    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)

    # Class details
    object_classes = sorted(vocab.get("object_answers", []))
    color_classes = sorted(vocab.get("color_answers", []))
    total_classes = len(vocab.get("answer_to_id", {}))

    # Overlap computations
    train_ids = set(s["image_id"] for s in train_samples)
    val_ids = set(s["image_id"] for s in val_samples)
    test_ids = set(s["image_id"] for s in test_samples)

    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids

    # Vocab overlap
    vocab_overlap = set(object_classes) & set(color_classes)

    # Answer counts per split and per type
    def get_counts_by_type(samples: list[dict]) -> dict:
        counts = {}
        for qtype in ["object", "color"]:
            counts[qtype] = Counter(s["answer"] for s in samples if s["type"] == qtype)
        return counts

    train_type_counts = get_counts_by_type(train_samples)
    val_type_counts = get_counts_by_type(val_samples)
    test_type_counts = get_counts_by_type(test_samples)

    # Min/max class counts per split
    def get_min_max(samples: list[dict], cls_list: list[str]) -> tuple[tuple[str, int], tuple[str, int]]:
        counts = Counter(s["answer"] for s in samples)
        # Ensure classes that have 0 counts are considered
        for cls in cls_list:
            if cls not in counts:
                counts[cls] = 0
        sorted_counts = counts.most_common()
        if not sorted_counts:
            return ("", 0), ("", 0)
        return sorted_counts[-1], sorted_counts[0]  # (min, max)

    train_min_max = get_min_max(train_samples, object_classes + color_classes)
    val_min_max = get_min_max(val_samples, object_classes + color_classes)
    test_min_max = get_min_max(test_samples, object_classes + color_classes)

    config = stats.get("config", {})
    image_size_config = stats.get("diagnostics", {}).get("image_size_config", {})
    teacher_image_size = image_size_config.get("teacher_image_size", config.get("teacher_image_size", 224))
    student_image_size = image_size_config.get("student_image_size", config.get("student_image_size", 128))
    padding_fill_rgb = image_size_config.get("padding_fill_rgb", config.get("padding_fill_rgb", [123, 116, 103]))

    print("\n==================================================")
    print("FINAL COCO-QA DATASET SUMMARY REPORT")
    print("==================================================")
    print(f"Total number of answer classes: {total_classes}")
    print(f"Number of object classes:       {len(object_classes)}")
    print(f"Number of color classes:        {len(color_classes)}")
    print(f"\nFull object class list:")
    print(json.dumps(object_classes, indent=2))
    print(f"\nFull color class list:")
    print(json.dumps(color_classes, indent=2))
    
    print("\n--------------------------------------------------")
    print("Split Sample and Image Statistics:")
    print("--------------------------------------------------")
    print(f"Train split: {len(train_samples):<6} samples | {len(train_ids):<6} unique images")
    print(f"Val split:   {len(val_samples):<6} samples | {len(val_ids):<6} unique images")
    print(f"Test split:  {len(test_samples):<6} samples | {len(test_ids):<6} unique images")

    print("\n--------------------------------------------------")
    print("Overlap Verifications:")
    print("--------------------------------------------------")
    print(f"Train ∩ Val image_ids overlap count: {len(train_val_overlap)}")
    print(f"Train ∩ Test image_ids overlap count: {len(train_test_overlap)}")
    print(f"Val ∩ Test image_ids overlap count:   {len(val_test_overlap)}")
    print(f"Object ∩ Color vocabulary overlap:    {len(vocab_overlap)}")

    print("\n--------------------------------------------------")
    print("Answer Counts Per Split and Type (Top 5 / Bottom 5):")
    print("--------------------------------------------------")
    for split_name, t_counts in [("Train", train_type_counts), ("Val", val_type_counts), ("Test", test_type_counts)]:
        print(f"\n[{split_name} Split]")
        for qtype in ["object", "color"]:
            type_counts = t_counts[qtype]
            print(f"  Type '{qtype}': total_samples={sum(type_counts.values())}")
            sorted_tc = type_counts.most_common()
            if sorted_tc:
                print(f"    Top 5:    {sorted_tc[:5]}")
                print(f"    Bottom 5: {sorted_tc[-5:]}")

    print("\n--------------------------------------------------")
    print("Min / Max Class Counts Per Split:")
    print("--------------------------------------------------")
    print(f"Train: min class='{train_min_max[0][0]}' ({train_min_max[0][1]}), max class='{train_min_max[1][0]}' ({train_min_max[1][1]})")
    print(f"Val:   min class='{val_min_max[0][0]}' ({val_min_max[0][1]}), max class='{val_min_max[1][0]}' ({val_min_max[1][1]})")
    print(f"Test:  min class='{test_min_max[0][0]}' ({test_min_max[0][1]}), max class='{test_min_max[1][0]}' ({test_min_max[1][1]})")

    print("\n--------------------------------------------------")
    print("Pipeline & Transform Configuration:")
    print("--------------------------------------------------")
    print(f"teacher_image_size: {teacher_image_size}")
    print(f"student_image_size: {student_image_size}")
    print(f"padding_fill_rgb:   {padding_fill_rgb}")
    print("==================================================\n")

    # PROGRAMMATIC ASSERTIONS FOR DATASET INTEGRITY
    print("Running programmatic assertions...")
    
    # Assert exactly 40 object classes
    assert len(object_classes) == 40, f"Error: expected 40 object classes, got {len(object_classes)}"
    
    # Assert exactly 10 color classes
    assert len(color_classes) == 10, f"Error: expected 10 color classes, got {len(color_classes)}"
    
    # Assert exactly 50 total answer classes
    assert total_classes == 50, f"Error: expected 50 total classes, got {total_classes}"
    
    # Assert zero train/val/test image overlap
    assert len(train_val_overlap) == 0, f"Error: Train ∩ Val overlap detected ({len(train_val_overlap)} images)"
    assert len(train_test_overlap) == 0, f"Error: Train ∩ Test overlap detected ({len(train_test_overlap)} images)"
    assert len(val_test_overlap) == 0, f"Error: Val ∩ Test overlap detected ({len(val_test_overlap)} images)"
    
    # Assert zero object/color vocab overlap
    assert len(vocab_overlap) == 0, f"Error: Object and Color vocab overlap detected! Overlapping: {vocab_overlap}"

    print("All dataset integrity assertions passed successfully!")


if __name__ == "__main__":
    main()
