#!/usr/bin/env python3
"""
Inspect pruned COCO-QA datasets and print diagnostics.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

# Paths
DATA_DIR = Path("data/processed")
TRAIN_PATH = DATA_DIR / "cocoqa_train_pruned.jsonl"
VAL_PATH = DATA_DIR / "cocoqa_val_pruned.jsonl"
TEST_PATH = DATA_DIR / "cocoqa_test_pruned.jsonl"
VOCAB_PATH = DATA_DIR / "answer_vocab.json"

SUSPICIOUS_CLASSES = ["room", "bathroom", "kitchen", "road", "meal", "picture", "tower", "desk"]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def main() -> None:
    print("Loading pruned splits...")
    train_samples = read_jsonl(TRAIN_PATH)
    val_samples = read_jsonl(VAL_PATH)
    test_samples = read_jsonl(TEST_PATH)

    all_samples = train_samples + val_samples + test_samples

    print(f"Loaded: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}, total={len(all_samples)}")

    # 1. Load Vocab or build object / color classes
    if VOCAB_PATH.exists():
        with VOCAB_PATH.open("r", encoding="utf-8") as f:
            vocab = json.load(f)
        object_classes = vocab.get("object_answers", [])
        color_classes = vocab.get("color_answers", [])
    else:
        # Fallback if vocab file doesn't exist
        object_classes = sorted(list(set(s["answer"] for s in all_samples if s["type"] == "object")))
        color_classes = sorted(list(set(s["answer"] for s in all_samples if s["type"] == "color")))

    print("\n==================================================")
    print("1. Classes After Pruning")
    print("==================================================")
    print(f"Object Classes ({len(object_classes)}):")
    print(json.dumps(object_classes, indent=2))
    print(f"\nColor Classes ({len(color_classes)}):")
    print(json.dumps(color_classes, indent=2))

    # 2. Train/val/test counts per class
    counts = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for s in train_samples:
        counts[s["answer"]]["train"] += 1
    for s in val_samples:
        counts[s["answer"]]["val"] += 1
    for s in test_samples:
        counts[s["answer"]]["test"] += 1

    print("\n==================================================")
    print("2. Split Counts Per Class")
    print("==================================================")
    header = f"{'Class':<20} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Total':<8}"
    print(header)
    print("-" * len(header))
    
    # Sort by total count descending
    sorted_classes = sorted(
        counts.keys(),
        key=lambda k: counts[k]["train"] + counts[k]["val"] + counts[k]["test"],
        reverse=True
    )
    for cls in sorted_classes:
        t = counts[cls]["train"]
        v = counts[cls]["val"]
        te = counts[cls]["test"]
        tot = t + v + te
        print(f"{cls:<20} | {t:<8} | {v:<8} | {te:<8} | {tot:<8}")

    # 3. Warning list for non-COCO-object-like answers
    print("\n==================================================")
    print("3. Warnings for Non-COCO-Object-Like Answers")
    print("==================================================")
    warnings = [c for c in SUSPICIOUS_CLASSES if c in counts]
    if warnings:
        print(f"WARNING: The following {len(warnings)} non-COCO-object-like answers are present in the vocabulary:")
        for w in warnings:
            t = counts[w]["train"]
            v = counts[w]["val"]
            te = counts[w]["test"]
            print(f"  - '{w}': train={t}, val={v}, test={te}, total={t+v+te}")
    else:
        print("No suspicious classes found in the vocabulary.")

    # 4. 10 random question examples for each suspicious class
    print("\n==================================================")
    print("4. Random Examples for Suspicious Classes (Up to 10)")
    print("==================================================")
    
    rng = random.Random(42)
    
    # Group samples by class
    samples_by_class = defaultdict(list)
    for s in train_samples:
        s = dict(s)
        s["split"] = "train"
        samples_by_class[s["answer"]].append(s)
    for s in val_samples:
        s = dict(s)
        s["split"] = "val"
        samples_by_class[s["answer"]].append(s)
    for s in test_samples:
        s = dict(s)
        s["split"] = "test"
        samples_by_class[s["answer"]].append(s)

    for w in SUSPICIOUS_CLASSES:
        cls_samples = samples_by_class[w]
        if not cls_samples:
            print(f"\nNo samples found for suspicious class '{w}'.")
            continue
            
        print(f"\n--- Examples for '{w}' (Total={len(cls_samples)}) ---")
        # Draw up to 10 random examples
        subset = rng.sample(cls_samples, min(10, len(cls_samples)))
        for idx, sample in enumerate(subset, 1):
            print(f" {idx}. Question: {sample['question']}")
            print(f"    Answer:   {sample['answer']}")
            print(f"    Image ID: {sample['image_id']}")
            print(f"    Split:    {sample['split']}")
            print()


if __name__ == "__main__":
    main()
