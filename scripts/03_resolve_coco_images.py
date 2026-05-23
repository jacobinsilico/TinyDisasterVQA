#!/usr/bin/env python3
"""
Resolve COCO-QA image IDs to COCO 2014 image metadata.

Input:
  data/processed/cocoqa_train_pruned.jsonl
  data/processed/cocoqa_val_pruned.jsonl
  data/processed/cocoqa_test_pruned.jsonl

COCO annotation input:
  data/coco-annotations/train/instances_train2014.json
  data/coco-annotations/val/instances_val2014.json

Output:
  data/processed/cocoqa_train_resolved.jsonl
  data/processed/cocoqa_val_resolved.jsonl
  data/processed/cocoqa_test_resolved.jsonl
  data/processed/required_images.jsonl
  data/processed/cocoqa_resolved_stats.json
"""

import argparse
import json
from collections import Counter
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


def write_jsonl(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def load_coco_image_map(annotation_path: Path, coco_split: str) -> dict[int, dict]:
    """
    Load COCO image metadata from annotation JSON.

    Returns:
      image_id -> {
        image_id,
        coco_split,
        file_name,
        coco_url,
        width,
        height
      }
    """
    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing COCO annotation file: {annotation_path}")

    print(f"Loading COCO annotations: {annotation_path}")

    with annotation_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "images" not in data:
        raise ValueError(f"No 'images' field found in {annotation_path}")

    image_map = {}

    for img in data["images"]:
        image_id = int(img["id"])
        file_name = img["file_name"]

        # COCO annotations usually include coco_url.
        # If not, construct the standard COCO URL.
        coco_url = img.get(
            "coco_url",
            f"http://images.cocodataset.org/{coco_split}/{file_name}",
        )

        image_map[image_id] = {
            "image_id": image_id,
            "coco_split": coco_split,
            "file_name": file_name,
            "coco_url": coco_url,
            "width": img.get("width"),
            "height": img.get("height"),
        }

    print(f"Loaded {len(image_map)} images for {coco_split}")
    return image_map


def build_global_image_map(
    train_image_map: dict[int, dict],
    val_image_map: dict[int, dict],
) -> dict[int, dict]:
    overlap = set(train_image_map.keys()) & set(val_image_map.keys())

    if overlap:
        raise RuntimeError(
            f"Unexpected image_id overlap between train2014 and val2014: "
            f"{len(overlap)} IDs"
        )

    global_map = {}
    global_map.update(train_image_map)
    global_map.update(val_image_map)

    return global_map


def resolve_samples(
    samples: list[dict],
    global_image_map: dict[int, dict],
    images_root: Path,
) -> tuple[list[dict], list[int]]:
    resolved = []
    missing_ids = []

    for sample in samples:
        image_id = int(sample["image_id"])

        if image_id not in global_image_map:
            missing_ids.append(image_id)
            continue

        meta = global_image_map[image_id]
        image_path = images_root / meta["coco_split"] / meta["file_name"]

        sample_out = dict(sample)
        sample_out.update(
            {
                "coco_split": meta["coco_split"],
                "file_name": meta["file_name"],
                "coco_url": meta["coco_url"],
                "width": meta["width"],
                "height": meta["height"],
                "image_path": str(image_path),
            }
        )

        resolved.append(sample_out)

    return resolved, missing_ids


def build_required_images(
    resolved_splits: dict[str, list[dict]],
) -> list[dict]:
    """
    Build one unique image entry per image_id across all resolved splits.
    """
    by_image_id = {}

    for split_name, samples in resolved_splits.items():
        for sample in samples:
            image_id = int(sample["image_id"])

            if image_id not in by_image_id:
                by_image_id[image_id] = {
                    "image_id": image_id,
                    "coco_split": sample["coco_split"],
                    "file_name": sample["file_name"],
                    "coco_url": sample["coco_url"],
                    "image_path": sample["image_path"],
                    "width": sample["width"],
                    "height": sample["height"],
                    "used_by_splits": set(),
                    "num_qa_pairs": 0,
                }

            by_image_id[image_id]["used_by_splits"].add(split_name)
            by_image_id[image_id]["num_qa_pairs"] += 1

    required_images = []

    for image_id in sorted(by_image_id.keys()):
        item = by_image_id[image_id]
        item["used_by_splits"] = sorted(item["used_by_splits"])
        required_images.append(item)

    return required_images


def split_stats(samples: list[dict]) -> dict:
    image_counter = Counter(s["image_id"] for s in samples)
    answer_counter = Counter(s["answer"] for s in samples)
    type_counter = Counter(s["type"] for s in samples)
    coco_split_counter = Counter(s["coco_split"] for s in samples)

    return {
        "num_samples": len(samples),
        "num_unique_images": len(image_counter),
        "num_unique_answers": len(answer_counter),
        "type_counts": dict(type_counter),
        "coco_split_counts_by_qa_pairs": dict(coco_split_counter),
    }


def required_images_stats(required_images: list[dict]) -> dict:
    coco_split_counter = Counter(img["coco_split"] for img in required_images)

    used_by_counter = Counter(
        "+".join(img["used_by_splits"]) for img in required_images
    )

    return {
        "num_unique_images_total": len(required_images),
        "coco_split_counts_by_unique_images": dict(coco_split_counter),
        "used_by_splits_counts": dict(used_by_counter),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-ann",
        type=Path,
        default=Path("data/coco-annotations/train/instances_train2014.json"),
        help="Path to COCO 2014 train instances annotation JSON.",
    )
    parser.add_argument(
        "--val-ann",
        type=Path,
        default=Path("data/coco-annotations/val/instances_val2014.json"),
        help="Path to COCO 2014 val instances annotation JSON.",
    )
    parser.add_argument(
        "--in-dir",
        type=Path,
        default=Path("data/processed"),
        help="Input directory containing pruned COCO-QA JSONL files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for resolved JSONL files.",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("data/images"),
        help="Root directory where downloaded COCO images will be stored.",
    )

    args = parser.parse_args()

    train_in = args.in_dir / "cocoqa_train_pruned.jsonl"
    val_in = args.in_dir / "cocoqa_val_pruned.jsonl"
    test_in = args.in_dir / "cocoqa_test_pruned.jsonl"

    print("Loading pruned COCO-QA manifests...")
    train_samples = read_jsonl(train_in)
    val_samples = read_jsonl(val_in)
    test_samples = read_jsonl(test_in)

    train_image_map = load_coco_image_map(args.train_ann, "train2014")
    val_image_map = load_coco_image_map(args.val_ann, "val2014")
    global_image_map = build_global_image_map(train_image_map, val_image_map)

    print(f"Global COCO image map size: {len(global_image_map)}")

    print("Resolving train samples...")
    train_resolved, train_missing = resolve_samples(
        train_samples,
        global_image_map,
        args.images_root,
    )

    print("Resolving val samples...")
    val_resolved, val_missing = resolve_samples(
        val_samples,
        global_image_map,
        args.images_root,
    )

    print("Resolving test samples...")
    test_resolved, test_missing = resolve_samples(
        test_samples,
        global_image_map,
        args.images_root,
    )

    missing_total = len(train_missing) + len(val_missing) + len(test_missing)

    if missing_total > 0:
        print("WARNING: Some image IDs could not be resolved.")
        print(f"Train missing QA rows: {len(train_missing)}")
        print(f"Val missing QA rows:   {len(val_missing)}")
        print(f"Test missing QA rows:  {len(test_missing)}")
        print("First missing IDs:")
        print((train_missing + val_missing + test_missing)[:20])
    else:
        print("All image IDs resolved successfully.")

    resolved_splits = {
        "train": train_resolved,
        "val": val_resolved,
        "test": test_resolved,
    }

    required_images = build_required_images(resolved_splits)

    train_out = args.out_dir / "cocoqa_train_resolved.jsonl"
    val_out = args.out_dir / "cocoqa_val_resolved.jsonl"
    test_out = args.out_dir / "cocoqa_test_resolved.jsonl"
    required_images_out = args.out_dir / "required_images.jsonl"
    stats_out = args.out_dir / "cocoqa_resolved_stats.json"

    print("Writing resolved manifests...")
    write_jsonl(train_resolved, train_out)
    write_jsonl(val_resolved, val_out)
    write_jsonl(test_resolved, test_out)
    write_jsonl(required_images, required_images_out)

    stats = {
        "train": split_stats(train_resolved),
        "val": split_stats(val_resolved),
        "test": split_stats(test_resolved),
        "required_images": required_images_stats(required_images),
        "missing": {
            "train_missing_qa_rows": len(train_missing),
            "val_missing_qa_rows": len(val_missing),
            "test_missing_qa_rows": len(test_missing),
            "total_missing_qa_rows": missing_total,
        },
        "paths": {
            "train_resolved": str(train_out),
            "val_resolved": str(val_out),
            "test_resolved": str(test_out),
            "required_images": str(required_images_out),
        },
    }

    with stats_out.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("Done.")
    print(f"Train resolved:    {train_out}")
    print(f"Val resolved:      {val_out}")
    print(f"Test resolved:     {test_out}")
    print(f"Required images:   {required_images_out}")
    print(f"Stats:             {stats_out}")
    print()
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()