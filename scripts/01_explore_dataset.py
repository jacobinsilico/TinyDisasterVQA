#!/usr/bin/env python3
"""
01_explore_dataset.py

Basic exploration script for FloodNet-VQA inside TinyDisasterVQA.

Expected dataset structure:

dataset/
  data/
    train_annotations.json
    valid_annotations.json
    test_annotations.json
    class_to_label.json
    word_to_token.json
  images/
    train_images/
    valid_images/
    test_images/

This script does NOT modify the dataset.
It only reads annotations/images and writes exploration outputs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


SPLITS = {
    "train": {
        "annotation_file": "train_annotations.json",
        "image_dir": "train_images",
    },
    "valid": {
        "annotation_file": "valid_annotations.json",
        "image_dir": "valid_images",
    },
    "test": {
        "annotation_file": "test_annotations.json",
        "image_dir": "test_images",
    },
}


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_answer(answer: Any) -> str:
    return str(answer).strip().lower()


def is_counting_question(question_type: str) -> bool:
    return "counting" in question_type.lower()


def parse_count(answer: Any) -> int | None:
    answer_str = str(answer).strip()
    if answer_str.isdigit():
        return int(answer_str)
    return None


def capped_count(answer: Any, cap: int = 10) -> str | None:
    value = parse_count(answer)
    if value is None:
        return None
    if value > cap:
        return f"{cap}+"
    return str(value)


def find_image_path(dataset_root: Path, split: str, image_id: str) -> tuple[Path, bool]:
    image_dir = dataset_root / "images" / SPLITS[split]["image_dir"]
    expected_path = image_dir / image_id

    if expected_path.exists():
        return expected_path, True

    # Fallback: case-insensitive search, useful if extensions differ as .jpg/.JPG.
    if image_dir.exists():
        target = image_id.lower()
        for candidate in image_dir.iterdir():
            if candidate.name.lower() == target:
                return candidate, True

    return expected_path, False


def load_split_dataframe(dataset_root: Path, split: str, count_cap: int) -> pd.DataFrame:
    annotation_path = dataset_root / "data" / SPLITS[split]["annotation_file"]
    records = load_json(annotation_path)

    if not isinstance(records, list):
        raise ValueError(f"Expected list of annotations in {annotation_path}")

    rows = []
    for row in records:
        image_id = str(row.get("Image_ID", "")).strip()
        question = str(row.get("Question", "")).strip()
        question_type = str(row.get("Question_Type", "")).strip()
        answer_raw = row.get("Ground_Truth", "")

        image_path, image_exists = find_image_path(dataset_root, split, image_id)
        counting = is_counting_question(question_type)

        rows.append(
            {
                "split": split,
                "image_id": image_id,
                "image_path": str(image_path),
                "image_exists": image_exists,
                "question_id": row.get("Question_ID"),
                "question": question,
                "question_norm": question.lower().strip(),
                "question_type": question_type,
                "answer_raw": str(answer_raw).strip(),
                "answer_norm": normalize_answer(answer_raw),
                "is_counting": counting,
                "count_value": parse_count(answer_raw) if counting else None,
                "count_capped": capped_count(answer_raw, cap=count_cap) if counting else None,
                "attention_map_dir": row.get("AttentionMap_dir"),
                "grad_cam_attention": row.get("grad_cam_attention"),
            }
        )

    return pd.DataFrame(rows)


def save_counter_csv(counter: Counter, path: Path, key_name: str, count_name: str = "count") -> None:
    df = pd.DataFrame(counter.items(), columns=[key_name, count_name])
    df = df.sort_values(count_name, ascending=False)
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
        help="Path to FloodNet dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/exploration"),
        help="Directory where exploration outputs will be written.",
    )
    parser.add_argument(
        "--count-cap",
        type=int,
        default=10,
        help="Cap for counting labels, e.g. values >10 become 10+.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TinyDisasterVQA / FloodNet-VQA Dataset Exploration")
    print("=" * 80)
    print(f"Dataset root: {dataset_root.resolve()}")
    print(f"Output dir:   {output_dir.resolve()}")
    print(f"Count cap:    {args.count_cap}+")
    print()

    # Load optional metadata files.
    class_to_label_path = dataset_root / "data" / "class_to_label.json"
    word_to_token_path = dataset_root / "data" / "word_to_token.json"

    class_to_label = load_json(class_to_label_path) if class_to_label_path.exists() else {}
    word_to_token = load_json(word_to_token_path) if word_to_token_path.exists() else {}

    # Load annotations.
    split_dfs = []
    for split in SPLITS:
        df_split = load_split_dataframe(dataset_root, split, args.count_cap)
        split_dfs.append(df_split)

    df = pd.concat(split_dfs, ignore_index=True)

    # Save full normalized annotations for inspection.
    df.to_csv(output_dir / "all_annotations_normalized.csv", index=False)

    # Split stats.
    split_stats = []
    for split, df_split in df.groupby("split"):
        split_stats.append(
            {
                "split": split,
                "num_qa_samples": len(df_split),
                "num_unique_images": df_split["image_id"].nunique(),
                "num_unique_questions": df_split["question_norm"].nunique(),
                "num_unique_question_types": df_split["question_type"].nunique(),
                "num_unique_answers": df_split["answer_norm"].nunique(),
                "num_missing_images": int((~df_split["image_exists"]).sum()),
            }
        )

    split_stats_df = pd.DataFrame(split_stats).sort_values("split")
    split_stats_df.to_csv(output_dir / "split_stats.csv", index=False)

    # Question types.
    question_type_counts = (
        df.groupby(["split", "question_type"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "count"], ascending=[True, False])
    )
    question_type_counts.to_csv(output_dir / "question_type_counts_by_split.csv", index=False)

    question_type_counts_total = (
        df.groupby("question_type")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    question_type_counts_total["percentage"] = (
        100.0 * question_type_counts_total["count"] / len(df)
    )
    question_type_counts_total.to_csv(output_dir / "question_type_counts_total.csv", index=False)

    # Question templates.
    question_templates = (
        df.groupby(["question_norm", "question_type"])
        .agg(
            count=("question_norm", "size"),
            example_question=("question", "first"),
            num_unique_answers=("answer_norm", "nunique"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )
    question_templates.insert(0, "question_template_id", range(len(question_templates)))
    question_templates.to_csv(output_dir / "question_templates.csv", index=False)

    # Answer distributions.
    answer_counts = (
        df.groupby("answer_norm")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    answer_counts["percentage"] = 100.0 * answer_counts["count"] / len(df)
    answer_counts.to_csv(output_dir / "answer_counts_total.csv", index=False)

    answer_counts_by_split = (
        df.groupby(["split", "answer_norm"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "count"], ascending=[True, False])
    )
    answer_counts_by_split.to_csv(output_dir / "answer_counts_by_split.csv", index=False)

    answer_counts_by_type = (
        df.groupby(["question_type", "answer_norm"])
        .size()
        .reset_index(name="count")
        .sort_values(["question_type", "count"], ascending=[True, False])
    )
    answer_counts_by_type.to_csv(output_dir / "answer_counts_by_type.csv", index=False)

    # Counting-specific stats.
    df_count = df[df["is_counting"]].copy()
    count_distribution = (
        df_count.groupby(["split", "answer_norm", "count_value", "count_capped"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "count_value"], ascending=[True, True])
    )
    count_distribution.to_csv(output_dir / "count_answer_distribution_by_split.csv", index=False)

    count_distribution_total = (
        df_count.groupby(["answer_norm", "count_value", "count_capped"])
        .size()
        .reset_index(name="count")
        .sort_values("count_value", ascending=True)
    )
    count_distribution_total.to_csv(output_dir / "count_answer_distribution_total.csv", index=False)

    # Count labels seen/unseen across splits.
    train_counts = set(df_count[df_count["split"] == "train"]["answer_norm"].dropna())
    valid_counts = set(df_count[df_count["split"] == "valid"]["answer_norm"].dropna())
    test_counts = set(df_count[df_count["split"] == "test"]["answer_norm"].dropna())

    unseen_count_rows = []
    for split_name, split_counts in [("valid", valid_counts), ("test", test_counts)]:
        unseen = sorted(split_counts - train_counts, key=lambda x: int(x) if x.isdigit() else 9999)
        for label in unseen:
            unseen_count_rows.append(
                {
                    "split": split_name,
                    "count_label_unseen_in_train": label,
                }
            )

    pd.DataFrame(unseen_count_rows).to_csv(
        output_dir / "count_labels_unseen_in_train.csv", index=False
    )

    # Missing images.
    missing_images = df[~df["image_exists"]][
        ["split", "image_id", "image_path", "question_id", "question"]
    ].copy()
    missing_images.to_csv(output_dir / "missing_images.csv", index=False)

    # Duplicate question IDs.
    duplicate_question_ids = (
        df[df.duplicated("question_id", keep=False)]
        .sort_values("question_id")
        [["split", "question_id", "image_id", "question", "answer_raw"]]
    )
    duplicate_question_ids.to_csv(output_dir / "duplicate_question_ids_global.csv", index=False)

    duplicate_question_ids_by_split = (
        df[df.duplicated(["split", "question_id"], keep=False)]
        .sort_values(["split", "question_id"])
        [["split", "question_id", "image_id", "question", "answer_raw"]]
    )
    duplicate_question_ids_by_split.to_csv(
        output_dir / "duplicate_question_ids_by_split.csv", index=False
    )

    # Image overlap between splits.
    split_images = {
        split: set(df[df["split"] == split]["image_id"])
        for split in SPLITS
    }

    overlap_rows = []
    split_names = list(SPLITS.keys())
    for i, split_a in enumerate(split_names):
        for split_b in split_names[i + 1 :]:
            overlap = split_images[split_a] & split_images[split_b]
            overlap_rows.append(
                {
                    "split_a": split_a,
                    "split_b": split_b,
                    "num_overlapping_images": len(overlap),
                    "example_overlapping_images": ", ".join(sorted(list(overlap))[:20]),
                }
            )

    image_overlap_df = pd.DataFrame(overlap_rows)
    image_overlap_df.to_csv(output_dir / "image_overlap_between_splits.csv", index=False)

    # Class/token mapping summaries.
    mapping_summary = pd.DataFrame(
        [
            {
                "file": "class_to_label.json",
                "num_entries": len(class_to_label),
            },
            {
                "file": "word_to_token.json",
                "num_entries": len(word_to_token),
            },
        ]
    )
    mapping_summary.to_csv(output_dir / "mapping_summary.csv", index=False)

    # Majority baselines.
    train_df = df[df["split"] == "train"]
    valid_df = df[df["split"] == "valid"]
    test_df = df[df["split"] == "test"]

    global_majority_answer = train_df["answer_norm"].mode().iloc[0]

    def majority_acc(eval_df: pd.DataFrame, answer: str) -> float:
        return float((eval_df["answer_norm"] == answer).mean())

    majority_by_type = (
        train_df.groupby("question_type")["answer_norm"]
        .agg(lambda x: x.mode().iloc[0])
        .to_dict()
    )

    def majority_by_type_acc(eval_df: pd.DataFrame) -> float:
        preds = eval_df["question_type"].map(majority_by_type)
        return float((preds == eval_df["answer_norm"]).mean())

    baseline_rows = [
        {
            "baseline": f"global_majority_train_answer={global_majority_answer}",
            "valid_accuracy": majority_acc(valid_df, global_majority_answer),
            "test_accuracy": majority_acc(test_df, global_majority_answer),
        },
        {
            "baseline": "majority_answer_per_question_type",
            "valid_accuracy": majority_by_type_acc(valid_df),
            "test_accuracy": majority_by_type_acc(test_df),
        },
    ]
    pd.DataFrame(baseline_rows).to_csv(output_dir / "majority_baselines.csv", index=False)

    # Human-readable summary.
    summary_lines = []
    summary_lines.append("TinyDisasterVQA / FloodNet-VQA Dataset Exploration")
    summary_lines.append("=" * 80)
    summary_lines.append(f"Dataset root: {dataset_root.resolve()}")
    summary_lines.append(f"Total QA samples: {len(df)}")
    summary_lines.append(f"Total unique images: {df['image_id'].nunique()}")
    summary_lines.append(f"Total unique question templates: {df['question_norm'].nunique()}")
    summary_lines.append(f"Total unique question types: {df['question_type'].nunique()}")
    summary_lines.append(f"Total unique answers: {df['answer_norm'].nunique()}")
    summary_lines.append(f"Question vocabulary size from word_to_token.json: {len(word_to_token)}")
    summary_lines.append(f"Answer class count from class_to_label.json: {len(class_to_label)}")
    summary_lines.append("")

    summary_lines.append("Split stats:")
    summary_lines.append(split_stats_df.to_string(index=False))
    summary_lines.append("")

    summary_lines.append("Question type counts:")
    summary_lines.append(question_type_counts_total.to_string(index=False))
    summary_lines.append("")

    summary_lines.append("Top 20 answers:")
    summary_lines.append(answer_counts.head(20).to_string(index=False))
    summary_lines.append("")

    summary_lines.append("Counting summary:")
    summary_lines.append(f"Counting QA samples: {len(df_count)}")
    summary_lines.append(
        f"Counting labels unseen in valid vs train: {sorted(valid_counts - train_counts, key=lambda x: int(x) if x.isdigit() else 9999)}"
    )
    summary_lines.append(
        f"Counting labels unseen in test vs train: {sorted(test_counts - train_counts, key=lambda x: int(x) if x.isdigit() else 9999)}"
    )
    summary_lines.append("")

    summary_lines.append("Image split overlap:")
    summary_lines.append(image_overlap_df.to_string(index=False))
    summary_lines.append("")

    summary_lines.append("Missing images:")
    summary_lines.append(f"Total missing image references: {len(missing_images)}")
    summary_lines.append("")

    summary_lines.append("Majority baselines:")
    summary_lines.append(pd.DataFrame(baseline_rows).to_string(index=False))
    summary_lines.append("")

    summary_path = output_dir / "dataset_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n".join(summary_lines))
    print()
    print(f"Done. Wrote exploration outputs to: {output_dir}")


if __name__ == "__main__":
    main()