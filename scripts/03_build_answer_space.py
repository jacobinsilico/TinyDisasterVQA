#!/usr/bin/env python3
"""
03_build_answer_space.py

Builds final answer-space metadata for TinyDisasterVQA training.

Input:
  outputs/processed/floodnet_manifest.csv

Outputs:
  outputs/answer_space/
    answer_space.json
    edge_class_counts.csv
    edge_head_counts.csv
    edge_head_label_counts.csv
    original_answer_counts.csv
    class_weights_edge_global.json
    class_weights_by_head.json
    answer_space_summary.txt

This script does not modify the dataset or manifest.
It only derives target metadata for training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


HEAD_ORDER = ["binary", "condition", "density", "count"]


def save_json(obj: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def normalize_weights(counts: dict[str, int], mode: str = "inverse_sqrt") -> dict[str, float]:
    """
    Creates class weights for optional weighted CE loss.

    inverse:
      w_c = 1 / count_c

    inverse_sqrt:
      w_c = 1 / sqrt(count_c)

    Then normalize weights to have mean 1.
    """
    if not counts:
        return {}

    raw = {}

    for label, count in counts.items():
        count = max(int(count), 1)

        if mode == "inverse":
            raw[label] = 1.0 / count
        elif mode == "inverse_sqrt":
            raw[label] = 1.0 / (count ** 0.5)
        else:
            raise ValueError(f"Unknown weight mode: {mode}")

    mean_weight = sum(raw.values()) / len(raw)
    return {label: weight / mean_weight for label, weight in raw.items()}


def check_required_columns(df: pd.DataFrame) -> None:
    required = [
        "split",
        "answer_norm",
        "original_answer_label",
        "edge_head",
        "edge_answer",
        "edge_class",
        "edge_global_label",
        "edge_head_label",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")


def build_edge_global_maps(df: pd.DataFrame) -> tuple[dict[str, int], dict[str, str]]:
    pairs = (
        df[["edge_class", "edge_global_label"]]
        .drop_duplicates()
        .sort_values("edge_global_label")
    )

    edge_class_to_label = {
        str(row["edge_class"]): int(row["edge_global_label"])
        for _, row in pairs.iterrows()
    }

    edge_label_to_class = {
        str(int(row["edge_global_label"])): str(row["edge_class"])
        for _, row in pairs.iterrows()
    }

    expected_labels = list(range(len(edge_class_to_label)))
    actual_labels = sorted(edge_class_to_label.values())

    if actual_labels != expected_labels:
        raise ValueError(
            f"edge_global_label must be contiguous from 0 to N-1. Got {actual_labels}"
        )

    return edge_class_to_label, edge_label_to_class


def build_head_maps(df: pd.DataFrame) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, str]]]:
    head_label_maps: dict[str, dict[str, int]] = {}
    head_idx_to_answer: dict[str, dict[str, str]] = {}

    for head in HEAD_ORDER:
        head_df = df[df["edge_head"] == head]

        if len(head_df) == 0:
            continue

        pairs = (
            head_df[["edge_answer", "edge_head_label"]]
            .drop_duplicates()
            .sort_values("edge_head_label")
        )

        label_map = {
            str(row["edge_answer"]): int(row["edge_head_label"])
            for _, row in pairs.iterrows()
        }

        idx_map = {
            str(int(row["edge_head_label"])): str(row["edge_answer"])
            for _, row in pairs.iterrows()
        }

        expected = list(range(len(label_map)))
        actual = sorted(label_map.values())

        if actual != expected:
            raise ValueError(
                f"edge_head_label for head '{head}' must be contiguous from 0 to N-1. "
                f"Got {actual}"
            )

        head_label_maps[head] = label_map
        head_idx_to_answer[head] = idx_map

    return head_label_maps, head_idx_to_answer


def build_original_maps(df: pd.DataFrame) -> tuple[dict[str, int], dict[str, str]]:
    pairs = (
        df[["answer_norm", "original_answer_label"]]
        .drop_duplicates()
        .sort_values("original_answer_label")
    )

    answer_to_label = {
        str(row["answer_norm"]): int(row["original_answer_label"])
        for _, row in pairs.iterrows()
    }

    label_to_answer = {
        str(int(row["original_answer_label"])): str(row["answer_norm"])
        for _, row in pairs.iterrows()
    }

    return answer_to_label, label_to_answer


def split_set(df: pd.DataFrame, split: str, column: str) -> set[str]:
    return set(df[df["split"] == split][column].astype(str).unique())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/processed/floodnet_manifest.csv"),
        help="Path to floodnet_manifest.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/answer_space"),
        help="Directory where answer-space files are written.",
    )
    parser.add_argument(
        "--weight-mode",
        type=str,
        default="inverse_sqrt",
        choices=["inverse", "inverse_sqrt"],
        help="Class weighting strategy.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TinyDisasterVQA / Build Answer Space")
    print("=" * 80)
    print(f"Manifest:   {args.manifest.resolve()}")
    print(f"Output dir: {output_dir.resolve()}")
    print(f"Weights:    {args.weight_mode}")
    print()

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    df = pd.read_csv(args.manifest)
    check_required_columns(df)

    edge_class_to_label, edge_label_to_class = build_edge_global_maps(df)
    head_label_maps, head_idx_to_answer = build_head_maps(df)
    original_answer_to_label, original_label_to_answer = build_original_maps(df)

    # Counts.
    edge_class_counts = (
        df.groupby(["split", "edge_class", "edge_global_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "edge_global_label"])
    )

    edge_head_counts = (
        df.groupby(["split", "edge_head"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "edge_head"])
    )

    edge_head_label_counts = (
        df.groupby(["split", "edge_head", "edge_answer", "edge_head_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "edge_head", "edge_head_label"])
    )

    original_answer_counts = (
        df.groupby(["split", "answer_norm", "original_answer_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "original_answer_label"])
    )

    edge_class_counts.to_csv(output_dir / "edge_class_counts.csv", index=False)
    edge_head_counts.to_csv(output_dir / "edge_head_counts.csv", index=False)
    edge_head_label_counts.to_csv(output_dir / "edge_head_label_counts.csv", index=False)
    original_answer_counts.to_csv(output_dir / "original_answer_counts.csv", index=False)

    # Train-only class weights.
    train_df = df[df["split"] == "train"]

    train_edge_counts = (
        train_df.groupby("edge_class")
        .size()
        .astype(int)
        .to_dict()
    )

    class_weights_edge_global_by_class = normalize_weights(
        {str(k): int(v) for k, v in train_edge_counts.items()},
        mode=args.weight_mode,
    )

    # Convert class-name weights to index-ordered weights.
    class_weights_edge_global_by_label = {
        str(edge_class_to_label[edge_class]): class_weights_edge_global_by_class[edge_class]
        for edge_class in edge_class_to_label
    }

    class_weights_by_head: dict[str, dict[str, float]] = {}
    class_weights_by_head_label: dict[str, dict[str, float]] = {}

    for head in head_label_maps:
        head_train = train_df[train_df["edge_head"] == head]

        answer_counts = (
            head_train.groupby("edge_answer")
            .size()
            .astype(int)
            .to_dict()
        )

        weights_by_answer = normalize_weights(
            {str(k): int(v) for k, v in answer_counts.items()},
            mode=args.weight_mode,
        )

        weights_by_label = {
            str(head_label_maps[head][answer]): weights_by_answer[answer]
            for answer in head_label_maps[head]
        }

        class_weights_by_head[head] = weights_by_answer
        class_weights_by_head_label[head] = weights_by_label

    save_json(class_weights_edge_global_by_class, output_dir / "class_weights_edge_global_by_class.json")
    save_json(class_weights_edge_global_by_label, output_dir / "class_weights_edge_global_by_label.json")
    save_json(class_weights_by_head, output_dir / "class_weights_by_head_by_answer.json")
    save_json(class_weights_by_head_label, output_dir / "class_weights_by_head_by_label.json")

    # Leakage / unseen checks.
    train_edge_classes = split_set(df, "train", "edge_class")
    valid_edge_classes = split_set(df, "valid", "edge_class")
    test_edge_classes = split_set(df, "test", "edge_class")

    valid_unseen_edge = sorted(valid_edge_classes - train_edge_classes)
    test_unseen_edge = sorted(test_edge_classes - train_edge_classes)

    train_original = split_set(df, "train", "answer_norm")
    valid_original = split_set(df, "valid", "answer_norm")
    test_original = split_set(df, "test", "answer_norm")

    valid_unseen_original = sorted(valid_original - train_original)
    test_unseen_original = sorted(test_original - train_original)

    # Final answer space object.
    answer_space = {
        "target_modes": {
            "original_global": {
                "description": "Original FloodNet answer labels. Not ideal because val/test contain unseen raw count labels.",
                "num_classes_used_in_manifest": len(original_answer_to_label),
                "answer_to_label": original_answer_to_label,
                "label_to_answer": original_label_to_answer,
                "valid_unseen_answers_vs_train": valid_unseen_original,
                "test_unseen_answers_vs_train": test_unseen_original,
            },
            "edge_global": {
                "description": "Single 19-class edge-clean target using answer groups and capped counts.",
                "num_classes": len(edge_class_to_label),
                "class_to_label": edge_class_to_label,
                "label_to_class": edge_label_to_class,
                "valid_unseen_classes_vs_train": valid_unseen_edge,
                "test_unseen_classes_vs_train": test_unseen_edge,
            },
            "edge_multihead": {
                "description": "Question-type-aware multi-head target. Use edge_head and edge_head_label.",
                "num_heads": len(head_label_maps),
                "head_order": [head for head in HEAD_ORDER if head in head_label_maps],
                "head_label_maps": head_label_maps,
                "head_idx_to_answer": head_idx_to_answer,
            },
        },
        "recommended_training_targets": {
            "teacher_fair_tinyvqa_style": "original_global or edge_global with LSTM question encoder",
            "teacher_strong": "edge_global or edge_multihead with ConvNeXt/EfficientNet image encoder and LSTM question encoder",
            "student_gap9": "edge_multihead with tiny CNN and template/question embedding",
            "deployment_main": "edge_multihead",
        },
        "class_weights": {
            "weight_mode": args.weight_mode,
            "edge_global_by_class_file": "class_weights_edge_global_by_class.json",
            "edge_global_by_label_file": "class_weights_edge_global_by_label.json",
            "by_head_answer_file": "class_weights_by_head_by_answer.json",
            "by_head_label_file": "class_weights_by_head_by_label.json",
        },
    }

    save_json(answer_space, output_dir / "answer_space.json")

    # Summary.
    summary_lines = []
    summary_lines.append("TinyDisasterVQA / Answer Space Summary")
    summary_lines.append("=" * 80)
    summary_lines.append(f"Manifest: {args.manifest}")
    summary_lines.append(f"Total samples: {len(df)}")
    summary_lines.append("")
    summary_lines.append("Original global answer space:")
    summary_lines.append(f"  Classes used in manifest: {len(original_answer_to_label)}")
    summary_lines.append(f"  Valid unseen answers vs train: {valid_unseen_original}")
    summary_lines.append(f"  Test unseen answers vs train:  {test_unseen_original}")
    summary_lines.append("")
    summary_lines.append("Edge global answer space:")
    summary_lines.append(f"  Classes: {len(edge_class_to_label)}")
    summary_lines.append(f"  Valid unseen classes vs train: {valid_unseen_edge}")
    summary_lines.append(f"  Test unseen classes vs train:  {test_unseen_edge}")
    summary_lines.append("")
    summary_lines.append("Edge global classes:")
    for edge_class, idx in edge_class_to_label.items():
        train_count = int(train_edge_counts.get(edge_class, 0))
        summary_lines.append(f"  {idx:02d}: {edge_class:<25} train_count={train_count}")
    summary_lines.append("")
    summary_lines.append("Edge multi-head answer space:")
    for head in [h for h in HEAD_ORDER if h in head_label_maps]:
        summary_lines.append(f"  {head}: {len(head_label_maps[head])} classes")
        for answer, idx in head_label_maps[head].items():
            count = int(
                len(train_df[(train_df["edge_head"] == head) & (train_df["edge_answer"] == answer)])
            )
            summary_lines.append(f"    {idx:02d}: {answer:<12} train_count={count}")
    summary_lines.append("")
    summary_lines.append("Recommended:")
    summary_lines.append("  Use edge_global for first simple baseline.")
    summary_lines.append("  Use edge_multihead for final TinyDisasterVQA/GAP9 model.")
    summary_lines.append("  Keep original_global only for TinyVQA-style comparison, not deployment.")
    summary_lines.append("")

    summary_path = output_dir / "answer_space_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n".join(summary_lines))
    print(f"Done. Wrote answer-space outputs to: {output_dir}")


if __name__ == "__main__":
    main()