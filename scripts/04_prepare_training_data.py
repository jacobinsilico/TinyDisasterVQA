#!/usr/bin/env python3
"""
04_prepare_training_data.py

Prepares final train/valid/test CSVs for TinyDisasterVQA training.

Current submission formulation:
  Main target is target_edge_global:
    - cap5  -> 14-class single-head task
    - cap10 -> 19-class single-head task

Input:
  outputs/processed/floodnet_manifest.csv
  outputs/processed/word_to_token.json
  outputs/answer_space/answer_space.json

Outputs:
  outputs/training_data/
    train.csv
    valid.csv
    test.csv
    metadata.json
    question_token_oov.csv
    training_data_summary.txt

This script does NOT copy or preprocess images.
It only prepares clean tabular training files.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_HEAD_TO_ID = {
    "binary": 0,
    "condition": 1,
    "density": 2,
    "count": 3,
}


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def tokenize_question(question: str) -> list[str]:
    """
    Simple tokenizer matching the tiny FloodNet vocabulary style.

    Example:
      "Is the entire road non flooded?"
      -> ["is", "the", "entire", "road", "non", "flooded"]
    """
    question = str(question).strip().lower()
    return re.findall(r"[a-z0-9]+", question)


def encode_question(
    question: str,
    word_to_token: dict[str, int],
    max_len: int,
    pad_id: int = 0,
    unk_id: int = 1,
) -> tuple[list[int], int, list[str]]:
    tokens = tokenize_question(question)
    original_len = len(tokens)

    oov_tokens = [tok for tok in tokens if tok not in word_to_token]
    token_ids = [int(word_to_token.get(tok, unk_id)) for tok in tokens]

    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]

    question_len = min(original_len, max_len)

    if len(token_ids) < max_len:
        token_ids = token_ids + [pad_id] * (max_len - len(token_ids))

    return token_ids, question_len, oov_tokens


def check_required_columns(df: pd.DataFrame) -> None:
    required = [
        "split",
        "image_id",
        "image_rel_path",
        "question_id",
        "question",
        "question_norm",
        "question_type",
        "question_template_id",
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
        raise ValueError(f"Manifest missing required columns: {missing}")


def get_edge_global_space(answer_space: dict[str, Any]) -> dict[str, Any]:
    try:
        return answer_space["target_modes"]["edge_global"]
    except KeyError as exc:
        raise KeyError(
            "answer_space.json must contain target_modes.edge_global. "
            "Run scripts/03_build_answer_space.py first."
        ) from exc


def get_head_space(answer_space: dict[str, Any]) -> dict[str, Any]:
    target_modes = answer_space.get("target_modes", {})

    if "edge_head_local" in target_modes:
        return target_modes["edge_head_local"]

    # Backward compatibility with old answer_space.json.
    if "edge_multihead" in target_modes:
        return target_modes["edge_multihead"]

    return {}


def build_head_to_id(answer_space: dict[str, Any]) -> dict[str, int]:
    head_order = answer_space.get("head_order")

    if not head_order:
        head_space = get_head_space(answer_space)
        head_order = head_space.get("head_order")

    if not head_order:
        return DEFAULT_HEAD_TO_ID.copy()

    return {
        str(head): idx
        for idx, head in enumerate(head_order)
    }


def validate_answer_space_against_manifest(
    df: pd.DataFrame,
    answer_space: dict[str, Any],
) -> None:
    edge_global = get_edge_global_space(answer_space)

    num_classes = int(edge_global["num_classes"])
    class_to_label = {
        str(k): int(v)
        for k, v in edge_global["class_to_label"].items()
    }

    manifest_class_to_label = (
        df[["edge_class", "edge_global_label"]]
        .drop_duplicates()
        .sort_values("edge_global_label")
    )

    manifest_map = {
        str(row["edge_class"]): int(row["edge_global_label"])
        for _, row in manifest_class_to_label.iterrows()
    }

    if manifest_map != class_to_label:
        raise ValueError(
            "Manifest edge_class -> edge_global_label mapping does not match "
            "answer_space.json. Re-run scripts 02 and 03 consistently."
        )

    labels = sorted(df["edge_global_label"].astype(int).unique().tolist())
    expected = list(range(num_classes))

    if labels != expected:
        raise ValueError(
            f"edge_global_label values must be contiguous 0..{num_classes - 1}. "
            f"Got {labels}."
        )


def validate_splits(df: pd.DataFrame) -> None:
    expected_splits = {"train", "valid", "test"}
    observed_splits = set(df["split"].astype(str).unique())

    if observed_splits != expected_splits:
        raise ValueError(
            f"Expected splits {sorted(expected_splits)}, got {sorted(observed_splits)}."
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/processed/floodnet_manifest.csv"),
    )
    parser.add_argument(
        "--word-to-token",
        type=Path,
        default=Path("outputs/processed/word_to_token.json"),
    )
    parser.add_argument(
        "--answer-space",
        type=Path,
        default=Path("outputs/answer_space/answer_space.json"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/training_data"),
    )
    parser.add_argument(
        "--max-question-len",
        type=int,
        default=11,
        help="Fixed padded question length. TinyVQA paper mentions max length 11.",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TinyDisasterVQA / Prepare Training Data")
    print("=" * 80)
    print(f"Manifest:         {args.manifest.resolve()}")
    print(f"Word-to-token:    {args.word_to_token.resolve()}")
    print(f"Answer space:     {args.answer_space.resolve()}")
    print(f"Dataset root:     {args.dataset_root.resolve()}")
    print(f"Output dir:       {args.output_dir.resolve()}")
    print(f"Max question len: {args.max_question_len}")
    print()

    if args.max_question_len < 1:
        raise ValueError("--max-question-len must be >= 1.")

    df = pd.read_csv(args.manifest)
    check_required_columns(df)
    validate_splits(df)

    word_to_token = load_json(args.word_to_token)
    word_to_token = {
        str(k): int(v)
        for k, v in word_to_token.items()
    }

    answer_space = load_json(args.answer_space)
    validate_answer_space_against_manifest(df, answer_space)

    edge_global_space = get_edge_global_space(answer_space)
    head_space = get_head_space(answer_space)

    target_formulation = answer_space.get("target_formulation", "single_head_edge_global")
    count_cap = answer_space.get("count_cap", edge_global_space.get("count_cap"))
    num_edge_global_classes = int(edge_global_space["num_classes"])

    edge_class_to_label = {
        str(k): int(v)
        for k, v in edge_global_space["class_to_label"].items()
    }

    edge_label_to_class = {
        str(k): str(v)
        for k, v in edge_global_space["label_to_class"].items()
    }

    head_label_maps = head_space.get("head_label_maps", {})
    head_idx_to_answer = head_space.get("head_idx_to_answer", {})

    head_to_id = build_head_to_id(answer_space)
    id_to_head = {
        str(v): k
        for k, v in head_to_id.items()
    }

    pad_id = 0
    unk_id = int(word_to_token.get("unk", word_to_token.get("UNK", 1)))

    encoded_questions = []
    question_lengths = []
    token_strings = []
    oov_counter: Counter[str] = Counter()
    too_long_count = 0
    max_observed_len = 0

    for question in df["question"]:
        tokens = tokenize_question(question)
        max_observed_len = max(max_observed_len, len(tokens))

        if len(tokens) > args.max_question_len:
            too_long_count += 1

        token_ids, question_len, oov_tokens = encode_question(
            question=question,
            word_to_token=word_to_token,
            max_len=args.max_question_len,
            pad_id=pad_id,
            unk_id=unk_id,
        )

        for tok in oov_tokens:
            oov_counter[tok] += 1

        encoded_questions.append(token_ids)
        question_lengths.append(question_len)
        token_strings.append(" ".join(str(x) for x in token_ids))

    df["question_token_ids"] = token_strings
    df["question_length"] = question_lengths

    df["head_id"] = df["edge_head"].map(head_to_id)

    if df["head_id"].isna().any():
        bad_heads = sorted(df[df["head_id"].isna()]["edge_head"].unique())
        raise ValueError(f"Unknown edge_head values: {bad_heads}")

    df["head_id"] = df["head_id"].astype(int)

    # Standard target aliases used by training scripts.
    df["target_original"] = df["original_answer_label"].astype(int)
    df["target_edge_global"] = df["edge_global_label"].astype(int)
    df["target_edge_head"] = df["edge_head_label"].astype(int)

    # Main target alias for the current formulation.
    df["target"] = df["target_edge_global"].astype(int)

    if int(df["target"].min()) < 0 or int(df["target"].max()) >= num_edge_global_classes:
        raise ValueError(
            f"target labels must be in [0, {num_edge_global_classes - 1}], "
            f"got min={df['target'].min()}, max={df['target'].max()}."
        )

    # Add full image path for convenience.
    df["image_path"] = df["image_rel_path"].apply(
        lambda p: str(args.dataset_root / str(p))
    )

    # Verify image files exist.
    missing_images = []

    for _, row in df.iterrows():
        if not Path(row["image_path"]).exists():
            missing_images.append(row["image_path"])

    if missing_images:
        raise FileNotFoundError(
            f"Found {len(missing_images)} missing images. Example: {missing_images[:5]}"
        )

    # Keep training columns clean and explicit.
    keep_cols = [
        "split",
        "image_id",
        "image_path",
        "image_rel_path",
        "question_id",
        "question",
        "question_norm",
        "question_token_ids",
        "question_length",
        "question_template_id",
        "question_type",
        "edge_head",
        "head_id",
        "edge_answer",
        "edge_class",
        "answer_norm",
        "target",
        "target_original",
        "target_edge_global",
        "target_edge_head",
        "count_value",
        "count_capped",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]

    train_df = df[df["split"] == "train"][keep_cols].copy()
    valid_df = df[df["split"] == "valid"][keep_cols].copy()
    test_df = df[df["split"] == "test"][keep_cols].copy()

    train_df.to_csv(args.output_dir / "train.csv", index=False)
    valid_df.to_csv(args.output_dir / "valid.csv", index=False)
    test_df.to_csv(args.output_dir / "test.csv", index=False)

    if oov_counter:
        oov_df = pd.DataFrame(
            [
                {"token": tok, "count": count}
                for tok, count in oov_counter.items()
            ]
        ).sort_values("count", ascending=False)
    else:
        oov_df = pd.DataFrame(columns=["token", "count"])

    oov_df.to_csv(args.output_dir / "question_token_oov.csv", index=False)

    num_question_templates = int(df["question_template_id"].nunique())

    split_sizes = {
        "train": int(len(train_df)),
        "valid": int(len(valid_df)),
        "test": int(len(test_df)),
    }

    metadata = {
        "dataset_root": str(args.dataset_root),
        "target_formulation": target_formulation,
        "main_target_mode": "edge_global",
        "main_target_column": "target_edge_global",
        "target_alias_column": "target",
        "count_cap": count_cap,
        "num_edge_global_classes": num_edge_global_classes,
        "num_classes": num_edge_global_classes,
        "num_question_templates": num_question_templates,
        "max_question_len": args.max_question_len,
        "max_observed_question_len": max_observed_len,
        "questions_longer_than_max_len": too_long_count,
        "pad_id": pad_id,
        "unk_id": unk_id,
        "vocab_size_with_pad": max(word_to_token.values()) + 1,
        "word_to_token": word_to_token,
        "head_to_id": head_to_id,
        "id_to_head": id_to_head,
        "edge_class_to_label": edge_class_to_label,
        "edge_label_to_class": edge_label_to_class,
        "head_label_maps": head_label_maps,
        "head_idx_to_answer": head_idx_to_answer,
        "answer_space": answer_space,
        "split_sizes": split_sizes,
        "files": {
            "train_csv": "train.csv",
            "valid_csv": "valid.csv",
            "test_csv": "test.csv",
            "question_token_oov_csv": "question_token_oov.csv",
        },
        "targets": {
            "target": "Alias for target_edge_global; main single-head edge-clean class.",
            "target_original": "Original FloodNet answer label. Not recommended for main training.",
            "target_edge_global": "Main compact single-head edge-clean global class.",
            "target_edge_head": (
                "Local label inside the selected edge_head. Useful for diagnostics "
                "or count auxiliary loss."
            ),
        },
        "columns": {
            "question_token_ids": "Space-separated fixed-length token IDs.",
            "question_length": "Unpadded length clipped to max_question_len.",
            "question_template_id": "Template ID for template-based question encoder.",
            "edge_head": "Semantic answer family: binary/condition/density/count.",
            "head_id": "Integer ID for edge_head.",
        },
    }

    save_json(metadata, args.output_dir / "metadata.json")

    # Diagnostics.
    target_counts = (
        df.groupby(["split", "edge_class", "target_edge_global"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "target_edge_global"])
    )

    head_distribution = (
        df.groupby(["split", "edge_head"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "edge_head"])
    )

    template_distribution = (
        df.groupby(["split", "question_template_id", "question_type", "question_norm"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "question_template_id"])
    )

    target_counts.to_csv(args.output_dir / "target_counts.csv", index=False)
    head_distribution.to_csv(args.output_dir / "head_distribution.csv", index=False)
    template_distribution.to_csv(args.output_dir / "question_template_distribution.csv", index=False)

    # Summary.
    summary_lines = []

    summary_lines.append("TinyDisasterVQA / Training Data Summary")
    summary_lines.append("=" * 80)
    summary_lines.append("Task formulation:")
    summary_lines.append(f"  target_formulation: {target_formulation}")
    summary_lines.append("  main target: target_edge_global")
    summary_lines.append(f"  count_cap: {count_cap}+")
    summary_lines.append(f"  num_edge_global_classes: {num_edge_global_classes}")
    summary_lines.append("")
    summary_lines.append(f"Total samples: {len(df)}")
    summary_lines.append(f"Train samples: {len(train_df)}")
    summary_lines.append(f"Valid samples: {len(valid_df)}")
    summary_lines.append(f"Test samples:  {len(test_df)}")
    summary_lines.append("")
    summary_lines.append(f"Question templates: {num_question_templates}")
    summary_lines.append(f"Vocabulary size with PAD=0: {metadata['vocab_size_with_pad']}")
    summary_lines.append(f"UNK id: {unk_id}")
    summary_lines.append(f"PAD id: {pad_id}")
    summary_lines.append(f"Max observed question length: {max_observed_len}")
    summary_lines.append(f"Fixed max question length: {args.max_question_len}")
    summary_lines.append(f"Questions longer than max length: {too_long_count}")
    summary_lines.append(f"OOV token count: {sum(oov_counter.values())}")
    summary_lines.append("")
    summary_lines.append("Head distribution:")
    summary_lines.append(head_distribution.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Target counts:")
    summary_lines.append(target_counts.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Outputs:")
    summary_lines.append(f"  {args.output_dir / 'train.csv'}")
    summary_lines.append(f"  {args.output_dir / 'valid.csv'}")
    summary_lines.append(f"  {args.output_dir / 'test.csv'}")
    summary_lines.append(f"  {args.output_dir / 'metadata.json'}")
    summary_lines.append(f"  {args.output_dir / 'target_counts.csv'}")
    summary_lines.append(f"  {args.output_dir / 'head_distribution.csv'}")
    summary_lines.append(f"  {args.output_dir / 'question_template_distribution.csv'}")
    summary_lines.append("")

    summary_path = args.output_dir / "training_data_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n".join(summary_lines))
    print(f"Done. Wrote training data to: {args.output_dir}")


if __name__ == "__main__":
    main()