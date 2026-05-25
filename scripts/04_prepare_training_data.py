#!/usr/bin/env python3
"""
04_prepare_training_data.py

Prepares final train/valid/test CSVs for TinyDisasterVQA training.

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


HEAD_TO_ID = {
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
        "edge_global_label",
        "edge_head_label",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")


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

    df = pd.read_csv(args.manifest)
    check_required_columns(df)

    word_to_token = load_json(args.word_to_token)
    word_to_token = {str(k): int(v) for k, v in word_to_token.items()}

    answer_space = load_json(args.answer_space)

    pad_id = 0
    unk_id = int(word_to_token.get("unk", 1))

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

        token_ids, q_len, oov_tokens = encode_question(
            question=question,
            word_to_token=word_to_token,
            max_len=args.max_question_len,
            pad_id=pad_id,
            unk_id=unk_id,
        )

        for tok in oov_tokens:
            oov_counter[tok] += 1

        encoded_questions.append(token_ids)
        question_lengths.append(q_len)
        token_strings.append(" ".join(str(x) for x in token_ids))

    df["question_token_ids"] = token_strings
    df["question_length"] = question_lengths
    df["head_id"] = df["edge_head"].map(HEAD_TO_ID)

    if df["head_id"].isna().any():
        bad_heads = sorted(df[df["head_id"].isna()]["edge_head"].unique())
        raise ValueError(f"Unknown edge_head values: {bad_heads}")

    df["head_id"] = df["head_id"].astype(int)

    # Standard target aliases used by training scripts.
    df["target_original"] = df["original_answer_label"].astype(int)
    df["target_edge_global"] = df["edge_global_label"].astype(int)
    df["target_edge_head"] = df["edge_head_label"].astype(int)

    # Add full image path for convenience.
    df["image_path"] = df["image_rel_path"].apply(
        lambda p: str(args.dataset_root / p)
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
        "answer_norm",
        "target_original",
        "target_edge_global",
        "target_edge_head",
        "count_value",
        "count_capped",
    ]

    # Some pandas columns may be absent/NaN depending on prior saves.
    keep_cols = [c for c in keep_cols if c in df.columns]
    train_df = df[df["split"] == "train"][keep_cols].copy()
    valid_df = df[df["split"] == "valid"][keep_cols].copy()
    test_df = df[df["split"] == "test"][keep_cols].copy()

    train_df.to_csv(args.output_dir / "train.csv", index=False)
    valid_df.to_csv(args.output_dir / "valid.csv", index=False)
    test_df.to_csv(args.output_dir / "test.csv", index=False)

    oov_df = pd.DataFrame(
        [{"token": tok, "count": count} for tok, count in oov_counter.items()]
    ).sort_values("count", ascending=False) if oov_counter else pd.DataFrame(columns=["token", "count"])

    oov_df.to_csv(args.output_dir / "question_token_oov.csv", index=False)

    metadata = {
        "dataset_root": str(args.dataset_root),
        "max_question_len": args.max_question_len,
        "max_observed_question_len": max_observed_len,
        "pad_id": pad_id,
        "unk_id": unk_id,
        "vocab_size_with_pad": max(word_to_token.values()) + 1,
        "word_to_token": word_to_token,
        "head_to_id": HEAD_TO_ID,
        "id_to_head": {str(v): k for k, v in HEAD_TO_ID.items()},
        "answer_space": answer_space,
        "files": {
            "train_csv": "train.csv",
            "valid_csv": "valid.csv",
            "test_csv": "test.csv",
        },
        "targets": {
            "target_original": "Original FloodNet answer label.",
            "target_edge_global": "Single edge-clean global class.",
            "target_edge_head": "Label inside the selected edge_head.",
        },
    }

    save_json(metadata, args.output_dir / "metadata.json")

    # Summary.
    summary_lines = []
    summary_lines.append("TinyDisasterVQA / Training Data Summary")
    summary_lines.append("=" * 80)
    summary_lines.append(f"Total samples: {len(df)}")
    summary_lines.append(f"Train samples: {len(train_df)}")
    summary_lines.append(f"Valid samples: {len(valid_df)}")
    summary_lines.append(f"Test samples:  {len(test_df)}")
    summary_lines.append("")
    summary_lines.append(f"Vocabulary size with PAD=0: {metadata['vocab_size_with_pad']}")
    summary_lines.append(f"UNK id: {unk_id}")
    summary_lines.append(f"PAD id: {pad_id}")
    summary_lines.append(f"Max observed question length: {max_observed_len}")
    summary_lines.append(f"Fixed max question length: {args.max_question_len}")
    summary_lines.append(f"Questions longer than max length: {too_long_count}")
    summary_lines.append(f"OOV token count: {sum(oov_counter.values())}")
    summary_lines.append("")
    summary_lines.append("Head distribution:")
    summary_lines.append(
        df.groupby(["split", "edge_head"])
        .size()
        .reset_index(name="count")
        .to_string(index=False)
    )
    summary_lines.append("")
    summary_lines.append("Outputs:")
    summary_lines.append(f"  {args.output_dir / 'train.csv'}")
    summary_lines.append(f"  {args.output_dir / 'valid.csv'}")
    summary_lines.append(f"  {args.output_dir / 'test.csv'}")
    summary_lines.append(f"  {args.output_dir / 'metadata.json'}")
    summary_lines.append("")

    summary_path = args.output_dir / "training_data_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n".join(summary_lines))
    print(f"Done. Wrote training data to: {args.output_dir}")


if __name__ == "__main__":
    main()