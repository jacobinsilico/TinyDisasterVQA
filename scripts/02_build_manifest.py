#!/usr/bin/env python3
"""
02_build_manifest.py

Builds a clean FloodNet-VQA manifest for TinyDisasterVQA.

Current submission formulation:
  FloodNet-VQA raw answers are converted into a compact single-head
  edge answer space:

    binary:    no / yes
    condition: flooded / mixed / non_flooded
    density:   low / moderate / high
    count:     1 / 2 / ... / count_cap / count_cap+

This script supports both:
  --count-cap 5   -> 14-class single-head task
  --count-cap 10  -> 19-class single-head task

Expected input structure:

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

Outputs:

outputs/processed/
  floodnet_manifest.csv
  floodnet_manifest.jsonl
  question_templates.csv
  question_template_to_id.json
  original_answer_to_label.json
  edge_class_to_label.json
  edge_label_to_class.json
  head_label_maps.json
  preprocessing_config.json
  preprocessing_summary.txt
"""

from __future__ import annotations

import argparse
import json
import re
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


HEAD_ORDER = ["binary", "condition", "density", "count"]


CONDITION_ANSWERS = {
    "flooded": "flooded",
    "non flooded": "non_flooded",
    "flooded,non flooded": "mixed",
    "flooded, non flooded": "mixed",
    "non flooded,flooded": "mixed",
    "non flooded, flooded": "mixed",
}

DENSITY_ANSWERS = {
    "low": "low",
    "moderate": "moderate",
    "high": "high",
}

BINARY_ANSWERS = {
    "yes": "yes",
    "no": "no",
}


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def normalize_text(text: Any) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_answer(answer: Any) -> str:
    return normalize_text(answer)


def is_counting_question(question_type: str) -> bool:
    return "counting" in normalize_text(question_type)


def parse_count(answer: Any) -> int | None:
    answer_str = str(answer).strip()

    if answer_str.isdigit():
        return int(answer_str)

    return None


def cap_count(value: int, cap: int) -> str:
    if value > cap:
        return f"{cap}+"

    return str(value)


def get_image_rel_path(split: str, image_id: str) -> Path:
    return Path("images") / SPLITS[split]["image_dir"] / image_id


def image_exists_case_insensitive(dataset_root: Path, rel_path: Path) -> tuple[bool, Path]:
    full_path = dataset_root / rel_path

    if full_path.exists():
        return True, rel_path

    image_dir = full_path.parent
    target_name = full_path.name.lower()

    if image_dir.exists():
        for candidate in image_dir.iterdir():
            if candidate.name.lower() == target_name:
                corrected_rel = rel_path.parent / candidate.name
                return True, corrected_rel

    return False, rel_path


def derive_edge_fields(question_type: str, answer_norm: str, count_cap: int) -> dict[str, Any]:
    """
    Creates the compact edge-clean target.

    edge_head:
      - binary
      - condition
      - density
      - count

    edge_answer:
      - yes/no
      - flooded/non_flooded/mixed
      - low/moderate/high
      - 1...count_cap/count_cap+

    edge_class:
      - binary:yes
      - condition:non_flooded
      - count:5+
      - count:10+
      etc.
    """
    if is_counting_question(question_type):
        count_value = parse_count(answer_norm)

        if count_value is None:
            raise ValueError(f"Counting question has non-integer answer: {answer_norm}")

        edge_head = "count"
        edge_answer = cap_count(count_value, count_cap)
        edge_class = f"{edge_head}:{edge_answer}"

        return {
            "edge_head": edge_head,
            "edge_answer": edge_answer,
            "edge_class": edge_class,
            "count_value": count_value,
            "count_capped": edge_answer,
        }

    if answer_norm in BINARY_ANSWERS:
        edge_head = "binary"
        edge_answer = BINARY_ANSWERS[answer_norm]
        edge_class = f"{edge_head}:{edge_answer}"

        return {
            "edge_head": edge_head,
            "edge_answer": edge_answer,
            "edge_class": edge_class,
            "count_value": None,
            "count_capped": None,
        }

    if answer_norm in CONDITION_ANSWERS:
        edge_head = "condition"
        edge_answer = CONDITION_ANSWERS[answer_norm]
        edge_class = f"{edge_head}:{edge_answer}"

        return {
            "edge_head": edge_head,
            "edge_answer": edge_answer,
            "edge_class": edge_class,
            "count_value": None,
            "count_capped": None,
        }

    if answer_norm in DENSITY_ANSWERS:
        edge_head = "density"
        edge_answer = DENSITY_ANSWERS[answer_norm]
        edge_class = f"{edge_head}:{edge_answer}"

        return {
            "edge_head": edge_head,
            "edge_answer": edge_answer,
            "edge_class": edge_class,
            "count_value": None,
            "count_capped": None,
        }

    raise ValueError(
        f"Unknown answer type: '{answer_norm}' for question type '{question_type}'"
    )


def build_original_answer_map(class_to_label: dict[str, Any]) -> dict[str, int]:
    """
    Normalize class_to_label keys and convert labels to int.
    """
    answer_to_label = {}

    for answer, label in class_to_label.items():
        answer_to_label[normalize_answer(answer)] = int(label)

    return answer_to_label


def load_raw_annotations(dataset_root: Path) -> pd.DataFrame:
    rows = []

    for split, info in SPLITS.items():
        ann_path = dataset_root / "data" / info["annotation_file"]
        records = load_json(ann_path)

        if not isinstance(records, list):
            raise ValueError(f"Expected list in {ann_path}")

        for item in records:
            image_id = str(item.get("Image_ID", "")).strip()
            image_rel_path = get_image_rel_path(split, image_id)

            image_exists, corrected_rel_path = image_exists_case_insensitive(
                dataset_root=dataset_root,
                rel_path=image_rel_path,
            )

            question = str(item.get("Question", "")).strip()
            question_norm = normalize_text(question)

            question_type = str(item.get("Question_Type", "")).strip()
            question_type_norm = normalize_text(question_type)

            answer_raw = str(item.get("Ground_Truth", "")).strip()
            answer_norm = normalize_answer(answer_raw)

            rows.append(
                {
                    "split": split,
                    "image_id": image_id,
                    "image_rel_path": str(corrected_rel_path),
                    "image_exists": image_exists,
                    "question_id": item.get("Question_ID"),
                    "question": question,
                    "question_norm": question_norm,
                    "question_type": question_type,
                    "question_type_norm": question_type_norm,
                    "answer_raw": answer_raw,
                    "answer_norm": answer_norm,
                    "attention_map_dir": item.get("AttentionMap_dir"),
                    "grad_cam_attention": item.get("grad_cam_attention"),
                }
            )

    return pd.DataFrame(rows)


def make_template_key(question_type: str, question_norm: str) -> str:
    return f"{question_type} ||| {question_norm}"


def build_question_templates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Builds stable question_template_id values.

    Sorting by question_type and question_norm gives deterministic template IDs.
    """
    template_df = (
        df.groupby(["question_type", "question_norm"])
        .agg(
            count=("question_norm", "size"),
            example_question=("question", "first"),
            num_unique_answers=("answer_norm", "nunique"),
        )
        .reset_index()
        .sort_values(["question_type", "question_norm"])
        .reset_index(drop=True)
    )

    template_df.insert(0, "question_template_id", range(len(template_df)))

    template_to_id = {}

    for _, row in template_df.iterrows():
        key = make_template_key(row["question_type"], row["question_norm"])
        template_to_id[key] = int(row["question_template_id"])

    return template_df, template_to_id


def build_preferred_answer_order(count_cap: int) -> dict[str, list[str]]:
    """
    Defines explicit, stable local answer ordering for each semantic head.

    For count_cap=5:
      count: 1,2,3,4,5,5+

    For count_cap=10:
      count: 1,2,3,4,5,6,7,8,9,10,10+
    """
    if count_cap < 1:
        raise ValueError(f"count_cap must be >= 1, got {count_cap}")

    return {
        "binary": ["no", "yes"],
        "condition": ["flooded", "mixed", "non_flooded"],
        "density": ["low", "moderate", "high"],
        "count": [str(i) for i in range(1, count_cap + 1)] + [f"{count_cap}+"],
    }


def sort_extra_labels(head: str, labels: set[str]) -> list[str]:
    """
    Deterministic fallback ordering for unexpected labels.

    This should usually not be needed, but it keeps the script robust if
    FloodNet contains a future label not covered by the preferred order.
    """
    if head == "count":
        def key_fn(label: str) -> tuple[int, int | str]:
            if label.endswith("+") and label[:-1].isdigit():
                return (1, int(label[:-1]))
            if label.isdigit():
                return (0, int(label))
            return (2, label)

        return sorted(labels, key=key_fn)

    return sorted(labels)


def build_edge_label_maps(
    df: pd.DataFrame,
    count_cap: int,
) -> tuple[dict[str, int], dict[str, str], dict[str, dict[str, int]]]:
    """
    Builds:

      edge_class_to_label:
        binary:no -> 0
        binary:yes -> 1
        condition:flooded -> 2
        ...
        count:5+ -> 13      for count_cap=5
        count:10+ -> 18     for count_cap=10

      edge_label_to_class:
        "0" -> binary:no
        ...

      head_label_maps:
        binary:
          no -> 0
          yes -> 1
        count:
          1 -> 0
          ...
          5+ -> 5
    """
    preferred_order = build_preferred_answer_order(count_cap=count_cap)

    head_label_maps: dict[str, dict[str, int]] = {}

    observed_heads = set(str(x) for x in df["edge_head"].unique())

    for head in HEAD_ORDER:
        if head not in observed_heads:
            continue

        observed = set(
            str(x) for x in df[df["edge_head"] == head]["edge_answer"].unique()
        )

        ordered = []

        for label in preferred_order.get(head, []):
            if label in observed:
                ordered.append(label)

        extras = observed - set(ordered)
        ordered.extend(sort_extra_labels(head=head, labels=extras))

        head_label_maps[head] = {
            label: idx
            for idx, label in enumerate(ordered)
        }

    # Explicit global order: binary, condition, density, count.
    edge_classes = []

    for head in HEAD_ORDER:
        if head not in head_label_maps:
            continue

        for answer in head_label_maps[head]:
            edge_classes.append(f"{head}:{answer}")

    edge_class_to_label = {
        edge_class: idx
        for idx, edge_class in enumerate(edge_classes)
    }

    edge_label_to_class = {
        str(idx): edge_class
        for edge_class, idx in edge_class_to_label.items()
    }

    return edge_class_to_label, edge_label_to_class, head_label_maps


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in df.to_dict(orient="records"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
        default=Path("outputs/processed"),
        help="Directory where processed manifest files will be written.",
    )
    parser.add_argument(
        "--count-cap",
        type=int,
        default=5,
        help="Count labels above this value become '<cap>+'. Use 5 or 10 for official ablations.",
    )

    args = parser.parse_args()

    dataset_root = args.dataset_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TinyDisasterVQA / Build FloodNet Manifest")
    print("=" * 80)
    print(f"Dataset root: {dataset_root.resolve()}")
    print(f"Output dir:   {output_dir.resolve()}")
    print(f"Count cap:    {args.count_cap}+")
    print()

    class_to_label_path = dataset_root / "data" / "class_to_label.json"
    word_to_token_path = dataset_root / "data" / "word_to_token.json"

    class_to_label = load_json(class_to_label_path)
    word_to_token = load_json(word_to_token_path)

    original_answer_to_label = build_original_answer_map(class_to_label)

    df = load_raw_annotations(dataset_root)

    if not df["image_exists"].all():
        missing = df[~df["image_exists"]]
        raise FileNotFoundError(
            f"Found {len(missing)} missing image references. "
            f"Run scripts/01_explore_dataset.py and inspect missing_images.csv."
        )

    template_df, template_to_id = build_question_templates(df)

    df["question_template_id"] = df.apply(
        lambda row: template_to_id[
            make_template_key(row["question_type"], row["question_norm"])
        ],
        axis=1,
    )

    # Original FloodNet answer labels.
    df["original_answer_label"] = df["answer_norm"].map(original_answer_to_label)

    missing_original_labels = (
        df[df["original_answer_label"].isna()]["answer_norm"]
        .dropna()
        .unique()
    )

    if len(missing_original_labels) > 0:
        raise ValueError(
            f"Some answers are missing from class_to_label.json: "
            f"{missing_original_labels}"
        )

    df["original_answer_label"] = df["original_answer_label"].astype(int)

    # Compact single-head edge labels.
    edge_rows = []

    for _, row in df.iterrows():
        edge_rows.append(
            derive_edge_fields(
                question_type=row["question_type"],
                answer_norm=row["answer_norm"],
                count_cap=args.count_cap,
            )
        )

    edge_df = pd.DataFrame(edge_rows)
    df = pd.concat([df.reset_index(drop=True), edge_df.reset_index(drop=True)], axis=1)

    edge_class_to_label, edge_label_to_class, head_label_maps = build_edge_label_maps(
        df=df,
        count_cap=args.count_cap,
    )

    df["edge_global_label"] = df["edge_class"].map(edge_class_to_label).astype(int)

    df["edge_head_label"] = df.apply(
        lambda row: head_label_maps[row["edge_head"]][row["edge_answer"]],
        axis=1,
    ).astype(int)

    # Useful boolean fields.
    df["is_counting"] = df["edge_head"] == "count"
    df["is_binary"] = df["edge_head"] == "binary"
    df["is_condition"] = df["edge_head"] == "condition"
    df["is_density"] = df["edge_head"] == "density"

    # Reorder columns.
    column_order = [
        "split",
        "image_id",
        "image_rel_path",
        "image_exists",
        "question_id",
        "question",
        "question_norm",
        "question_type",
        "question_type_norm",
        "question_template_id",
        "answer_raw",
        "answer_norm",
        "original_answer_label",
        "edge_head",
        "edge_answer",
        "edge_class",
        "edge_global_label",
        "edge_head_label",
        "is_counting",
        "is_binary",
        "is_condition",
        "is_density",
        "count_value",
        "count_capped",
        "attention_map_dir",
        "grad_cam_attention",
    ]

    df = df[column_order]

    # Save main outputs.
    manifest_csv = output_dir / "floodnet_manifest.csv"
    manifest_jsonl = output_dir / "floodnet_manifest.jsonl"

    df.to_csv(manifest_csv, index=False)
    write_jsonl(df, manifest_jsonl)

    template_df.to_csv(output_dir / "question_templates.csv", index=False)

    save_json(template_to_id, output_dir / "question_template_to_id.json")
    save_json(original_answer_to_label, output_dir / "original_answer_to_label.json")
    save_json(edge_class_to_label, output_dir / "edge_class_to_label.json")
    save_json(edge_label_to_class, output_dir / "edge_label_to_class.json")
    save_json(head_label_maps, output_dir / "head_label_maps.json")
    save_json(word_to_token, output_dir / "word_to_token.json")

    preprocessing_config = {
        "dataset_root": str(dataset_root),
        "count_cap": args.count_cap,
        "target_formulation": "single_head_edge_global",
        "num_edge_global_classes": len(edge_class_to_label),
        "head_order": HEAD_ORDER,
        "edge_class_to_label_file": "edge_class_to_label.json",
        "edge_label_to_class_file": "edge_label_to_class.json",
        "head_label_maps_file": "head_label_maps.json",
    }

    save_json(preprocessing_config, output_dir / "preprocessing_config.json")

    # Diagnostics.
    split_stats = (
        df.groupby("split")
        .agg(
            num_samples=("split", "size"),
            num_images=("image_id", "nunique"),
            num_question_templates=("question_template_id", "nunique"),
            num_original_answers=("answer_norm", "nunique"),
            num_edge_classes=("edge_class", "nunique"),
        )
        .reset_index()
    )

    edge_head_counts = (
        df.groupby(["split", "edge_head"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "edge_head"])
    )

    edge_class_counts = (
        df.groupby(["split", "edge_class", "edge_global_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "edge_global_label"])
    )

    edge_head_label_counts = (
        df.groupby(["split", "edge_head", "edge_answer", "edge_head_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "edge_head", "edge_head_label"])
    )

    split_stats.to_csv(output_dir / "manifest_split_stats.csv", index=False)
    edge_head_counts.to_csv(output_dir / "edge_head_counts.csv", index=False)
    edge_class_counts.to_csv(output_dir / "edge_class_counts.csv", index=False)
    edge_head_label_counts.to_csv(output_dir / "edge_head_label_counts.csv", index=False)

    # Check unseen edge classes.
    train_edge_classes = set(df[df["split"] == "train"]["edge_class"])
    valid_edge_classes = set(df[df["split"] == "valid"]["edge_class"])
    test_edge_classes = set(df[df["split"] == "test"]["edge_class"])

    valid_unseen_edge = sorted(valid_edge_classes - train_edge_classes)
    test_unseen_edge = sorted(test_edge_classes - train_edge_classes)

    # Check original unseen labels.
    train_original_answers = set(df[df["split"] == "train"]["answer_norm"])
    valid_original_answers = set(df[df["split"] == "valid"]["answer_norm"])
    test_original_answers = set(df[df["split"] == "test"]["answer_norm"])

    valid_unseen_original = sorted(valid_original_answers - train_original_answers)
    test_unseen_original = sorted(test_original_answers - train_original_answers)

    # Count-specific diagnostics.
    count_df = df[df["edge_head"] == "count"].copy()

    count_capped_counts = (
        count_df.groupby(["split", "edge_answer", "edge_global_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "edge_global_label"])
    )

    count_capped_counts.to_csv(output_dir / "count_capped_counts.csv", index=False)

    # Summary.
    summary_lines = []

    summary_lines.append("TinyDisasterVQA / FloodNet Manifest Summary")
    summary_lines.append("=" * 80)
    summary_lines.append(f"Dataset root: {dataset_root.resolve()}")
    summary_lines.append(f"Manifest CSV: {manifest_csv}")
    summary_lines.append(f"Manifest JSONL: {manifest_jsonl}")
    summary_lines.append("")
    summary_lines.append("Task formulation:")
    summary_lines.append("  single_head_edge_global")
    summary_lines.append(f"  count_cap: {args.count_cap}+")
    summary_lines.append(f"  edge_global_classes: {len(edge_class_to_label)}")
    summary_lines.append("")
    summary_lines.append(f"Total samples: {len(df)}")
    summary_lines.append(f"Unique images: {df['image_id'].nunique()}")
    summary_lines.append(f"Question templates: {df['question_template_id'].nunique()}")
    summary_lines.append(f"Original answer classes used in data: {df['answer_norm'].nunique()}")
    summary_lines.append(f"Original answer classes in class_to_label.json: {len(original_answer_to_label)}")
    summary_lines.append("")
    summary_lines.append("Split stats:")
    summary_lines.append(split_stats.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Edge heads:")
    summary_lines.append(edge_head_counts.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Edge class counts:")
    summary_lines.append(edge_class_counts.to_string(index=False))
    summary_lines.append("")
    summary_lines.append("Head label maps:")
    summary_lines.append(json.dumps(head_label_maps, indent=2, sort_keys=True))
    summary_lines.append("")
    summary_lines.append("Edge class to label:")
    summary_lines.append(json.dumps(edge_class_to_label, indent=2, sort_keys=True))
    summary_lines.append("")
    summary_lines.append("Original labels unseen in valid compared to train:")
    summary_lines.append(str(valid_unseen_original))
    summary_lines.append("")
    summary_lines.append("Original labels unseen in test compared to train:")
    summary_lines.append(str(test_unseen_original))
    summary_lines.append("")
    summary_lines.append("Edge classes unseen in valid compared to train:")
    summary_lines.append(str(valid_unseen_edge))
    summary_lines.append("")
    summary_lines.append("Edge classes unseen in test compared to train:")
    summary_lines.append(str(test_unseen_edge))
    summary_lines.append("")
    summary_lines.append("Count capped counts:")
    summary_lines.append(count_capped_counts.to_string(index=False))
    summary_lines.append("")

    summary_path = output_dir / "preprocessing_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n".join(summary_lines))
    print(f"Done. Wrote manifest outputs to: {output_dir}")


if __name__ == "__main__":
    main()