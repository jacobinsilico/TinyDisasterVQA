#!/usr/bin/env python3
"""
09_make_gap9_demo_input.py

Create GAP9 demo input binaries for generated TinyDisasterVQA TDM apps.

Final v2 deployment defaults:
  - checkpoint: checkpoints/tdm_fast_128_ce_best.pt
  - ONNX:       onnx/tdm_fast_128_ce_best.onnx
  - metadata:   outputs/training_data_cap5/metadata.json
  - CSV:        outputs/training_data_cap5/test.csv
  - output dir: gap9_generated_final/<model_stem>
  - image size: 128
  - classes:    14 cap5 edge_global classes

Generated files:
  Input_1.bin              image input for GAP9 app
  Input_2.bin              question_template_id input for GAP9 app
  Input_1_float32.bin      debug/reference preprocessed float32 image
  Input_1_uint8.bin        debug/reference quantized uint8 image
  demo_expected.json       row info + PyTorch/ONNX predictions

By default, Input_1.bin is written as uint8 using the provided input
quantization parameters. This matches the earlier GAP9 pipeline behavior.

Example:

  PYTHONPATH=src python scripts/09_make_gap9_demo_input.py

For XS baseline:

  PYTHONPATH=src python scripts/09_make_gap9_demo_input.py \\
    --checkpoint checkpoints/tdm_xs_128_ce_best.pt \\
    --onnx onnx/tdm_xs_128_ce_best.onnx

Multiple samples:

  PYTHONPATH=src python scripts/09_make_gap9_demo_input.py \\
    --num-samples 20 \\
    --edge-head count
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tinydisastervqa.data import load_json  # noqa: E402
from tinydisastervqa.models import build_tdm_from_metadata, describe_model  # noqa: E402


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TARGET_KEYS = (
    "target_edge_global",
    "edge_global_label",
    "target",
    "label",
    "class_id",
)

ANSWER_KEYS = (
    "answer_norm",
    "answer",
    "edge_global_class",
    "target_name",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/tdm_fast_128_ce_best.pt"),
        help="Student checkpoint used for PyTorch reference prediction.",
    )
    p.add_argument(
        "--onnx",
        type=Path,
        default=Path("onnx/tdm_fast_128_ce_best.onnx"),
        help="Exported ONNX model used for ONNXRuntime reference prediction.",
    )
    p.add_argument(
        "--metadata",
        type=Path,
        default=Path("outputs/training_data_cap5/metadata.json"),
        help="Training metadata JSON.",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/training_data_cap5/test.csv"),
        help="CSV split from which demo samples are selected.",
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
        help="Dataset root used to resolve image_rel_path.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Where to write demo inputs. "
            "If omitted, uses gap9_generated_final/<onnx_stem>."
        ),
    )
    p.add_argument(
        "--row-index",
        type=int,
        default=0,
        help="First CSV row index to consider.",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help=(
            "Number of usable demo samples to generate. "
            "If >1, samples are placed under output_dir/demo_inputs/."
        ),
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Input image size. Final v2 deployment uses 128.",
    )
    p.add_argument(
        "--num-classes",
        type=int,
        default=14,
        help="Number of edge_global output classes. Final cap5 setup uses 14.",
    )
    p.add_argument(
        "--num-question-templates",
        type=int,
        default=31,
        help="Number of question templates.",
    )

    p.add_argument(
        "--edge-head",
        type=str,
        default=None,
        help="Optional filter, e.g. binary, condition, count, density.",
    )
    p.add_argument(
        "--question-type",
        type=str,
        default=None,
        help="Optional question_type filter if present in CSV.",
    )
    p.add_argument(
        "--template-id",
        type=int,
        default=None,
        help="Optional question_template_id filter.",
    )

    p.add_argument(
        "--input-format",
        choices=("uint8", "float32"),
        default="uint8",
        help=(
            "Format used for Input_1.bin. "
            "uint8 is the default because the GAP9 generated app usually expects quantized input."
        ),
    )
    p.add_argument(
        "--input-scale",
        type=float,
        default=0.01865844801068306,
        help=(
            "Input image quantization scale used when --input-format uint8. "
            "Update this if NNTool reports a different input quantization."
        ),
    )
    p.add_argument(
        "--input-zero-point",
        type=int,
        default=114,
        help=(
            "Input image quantization zero-point used when --input-format uint8. "
            "Update this if NNTool reports a different input quantization."
        ),
    )

    p.add_argument(
        "--force",
        action="store_true",
        help="Delete existing demo_inputs subdirectory before writing multiple samples.",
    )

    return p.parse_args()


def infer_variant_from_name(path: Path) -> str:
    name = path.name.lower()

    if "tdm_fast" in name or "tdm-fast" in name:
        return "tdm_fast"
    if "tdm_xxs" in name or "tdm-xxs" in name:
        return "tdm_xxs"
    if "tdm_xs" in name or "tdm-xs" in name:
        return "tdm_xs"
    if "tdm_s" in name or "tdm-s" in name:
        return "tdm_s"
    if "tdm_m" in name or "tdm-m" in name:
        return "tdm_m"
    if "tdm_l" in name or "tdm-l" in name:
        return "tdm_l"

    raise ValueError(f"Cannot infer TDM variant from filename: {path.name}")


def get_checkpoint_config(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
        return dict(checkpoint["config"])
    return {}


def get_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value

    if isinstance(checkpoint, dict) and all(
        isinstance(v, torch.Tensor) for v in checkpoint.values()
    ):
        return checkpoint

    raise ValueError(
        "Could not find model weights. Expected checkpoint['model_state_dict'] "
        "or a raw state_dict."
    )


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}

    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value

    return cleaned


def resolve_model_config(
    checkpoint_path: Path,
    checkpoint: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    ckpt_config = get_checkpoint_config(checkpoint)

    variant = (
        ckpt_config.get("student_variant")
        or ckpt_config.get("variant")
        or infer_variant_from_name(checkpoint_path)
    )

    image_size = int(ckpt_config.get("image_size") or args.image_size)
    num_classes = int(ckpt_config.get("num_classes") or args.num_classes)
    num_question_templates = int(
        ckpt_config.get("num_question_templates") or args.num_question_templates
    )

    template_embed_dim = ckpt_config.get("template_embed_dim", None)
    fusion_hidden_dim = ckpt_config.get("fusion_hidden_dim", None)
    fusion_dropout = ckpt_config.get("fusion_dropout", None)
    fusion_layers = ckpt_config.get("fusion_layers", None)

    return {
        "variant": str(variant),
        "image_size": image_size,
        "num_classes": num_classes,
        "num_question_templates": num_question_templates,
        "template_embed_dim": None if template_embed_dim is None else int(template_embed_dim),
        "fusion_hidden_dim": None if fusion_hidden_dim is None else int(fusion_hidden_dim),
        "fusion_dropout": None if fusion_dropout is None else float(fusion_dropout),
        "fusion_layers": None if fusion_layers is None else int(fusion_layers),
    }


def build_model(
    metadata: dict[str, Any],
    checkpoint_path: Path,
    checkpoint: Any,
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    config = resolve_model_config(
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        args=args,
    )

    model = build_tdm_from_metadata(
        metadata=metadata,
        variant=config["variant"],
        num_classes=config["num_classes"],
        num_question_templates=config["num_question_templates"],
        question_template_embed_dim=config["template_embed_dim"],
        fusion_hidden_dim=config["fusion_hidden_dim"],
        fusion_dropout=config["fusion_dropout"],
        fusion_layers=config["fusion_layers"],
    )

    state_dict = clean_state_dict(get_state_dict(checkpoint))
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, config


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    return Path("gap9_generated_final") / args.onnx.stem


def resolve_image_path(row: dict[str, str], dataset_root: Path) -> Path:
    image_path_raw = row.get("image_path", "")
    image_rel_path_raw = row.get("image_rel_path", "")

    if image_path_raw:
        image_path = Path(image_path_raw)
        if image_path.exists():
            return image_path

    if image_rel_path_raw:
        fallback = dataset_root / image_rel_path_raw
        if fallback.exists():
            return fallback

    raise FileNotFoundError(
        f"Could not resolve image. image_path={image_path_raw}, "
        f"image_rel_path={image_rel_path_raw}, dataset_root={dataset_root}"
    )


def preprocess_image(path: Path, image_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), Image.BILINEAR)

    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)

    return arr.astype(np.float32)


def parse_optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def get_target(row: dict[str, str]) -> int | None:
    for key in TARGET_KEYS:
        value = parse_optional_int(row.get(key))
        if value is not None:
            return value
    return None


def get_answer_string(row: dict[str, str]) -> str:
    for key in ANSWER_KEYS:
        value = row.get(key, "")
        if value:
            return value
    return ""


def row_matches_filters(row: dict[str, str], args: argparse.Namespace) -> bool:
    if args.edge_head is not None:
        if row.get("edge_head", "").lower() != args.edge_head.lower():
            return False

    if args.question_type is not None:
        if row.get("question_type", "").lower() != args.question_type.lower():
            return False

    if args.template_id is not None:
        if parse_optional_int(row.get("question_template_id")) != args.template_id:
            return False

    return True


def row_is_usable(row: dict[str, str], dataset_root: Path) -> bool:
    try:
        _ = resolve_image_path(row, dataset_root)
    except FileNotFoundError:
        return False

    qtid = parse_optional_int(row.get("question_template_id"))
    if qtid is None:
        return False

    return 0 <= qtid < 31


def load_selected_rows(
    csv_path: Path,
    dataset_root: Path,
    row_index: int,
    num_samples: int,
    args: argparse.Namespace,
) -> list[tuple[int, dict[str, str]]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if row_index < 0 or row_index >= len(rows):
        raise IndexError(f"row_index={row_index} out of range for {len(rows)} rows")

    selected: list[tuple[int, dict[str, str]]] = []

    for idx in range(row_index, len(rows)):
        row = rows[idx]

        if not row_matches_filters(row, args):
            continue

        if not row_is_usable(row, dataset_root):
            continue

        selected.append((idx, row))

        if len(selected) >= num_samples:
            break

    if not selected:
        raise RuntimeError(
            "No usable row found. Try a different --row-index or remove filters."
        )

    return selected


def quantize_image_uint8(
    image_np: np.ndarray,
    input_scale: float,
    input_zero_point: int,
) -> np.ndarray:
    image_q = np.round(image_np / input_scale + input_zero_point)
    image_q = np.clip(image_q, 0, 255).astype(np.uint8)
    return image_q


def run_pytorch(
    model: torch.nn.Module,
    image_np: np.ndarray,
    qtid_np: np.ndarray,
) -> tuple[int, np.ndarray]:
    image_t = torch.from_numpy(image_np)
    qtid_t = torch.from_numpy(qtid_np)

    with torch.no_grad():
        logits = model(images=image_t, question_template_ids=qtid_t)
        logits_np = logits.detach().cpu().numpy()[0]
        argmax = int(logits_np.argmax())

    return argmax, logits_np


def run_onnx(
    onnx_path: Path,
    image_np: np.ndarray,
    qtid_np: np.ndarray,
) -> tuple[int | None, np.ndarray | None, str | None]:
    try:
        import onnxruntime as ort
    except Exception as exc:
        return None, None, f"ONNXRuntime import failed: {exc}"

    try:
        session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )

        inputs = session.get_inputs()
        if len(inputs) != 2:
            raise RuntimeError(f"Expected 2 ONNX inputs, got {len(inputs)}")

        feed = {
            inputs[0].name: image_np,
            inputs[1].name: qtid_np,
        }

        outs = session.run(None, feed)
        logits_np = outs[0][0]
        argmax = int(logits_np.argmax())

        return argmax, logits_np, None

    except Exception as exc:
        return None, None, str(exc)


def write_one_sample(
    *,
    sample_dir: Path,
    row_idx: int,
    row: dict[str, str],
    image_path: Path,
    image_np: np.ndarray,
    image_q: np.ndarray,
    qtid_np: np.ndarray,
    input_format: str,
    input_scale: float,
    input_zero_point: int,
    pt_argmax: int,
    pt_logits: np.ndarray,
    onnx_argmax: int | None,
    onnx_logits: np.ndarray | None,
    onnx_error: str | None,
    model_config: dict[str, Any],
) -> dict[str, Any]:
    sample_dir.mkdir(parents=True, exist_ok=True)

    if input_format == "uint8":
        input_1 = image_q
    elif input_format == "float32":
        input_1 = image_np
    else:
        raise ValueError(f"Unknown input_format: {input_format}")

    input_1_path = sample_dir / "Input_1.bin"
    input_2_path = sample_dir / "Input_2.bin"

    input_1_path.write_bytes(input_1.tobytes())
    input_2_path.write_bytes(qtid_np.tobytes())

    # Always save debug copies.
    (sample_dir / "Input_1_float32.bin").write_bytes(image_np.tobytes())
    (sample_dir / "Input_1_uint8.bin").write_bytes(image_q.tobytes())

    target = get_target(row)
    answer = get_answer_string(row)

    info: dict[str, Any] = {
        "row_index": row_idx,
        "image_path": str(image_path),
        "image_rel_path": row.get("image_rel_path", ""),
        "question": row.get("question", ""),
        "question_template_id": int(qtid_np[0]),
        "question_type": row.get("question_type", ""),
        "edge_head": row.get("edge_head", ""),
        "target_edge_global": target,
        "answer": answer,
        "pytorch_argmax": pt_argmax,
        "onnx_argmax": onnx_argmax,
        "onnx_error": onnx_error,
        "prediction_match_pytorch_onnx": (
            None if onnx_argmax is None else bool(pt_argmax == onnx_argmax)
        ),
        "pytorch_logits_first10": pt_logits[:10].astype(float).tolist(),
        "onnx_logits_first10": (
            None if onnx_logits is None else onnx_logits[:10].astype(float).tolist()
        ),
        "input_1_file": "Input_1.bin",
        "input_1_format": input_format,
        "input_1_shape_float_reference": list(image_np.shape),
        "input_1_dtype_float_reference": str(image_np.dtype),
        "input_1_shape_written": list(input_1.shape),
        "input_1_dtype_written": str(input_1.dtype),
        "input_1_bytes": input_1_path.stat().st_size,
        "input_2_file": "Input_2.bin",
        "input_2_shape": list(qtid_np.shape),
        "input_2_dtype": str(qtid_np.dtype),
        "input_2_bytes": input_2_path.stat().st_size,
        "input_quantization": {
            "scale": input_scale,
            "zero_point": input_zero_point,
        },
        "model_config": model_config,
    }

    with (sample_dir / "demo_expected.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    return info


def copy_primary_sample_to_root(sample_dir: Path, output_dir: Path) -> None:
    for filename in (
        "Input_1.bin",
        "Input_2.bin",
        "Input_1_float32.bin",
        "Input_1_uint8.bin",
        "demo_expected.json",
    ):
        shutil.copy2(sample_dir / filename, output_dir / filename)


def print_sample_summary(info: dict[str, Any], output_dir: Path) -> None:
    print("-" * 80)
    print(f"Sample dir:           {output_dir}")
    print(f"Input_1.bin bytes:    {info['input_1_bytes']}")
    print(f"Input_2.bin bytes:    {info['input_2_bytes']}")
    print(f"Row index:            {info['row_index']}")
    print(f"Image:                {info['image_path']}")
    print(f"Question:             {info['question']}")
    print(f"Question template id: {info['question_template_id']}")
    print(f"Question type:        {info['question_type']}")
    print(f"Edge head:            {info['edge_head']}")
    print(f"Target:               {info['target_edge_global']} ({info['answer']})")
    print(f"PyTorch argmax:       {info['pytorch_argmax']}")
    print(f"ONNX argmax:          {info['onnx_argmax']}")
    print(f"PyTorch/ONNX match:   {info['prediction_match_pytorch_onnx']}")


def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    if not args.onnx.exists():
        raise FileNotFoundError(f"Missing ONNX model: {args.onnx}")

    if not args.metadata.exists():
        raise FileNotFoundError(f"Missing metadata: {args.metadata}")

    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")

    selected_rows = load_selected_rows(
        csv_path=args.csv,
        dataset_root=args.dataset_root,
        row_index=args.row_index,
        num_samples=args.num_samples,
        args=args,
    )

    metadata = load_json(args.metadata)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model, model_config = build_model(
        metadata=metadata,
        checkpoint_path=args.checkpoint,
        checkpoint=checkpoint,
        args=args,
    )

    print("=" * 80)
    print("TinyDisasterVQA / GAP9 demo input generation")
    print("=" * 80)
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"ONNX:        {args.onnx}")
    print(f"Metadata:    {args.metadata}")
    print(f"CSV:         {args.csv}")
    print(f"Dataset:     {args.dataset_root}")
    print(f"Output dir:  {output_dir}")
    print(f"Image size:  {model_config['image_size']}")
    print(f"Input fmt:   {args.input_format}")
    print(f"Input scale: {args.input_scale}")
    print(f"Input zp:    {args.input_zero_point}")
    print()
    print(describe_model(model))
    print()

    if args.num_samples > 1:
        demo_inputs_dir = output_dir / "demo_inputs"
        if demo_inputs_dir.exists() and args.force:
            shutil.rmtree(demo_inputs_dir)
        demo_inputs_dir.mkdir(parents=True, exist_ok=True)
    else:
        demo_inputs_dir = output_dir

    all_infos = []

    for sample_idx, (row_idx, row) in enumerate(selected_rows):
        image_path = resolve_image_path(row, args.dataset_root)

        image_np = preprocess_image(
            image_path,
            image_size=int(model_config["image_size"]),
        )
        image_q = quantize_image_uint8(
            image_np,
            input_scale=args.input_scale,
            input_zero_point=args.input_zero_point,
        )
        qtid_np = np.array(
            [int(row["question_template_id"])],
            dtype=np.int64,
        )

        pt_argmax, pt_logits = run_pytorch(
            model=model,
            image_np=image_np,
            qtid_np=qtid_np,
        )

        onnx_argmax, onnx_logits, onnx_error = run_onnx(
            onnx_path=args.onnx,
            image_np=image_np,
            qtid_np=qtid_np,
        )

        if args.num_samples == 1:
            sample_dir = output_dir
        else:
            sample_dir = demo_inputs_dir / f"sample_{sample_idx:04d}_row_{row_idx:05d}"

        info = write_one_sample(
            sample_dir=sample_dir,
            row_idx=row_idx,
            row=row,
            image_path=image_path,
            image_np=image_np,
            image_q=image_q,
            qtid_np=qtid_np,
            input_format=args.input_format,
            input_scale=args.input_scale,
            input_zero_point=args.input_zero_point,
            pt_argmax=pt_argmax,
            pt_logits=pt_logits,
            onnx_argmax=onnx_argmax,
            onnx_logits=onnx_logits,
            onnx_error=onnx_error,
            model_config=model_config,
        )

        all_infos.append(info)

        if args.num_samples > 1 and sample_idx == 0:
            copy_primary_sample_to_root(sample_dir=sample_dir, output_dir=output_dir)

        print_sample_summary(info=info, output_dir=sample_dir)

    summary_path = output_dir / "demo_expected_all.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_infos, f, indent=2)

    print()
    print("=" * 80)
    print("GAP9 demo input generation complete")
    print("=" * 80)
    print(f"Primary Input_1.bin:  {output_dir / 'Input_1.bin'}")
    print(f"Primary Input_2.bin:  {output_dir / 'Input_2.bin'}")
    print(f"Primary expected:     {output_dir / 'demo_expected.json'}")
    print(f"All expected:         {summary_path}")

    if args.input_format == "uint8":
        print()
        print("[NOTE] Input_1.bin was written as uint8 using:")
        print(f"       scale={args.input_scale}")
        print(f"       zero_point={args.input_zero_point}")
        print("       If NNTool reports different input quantization, rerun with updated values.")


if __name__ == "__main__":
    main()