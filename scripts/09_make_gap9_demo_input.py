#!/usr/bin/env python3
"""
09_make_gap9_demo_input.py

Create GAP9 demo input binaries for generated TDM app:
  Input_1.bin = preprocessed image [1, 3, H, W] float32
  Input_2.bin = question_template_id [1] int64

Also prints/saves PyTorch and ONNX expected prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tinydisastervqa.data import load_json
from tinydisastervqa.models import build_tdm_from_metadata


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("models/tdm_xxs_single_ce_224_best.pt"))
    p.add_argument("--onnx", type=Path, default=Path("models/tdm_xxs_single_ce_224_best.onnx"))
    p.add_argument("--metadata", type=Path, default=Path("outputs/training_data/metadata.json"))
    p.add_argument("--csv", type=Path, default=Path("outputs/training_data/test.csv"))
    p.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    p.add_argument("--output-dir", type=Path, default=Path("gap9_generated/tdm_xxs_single_ce_224_best"))
    p.add_argument("--row-index", type=int, default=0)
    p.add_argument("--image-size", type=int, default=224)
    return p.parse_args()


def resolve_image_path(row, dataset_root: Path) -> Path:
    p = Path(row["image_path"])
    if p.exists():
        return p

    p2 = dataset_root / row["image_rel_path"]
    if p2.exists():
        return p2

    raise FileNotFoundError(f"Could not resolve image: {p} or {p2}")


def preprocess_image(path: Path, image_size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr.astype(np.float32)


def load_row(csv_path: Path, dataset_root: Path, row_index: int):
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if row_index < 0 or row_index >= len(rows):
        raise IndexError(f"row_index={row_index} out of range for {len(rows)} rows")

    # Find first row at or after row_index with existing image.
    for i in range(row_index, len(rows)):
        try:
            _ = resolve_image_path(rows[i], dataset_root)
            return i, rows[i]
        except FileNotFoundError:
            continue

    raise RuntimeError("No usable row found.")


def infer_config_from_filename(path: Path):
    name = path.name.lower()
    if "tdm_xxs" in name:
        variant = "tdm_xxs"
    elif "tdm_xs" in name:
        variant = "tdm_xs"
    elif "tdm_s" in name:
        variant = "tdm_s"
    elif "tdm_m" in name:
        variant = "tdm_m"
    else:
        raise ValueError(f"Cannot infer variant from {path}")

    head_type = "multihead" if "multihead" in name else "single"
    return variant, head_type


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    row_idx, row = load_row(args.csv, args.dataset_root, args.row_index)
    image_path = resolve_image_path(row, args.dataset_root)

    image_np = preprocess_image(image_path, args.image_size)
    qtid_np = np.array([int(row["question_template_id"])], dtype=np.int64)

    # Write GAP9 input binaries.
    input_scale = 0.01865844801068306
    input_zero_point = 114

    image_q = np.round(image_np / input_scale + input_zero_point)
    image_q = np.clip(image_q, 0, 255).astype(np.uint8)

    (args.output_dir / "Input_1.bin").write_bytes(image_q.tobytes())
    (args.output_dir / "Input_2.bin").write_bytes(qtid_np.tobytes())

    metadata = load_json(args.metadata)
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    variant, head_type = infer_config_from_filename(args.checkpoint)

    model = build_tdm_from_metadata(
        metadata=metadata,
        variant=variant,
        num_classes=19,
        num_question_templates=31,
        head_type=head_type,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    image_t = torch.from_numpy(image_np)
    qtid_t = torch.from_numpy(qtid_np)

    with torch.no_grad():
        logits = model(images=image_t, question_template_ids=qtid_t)
        pt_logits = logits.detach().cpu().numpy()[0]
        pt_argmax = int(pt_logits.argmax())

    onnx_argmax = None
    onnx_logits = None
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
        outs = sess.run(None, {
            "image": image_np,
            "question_template_id": qtid_np,
        })
        onnx_logits = outs[0][0]
        onnx_argmax = int(onnx_logits.argmax())
    except Exception as e:
        print(f"[WARN] ONNXRuntime check skipped/failed: {e}")

    info = {
        "row_index": row_idx,
        "image_path": str(image_path),
        "question": row.get("question", ""),
        "question_template_id": int(row["question_template_id"]),
        "question_type": row.get("question_type", ""),
        "edge_head": row.get("edge_head", ""),
        "target_edge_global": int(row["target_edge_global"]),
        "answer_norm": row.get("answer_norm", ""),
        "pytorch_argmax": pt_argmax,
        "onnx_argmax": onnx_argmax,
        "pytorch_logits_first10": pt_logits[:10].astype(float).tolist(),
        "onnx_logits_first10": None if onnx_logits is None else onnx_logits[:10].astype(float).tolist(),
        "input_1_shape": list(image_np.shape),
        "input_1_dtype": str(image_np.dtype),
        "input_2_shape": list(qtid_np.shape),
        "input_2_dtype": str(qtid_np.dtype),
    }

    with (args.output_dir / "demo_expected.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print("=" * 80)
    print("Wrote GAP9 demo inputs")
    print("=" * 80)
    print(f"Output dir:           {args.output_dir}")
    print(f"Input_1.bin bytes:    {(args.output_dir / 'Input_1.bin').stat().st_size}")
    print(f"Input_2.bin bytes:    {(args.output_dir / 'Input_2.bin').stat().st_size}")
    print(f"Row index:            {row_idx}")
    print(f"Image:                {image_path}")
    print(f"Question:             {info['question']}")
    print(f"Question template id: {info['question_template_id']}")
    print(f"Target:               {info['target_edge_global']} ({info['answer_norm']})")
    print(f"PyTorch argmax:       {pt_argmax}")
    print(f"ONNX argmax:          {onnx_argmax}")
    print("Saved:", args.output_dir / "demo_expected.json")


if __name__ == "__main__":
    main()