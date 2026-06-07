#!/usr/bin/env python3
"""
08_generate_gap9_artifacts.py

Generate GAP9 NNTool/Autotiler artifacts from exported TinyDisasterVQA ONNX.

First target:
  models/tdm_xxs_single_ce_224_best.onnx

Run inside the GAP9 Docker container, with GAP9 env sourced:

  source /app/install/gap9-sdk/.gap9-venv/bin/activate
  source /app/install/gap9-sdk/configs/gap9_evk_audio.sh
  export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation

  cd /app/TinyDisasterVQA
  PYTHONPATH=src python scripts/08_generate_gap9_artifacts.py

Outputs:
  gap9_generated/tdm_xxs_single_ce_224_best/
    at_model/
    tensors/
    Input_*.bin, depending on NNTool generation
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from nntool.api import NNGraph
from nntool.api.utils import model_settings, quantization_options


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("models/tdm_xxs_single_ce_224_best.onnx"),
        help="Path to exported ONNX model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("gap9_generated/tdm_xxs_single_ce_224_best"),
        help="Directory where NNTool/Autotiler artifacts will be generated.",
    )
    parser.add_argument(
        "--calib-csv",
        type=Path,
        default=Path("outputs/training_data/valid.csv"),
        help="CSV with calibration samples.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
        help="Dataset root used to resolve image_rel_path.",
    )
    parser.add_argument(
        "--num-calib",
        type=int,
        default=32,
        help="Number of representative calibration samples.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override image size. If omitted, inferred from ONNX input shape.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete output directory if it already exists.",
    )
    parser.add_argument(
        "--no-ne16",
        action="store_true",
        help="Disable NE16 quantization option.",
    )
    parser.add_argument(
        "--privileged-l3-flash-size",
        type=int,
        default=1800000,
        help="Privileged L3 flash size passed to model_settings.",
    )

    return parser.parse_args()


def resolve_image_path(row: dict[str, str], dataset_root: Path) -> Path:
    image_path = Path(row["image_path"])

    if image_path.exists():
        return image_path

    fallback = dataset_root / row["image_rel_path"]
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Could not resolve image. image_path={image_path}, fallback={fallback}"
    )


def preprocess_image(path: Path, image_size: int) -> np.ndarray:
    """
    Match data.py eval transform:
      Resize((image_size, image_size))
      ToTensor()
      Normalize(ImageNet mean/std)

    Returns:
      image [1, 3, H, W], float32
    """
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), Image.BILINEAR)

    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    arr = np.expand_dims(arr, axis=0)   # CHW -> NCHW

    return arr.astype(np.float32)


def load_rows(csv_path: Path, dataset_root: Path, limit: int) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing calibration CSV: {csv_path}")

    rows: list[dict[str, str]] = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required = {"image_path", "image_rel_path", "question_template_id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

        for row in reader:
            # Only keep rows whose images actually resolve.
            try:
                _ = resolve_image_path(row, dataset_root)
            except FileNotFoundError:
                continue

            rows.append(row)

            if len(rows) >= limit:
                break

    if not rows:
        raise RuntimeError("No usable calibration rows found.")

    return rows


def make_inputs(row: dict[str, str], dataset_root: Path, image_size: int) -> list[np.ndarray]:
    image_path = resolve_image_path(row, dataset_root)
    image = preprocess_image(image_path, image_size=image_size)

    question_template_id = np.array(
        [int(row["question_template_id"])],
        dtype=np.int64,
    )

    return [image, question_template_id]


def representative_dataset(
    rows: list[dict[str, str]],
    dataset_root: Path,
    image_size: int,
) -> Iterator[list[np.ndarray]]:
    for row in rows:
        yield make_inputs(row, dataset_root=dataset_root, image_size=image_size)


def infer_image_size_from_graph(G: NNGraph) -> int:
    inputs = G.inputs()
    if not inputs:
        raise RuntimeError("Graph has no inputs.")

    shape = inputs[0].out_dims[0].shape

    # Expected NCHW: [1, 3, H, W]
    if len(shape) != 4:
        raise RuntimeError(f"Expected image input shape [1,3,H,W], got {shape}")

    if int(shape[2]) != int(shape[3]):
        raise RuntimeError(f"Expected square image input, got {shape}")

    return int(shape[2])


def main() -> None:
    args = parse_args()

    if not args.onnx.exists():
        raise FileNotFoundError(f"Missing ONNX model: {args.onnx}")

    if args.output_dir.exists():
        if args.force:
            shutil.rmtree(args.output_dir)
        else:
            raise FileExistsError(
                f"Output directory already exists: {args.output_dir}. "
                f"Use --force to overwrite."
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TinyDisasterVQA / GAP9 artifact generation")
    print("=" * 80)
    print(f"ONNX:       {args.onnx}")
    print(f"Output dir: {args.output_dir}")
    print(f"Calib CSV:  {args.calib_csv}")
    print(f"Dataset:    {args.dataset_root}")
    print(f"Num calib:  {args.num_calib}")
    print(f"Use NE16:   {not args.no_ne16}")
    print()

    print("[1/6] Loading ONNX into NNTool...")
    G = NNGraph.load_graph(str(args.onnx))
    print("[OK] Loaded graph")

    print("[2/6] Adjusting graph order and applying fusions...")
    G.adjust_order()
    G.fusions()
    print("[OK] Graph adjusted/fused")

    image_size = args.image_size or infer_image_size_from_graph(G)
    print(f"[INFO] Image size: {image_size}")

    print("[INFO] Graph inputs:")
    for node in G.inputs():
        print(f"  {node.name}: {node.out_dims}")

    print("[INFO] Graph outputs:")
    for node in G.outputs():
        print(f"  {node.name}: {node.in_dims}")

    print("[3/6] Loading representative calibration rows...")
    rows = load_rows(
        csv_path=args.calib_csv,
        dataset_root=args.dataset_root,
        limit=args.num_calib,
    )
    print(f"[OK] Loaded {len(rows)} calibration rows")

    first_inputs = make_inputs(rows[0], dataset_root=args.dataset_root, image_size=image_size)
    print("[INFO] First input shapes:")
    for idx, arr in enumerate(first_inputs):
        print(f"  input {idx}: shape={arr.shape}, dtype={arr.dtype}")

    print("[4/6] Collecting statistics...")
    stats = G.collect_statistics(
        representative_dataset(
            rows=rows,
            dataset_root=args.dataset_root,
            image_size=image_size,
        )
    )
    print("[OK] Statistics collected")

    print("[5/6] Quantizing graph...")
    G.quantize(
        statistics=stats,
        graph_options=quantization_options(
            use_ne16=(not args.no_ne16),
        ),
    )
    print("[OK] Quantized")

    print("[INFO] Running quantized graph once in NNTool...")
    qout = G.execute(first_inputs, quantize=True)
    print(f"[OK] NNTool execute produced {len(qout)} tensor(s)")
    if qout:
        final = qout[-1]
        print(f"[INFO] Final output shape={getattr(final, 'shape', None)}, dtype={getattr(final, 'dtype', None)}")
        try:
            flat = np.asarray(final).reshape(-1)
            print(f"[INFO] Final output argmax={int(np.argmax(flat))}")
            print(f"[INFO] First 10 output values={flat[:10]}")
        except Exception as exc:
            print(f"[WARN] Could not summarize output: {exc}")

    print("[6/6] Generating GAP9 Autotiler artifacts...")
    res = G.execute_on_target(
        directory=str(args.output_dir),
        finput_tensors=first_inputs,
        output_tensors=1,
        print_output=True,
        settings=model_settings(
            tensor_directory="tensors",
            model_directory="at_model",

            l3_ram_device="AT_MEM_L3_DEFAULTRAM",
            l3_flash_device="AT_MEM_L3_DEFAULTFLASH",

            privileged_l3_flash_device="AT_MEM_L3_MRAMFLASH",
            privileged_l3_flash_size=args.privileged_l3_flash_size,

            graph_size_opt=2,
            graph_dump_tensor_to_file=True,
            graph_const_exec_from_flash=True,
            graph_l1_promotion=2,
        ),
    )

    print("[OK] execute_on_target completed")
    print(f"[INFO] Generated directory: {args.output_dir}")

    print()
    print("Generated files preview:")
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(args.output_dir)}")

    print()
    print("=" * 80)
    print("GAP9 artifact generation complete")
    print("=" * 80)


if __name__ == "__main__":
    main()