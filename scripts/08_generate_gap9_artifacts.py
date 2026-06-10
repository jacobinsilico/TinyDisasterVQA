#!/usr/bin/env python3
"""
08_generate_gap9_artifacts.py

Generate GAP9 NNTool/Autotiler artifacts from exported TinyDisasterVQA ONNX models.

Final v2 deployment defaults:
  - input ONNX models from: onnx/
  - calibration CSV: outputs/training_data_cap5/valid.csv
  - image size: 128
  - output root: gap9_generated_final/
  - target models:
      tdm_fast_128_ce_best.onnx
      tdm_xs_128_ce_best.onnx

Run inside the GAP9 Docker container, with GAP9 env sourced:

  source /app/install/gap9-sdk/.gap9-venv/bin/activate
  source /app/install/gap9-sdk/configs/gap9_evk_audio.sh
  export GVSOC_INSTALL_DIR=/app/install/gap9-sdk/install/workstation

  cd /app/TinyDisasterVQA
  PYTHONPATH=src python scripts/08_generate_gap9_artifacts.py --force

Single model:

  PYTHONPATH=src python scripts/08_generate_gap9_artifacts.py \
    --onnx onnx/tdm_fast_128_ce_best.onnx \
    --force

Outputs:

  gap9_generated_final/
    tdm_fast_128_ce_best/
      at_model/
      tensors/
      manifest.json
      calibration_manifest.csv
      ...

    tdm_xs_128_ce_best/
      at_model/
      tensors/
      manifest.json
      calibration_manifest.csv
      ...
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from nntool.api import NNGraph
from nntool.api.utils import model_settings, quantization_options


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_FINAL_ONNX_NAMES = (
    "tdm_fast_128_ce_best.onnx",
    "tdm_xs_128_ce_best.onnx",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help=(
            "Path to one exported ONNX model. "
            "If omitted, the script searches --onnx-dir for final deployment ONNX files."
        ),
    )
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=Path("onnx"),
        help="Directory containing exported ONNX models.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("gap9_generated_final"),
        help="Root directory where per-model GAP9 artifacts are generated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Explicit output directory for a single --onnx model. "
            "If omitted, uses --output-root/<onnx_stem>."
        ),
    )
    parser.add_argument(
        "--calib-csv",
        type=Path,
        default=Path("outputs/training_data_cap5/valid.csv"),
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
        default=64,
        help="Number of representative calibration samples.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help=(
            "Input image size. Final v2 deployment uses 128. "
            "If set to <=0, inferred from ONNX input shape."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete output directories if they already exist.",
    )
    parser.add_argument(
        "--no-ne16",
        action="store_true",
        help="Disable NE16 quantization option.",
    )
    parser.add_argument(
        "--privileged-l3-flash-size",
        type=int,
        default=1_800_000,
        help="Privileged L3 flash size passed to model_settings.",
    )
    parser.add_argument(
        "--max-preview-files",
        type=int,
        default=80,
        help="Maximum number of generated files to print per model.",
    )

    return parser.parse_args()


def discover_onnx_models(args: argparse.Namespace) -> list[Path]:
    if args.onnx is not None:
        if not args.onnx.exists():
            raise FileNotFoundError(f"Missing ONNX model: {args.onnx}")
        return [args.onnx]

    if not args.onnx_dir.exists():
        raise FileNotFoundError(f"Missing ONNX directory: {args.onnx_dir}")

    final_models = [
        args.onnx_dir / name
        for name in DEFAULT_FINAL_ONNX_NAMES
        if (args.onnx_dir / name).exists()
    ]

    if final_models:
        return final_models

    fallback = sorted(args.onnx_dir.glob("*.onnx"))
    if not fallback:
        raise FileNotFoundError(f"No .onnx models found in {args.onnx_dir}")

    return fallback


def resolve_output_dir(args: argparse.Namespace, onnx_path: Path) -> Path:
    if args.output_dir is not None:
        if args.onnx is None:
            raise ValueError("--output-dir may only be used together with a single --onnx model.")
        return args.output_dir

    return args.output_root / onnx_path.stem


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
    """
    Match TinyDisasterVQA eval transform:
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
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)

    return arr.astype(np.float32)


def row_is_usable(row: dict[str, str], dataset_root: Path) -> bool:
    try:
        _ = resolve_image_path(row, dataset_root)
    except FileNotFoundError:
        return False

    try:
        qtid = int(row["question_template_id"])
    except Exception:
        return False

    return 0 <= qtid < 31


def load_all_usable_rows(csv_path: Path, dataset_root: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing calibration CSV: {csv_path}")

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        fieldnames = set(reader.fieldnames or [])
        required = {"question_template_id"}
        missing = required - fieldnames
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

        if "image_path" not in fieldnames and "image_rel_path" not in fieldnames:
            raise ValueError(
                f"{csv_path} must contain at least one of: image_path, image_rel_path"
            )

        rows = [
            row
            for row in reader
            if row_is_usable(row, dataset_root=dataset_root)
        ]

    if not rows:
        raise RuntimeError(f"No usable calibration rows found in {csv_path}")

    return rows


def stratified_select_rows(
    rows: list[dict[str, str]],
    limit: int,
) -> list[dict[str, str]]:
    """
    Deterministically choose calibration rows.

    Priority:
      1. Spread across edge_head/question_type if available.
      2. Spread across question_template_id.
      3. Keep original CSV order within each bucket.

    This is better than simply taking the first N rows because calibration should
    see binary, condition, count, and density-like questions when possible.
    """
    if limit <= 0:
        raise ValueError("--num-calib must be positive.")

    bucketed: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        if row.get("edge_head"):
            key = f"edge_head={row['edge_head']}"
        elif row.get("question_type"):
            key = f"question_type={row['question_type']}"
        else:
            key = f"template={row['question_template_id']}"

        bucketed[key].append(row)

    selected: list[dict[str, str]] = []
    bucket_keys = sorted(bucketed.keys())

    cursor = 0
    while len(selected) < limit:
        made_progress = False

        for key in bucket_keys:
            bucket = bucketed[key]
            if cursor < len(bucket):
                selected.append(bucket[cursor])
                made_progress = True

                if len(selected) >= limit:
                    break

        if not made_progress:
            break

        cursor += 1

    return selected

def make_question_onehot(question_template_id: int, num_question_templates: int = 31) -> np.ndarray:
    if question_template_id < 0 or question_template_id >= num_question_templates:
        raise ValueError(
            f"question_template_id={question_template_id} outside "
            f"[0, {num_question_templates - 1}]"
        )

    onehot = np.zeros((1, num_question_templates), dtype=np.float32)
    onehot[0, question_template_id] = 1.0
    return onehot

def make_inputs(row: dict[str, str], dataset_root: Path, image_size: int) -> list[np.ndarray]:
    image_path = resolve_image_path(row, dataset_root)
    image = preprocess_image(image_path, image_size=image_size)

    question_template_id = int(row["question_template_id"])
    question_template_onehot = make_question_onehot(question_template_id)

    return [image, question_template_onehot]


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


def write_calibration_manifest(
    rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    manifest_path = output_dir / "calibration_manifest.csv"

    preferred_columns = [
        "image_rel_path",
        "image_path",
        "question",
        "answer",
        "question_template_id",
        "edge_head",
        "question_type",
        "edge_global_label",
        "edge_global_class",
        "target",
    ]

    available_columns = []
    for col in preferred_columns:
        if any(col in row for row in rows):
            available_columns.append(col)

    extra_columns = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key not in available_columns
        }
    )

    fieldnames = available_columns + extra_columns

    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_manifest_json(
    *,
    output_dir: Path,
    onnx_path: Path,
    calib_csv: Path,
    dataset_root: Path,
    image_size: int,
    num_calib: int,
    use_ne16: bool,
    privileged_l3_flash_size: int,
    graph_input_summary: list[str],
    graph_output_summary: list[str],
) -> None:
    manifest = {
        "model_name": onnx_path.stem,
        "onnx_path": str(onnx_path),
        "calib_csv": str(calib_csv),
        "dataset_root": str(dataset_root),
        "image_size": image_size,
        "num_calib": num_calib,
        "use_ne16": use_ne16,
        "privileged_l3_flash_size": privileged_l3_flash_size,
        "expected_inputs": {
            "image": [1, 3, image_size, image_size],
            "question_template_onehot": [1, 31],
        },
        "expected_output": {
            "logits": [1, 14],
        },
        "graph_inputs": graph_input_summary,
        "graph_outputs": graph_output_summary,
    }

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def summarize_graph_inputs_outputs(G: NNGraph) -> tuple[list[str], list[str]]:
    graph_inputs = []
    graph_outputs = []

    for node in G.inputs():
        graph_inputs.append(f"{node.name}: {node.out_dims}")

    for node in G.outputs():
        graph_outputs.append(f"{node.name}: {node.in_dims}")

    return graph_inputs, graph_outputs


def print_generated_preview(output_dir: Path, max_files: int) -> None:
    files = [path for path in sorted(output_dir.rglob("*")) if path.is_file()]

    print()
    print("Generated files preview:")
    for path in files[:max_files]:
        print(f"  {path.relative_to(output_dir)}")

    if len(files) > max_files:
        print(f"  ... {len(files) - max_files} more file(s)")


def generate_for_one_model(
    *,
    onnx_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    if output_dir.exists():
        if args.force:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                f"Use --force to overwrite."
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 80)
    print("TinyDisasterVQA / GAP9 artifact generation")
    print("=" * 80)
    print(f"ONNX:       {onnx_path}")
    print(f"Output dir: {output_dir}")
    print(f"Calib CSV:  {args.calib_csv}")
    print(f"Dataset:    {args.dataset_root}")
    print(f"Num calib:  {args.num_calib}")
    print(f"Use NE16:   {not args.no_ne16}")
    print()

    print("[1/6] Loading ONNX into NNTool...")
    G = NNGraph.load_graph(str(onnx_path))
    print("[OK] Loaded graph")

    print("[2/6] Adjusting graph order and applying fusions...")
    G.adjust_order()
    G.fusions()
    print("[OK] Graph adjusted/fused")

    image_size = (
        int(args.image_size)
        if args.image_size is not None and int(args.image_size) > 0
        else infer_image_size_from_graph(G)
    )

    print(f"[INFO] Image size: {image_size}")

    graph_inputs, graph_outputs = summarize_graph_inputs_outputs(G)

    print("[INFO] Graph inputs:")
    for item in graph_inputs:
        print(f"  {item}")

    print("[INFO] Graph outputs:")
    for item in graph_outputs:
        print(f"  {item}")

    print("[3/6] Loading representative calibration rows...")
    all_rows = load_all_usable_rows(
        csv_path=args.calib_csv,
        dataset_root=args.dataset_root,
    )
    rows = stratified_select_rows(all_rows, limit=args.num_calib)
    print(f"[OK] Loaded {len(rows)} calibration rows from {len(all_rows)} usable rows")

    write_calibration_manifest(rows, output_dir=output_dir)

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
        print(
            "[INFO] Final output "
            f"shape={getattr(final, 'shape', None)}, "
            f"dtype={getattr(final, 'dtype', None)}"
        )

        try:
            flat = np.asarray(final).reshape(-1)
            print(f"[INFO] Final output argmax={int(np.argmax(flat))}")
            print(f"[INFO] First 10 output values={flat[:10]}")
        except Exception as exc:
            print(f"[WARN] Could not summarize output: {exc}")

    write_manifest_json(
        output_dir=output_dir,
        onnx_path=onnx_path,
        calib_csv=args.calib_csv,
        dataset_root=args.dataset_root,
        image_size=image_size,
        num_calib=len(rows),
        use_ne16=(not args.no_ne16),
        privileged_l3_flash_size=args.privileged_l3_flash_size,
        graph_input_summary=graph_inputs,
        graph_output_summary=graph_outputs,
    )

    print("[6/6] Generating GAP9 Autotiler artifacts...")
    _ = G.execute_on_target(
        directory=str(output_dir),
        finput_tensors=first_inputs,
        print_output=False,
        settings=model_settings(
            tensor_directory="tensors",
            model_directory="at_model",

            l3_ram_device="AT_MEM_L3_DEFAULTRAM",
            l3_flash_device="AT_MEM_L3_DEFAULTFLASH",

            privileged_l3_flash_device="AT_MEM_L3_MRAMFLASH",
            privileged_l3_flash_size=args.privileged_l3_flash_size,

            graph_size_opt=2,
            graph_dump_tensor_to_file=False,
            graph_const_exec_from_flash=True,
            graph_l1_promotion=2,
        ),
    )

    print("[OK] execute_on_target completed")
    print(f"[INFO] Generated directory: {output_dir}")

    print_generated_preview(output_dir=output_dir, max_files=args.max_preview_files)

    print()
    print("=" * 80)
    print(f"GAP9 artifact generation complete: {onnx_path.stem}")
    print("=" * 80)


def main() -> None:
    args = parse_args()

    onnx_models = discover_onnx_models(args)

    print("=" * 80)
    print("TinyDisasterVQA / GAP9 artifact generation batch")
    print("=" * 80)
    print(f"Found {len(onnx_models)} ONNX model(s):")
    for path in onnx_models:
        print(f"  - {path}")

    for onnx_path in onnx_models:
        output_dir = resolve_output_dir(args, onnx_path)
        generate_for_one_model(
            onnx_path=onnx_path,
            output_dir=output_dir,
            args=args,
        )

    print()
    print("=" * 80)
    print("All GAP9 artifact generation complete")
    print("=" * 80)


if __name__ == "__main__":
    main()