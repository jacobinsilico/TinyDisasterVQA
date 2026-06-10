#!/usr/bin/env python3
"""
04_test_student_ablation.py

One-epoch overfit/smoke test for the planned student ablation.

Tests:
  S1  tdm_s    CE
  S2  tdm_m    CE
  S3  tdm_l    CE
  S4  tdm_fast CE
  S5  tdm_m    weighted CE
  S6  tdm_l    weighted CE

Optional KD tests, enabled when teacher checkpoints are provided:
  S7  tdm_m    KD from T5
  S8  tdm_m    KD from T6
  S9  tdm_l    KD from T5
  S10 tdm_fast KD from T5

Default:
  - cap5 only
  - 1 epoch
  - overfit-samples 32
  - no augmentation
  - num-workers 0
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--overfit-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--data-dir", type=Path, default=Path("outputs/training_data_cap5"))
    parser.add_argument("--answer-space-dir", type=Path, default=Path("outputs/answer_space_cap5"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/testing_student_ablation"))

    parser.add_argument("--teacher-t5-checkpoint", type=Path, default=None)
    parser.add_argument("--teacher-t6-checkpoint", type=Path, default=None)

    parser.add_argument("--skip-kd", action="store_true", default=False)
    parser.add_argument("--continue-on-error", action="store_true", default=False)

    return parser.parse_args()


def check_required_files(args: argparse.Namespace) -> None:
    required = [
        args.data_dir / "train.csv",
        args.data_dir / "valid.csv",
        args.data_dir / "test.csv",
        args.data_dir / "metadata.json",
        args.answer_space_dir / "class_weights_edge_global_by_label.json",
        Path("scripts/06_train_student.py"),
    ]

    missing = [REPO_ROOT / path for path in required if not (REPO_ROOT / path).exists()]

    if missing:
        raise FileNotFoundError(
            "Missing required file(s):\n"
            + "\n".join(f"  {path}" for path in missing)
            + "\n\nRun preprocessing scripts 02-04 for cap5 first."
        )


def base_cmd(args: argparse.Namespace, run_stamp: str) -> list[str]:
    return [
        sys.executable,
        "-u",
        "scripts/06_train_student.py",
        "--dataset-root",
        str(args.dataset_root),
        "--runs-dir",
        str(args.runs_dir),
        "--train-csv",
        str(args.data_dir / "train.csv"),
        "--valid-csv",
        str(args.data_dir / "valid.csv"),
        "--test-csv",
        str(args.data_dir / "test.csv"),
        "--metadata",
        str(args.data_dir / "metadata.json"),
        "--class-weights",
        str(args.answer_space_dir / "class_weights_edge_global_by_label.json"),
        "--image-size",
        str(args.image_size),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--overfit-samples",
        str(args.overfit_samples),
        "--no-augment-train",
        "--patience",
        "0",
        "--log-interval",
        "1",
        "--run-name",
        run_stamp,  # overwritten per spec below
    ]


def make_cmd(args: argparse.Namespace, spec: dict[str, str], run_stamp: str) -> list[str]:
    cmd = base_cmd(args, run_stamp)

    # Replace placeholder run-name.
    run_name_idx = cmd.index("--run-name") + 1
    cmd[run_name_idx] = f"{spec['id']}_{spec['variant']}_{spec['mode']}_{run_stamp}"

    cmd += [
        "--student-variant",
        spec["variant"],
        "--mode",
        spec["mode"],
    ]

    if spec["mode"] == "kd":
        teacher_key = spec["teacher"]

        if teacher_key == "T5":
            teacher_path = args.teacher_t5_checkpoint
        elif teacher_key == "T6":
            teacher_path = args.teacher_t6_checkpoint
        else:
            raise ValueError(f"Unknown teacher key: {teacher_key}")

        if teacher_path is None:
            raise ValueError(f"{spec['id']} requires --teacher-{teacher_key.lower()}-checkpoint.")

        cmd += [
            "--teacher-checkpoint",
            str(teacher_path),
            "--kd-alpha",
            "0.7",
            "--kd-temperature",
            "4.0",
        ]

    return cmd


def main() -> None:
    args = parse_args()
    check_required_files(args)

    specs: list[dict[str, str]] = [
        {"id": "S1", "variant": "tdm_s", "mode": "ce"},
        {"id": "S2", "variant": "tdm_m", "mode": "ce"},
        {"id": "S3", "variant": "tdm_l", "mode": "ce"},
        {"id": "S4", "variant": "tdm_fast", "mode": "ce"},
        {"id": "S5", "variant": "tdm_m", "mode": "weighted_ce"},
        {"id": "S6", "variant": "tdm_l", "mode": "weighted_ce"},
    ]

    kd_specs: list[dict[str, str]] = [
        {"id": "S7", "variant": "tdm_m", "mode": "kd", "teacher": "T5"},
        {"id": "S8", "variant": "tdm_m", "mode": "kd", "teacher": "T6"},
        {"id": "S9", "variant": "tdm_l", "mode": "kd", "teacher": "T5"},
        {"id": "S10", "variant": "tdm_fast", "mode": "kd", "teacher": "T5"},
    ]

    if not args.skip_kd:
        if args.teacher_t5_checkpoint is not None and args.teacher_t6_checkpoint is not None:
            specs.extend(kd_specs)
        else:
            print("Skipping KD specs because T5/T6 teacher checkpoints were not both provided.")
            print("Use --teacher-t5-checkpoint and --teacher-t6-checkpoint to enable S7-S10.")

    run_stamp = time.strftime("%Y%m%d_%H%M%S")

    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"

    print("=" * 80)
    print("TinyDisasterVQA / Student Ablation Smoke + Overfit Test")
    print("=" * 80)
    print(f"Repo root:        {REPO_ROOT}")
    print(f"Data dir:         {args.data_dir}")
    print(f"Runs dir:         {args.runs_dir}")
    print(f"Epochs:           {args.epochs}")
    print(f"Overfit samples:  {args.overfit_samples}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Device:           {args.device}")
    print(f"Run stamp:        {run_stamp}")
    print(f"PYTHONPATH:       {env['PYTHONPATH']}")
    print()

    failed: list[str] = []

    for idx, spec in enumerate(specs, start=1):
        name = f"{spec['id']} {spec['variant']} {spec['mode']}"
        cmd = make_cmd(args, spec, run_stamp)

        print()
        print("=" * 80)
        print(f"[{idx}/{len(specs)}] Running {name}")
        print("=" * 80)
        print(" ".join(cmd))
        print()

        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            check=False,
        )

        if result.returncode != 0:
            failed.append(name)
            print()
            print(f"FAILED: {name} returned exit code {result.returncode}")

            if not args.continue_on_error:
                raise SystemExit(result.returncode)
        else:
            print()
            print(f"PASSED: {name}")

    print()
    print("=" * 80)

    if failed:
        print("Student ablation smoke/overfit test finished with failures:")
        for name in failed:
            print(f"  - {name}")
        raise SystemExit(1)

    print("All requested student smoke/overfit tests passed.")
    print("=" * 80)


if __name__ == "__main__":
    main()