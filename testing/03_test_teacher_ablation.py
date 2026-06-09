#!/usr/bin/env python3
"""
03_test_teacher_ablation.py

Smoke-test the official T1-T6 teacher ablation.

Each run trains for 1 epoch by default.

T1: cap10 + LSTM     + CE
T2: cap5  + LSTM     + CE
T3: cap10 + template + CE
T4: cap5  + template + CE
T5: cap5  + template + weighted CE
T6: cap5  + template + count auxiliary loss

Default uses --no-pretrained to avoid downloads and keep the test lightweight.
Use --pretrained if you also want to check torchvision pretrained weight loading.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--backbone", type=str, default="convnext_tiny")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/testing_teacher_ablation"))
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--pretrained", action="store_true", default=False)
    parser.add_argument("--continue-on-error", action="store_true", default=False)

    return parser.parse_args()


def check_required_files() -> None:
    required = [
        "outputs/training_data_cap5/train.csv",
        "outputs/training_data_cap5/valid.csv",
        "outputs/training_data_cap5/test.csv",
        "outputs/training_data_cap5/metadata.json",
        "outputs/answer_space_cap5/class_weights_edge_global_by_label.json",
        "outputs/training_data_cap10/train.csv",
        "outputs/training_data_cap10/valid.csv",
        "outputs/training_data_cap10/test.csv",
        "outputs/training_data_cap10/metadata.json",
        "outputs/answer_space_cap10/class_weights_edge_global_by_label.json",
        "scripts/05_train_teacher.py",
    ]

    missing = [REPO_ROOT / path for path in required if not (REPO_ROOT / path).exists()]

    if missing:
        raise FileNotFoundError(
            "Missing required file(s):\n"
            + "\n".join(f"  {path}" for path in missing)
            + "\n\nRun scripts 02-04 for cap5 and cap10 first."
        )


def base_train_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/05_train_teacher.py",
        "--dataset-root",
        str(args.dataset_root),
        "--runs-dir",
        str(args.runs_dir),
        "--backbone",
        args.backbone,
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
        "--no-augment-train",
        "--early-stopping-patience",
        "0",
        "--log-interval",
        "999999",
    ]

    if args.pretrained:
        cmd.append("--pretrained")
    else:
        cmd.append("--no-pretrained")

    return cmd


def cap_paths(cap: int) -> dict[str, str]:
    return {
        "train_csv": f"outputs/training_data_cap{cap}/train.csv",
        "valid_csv": f"outputs/training_data_cap{cap}/valid.csv",
        "test_csv": f"outputs/training_data_cap{cap}/test.csv",
        "metadata": f"outputs/training_data_cap{cap}/metadata.json",
        "class_weights": f"outputs/answer_space_cap{cap}/class_weights_edge_global_by_label.json",
    }


def make_run_cmd(args: argparse.Namespace, spec: dict[str, str | int | float]) -> list[str]:
    cap = int(spec["cap"])
    paths = cap_paths(cap)

    cmd = base_train_cmd(args)

    cmd += [
        "--train-csv",
        paths["train_csv"],
        "--valid-csv",
        paths["valid_csv"],
        "--test-csv",
        paths["test_csv"],
        "--metadata",
        paths["metadata"],
        "--class-weights",
        paths["class_weights"],
        "--run-name",
        str(spec["run_name"]),
        "--question-encoder",
        str(spec["question_encoder"]),
        "--loss-mode",
        str(spec["loss_mode"]),
    ]

    if spec["loss_mode"] == "count_aux":
        cmd += [
            "--count-aux-weight",
            str(spec.get("count_aux_weight", 0.5)),
        ]

    return cmd


def main() -> None:
    args = parse_args()
    check_required_files()

    specs: list[dict[str, str | int | float]] = [
        {
            "run_name": "smoke_T1_cap10_lstm_ce",
            "cap": 10,
            "question_encoder": "lstm",
            "loss_mode": "ce",
        },
        {
            "run_name": "smoke_T2_cap5_lstm_ce",
            "cap": 5,
            "question_encoder": "lstm",
            "loss_mode": "ce",
        },
        {
            "run_name": "smoke_T3_cap10_template_ce",
            "cap": 10,
            "question_encoder": "template",
            "loss_mode": "ce",
        },
        {
            "run_name": "smoke_T4_cap5_template_ce",
            "cap": 5,
            "question_encoder": "template",
            "loss_mode": "ce",
        },
        {
            "run_name": "smoke_T5_cap5_template_weighted_ce",
            "cap": 5,
            "question_encoder": "template",
            "loss_mode": "weighted_ce",
        },
        {
            "run_name": "smoke_T6_cap5_template_count_aux",
            "cap": 5,
            "question_encoder": "template",
            "loss_mode": "count_aux",
            "count_aux_weight": 0.5,
        },
    ]

    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    print("=" * 80)
    print("TinyDisasterVQA / Teacher Ablation Smoke Test")
    print("=" * 80)
    print(f"Repo root:   {REPO_ROOT}")
    print(f"Runs dir:    {args.runs_dir}")
    print(f"Pretrained:  {args.pretrained}")
    print(f"Epochs:      {args.epochs}")
    print(f"Batch size:  {args.batch_size}")
    print(f"PYTHONPATH:  {env['PYTHONPATH']}")
    print()

    failed: list[str] = []

    for idx, spec in enumerate(specs, start=1):
        run_name = str(spec["run_name"])
        cmd = make_run_cmd(args, spec)

        print()
        print("=" * 80)
        print(f"[{idx}/{len(specs)}] Running {run_name}")
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
            failed.append(run_name)
            print()
            print(f"FAILED: {run_name} returned exit code {result.returncode}")

            if not args.continue_on_error:
                raise SystemExit(result.returncode)

        else:
            print()
            print(f"PASSED: {run_name}")

    print()
    print("=" * 80)

    if failed:
        print("Teacher ablation smoke test finished with failures:")
        for name in failed:
            print(f"  - {name}")
        raise SystemExit(1)

    print("All T1-T6 teacher smoke tests passed.")
    print("=" * 80)


if __name__ == "__main__":
    main()