#!/usr/bin/env python3
"""
04_test_student_models.py

Sanity-test TinyDisasterVQA student model variants.

Checks:
  - model construction for tdm_xxs / tdm_xs / tdm_s / tdm_m
  - single-head and multi-head modes
  - parameter counts and rough int8/fp32 sizes
  - forward pass returns [B, 19] logits
  - CE loss + backward pass works

Run from repo root:

PYTHONPATH=src python scripts/04_test_student_models.py

Optional:

PYTHONPATH=src python scripts/04_test_student_models.py --verbose
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from tinydisastervqa.models import (  # noqa: E402
    count_parameters,
    describe_model,
    estimate_int8_model_size_kb,
    estimate_model_size_mb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--metadata", type=Path, default=Path("outputs/training_data/metadata.json"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--num-question-templates", type=int, default=31)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def get_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_metadata(path: Path) -> dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    print(f"[WARN] Metadata not found at {path}. Using minimal dummy metadata.")
    return {
        "vocab_size_with_pad": 64,
        "pad_id": 0,
    }


def get_builder():
    """
    Expected after the multi-head models.py patch:
      build_tdm_from_metadata(...)

    Fails loudly if the patched builder is missing.
    """
    import tinydisastervqa.models as model_module

    if not hasattr(model_module, "build_tdm_from_metadata"):
        available = [name for name in dir(model_module) if name.startswith("build_tdm")]
        raise AttributeError(
            "Could not find build_tdm_from_metadata(...) in tinydisastervqa.models. "
            "Make sure you copied the patched multi-head models.py into "
            "src/tinydisastervqa/models.py. "
            f"Available build_tdm* functions: {available}"
        )

    return model_module.build_tdm_from_metadata


def build_student(
    builder,
    metadata: dict[str, Any],
    variant: str,
    head_type: str,
    num_classes: int,
    num_question_templates: int,
):
    """
    Calls build_tdm_from_metadata robustly even if the exact parameter name is
    variant or student_variant.
    """
    sig = inspect.signature(builder)
    kwargs: dict[str, Any] = {}

    if "metadata" in sig.parameters:
        kwargs["metadata"] = metadata

    if "variant" in sig.parameters:
        kwargs["variant"] = variant
    elif "student_variant" in sig.parameters:
        kwargs["student_variant"] = variant
    elif "student_size" in sig.parameters:
        # Backward-compatible fallback, but this will not cover xxs/xs.
        kwargs["student_size"] = variant.replace("tdm_", "")
    else:
        raise TypeError(
            "build_tdm_from_metadata must accept one of: variant, student_variant, student_size"
        )

    if "head_type" in sig.parameters:
        kwargs["head_type"] = head_type

    if "num_classes" in sig.parameters:
        kwargs["num_classes"] = num_classes

    if "num_question_templates" in sig.parameters:
        kwargs["num_question_templates"] = num_question_templates

    return builder(**kwargs)


def make_fake_batch(
    batch_size: int,
    image_size: int,
    num_classes: int,
    num_question_templates: int,
    vocab_size: int,
    seq_len: int,
    device: torch.device,
) -> dict[str, Any]:
    edge_heads = [
        "binary",
        "condition",
        "count",
        "density",
    ]
    repeated_edge_heads = [edge_heads[i % len(edge_heads)] for i in range(batch_size)]

    # Conventional ids used only if the model supports edge_head_ids.
    edge_head_to_id = {
        "binary": 0,
        "condition": 1,
        "count": 2,
        "density": 3,
    }

    return {
        "images": torch.randn(batch_size, 3, image_size, image_size, device=device),
        "question_tokens": torch.randint(
            low=1,
            high=max(vocab_size, 2),
            size=(batch_size, seq_len),
            device=device,
        ),
        "question_lengths": torch.full((batch_size,), seq_len, dtype=torch.long, device=device),
        "question_template_ids": (
            torch.arange(batch_size, device=device) % num_question_templates
        ).long(),
        "edge_heads": repeated_edge_heads,
        "edge_head_ids": torch.tensor(
            [edge_head_to_id[h] for h in repeated_edge_heads],
            dtype=torch.long,
            device=device,
        ),
        "targets": torch.randint(0, num_classes, (batch_size,), device=device),
    }


def forward_model(model: torch.nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    """
    Calls the student forward while adapting to the model's exact forward signature.
    """
    sig = inspect.signature(model.forward)
    params = sig.parameters

    kwargs: dict[str, Any] = {}

    if "images" in params:
        kwargs["images"] = batch["images"]
    else:
        # All current models should use images as a named argument.
        kwargs["images"] = batch["images"]

    if "question_tokens" in params:
        kwargs["question_tokens"] = batch["question_tokens"]

    if "question_lengths" in params:
        kwargs["question_lengths"] = batch["question_lengths"]

    if "question_template_ids" in params:
        kwargs["question_template_ids"] = batch["question_template_ids"]

    # Multi-head compatibility. Different patches may use slightly different names.
    if "edge_heads" in params:
        kwargs["edge_heads"] = batch["edge_heads"]
    if "edge_head" in params:
        kwargs["edge_head"] = batch["edge_heads"]
    if "edge_head_ids" in params:
        kwargs["edge_head_ids"] = batch["edge_head_ids"]
    if "edge_head_id" in params:
        kwargs["edge_head_id"] = batch["edge_head_ids"]

    return model(**kwargs)


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    torch.manual_seed(42)

    metadata = load_metadata(args.metadata)
    vocab_size = int(metadata.get("vocab_size_with_pad", 64))

    builder = get_builder()

    variants = ["tdm_xxs", "tdm_xs", "tdm_s", "tdm_m"]
    head_types = ["single", "multihead"]

    print("=" * 100)
    print("TinyDisasterVQA / Student model sanity test")
    print("=" * 100)
    print(f"Device:                 {device}")
    print(f"Image size:             {args.image_size}")
    print(f"Batch size:             {args.batch_size}")
    print(f"Num classes:            {args.num_classes}")
    print(f"Num question templates: {args.num_question_templates}")
    print(f"Vocab size:             {vocab_size}")
    print()

    rows: list[dict[str, Any]] = []

    for variant in variants:
        for head_type in head_types:
            name = f"{variant}_{head_type}"
            try:
                model = build_student(
                    builder=builder,
                    metadata=metadata,
                    variant=variant,
                    head_type=head_type,
                    num_classes=args.num_classes,
                    num_question_templates=args.num_question_templates,
                ).to(device)

                model.train()

                batch = make_fake_batch(
                    batch_size=args.batch_size,
                    image_size=args.image_size,
                    num_classes=args.num_classes,
                    num_question_templates=args.num_question_templates,
                    vocab_size=vocab_size,
                    seq_len=args.seq_len,
                    device=device,
                )

                logits = forward_model(model, batch)

                if logits.ndim != 2:
                    raise RuntimeError(f"Expected logits.ndim == 2, got shape {tuple(logits.shape)}")

                expected_shape = (args.batch_size, args.num_classes)
                if tuple(logits.shape) != expected_shape:
                    raise RuntimeError(
                        f"Expected logits shape {expected_shape}, got {tuple(logits.shape)}"
                    )

                if not torch.isfinite(logits).all():
                    raise RuntimeError("Logits contain NaN or Inf.")

                loss = F.cross_entropy(logits, batch["targets"])
                if not torch.isfinite(loss):
                    raise RuntimeError("Loss is NaN or Inf.")

                loss.backward()

                grad_params = sum(
                    1 for p in model.parameters() if p.requires_grad and p.grad is not None
                )
                if grad_params == 0:
                    raise RuntimeError("No gradients were produced.")

                total_params = count_parameters(model, trainable_only=False)
                trainable_params = count_parameters(model, trainable_only=True)
                fp32_mb = estimate_model_size_mb(model)
                int8_kb = estimate_int8_model_size_kb(model)

                rows.append(
                    {
                        "variant": variant,
                        "head_type": head_type,
                        "params": total_params,
                        "trainable": trainable_params,
                        "fp32_mb": fp32_mb,
                        "int8_kb": int8_kb,
                        "shape": tuple(logits.shape),
                        "loss": float(loss.item()),
                        "status": "OK",
                    }
                )

                if args.verbose:
                    print()
                    print("-" * 100)
                    print(name)
                    print("-" * 100)
                    print(describe_model(model))

            except Exception as exc:
                rows.append(
                    {
                        "variant": variant,
                        "head_type": head_type,
                        "params": -1,
                        "trainable": -1,
                        "fp32_mb": -1.0,
                        "int8_kb": -1.0,
                        "shape": "-",
                        "loss": -1.0,
                        "status": f"FAIL: {type(exc).__name__}: {exc}",
                    }
                )

    print(
        f"{'Variant':<10} {'Head':<10} {'Params':>12} {'Int8 KB':>10} "
        f"{'FP32 MB':>9} {'Output':>12} {'Loss':>9}  Status"
    )
    print("-" * 100)

    for row in rows:
        params = f"{row['params']:,}" if row["params"] >= 0 else "-"
        int8_kb = f"{row['int8_kb']:.1f}" if row["int8_kb"] >= 0 else "-"
        fp32_mb = f"{row['fp32_mb']:.3f}" if row["fp32_mb"] >= 0 else "-"
        loss = f"{row['loss']:.4f}" if row["loss"] >= 0 else "-"
        shape = str(row["shape"])

        print(
            f"{row['variant']:<10} {row['head_type']:<10} {params:>12} "
            f"{int8_kb:>10} {fp32_mb:>9} {shape:>12} {loss:>9}  {row['status']}"
        )

    print("-" * 100)

    failures = [row for row in rows if row["status"] != "OK"]
    if failures:
        print()
        print("Some student model checks failed:")
        for row in failures:
            print(f"  - {row['variant']} / {row['head_type']}: {row['status']}")
        raise SystemExit(1)

    print()
    print("All student model sanity checks passed.")


if __name__ == "__main__":
    main()
