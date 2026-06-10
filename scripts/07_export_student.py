#!/usr/bin/env python3
"""
07_export_student.py

Export all TinyDisasterVQA student .pt checkpoints in checkpoints/ to ONNX.

Default behavior:
  - finds all *.pt files in models/
  - reconstructs the correct TDM student from checkpoint config or filename
  - exports each checkpoint to models/<same_name>.onnx
  - uses fixed batch size 1 and fixed image resolution
  - exports multihead models with an ONNX-friendly wrapper

Example:
  PYTHONPATH=src python scripts/07_export_student.py

Specific file:
  PYTHONPATH=src python scripts/07_export_student.py \
  --checkpoint checkpoints/tdm_fast_128_ce_best.pt

Check with ONNXRuntime if installed:
  PYTHONPATH=src python scripts/07_export_student.py --verify
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tinydisastervqa.data import load_json  # noqa: E402
from tinydisastervqa.models import (  # noqa: E402
    EDGE_HEADS,
    TDMVQA,
    build_tdm_from_metadata,
    describe_model,
)


class SingleHeadExportWrapper(nn.Module):
    """
    GAP9/NNTool-friendly ONNX wrapper for single-head TDM models.

    Instead of exporting question_template_id -> OneHot -> Linear,
    this wrapper takes the one-hot template vector directly.

    Inputs:
      image: [1, 3, H, W], float32
      question_template_onehot: [1, num_question_templates], float32

    Output:
      logits: [1, num_classes], float32
    """

    def __init__(self, model: TDMVQA) -> None:
        super().__init__()
        self.model = model

        if not hasattr(self.model.question_encoder, "linear"):
            raise AttributeError(
                "Expected model.question_encoder.linear. "
                "This wrapper assumes one_hot + Linear template encoder."
            )

    def forward(
        self,
        image: torch.Tensor,
        question_template_onehot: torch.Tensor,
    ) -> torch.Tensor:
        image_features = self.model.image_encoder(image)

        question_features = self.model.question_encoder.linear(
            question_template_onehot.float()
        )

        fused = torch.cat([image_features, question_features], dim=1)
        logits = self.model.classifier(fused)

        return logits


class MultiHeadExportWrapper(nn.Module):
    """
    ONNX-friendly wrapper for multihead TDM models.

    The original MultiHeadGlobalClassifier.forward() uses Python control flow:
      if bool(mask.any()):
          logits[mask] = ...

    That is bad for ONNX export because tracing may freeze only the dummy head.
    This wrapper instead:
      1. computes shared image/question/fusion features
      2. computes all four heads
      3. stacks them into [B, 4, num_classes]
      4. gathers the selected head using edge_head_id

    Inputs:
      image: [1, 3, H, W], float32
      question_template_id: [1], int64
      edge_head_id: [1], int64
        0=binary, 1=condition, 2=count, 3=density

    Output:
      logits: [1, 19], float32
    """

    def __init__(self, model: TDMVQA) -> None:
        super().__init__()
        self.model = model

        if model.config.head_type != "multihead":
            raise ValueError("MultiHeadExportWrapper requires a multihead model.")

        self.num_classes = int(model.config.num_classes)
        self.edge_head_names = tuple(model.config.edge_head_names)

    def forward(
        self,
        image: torch.Tensor,
        question_template_id: torch.Tensor,
        edge_head_id: torch.Tensor,
    ) -> torch.Tensor:
        image_features = self.model.image_encoder(image)
        question_features = self.model.question_encoder(question_template_id)

        fused = torch.cat([image_features, question_features], dim=1)
        hidden = self.model.classifier.trunk(fused)

        head_logits = []
        for head_name in self.edge_head_names:
            head_logits.append(self.model.classifier.heads[head_name](hidden))

        # [B, 4, num_classes]
        all_logits = torch.stack(head_logits, dim=1)

        # [B] -> [B, 1, num_classes]
        gather_idx = edge_head_id.long().view(-1, 1, 1)
        gather_idx = gather_idx.expand(-1, 1, self.num_classes)

        # [B, 1, num_classes] -> [B, num_classes]
        selected_logits = torch.gather(all_logits, dim=1, index=gather_idx).squeeze(1)

        return selected_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--models-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=Path("outputs/training_data_cap5/metadata.json"))
    parser.add_argument("--onnx-dir", type=Path, default=Path("onnx"))

    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--verify", action="store_true", help="Verify exported ONNX with onnxruntime if available.")
    parser.add_argument("--overwrite", action="store_true", default=True)

    # Fallbacks used if checkpoint config and filename do not provide values.
    parser.add_argument("--default-image-size", type=int, default=128)
    parser.add_argument("--default-num-classes", type=int, default=14)
    parser.add_argument("--default-num-question-templates", type=int, default=31)

    return parser.parse_args()


def infer_variant_from_name(name: str) -> str:
    lowered = name.lower()

    if "tdm_fast" in lowered or "tdm-fast" in lowered:
        return "tdm_fast"
    if "tdm_xxs" in lowered or "tdm-xxs" in lowered:
        return "tdm_xxs"
    if "tdm_xs" in lowered or "tdm-xs" in lowered:
        return "tdm_xs"
    if "tdm_s" in lowered or "tdm-s" in lowered:
        return "tdm_s"
    if "tdm_m" in lowered or "tdm-m" in lowered:
        return "tdm_m"
    if "tdm_l" in lowered or "tdm-l" in lowered:
        return "tdm_l"

    raise ValueError(f"Could not infer student variant from filename: {name}")


def infer_head_type_from_name(name: str) -> str:
    lowered = name.lower()

    if "multihead" in lowered:
        return "multihead"
    if "single" in lowered:
        return "single"

    # Final v2 deployment uses single-head edge_global by default.
    return "single"


def infer_image_size_from_name(name: str, default: int) -> int:
    lowered = name.lower()

    # Matches names like:
    #   tdm_xs_multihead_ce_128_best.pt
    #   tdm_xxs_single_ce_224_best.pt
    match = re.search(r"_(128|160|224)(?:_|\.|$)", lowered)
    if match:
        return int(match.group(1))

    return int(default)


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
    """
    Removes common prefixes from saved checkpoints.
    """
    cleaned = {}

    for key, value in state_dict.items():
        new_key = key

        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]

        cleaned[new_key] = value

    return cleaned


def resolve_export_config(
    checkpoint_path: Path,
    checkpoint: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    ckpt_config = get_checkpoint_config(checkpoint)
    filename = checkpoint_path.name

    variant = (
        ckpt_config.get("student_variant")
        or ckpt_config.get("variant")
        or infer_variant_from_name(filename)
    )

    head_type = (
        ckpt_config.get("head_type")
        or infer_head_type_from_name(filename)
    )

    image_size = int(
        ckpt_config.get("image_size")
        or infer_image_size_from_name(filename, args.default_image_size)
    )

    num_classes = int(
        ckpt_config.get("num_classes")
        or args.default_num_classes
    )

    num_question_templates = int(
        ckpt_config.get("num_question_templates")
        or args.default_num_question_templates
    )

    template_embed_dim = ckpt_config.get("template_embed_dim", None)
    fusion_hidden_dim = ckpt_config.get("fusion_hidden_dim", None)
    fusion_dropout = ckpt_config.get("fusion_dropout", None)
    fusion_layers = ckpt_config.get("fusion_layers", None)

    return {
        "variant": str(variant),
        "head_type": str(head_type),
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
    export_config: dict[str, Any],
) -> TDMVQA:
    if export_config["head_type"] != "single":
        raise ValueError(
            "Final v2 GAP9 export supports only single-head student models. "
            f"Got head_type={export_config['head_type']!r}."
        )

    model = build_tdm_from_metadata(
        metadata=metadata,
        variant=export_config["variant"],
        num_classes=export_config["num_classes"],
        num_question_templates=export_config["num_question_templates"],
        question_template_embed_dim=export_config["template_embed_dim"],
        fusion_hidden_dim=export_config["fusion_hidden_dim"],
        fusion_dropout=export_config["fusion_dropout"],
        fusion_layers=export_config["fusion_layers"],
    )

    return model


@torch.no_grad()
def verify_export(
    wrapper: nn.Module,
    onnx_path: Path,
    inputs: tuple[torch.Tensor, ...],
) -> None:
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
    except Exception as exc:
        print(f"[WARN] Verification skipped: missing onnx/onnxruntime ({exc})")
        return

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    pt_output = wrapper(*inputs).detach().cpu().numpy()

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    feed = {}
    for input_info, tensor in zip(session.get_inputs(), inputs):
        feed[input_info.name] = tensor.detach().cpu().numpy()

    ort_output = session.run(None, feed)[0]

    max_abs_diff = float(np.max(np.abs(pt_output - ort_output)))
    pt_argmax = int(pt_output.argmax(axis=1)[0])
    ort_argmax = int(ort_output.argmax(axis=1)[0])

    print(f"[VERIFY] max_abs_diff={max_abs_diff:.6g}")
    print(f"[VERIFY] PyTorch argmax={pt_argmax}, ONNX argmax={ort_argmax}")

    if pt_argmax != ort_argmax:
        print("[WARN] PyTorch and ONNX argmax differ.")


def export_one(
    checkpoint_path: Path,
    metadata: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    print()
    print("=" * 80)
    print(f"Exporting: {checkpoint_path}")
    print("=" * 80)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    export_config = resolve_export_config(checkpoint_path, checkpoint, args)

    print("Resolved export config:")
    for key, value in export_config.items():
        print(f"  {key}: {value}")

    model = build_model(metadata=metadata, export_config=export_config)

    state_dict = clean_state_dict(get_state_dict(checkpoint))
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print()
    print(describe_model(model))
    print()

    image_size = export_config["image_size"]
    head_type = export_config["head_type"]

    dummy_image = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    num_question_templates = export_config["num_question_templates"]

    dummy_question_template_onehot = torch.zeros(
        1,
        num_question_templates,
        dtype=torch.float32,
    )
    dummy_question_template_onehot[0, 0] = 1.0

    args.onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.onnx_dir / checkpoint_path.with_suffix(".onnx").name

    if onnx_path.exists() and not args.overwrite:
        print(f"[SKIP] ONNX already exists: {onnx_path}")
        return

    if head_type == "multihead":
        wrapper = MultiHeadExportWrapper(model).eval()
        dummy_edge_head_id = torch.tensor([0], dtype=torch.long)

        input_names = ["image", "question_template_id", "edge_head_id"]
        inputs = (dummy_image, dummy_question_template_onehot, dummy_edge_head_id)

    elif head_type == "single":
        wrapper = SingleHeadExportWrapper(model).eval()

        input_names = ["image", "question_template_onehot"]
        inputs = (dummy_image, dummy_question_template_onehot)

    else:
        raise ValueError(f"Unknown head_type: {head_type}")

    print(f"Writing ONNX: {onnx_path}")

    torch.onnx.export(
        wrapper,
        inputs,
        str(onnx_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=None,  # static shapes are better for GAP9 / NNTool
        dynamo=False,
    )

    print(f"[OK] Exported {onnx_path}")

    if args.verify:
        verify_export(wrapper=wrapper, onnx_path=onnx_path, inputs=inputs)


def main() -> None:
    args = parse_args()

    if args.metadata.exists():
        metadata = load_json(args.metadata)
    else:
        print(f"[WARN] Metadata not found at {args.metadata}; using empty metadata.")
        metadata = {}

    if args.checkpoint is not None:
        checkpoints = [args.checkpoint]
    else:
        checkpoints = sorted(args.models_dir.glob("*.pt"))

    checkpoints = [
        path for path in checkpoints
        if not path.name.endswith(":Zone.Identifier")
    ]

    if not checkpoints:
        raise FileNotFoundError(f"No .pt checkpoints found in {args.models_dir}")

    print(f"Found {len(checkpoints)} checkpoint(s):")
    for path in checkpoints:
        print(f"  - {path}")

    for checkpoint_path in checkpoints:
        export_one(
            checkpoint_path=checkpoint_path,
            metadata=metadata,
            args=args,
        )

    print()
    print("=" * 80)
    print("ONNX export complete")
    print("=" * 80)


if __name__ == "__main__":
    main()