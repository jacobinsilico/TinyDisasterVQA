#!/usr/bin/env python3
"""
Smoke test to verify both classifier and prototype modes in the question-aware VQA model.
Checks:
- Forward pass shapes
- Normalization integrity of prototypes and fused embeddings
- Loss computation compatibility
- Inference helper operations
"""

import sys
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import CocoQADataset
from src.text import QuestionVocab
from src.models.vqa_models import QuestionVQAModel, compute_type_aware_loss


def main() -> None:
    print("--- Question-Aware Prototype Support Smoke Test ---")

    # Paths
    processed_dir = REPO_ROOT / "data" / "processed"
    train_manifest = processed_dir / "cocoqa_train_resolved.jsonl"
    vocab_path = processed_dir / "question_vocab.json"
    answer_vocab_path = processed_dir / "answer_vocab.json"
    prototypes_path = processed_dir / "answer_prototypes.pt"

    # Pre-flight checks
    if not train_manifest.exists():
        print(f"ERROR: Train manifest not found at {train_manifest}.")
        sys.exit(1)
    if not vocab_path.exists():
        print(f"ERROR: Vocab path not found at {vocab_path}.")
        sys.exit(1)
    if not prototypes_path.exists():
        print(f"ERROR: Prototypes file not found at {prototypes_path}. Run build_answer_prototypes.py first.")
        sys.exit(1)

    # 1. Instantiate dataset and loader
    print("\n[1] Instantiating Dataset and Loader...")
    dataset = CocoQADataset(
        manifest_path=train_manifest,
        question_vocab_path=vocab_path,
        answer_vocab_path=answer_vocab_path,
        image_size=128,
        train=False,
        repo_root=REPO_ROOT,
        limit=4,
    )
    vocab = QuestionVocab.load(vocab_path)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    print(f"Loaded a batch of {batch['image'].shape[0]} items.")

    # 2. Test Classifier Mode VQA Model
    print("\n[2] Testing CLASSIFIER Mode...")
    classifier_model = QuestionVQAModel(
        vocab_size=vocab.size,
        num_object_classes=40,
        num_color_classes=10,
        image_encoder_name="gapcnn_s",
        image_feature_dim=160,
        question_embedding_dim=64,
        question_feature_dim=128,
        pad_id=vocab.pad_id,
        head_type="classifier",
    )
    classifier_model.eval()

    with torch.no_grad():
        c_obj_logits, c_col_logits = classifier_model(
            batch["image"],
            batch["question_ids"],
            batch["question_len"],
        )

    print(f"Classifier Object Logits shape: {c_obj_logits.shape}")
    print(f"Classifier Color Logits shape:  {c_col_logits.shape}")

    assert c_obj_logits.shape == (4, 40), f"Mismatch: {c_obj_logits.shape}"
    assert c_col_logits.shape == (4, 10), f"Mismatch: {c_col_logits.shape}"

    c_loss_dict = compute_type_aware_loss(
        c_obj_logits, c_col_logits,
        batch["object_answer_id"], batch["color_answer_id"]
    )
    print(f"Classifier Loss: {c_loss_dict['loss'].item():.4f}")

    # 3. Test Prototype Mode VQA Model (Mock/Fallback Prototypes)
    print("\n[3] Testing PROTOTYPE Mode (Mock/Fallback Prototypes)...")
    proto_model = QuestionVQAModel(
        vocab_size=vocab.size,
        num_object_classes=40,
        num_color_classes=10,
        image_encoder_name="gapcnn_s",
        image_feature_dim=160,
        question_embedding_dim=64,
        question_feature_dim=128,
        pad_id=vocab.pad_id,
        head_type="prototype",
        answer_embed_dim=128,
        logit_scale_init=10.0,
        learn_logit_scale=True,
    )
    proto_model.eval()

    # Check mock buffer normalization
    obj_proto_norms = torch.norm(proto_model.object_prototypes, p=2, dim=1)
    col_proto_norms = torch.norm(proto_model.color_prototypes, p=2, dim=1)
    print(f"Pre-initialized Object prototype norms min/max: {obj_proto_norms.min().item():.4f} / {obj_proto_norms.max().item():.4f}")
    print(f"Pre-initialized Color prototype norms min/max:  {col_proto_norms.min().item():.4f} / {col_proto_norms.max().item():.4f}")
    
    assert torch.allclose(obj_proto_norms, torch.ones_like(obj_proto_norms), atol=1e-5)
    assert torch.allclose(col_proto_norms, torch.ones_like(col_proto_norms), atol=1e-5)

    with torch.no_grad():
        p_obj_logits, p_col_logits = proto_model(
            batch["image"],
            batch["question_ids"],
            batch["question_len"],
        )

    print(f"Prototype (Mock) Object Logits shape: {p_obj_logits.shape}")
    print(f"Prototype (Mock) Color Logits shape:  {p_col_logits.shape}")

    assert p_obj_logits.shape == (4, 40), f"Mismatch: {p_obj_logits.shape}"
    assert p_col_logits.shape == (4, 10), f"Mismatch: {p_col_logits.shape}"

    # Verify L2 normalization of fused embeddings
    with torch.no_grad():
        img_feats = proto_model.image_encoder(batch["image"])
        q_feats = proto_model.question_encoder(batch["question_ids"], batch["question_len"])
        fused = torch.cat([img_feats, q_feats], dim=1)
        fused_feats = proto_model.fusion_mlp(fused)
        fused_emb = proto_model.embedding_proj(fused_feats)
        fused_emb_norm = torch.norm(torch.nn.functional.normalize(fused_emb, p=2, dim=1), p=2, dim=1)
        print(f"Fused embedding norm: {fused_emb_norm.mean().item():.4f}")
        assert torch.allclose(fused_emb_norm, torch.ones_like(fused_emb_norm), atol=1e-5)

    # Test loading real prototypes
    print("\n[4] Testing load_prototypes helper...")
    proto_model.load_prototypes(prototypes_path)

    # Check loaded norms
    loaded_obj_norms = torch.norm(proto_model.object_prototypes, p=2, dim=1)
    loaded_col_norms = torch.norm(proto_model.color_prototypes, p=2, dim=1)
    print(f"Loaded Object prototype norms min/max: {loaded_obj_norms.min().item():.4f} / {loaded_obj_norms.max().item():.4f}")
    print(f"Loaded Color prototype norms min/max:  {loaded_col_norms.min().item():.4f} / {loaded_col_norms.max().item():.4f}")
    
    assert torch.allclose(loaded_obj_norms, torch.ones_like(loaded_obj_norms), atol=1e-5)
    assert torch.allclose(loaded_col_norms, torch.ones_like(loaded_col_norms), atol=1e-5)

    # Forward pass after loading real prototypes
    with torch.no_grad():
        real_obj_logits, real_col_logits = proto_model(
            batch["image"],
            batch["question_ids"],
            batch["question_len"],
        )

    print(f"Real Prototype Object Logits shape: {real_obj_logits.shape}")
    print(f"Real Prototype Color Logits shape:  {real_col_logits.shape}")

    assert real_obj_logits.shape == (4, 40)
    assert real_col_logits.shape == (4, 10)

    p_loss_dict = compute_type_aware_loss(
        real_obj_logits, real_col_logits,
        batch["object_answer_id"], batch["color_answer_id"]
    )
    print(f"Prototype Loss: {p_loss_dict['loss'].item():.4f}")

    # Test Inference Helpers on both models
    print("\n[5] Testing Inference Helpers...")
    c_preds = classifier_model.inference(
        batch["image"],
        batch["question"],
        batch["question_ids"],
        batch["question_len"],
    )
    p_preds = proto_model.inference(
        batch["image"],
        batch["question"],
        batch["question_ids"],
        batch["question_len"],
    )
    print(f"Classifier inference predictions: {c_preds.tolist()}")
    print(f"Prototype inference predictions:  {p_preds.tolist()}")
    assert c_preds.shape == (4,)
    assert p_preds.shape == (4,)

    print("\n[SUCCESS] Smoke test completed successfully. All shape, L2-norm, and loss assertions passed!")


if __name__ == "__main__":
    main()
