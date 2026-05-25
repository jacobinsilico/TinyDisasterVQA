#!/usr/bin/env python3
"""
Smoke test to verify Knowledge Distillation (KD) shape matching, loss formulations, and forward loops.
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
from scripts.train_question_vqa import compute_kd_loss


def main() -> None:
    print("--- Question-Aware Knowledge Distillation (KD) Smoke Test ---")

    # Paths
    processed_dir = REPO_ROOT / "data" / "processed"
    train_manifest = processed_dir / "cocoqa_train_resolved.jsonl"
    vocab_path = processed_dir / "question_vocab.json"
    answer_vocab_path = processed_dir / "answer_vocab.json"

    # Pre-flight checks
    if not train_manifest.exists():
        print(f"ERROR: Train manifest not found at {train_manifest}.")
        sys.exit(1)
    if not vocab_path.exists():
        print(f"ERROR: Vocab path not found at {vocab_path}.")
        sys.exit(1)

    # 1. Instantiate dual-size dataset and loader
    print("\n[1] Instantiating Dataset in Distillation Mode...")
    vocab = QuestionVocab.load(vocab_path)
    dataset = CocoQADataset(
        manifest_path=train_manifest,
        question_vocab_path=vocab_path,
        answer_vocab_path=answer_vocab_path,
        image_size=128,
        train=False,
        repo_root=REPO_ROOT,
        limit=4,
        teacher_image_size=224,
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    print("Successfully loaded dual-size image batch.")

    # 2. Check loaded shapes
    print("\n[2] Verifying Loader Shapes...")
    student_imgs = batch["image"]
    teacher_imgs = batch["teacher_image"]
    print(f"  Student image tensor shape: {student_imgs.shape}")
    print(f"  Teacher image tensor shape: {teacher_imgs.shape}")

    assert student_imgs.shape == (4, 3, 128, 128), f"Mismatch student image: {student_imgs.shape}"
    assert teacher_imgs.shape == (4, 3, 224, 224), f"Mismatch teacher image: {teacher_imgs.shape}"

    # 3. Instantiate Student and Teacher Models
    print("\n[3] Instantiating Student and Teacher Models...")
    student_model = QuestionVQAModel(
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
    
    teacher_model = QuestionVQAModel(
        vocab_size=vocab.size,
        num_object_classes=40,
        num_color_classes=10,
        image_encoder_name="mobilenet_v3_large",
        image_feature_dim=256,
        question_embedding_dim=64,
        question_feature_dim=128,
        pad_id=vocab.pad_id,
        head_type="classifier",
        pretrained=False,
    )

    student_model.eval()
    teacher_model.eval()

    # 4. Forward Passes
    print("\n[4] Running Forward Passes...")
    with torch.no_grad():
        student_obj_logits, student_color_logits = student_model(
            student_imgs,
            batch["question_ids"],
            batch["question_len"],
        )
        teacher_obj_logits, teacher_col_logits = teacher_model(
            teacher_imgs,
            batch["question_ids"],
            batch["question_len"],
        )

    print(f"  Student Object Logits shape: {student_obj_logits.shape}")
    print(f"  Student Color Logits shape:  {student_color_logits.shape}")
    print(f"  Teacher Object Logits shape: {teacher_obj_logits.shape}")
    print(f"  Teacher Color Logits shape:  {teacher_col_logits.shape}")

    assert student_obj_logits.shape == (4, 40)
    assert student_color_logits.shape == (4, 10)
    assert teacher_obj_logits.shape == (4, 40)
    assert teacher_col_logits.shape == (4, 10)

    # 5. Compute type-aware hard and soft losses
    print("\n[5] Computing Losses...")
    
    # Supervised hard loss
    hard_loss_dict = compute_type_aware_loss(
        student_obj_logits, student_color_logits,
        batch["object_answer_id"], batch["color_answer_id"]
    )
    hard_loss = hard_loss_dict["loss"]
    print(f"  Type-Aware Hard Loss: {hard_loss.item():.4f}")

    # Soft distillation loss
    kd_temp = 4.0
    kd_alpha = 0.5
    
    obj_mask = (batch["object_answer_id"] != -1)
    col_mask = (batch["color_answer_id"] != -1)

    kd_obj_loss = compute_kd_loss(student_obj_logits, teacher_obj_logits, obj_mask, kd_temp)
    kd_col_loss = compute_kd_loss(student_color_logits, teacher_col_logits, col_mask, kd_temp)

    num_obj = obj_mask.sum().item()
    num_col = col_mask.sum().item()
    total_valid = num_obj + num_col

    if total_valid > 0:
        kd_loss = (kd_obj_loss * num_obj + kd_col_loss * num_col) / total_valid
    else:
        kd_loss = torch.tensor(0.0)

    print(f"  Type-Aware Soft KD Loss: {kd_loss.item():.4f}")

    # Total loss
    total_loss = (1.0 - kd_alpha) * hard_loss + kd_alpha * kd_loss
    print(f"  Type-Aware Combined Distillation Loss: {total_loss.item():.4f}")

    # Final asserts
    assert torch.isfinite(hard_loss), "Hard loss is not finite!"
    assert torch.isfinite(kd_loss), "KD loss is not finite!"
    assert torch.isfinite(total_loss), "Combined loss is not finite!"

    print("\n[SUCCESS] Knowledge distillation smoke test passed successfully!")


if __name__ == "__main__":
    main()
