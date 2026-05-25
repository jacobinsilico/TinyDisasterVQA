#!/usr/bin/env python3
"""
Smoke test script to verify question-aware model wiring, dataset outputs, and forward passes.
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
from src.text import QuestionVocab, build_question_vocab
from src.models.vqa_models import QuestionVQAModel, compute_type_aware_loss


def build_vocab_if_missing(train_manifest: Path, vocab_path: Path) -> QuestionVocab:
    if vocab_path.exists():
        print(f"Loading existing vocab from {vocab_path}")
        return QuestionVocab.load(vocab_path)

    print(f"Vocab not found. Building vocab from {train_manifest}...")
    samples = []
    with train_manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    questions = [s["question"] for s in samples]
    vocab = build_question_vocab(questions, max_vocab_size=2000, min_freq=1, max_length=24)
    vocab.save(vocab_path)
    print(f"Built and saved vocabulary of size {vocab.size} to {vocab_path}")
    return vocab


def main() -> None:
    print("--- Question-Aware Model Wiring Smoke Test ---")

    # Paths
    processed_dir = REPO_ROOT / "data" / "processed"
    train_manifest = processed_dir / "cocoqa_train_resolved.jsonl"
    vocab_path = processed_dir / "question_vocab.json"
    answer_vocab_path = processed_dir / "answer_vocab.json"

    # Pre-flight checks
    if not train_manifest.exists():
        print(f"ERROR: Train manifest not found at {train_manifest}.")
        print("Please ensure your dataset manifests are generated and located in data/processed.")
        sys.exit(1)

    if not answer_vocab_path.exists():
        print(f"ERROR: Answer vocab not found at {answer_vocab_path}.")
        sys.exit(1)

    # 1. Build or load question vocab
    vocab = build_vocab_if_missing(train_manifest, vocab_path)

    # 2. Instantiate datasets
    print("\n--- Instantiating Datasets ---")
    student_dataset = CocoQADataset(
        manifest_path=train_manifest,
        question_vocab_path=vocab_path,
        answer_vocab_path=answer_vocab_path,
        image_size=128,
        train=False,
        repo_root=REPO_ROOT,
        limit=8,  # Only load a few items for smoke test
    )

    teacher_dataset = CocoQADataset(
        manifest_path=train_manifest,
        question_vocab_path=vocab_path,
        answer_vocab_path=answer_vocab_path,
        image_size=224,
        train=False,
        repo_root=REPO_ROOT,
        limit=8,
    )

    print(f"Student dataset sample count: {len(student_dataset)}")
    print(f"Teacher dataset sample count: {len(teacher_dataset)}")

    # 3. Create Dataloaders
    student_loader = DataLoader(student_dataset, batch_size=4, shuffle=False)
    teacher_loader = DataLoader(teacher_dataset, batch_size=4, shuffle=False)

    student_batch = next(iter(student_loader))
    teacher_batch = next(iter(teacher_loader))

    # 4. Print one batch keys & examples
    print("\n--- Batch Exploration ---")
    print(f"Batch keys: {list(student_batch.keys())}")
    print(f"Student image tensor shape: {student_batch['image'].shape}")
    print(f"Teacher image tensor shape: {teacher_batch['image'].shape}")
    print(f"Question IDs shape:          {student_batch['question_ids'].shape}")
    print(f"Question Len shape:          {student_batch['question_len'].shape}")
    print(f"Object Answer ID shape:      {student_batch['object_answer_id'].shape}")
    print(f"Color Answer ID shape:       {student_batch['color_answer_id'].shape}")

    # Log examples
    print("\nExamples in batch:")
    for i in range(min(3, len(student_dataset))):
        print(f"  [{i}] Q: {student_batch['question'][i]} | Type: {student_batch['type'][i]}")
        print(f"      A: {student_batch['answer'][i]} | Global ID: {student_batch['answer_id'][i].item()}")
        print(f"      Obj Head ID: {student_batch['object_answer_id'][i].item()} | Color Head ID: {student_batch['color_answer_id'][i].item()}")

    # 5. Create Models
    print("\n--- Instantiating Models ---")
    student_model = QuestionVQAModel(
        vocab_size=vocab.size,
        num_object_classes=40,
        num_color_classes=10,
        image_encoder_name="gapcnn_s",
        image_feature_dim=160,
        question_embedding_dim=64,
        question_feature_dim=128,
        pad_id=vocab.pad_id,
        hidden_dim=192,
        dropout=0.1,
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
        hidden_dim=256,
        dropout=0.1,
        pretrained=False,  # Don't download weights during smoke test
    )

    print("Student model built successfully.")
    print("Teacher model built successfully.")

    # 6. Run Forward Pass through Student Model
    print("\n--- Running Student Forward Pass ---")
    student_model.eval()
    with torch.no_grad():
        student_obj_logits, student_color_logits = student_model(
            student_batch["image"],
            student_batch["question_ids"],
            student_batch["question_len"],
        )
    print(f"Student Object Logits shape: {student_obj_logits.shape}")
    print(f"Student Color Logits shape:  {student_color_logits.shape}")

    # Assert student outputs
    assert student_batch["image"].shape == (4, 3, 128, 128), f"Student image shape mismatch: {student_batch['image'].shape}"
    assert student_obj_logits.shape == (4, 40), f"Student object logits shape mismatch: {student_obj_logits.shape}"
    assert student_color_logits.shape == (4, 10), f"Student color logits shape mismatch: {student_color_logits.shape}"

    # 7. Run Forward Pass through Teacher Model
    print("\n--- Running Teacher Forward Pass ---")
    teacher_model.eval()
    with torch.no_grad():
        teacher_obj_logits, teacher_color_logits = teacher_model(
            teacher_batch["image"],
            teacher_batch["question_ids"],
            teacher_batch["question_len"],
        )
    print(f"Teacher Object Logits shape: {teacher_obj_logits.shape}")
    print(f"Teacher Color Logits shape:  {teacher_color_logits.shape}")

    # Assert teacher outputs
    assert teacher_batch["image"].shape == (4, 3, 224, 224), f"Teacher image shape mismatch: {teacher_batch['image'].shape}"
    assert teacher_obj_logits.shape == (4, 40), f"Teacher object logits shape mismatch: {teacher_obj_logits.shape}"
    assert teacher_color_logits.shape == (4, 10), f"Teacher color logits shape mismatch: {teacher_color_logits.shape}"

    # 8. Test compute_type_aware_loss
    print("\n--- Testing compute_type_aware_loss ---")
    loss_dict = compute_type_aware_loss(
        student_obj_logits,
        student_color_logits,
        student_batch["object_answer_id"],
        student_batch["color_answer_id"],
    )
    print(f"Type-Aware Loss: {loss_dict['loss'].item():.4f}")
    print(f"Object Acc:     {loss_dict['object_acc'].item():.4f}")
    print(f"Color Acc:      {loss_dict['color_acc'].item():.4f}")
    print(f"Total Acc:      {loss_dict['total_acc'].item():.4f}")

    # 9. Test inference helper
    print("\n--- Testing Inference Helper ---")
    inferred_preds = student_model.inference(
        student_batch["image"],
        student_batch["question"],
        student_batch["question_ids"],
        student_batch["question_len"],
    )
    print(f"Inference predictions tensor: {inferred_preds}")
    assert inferred_preds.shape == (4,), f"Inference predictions shape mismatch: {inferred_preds.shape}"

    print("\n[SUCCESS] Smoke test completed successfully. All shape and semantic assertions passed!")


if __name__ == "__main__":
    main()
