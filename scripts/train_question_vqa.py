#!/usr/bin/env python3
"""
Baseline training and overfitting script for the question-aware VQA model (QuestionVQAModel).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import CocoQADataset
from src.text import QuestionVocab
from src.models.vqa_models import QuestionVQAModel, compute_type_aware_loss
from src.utils import set_seed, count_parameters


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    num_object_samples = 0
    num_color_samples = 0
    object_correct = 0
    color_correct = 0

    start_time = time.time()

    for batch in loader:
        images = batch["image"].to(device)
        question_ids = batch["question_ids"].to(device)
        question_len = batch["question_len"].to(device)
        object_answer_id = batch["object_answer_id"].to(device)
        color_answer_id = batch["color_answer_id"].to(device)

        batch_size = images.shape[0]

        optimizer.zero_grad()
        object_logits, color_logits = model(images, question_ids, question_len)

        loss_dict = compute_type_aware_loss(
            object_logits=object_logits,
            color_logits=color_logits,
            object_answer_id=object_answer_id,
            color_answer_id=color_answer_id,
        )

        loss = loss_dict["loss"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Accumulate metrics
        obj_mask = (object_answer_id != -1)
        col_mask = (color_answer_id != -1)

        num_obj = obj_mask.sum().item()
        num_col = col_mask.sum().item()

        num_object_samples += num_obj
        num_color_samples += num_col

        if num_obj > 0:
            obj_preds = object_logits[obj_mask].argmax(dim=-1)
            obj_corr = (obj_preds == object_answer_id[obj_mask]).sum().item()
            object_correct += obj_corr
            total_correct += obj_corr

        if num_col > 0:
            col_preds = color_logits[col_mask].argmax(dim=-1)
            col_corr = (col_preds == color_answer_id[col_mask]).sum().item()
            color_correct += col_corr
            total_correct += col_corr

    elapsed = time.time() - start_time
    avg_loss = total_loss / max(total_samples, 1)
    total_acc = total_correct / max(total_samples, 1)
    object_acc = object_correct / max(num_object_samples, 1)
    color_acc = color_correct / max(num_color_samples, 1)

    return {
        "loss": avg_loss,
        "accuracy": total_acc,
        "object_acc": object_acc,
        "color_acc": color_acc,
        "time_sec": elapsed,
        "num_object": float(num_object_samples),
        "num_color": float(num_color_samples),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    num_object_samples = 0
    num_color_samples = 0
    object_correct = 0
    color_correct = 0

    start_time = time.time()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            question_ids = batch["question_ids"].to(device)
            question_len = batch["question_len"].to(device)
            object_answer_id = batch["object_answer_id"].to(device)
            color_answer_id = batch["color_answer_id"].to(device)

            batch_size = images.shape[0]

            object_logits, color_logits = model(images, question_ids, question_len)

            loss_dict = compute_type_aware_loss(
                object_logits=object_logits,
                color_logits=color_logits,
                object_answer_id=object_answer_id,
                color_answer_id=color_answer_id,
            )

            loss = loss_dict["loss"]

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Accumulate metrics
            obj_mask = (object_answer_id != -1)
            col_mask = (color_answer_id != -1)

            num_obj = obj_mask.sum().item()
            num_col = col_mask.sum().item()

            num_object_samples += num_obj
            num_color_samples += num_col

            if num_obj > 0:
                obj_preds = object_logits[obj_mask].argmax(dim=-1)
                obj_corr = (obj_preds == object_answer_id[obj_mask]).sum().item()
                object_correct += obj_corr
                total_correct += obj_corr

            if num_col > 0:
                col_preds = color_logits[col_mask].argmax(dim=-1)
                col_corr = (col_preds == color_answer_id[col_mask]).sum().item()
                color_correct += col_corr
                total_correct += col_corr

    elapsed = time.time() - start_time
    avg_loss = total_loss / max(total_samples, 1)
    total_acc = total_correct / max(total_samples, 1)
    object_acc = object_correct / max(num_object_samples, 1)
    color_acc = color_correct / max(num_color_samples, 1)

    return {
        "loss": avg_loss,
        "accuracy": total_acc,
        "object_acc": object_acc,
        "color_acc": color_acc,
        "time_sec": elapsed,
        "num_object": float(num_object_samples),
        "num_color": float(num_color_samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/question_vqa"))
    parser.add_argument("--image-encoder", type=str, default="gapcnn_s", choices=["gapcnn_s", "mobilenet_v3_large"])
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overfit-small", action="store_true", help="Run in a tiny overfit mode with 128 samples.")
    parser.add_argument("--num-workers", type=int, default=2)

    # Prototype heads arguments
    parser.add_argument("--head-type", type=str, default="classifier", choices=["classifier", "prototype"],
                        help="VQA model head classification style.")
    parser.add_argument("--answer-embed-dim", type=int, default=128, help="Size of projected prototype embeddings.")
    parser.add_argument("--learn-logit-scale", action="store_true", help="Make logit scale learnable during training.")
    parser.add_argument("--logit-scale-init", type=float, default=10.0, help="Initial scale value for cosine logits.")
    parser.add_argument("--prototype-path", type=Path, default=Path("data/processed/answer_prototypes.pt"),
                        help="Path to pre-generated answer_prototypes.pt")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("--- Question-Aware Model Training / Overfitting ---")
    print(f"Device: {device}")
    print(f"Image encoder: {args.image_encoder}")
    print(f"Image size: {args.image_size}")
    print(f"Overfit mode: {args.overfit_small}")
    print(f"Head type: {args.head_type}")

    train_manifest = args.processed_dir / "cocoqa_train_resolved.jsonl"
    val_manifest = args.processed_dir / "cocoqa_val_resolved.jsonl"
    vocab_path = args.processed_dir / "question_vocab.json"
    answer_vocab_path = args.processed_dir / "answer_vocab.json"

    # Set parameters for overfit mode
    limit = 128 if args.overfit_small else 0
    epochs = 50 if args.overfit_small else args.epochs
    augment = False if args.overfit_small else True

    # 1. Load Vocabulary
    vocab = QuestionVocab.load(vocab_path)

    # 2. Build Datasets
    train_dataset = CocoQADataset(
        manifest_path=train_manifest,
        question_vocab_path=vocab_path,
        answer_vocab_path=answer_vocab_path,
        image_size=args.image_size,
        train=augment,
        repo_root=REPO_ROOT,
        limit=limit,
    )

    val_dataset = None
    if not args.overfit_small:
        val_dataset = CocoQADataset(
            manifest_path=val_manifest,
            question_vocab_path=vocab_path,
            answer_vocab_path=answer_vocab_path,
            image_size=args.image_size,
            train=False,
            repo_root=REPO_ROOT,
        )

    print(f"Loaded {len(train_dataset)} training samples.")
    if val_dataset:
        print(f"Loaded {len(val_dataset)} validation samples.")

    # 3. Create Dataloaders
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size if not args.overfit_small else len(train_dataset),
        shuffle=True if not args.overfit_small else False,
        num_workers=args.num_workers if not args.overfit_small else 0,
        pin_memory=pin_memory,
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

    # 4. Instantiate Model
    model = QuestionVQAModel(
        vocab_size=vocab.size,
        num_object_classes=40,
        num_color_classes=10,
        image_encoder_name=args.image_encoder,
        image_feature_dim=160 if args.image_encoder == "gapcnn_s" else 256,
        question_embedding_dim=64,
        question_feature_dim=128,
        pad_id=vocab.pad_id,
        hidden_dim=192,
        dropout=0.1 if not args.overfit_small else 0.0,  # Zero dropout for overfitting
        head_type=args.head_type,
        answer_embed_dim=args.answer_embed_dim,
        logit_scale_init=args.logit_scale_init,
        learn_logit_scale=args.learn_logit_scale,
    ).to(device)

    # Load offline pre-projected prototypes if in prototype mode
    if args.head_type == "prototype":
        if args.prototype_path.exists():
            model.load_prototypes(args.prototype_path)
        else:
            print(f"[WARNING] Prototypes file not found at {args.prototype_path}. Model will fall back to randomized unit vectors.")

    print(f"Model parameters: {count_parameters(model):,}")

    # 5. Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr if not args.overfit_small else 2e-3,
        weight_decay=args.weight_decay if not args.overfit_small else 0.0,
    )

    # 6. Training Loop
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    print("\nStarting training loop...")
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        logit_scale_val = ""
        if args.head_type == "prototype":
            logit_scale_val = f" | Scale: {model.logit_scale.item():.4f}"

        if val_loader:
            val_metrics = evaluate(model, val_loader, device)
            print(
                f"Epoch {epoch:02d} | "
                f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}{logit_scale_val}"
            )
            # Save best checkpoint
            if val_metrics["accuracy"] > best_acc:
                best_acc = val_metrics["accuracy"]
                torch.save(model.state_dict(), args.checkpoint_dir / "best_model.pt")
        else:
            # Overfit mode output
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Loss: {train_metrics['loss']:.5f} | "
                f"Acc: {train_metrics['accuracy']:.4f} (Obj: {train_metrics['object_acc']:.4f}, Col: {train_metrics['color_acc']:.4f}){logit_scale_val}"
            )

            # Assert convergence if we are near the end of overfit mode
            if args.overfit_small and epoch == epochs and train_metrics["accuracy"] < 0.90:
                print("WARNING: Model did not achieve high overfit accuracy (>90%).")

    # Save last checkpoint
    torch.save(model.state_dict(), args.checkpoint_dir / "last_model.pt")
    print(f"\nTraining completed. Saved final model checkpoint to {args.checkpoint_dir / 'last_model.pt'}")


if __name__ == "__main__":
    main()
