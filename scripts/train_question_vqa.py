#!/usr/bin/env python3
"""
Baseline training and overfitting script for the question-aware VQA model (QuestionVQAModel).
Colab-Ready and fully compatible with mixed-precision (AMP) CUDA acceleration and teacher-student Knowledge Distillation.

================================================================================
COLAB SANITY COMMAND LIST (WORKSPACE GUIDE)
================================================================================
1. Install requirements:
   !pip install torch torchvision pillow sentence-transformers matplotlib

2. Run Manifest-Driven Image Downloader (Download whitelisted subset only):
   !python scripts/04_download_images.py --manifest-dir data/processed --image-root data/images

3. Build Answer Embedding Prototypes (SentenceTransformers or deterministic fallback):
   !python scripts/build_answer_prototypes.py

4. Run Shape & Norm Verification Smoke Tests:
   !python scripts/smoke_test_prototype_model.py

5. Train Classifier Student (gapcnn_s, 128x128):
   !python scripts/train_question_vqa.py --head-type classifier --image-encoder gapcnn_s --image-size 128 --epochs 20 --batch-size 64 --lr 1e-3 --run-name classifier_student --device cuda --amp

6. Train Classifier Teacher (mobilenet_v3_large, 224x224):
   !python scripts/train_question_vqa.py --head-type classifier --image-encoder mobilenet_v3_large --image-size 224 --epochs 20 --batch-size 64 --lr 3e-4 --run-name classifier_teacher --device cuda --amp

7. Train Prototype Student (gapcnn_s, 128x128, learnable logit scale):
   !python scripts/train_question_vqa.py --head-type prototype --image-encoder gapcnn_s --image-size 128 --epochs 20 --batch-size 64 --lr 1e-3 --run-name prototype_student --learn-logit-scale --device cuda --amp
================================================================================
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import CocoQADataset
from src.text import QuestionVocab
from src.models.vqa_models import QuestionVQAModel, compute_type_aware_loss
from src.utils import set_seed, count_parameters


def compute_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 4.0,
) -> torch.Tensor:
    """
    Computes KL divergence loss between softened student and teacher logits.
    """
    if mask.sum().item() == 0:
        return torch.tensor(0.0, device=student_logits.device)
        
    s_masked = student_logits[mask] / temperature
    t_masked = teacher_logits[mask] / temperature
    
    kl_loss_fn = nn.KLDivLoss(reduction="sum")
    
    log_probs = F.log_softmax(s_masked, dim=-1)
    targets = F.softmax(t_masked, dim=-1)
    
    loss = kl_loss_fn(log_probs, targets) * (temperature ** 2)
    return loss / s_masked.shape[0]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: torch.cuda.amp.GradScaler | None = None,
    use_amp: bool = False,
    teacher: nn.Module | None = None,
    kd_alpha: float = 0.5,
    kd_temperature: float = 4.0,
) -> dict[str, float]:
    model.train()
    if teacher is not None:
        teacher.eval()

    total_loss = 0.0
    total_hard_loss = 0.0
    total_kd_loss = 0.0
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

        # 1. Run Teacher Forward Pass (if distillation enabled)
        teacher_obj_logits = None
        teacher_col_logits = None
        if teacher is not None:
            teacher_images = batch["teacher_image"].to(device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_amp):
                    teacher_obj_logits, teacher_col_logits = teacher(
                        teacher_images,
                        question_ids,
                        question_len,
                    )

        optimizer.zero_grad()
        
        # 2. Run Student Forward Pass (mixed precision context)
        with torch.cuda.amp.autocast(enabled=use_amp):
            object_logits, color_logits = model(images, question_ids, question_len)
            
            # Hard supervised CrossEntropy loss
            loss_dict = compute_type_aware_loss(
                object_logits=object_logits,
                color_logits=color_logits,
                object_answer_id=object_answer_id,
                color_answer_id=color_answer_id,
            )
            hard_loss = loss_dict["loss"]

            # Compute softened KD loss
            kd_loss = torch.tensor(0.0, device=device)
            if teacher is not None:
                obj_mask = (object_answer_id != -1)
                col_mask = (color_answer_id != -1)

                kd_obj_loss = compute_kd_loss(object_logits, teacher_obj_logits, obj_mask, kd_temperature)
                kd_col_loss = compute_kd_loss(color_logits, teacher_col_logits, col_mask, kd_temperature)

                num_obj = obj_mask.sum().item()
                num_col = col_mask.sum().item()
                total_valid = num_obj + num_col

                if total_valid > 0:
                    kd_loss = (kd_obj_loss * num_obj + kd_col_loss * num_col) / total_valid

                loss = (1.0 - kd_alpha) * hard_loss + kd_alpha * kd_loss
            else:
                loss = hard_loss

        # Backpropagation
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * batch_size
        total_hard_loss += hard_loss.item() * batch_size
        total_kd_loss += kd_loss.item() * batch_size
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
    avg_hard_loss = total_hard_loss / max(total_samples, 1)
    avg_kd_loss = total_kd_loss / max(total_samples, 1)
    total_acc = total_correct / max(total_samples, 1)
    object_acc = object_correct / max(num_object_samples, 1)
    color_acc = color_correct / max(num_color_samples, 1)

    return {
        "loss": avg_loss,
        "hard_loss": avg_hard_loss,
        "kd_loss": avg_kd_loss,
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
    use_amp: bool = False,
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

            with torch.cuda.amp.autocast(enabled=use_amp):
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
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--run-name", type=str, default="question_vqa_run")
    parser.add_argument("--image-encoder", type=str, default="gapcnn_s", choices=["gapcnn_s", "mobilenet_v3_large"])
    parser.add_argument("--image-size", type=int, default=128, choices=[128, 224])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overfit-small", action="store_true", help="Run in a tiny overfit mode with 128 samples.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--amp", action="store_true", help="Use mixed precision training (autocast) if running on CUDA.")

    # Prototype heads arguments
    parser.add_argument("--head-type", type=str, default="classifier", choices=["classifier", "prototype"])
    parser.add_argument("--answer-embed-dim", type=int, default=128)
    parser.add_argument("--learn-logit-scale", action="store_true")
    parser.add_argument("--logit-scale-init", type=float, default=10.0)
    parser.add_argument("--prototype-path", type=Path, default=Path("data/processed/answer_prototypes.pt"))

    # Knowledge Distillation arguments
    parser.add_argument("--distill", action="store_true", help="Enable teacher-student knowledge distillation.")
    parser.add_argument("--teacher-checkpoint", type=Path, default=None, help="Path to trained teacher checkpoint.")
    parser.add_argument("--teacher-image-encoder", type=str, default="mobilenet_v3_large", help="Teacher image encoder name.")
    parser.add_argument("--teacher-image-size", type=int, default=224, help="Teacher input resolution.")
    parser.add_argument("--teacher-head-type", type=str, default=None, help="Teacher head style (classifier or prototype). Defaults to matches student's.")
    parser.add_argument("--kd-alpha", type=float, default=0.5, help="Distillation soft loss balance weight (0.0 to 1.0).")
    parser.add_argument("--kd-temperature", type=float, default=4.0, help="Logits softening temperature.")

    # Early stopping arguments
    parser.add_argument(
        "--early-stopping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable validation-based early stopping. Use --no-early-stopping to disable.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=6,
        help="Stop after this many validation epochs without improvement.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum validation accuracy improvement required to reset early stopping patience.",
    )

    args = parser.parse_args()

    set_seed(args.seed)
    
    # Device configuration
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    use_amp = args.amp and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    print("--- Question-Aware Model Training / Overfitting ---")
    print(f"Device: {device}")
    print(f"AMP/Mixed Precision: {use_amp}")
    print(f"Image encoder: {args.image_encoder}")
    print(f"Image size: {args.image_size}")
    print(f"Overfit mode: {args.overfit_small}")
    print(f"Head type: {args.head_type}")
    print(f"Run directory: {args.output_dir / args.run_name}")
    print(f"Early stopping: {args.early_stopping} (patience={args.patience}, min_delta={args.min_delta})")

    if args.distill:
        print("\n--- Distillation Configuration ---")
        print(f"  Teacher checkpoint: {args.teacher_checkpoint}")
        print(f"  Teacher encoder:    {args.teacher_image_encoder}")
        print(f"  Teacher resolution: {args.teacher_image_size}")
        print(f"  KD alpha:           {args.kd_alpha}")
        print(f"  KD temperature:     {args.kd_temperature}")

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
        teacher_image_size=args.teacher_image_size if args.distill else None,
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

    # 4.5 Instantiate and Load Teacher Model (if distillation enabled)
    teacher_model = None
    if args.distill:
        if not args.teacher_checkpoint:
            raise ValueError("--teacher-checkpoint is required when distillation is enabled.")
        
        teacher_head_type = args.teacher_head_type if args.teacher_head_type is not None else args.head_type
        if teacher_head_type != args.head_type:
            raise ValueError(
                f"Mismatched head configurations: teacher is '{teacher_head_type}' but student is '{args.head_type}'. "
                f"Distillation between different classification paradigms is not supported in this version."
            )

        print(f"Loading teacher model: encoder={args.teacher_image_encoder}, head_type={teacher_head_type}...")
        teacher_model = QuestionVQAModel(
            vocab_size=vocab.size,
            num_object_classes=40,
            num_color_classes=10,
            image_encoder_name=args.teacher_image_encoder,
            image_feature_dim=160 if args.teacher_image_encoder == "gapcnn_s" else 256,
            question_embedding_dim=64,
            question_feature_dim=128,
            pad_id=vocab.pad_id,
            hidden_dim=192,
            dropout=0.0,
            head_type=teacher_head_type,
            answer_embed_dim=args.answer_embed_dim,
        )

        state_dict = torch.load(args.teacher_checkpoint, map_location="cpu")
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        teacher_model.load_state_dict(state_dict)

        if teacher_head_type == "prototype":
            if args.prototype_path.exists():
                teacher_model.load_prototypes(args.prototype_path)
            else:
                raise FileNotFoundError(f"Prototypes file required for prototype teacher at {args.prototype_path}")

        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        print("Teacher model loaded and frozen successfully.")

    # 5. Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr if not args.overfit_small else 2e-3,
        weight_decay=args.weight_decay if not args.overfit_small else 0.0,
    )

    # 6. Training Loop
    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = run_dir / "best_model.pt"
    last_ckpt_path = run_dir / "last_model.pt"
    metrics_json_path = run_dir / "metrics.json"

    best_val_acc = 0.0
    epochs_without_improvement = 0
    history = []

    print("\nStarting training loop...")
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler,
            use_amp=use_amp,
            teacher=teacher_model,
            kd_alpha=args.kd_alpha,
            kd_temperature=args.kd_temperature,
        )

        logit_scale_val = ""
        if args.head_type == "prototype":
            logit_scale_val = f" | Scale: {model.logit_scale.item():.4f}"

        # Evaluation
        val_metrics = None
        is_best = False
        if val_loader:
            val_metrics = evaluate(model, val_loader, device, use_amp=use_amp)
            val_acc = val_metrics["accuracy"]

            # Save best checkpoint and update early stopping state.
            if val_acc > best_val_acc + args.min_delta:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                is_best = True
                torch.save(model.state_dict(), best_ckpt_path)
            else:
                epochs_without_improvement += 1

        # Log history
        metrics_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_hard_loss": train_metrics["hard_loss"],
            "train_kd_loss": train_metrics["kd_loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_object_acc": train_metrics["object_acc"],
            "train_color_acc": train_metrics["color_acc"],
        }
        if val_metrics:
            metrics_row.update({
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_object_acc": val_metrics["object_acc"],
                "val_color_acc": val_metrics["color_acc"],
                "best_val_accuracy": best_val_acc,
                "is_best": is_best,
                "epochs_without_improvement": epochs_without_improvement,
                "early_stopping_patience": args.patience,
                "early_stopping_min_delta": args.min_delta,
            })
        if args.distill:
            metrics_row.update({
                "kd_alpha": args.kd_alpha,
                "kd_temperature": args.kd_temperature,
                "teacher_checkpoint": str(args.teacher_checkpoint),
            })
        history.append(metrics_row)
        
        # Save metrics.json every epoch
        with metrics_json_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        # Print Epoch Report
        best_tag = " BEST" if is_best else ""
        print(f"\n--- Epoch {epoch:02d}/{epochs:02d}{best_tag} ---")
        if args.distill:
            print(f"  Train Total Loss: {train_metrics['loss']:.4f} | Hard Loss: {train_metrics['hard_loss']:.4f} | KD Loss: {train_metrics['kd_loss']:.4f}")
        else:
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Train Acc: {train_metrics['accuracy']:.4f} (Obj: {train_metrics['object_acc']:.4f}, Col: {train_metrics['color_acc']:.4f})")
        if val_metrics:
            print(f"  Val   Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} (Obj: {val_metrics['object_acc']:.4f}, Col: {val_metrics['color_acc']:.4f})")
            print(f"  Best Val Acc: {best_val_acc:.4f}{logit_scale_val}")
            if args.early_stopping:
                print(f"  Early stopping counter: {epochs_without_improvement}/{args.patience}")
                if epochs_without_improvement >= args.patience:
                    print(
                        f"\nEarly stopping triggered after epoch {epoch}. "
                        f"Best Val Acc: {best_val_acc:.4f}"
                    )
                    break
        else:
            print(f"  Overfit Mode{logit_scale_val}")

    # Save last checkpoint
    torch.save(model.state_dict(), last_ckpt_path)
    print(f"\nTraining completed. Saved final model checkpoints and metrics history to {run_dir}")


if __name__ == "__main__":
    main()
