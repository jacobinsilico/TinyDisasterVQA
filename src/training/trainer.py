import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.vocab import QuestionVocab
from src.evaluation.metrics import AccuracyTracker, AverageMeter
from src.evaluation.confusion import ConfusionTracker
from src.models.vqa_models import build_baseline_vqa_model
from src.utils.logging import format_metrics
from src.utils.model_info import count_parameters
from src.training.helpers import move_batch_to_device, forward_model


def compute_supervised_kd_loss(
    student_logits: torch.Tensor,
    targets: torch.Tensor,
    ce_criterion: nn.Module,
    teacher_logits: torch.Tensor | None = None,
    kd_alpha: float = 0.0,
    kd_temperature: float = 3.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute combined Supervised Cross-Entropy and Knowledge Distillation loss.
    """
    ce_loss = ce_criterion(student_logits, targets)

    if teacher_logits is None or kd_alpha <= 0.0:
        return ce_loss, {
            "loss_ce": float(ce_loss.detach().item()),
            "loss_kd": 0.0,
            "loss_total": float(ce_loss.detach().item()),
        }

    if kd_temperature <= 0.0:
        raise ValueError("kd_temperature must be > 0.")

    temperature = float(kd_temperature)

    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)

    kd_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    ) * (temperature * temperature)

    total_loss = (1.0 - kd_alpha) * ce_loss + kd_alpha * kd_loss

    return total_loss, {
        "loss_ce": float(ce_loss.detach().item()),
        "loss_kd": float(kd_loss.detach().item()),
        "loss_total": float(total_loss.detach().item()),
    }


def train_one_epoch(
    model: nn.Module,
    teacher: nn.Module | None,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ce_criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_interval: int,
    use_amp: bool,
    kd_alpha: float = 0.0,
    kd_temperature: float = 3.0,
) -> dict:
    """
    Train model for one epoch. Supports baseline, teacher, and student KD modes.
    """
    model.train()
    if teacher is not None:
        teacher.eval()

    loss_meter = AverageMeter()
    ce_meter = AverageMeter()
    kd_meter = AverageMeter()
    acc_tracker = AccuracyTracker()

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_time = time.time()

    for step, batch in enumerate(loader, start=1):
        batch_dev = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        # 1. Forward teacher if present
        with torch.no_grad():
            teacher_logits = None
            if teacher is not None:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    teacher_logits = forward_model(teacher, batch_dev)

        # 2. Forward student/model and calculate loss
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = forward_model(model, batch_dev)
            loss, loss_parts = compute_supervised_kd_loss(
                student_logits=logits,
                targets=batch_dev["answer_id"],
                ce_criterion=ce_criterion,
                teacher_logits=teacher_logits,
                kd_alpha=kd_alpha,
                kd_temperature=kd_temperature,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = batch_dev["images"].shape[0]
        loss_meter.update(loss_parts["loss_total"], n=batch_size)
        ce_meter.update(loss_parts["loss_ce"], n=batch_size)
        kd_meter.update(loss_parts["loss_kd"], n=batch_size)

        acc_tracker.update(logits.detach(), batch_dev["answer_id"], batch_dev["type_id"])

        if log_interval > 0 and step % log_interval == 0:
            partial_metrics = acc_tracker.compute()
            partial_metrics["loss"] = loss_meter.avg
            elapsed = time.time() - start_time

            extra = ""
            if teacher is not None:
                extra = f" | ce={ce_meter.avg:.4f} | kd={kd_meter.avg:.4f}"

            print(
                f"Epoch {epoch:03d} | step {step:05d}/{len(loader):05d} | "
                f"{format_metrics(partial_metrics, prefix='train')}"
                f"{extra} | time={elapsed:.1f}s"
            )

    metrics = acc_tracker.compute()
    metrics["loss"] = loss_meter.avg
    metrics["loss_ce"] = ce_meter.avg
    metrics["loss_kd"] = kd_meter.avg
    metrics["time_sec"] = time.time() - start_time

    return metrics


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    ce_criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    id_to_answer: dict[int, str] | None = None,
) -> tuple[dict, list[tuple[str, str, int]]]:
    """
    Evaluate model over validation/test data.
    """
    model.eval()

    loss_meter = AverageMeter()
    acc_tracker = AccuracyTracker()
    confusion_tracker = ConfusionTracker(id_to_answer) if id_to_answer is not None else None

    start_time = time.time()

    for batch in loader:
        batch_dev = move_batch_to_device(batch, device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = forward_model(model, batch_dev)
            loss = ce_criterion(logits, batch_dev["answer_id"])

        batch_size = batch_dev["images"].shape[0]
        loss_meter.update(loss.item(), n=batch_size)
        acc_tracker.update(logits, batch_dev["answer_id"], batch_dev["type_id"])

        if confusion_tracker is not None:
            confusion_tracker.update(logits, batch_dev["answer_id"])

    metrics = acc_tracker.compute()
    metrics["loss"] = loss_meter.avg
    metrics["time_sec"] = time.time() - start_time

    confusions = confusion_tracker.topk(20) if confusion_tracker is not None else []

    return metrics, confusions


def build_teacher_model(
    teacher_checkpoint_path: str | Path,
    question_vocab: QuestionVocab,
    num_answers: int,
    answer_ids_by_type: dict[str, list[int]],
    device: torch.device,
) -> nn.Module:
    """
    Load a compatible teacher VQA model from checkpoint.
    """
    teacher_checkpoint_path = Path(teacher_checkpoint_path)
    if not teacher_checkpoint_path.exists():
        raise FileNotFoundError(f"Missing teacher checkpoint: {teacher_checkpoint_path}")

    print(f"Loading teacher checkpoint: {teacher_checkpoint_path}")
    ckpt = torch.load(teacher_checkpoint_path, map_location="cpu")

    config = ckpt.get("config", {})
    teacher_model_name = config.get("model_name", "mobilenet_v2")
    teacher_head_type = config.get("head_type", "separate")

    checkpoint_num_answers = config.get("num_answer_classes", num_answers)
    if int(checkpoint_num_answers) != int(num_answers):
        raise ValueError(
            "Teacher checkpoint answer vocab size does not match current dataset.\n"
            f"  teacher num answers: {checkpoint_num_answers}\n"
            f"  current num answers: {num_answers}\n"
            "Train a new teacher first, or implement explicit vocab mapping."
        )

    print("Teacher config:")
    print(f"  model_name: {teacher_model_name}")
    print(f"  head_type:  {teacher_head_type}")

    # Build teacher Baseline VQA model
    teacher = build_baseline_vqa_model(
        vocab_size=question_vocab.size,
        num_answers=num_answers,
        pad_id=question_vocab.pad_id,
        model_name=teacher_model_name,
        pretrained=False,
        freeze_image_encoder=False,
        head_type=teacher_head_type,
        object_answer_ids=answer_ids_by_type["object"],
        color_answer_ids=answer_ids_by_type["color"],
        number_answer_ids=answer_ids_by_type.get("number"),
    )

    missing, unexpected = teacher.load_state_dict(
        ckpt["model_state_dict"],
        strict=False,
    )

    if missing:
        print("WARNING: Missing teacher keys:")
        for key in missing[:20]:
            print(f"  {key}")
        if len(missing) > 20:
            print(f"  ... {len(missing) - 20} more")

    if unexpected:
        print("WARNING: Unexpected teacher keys:")
        for key in unexpected[:20]:
            print(f"  {key}")
        if len(unexpected) > 20:
            print(f"  ... {len(unexpected) - 20} more")

    teacher.to(device)
    teacher.eval()

    for param in teacher.parameters():
        param.requires_grad = False

    print(
        f"Teacher parameters: {count_parameters(teacher, trainable_only=False):,}"
    )

    return teacher
