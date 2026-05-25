import argparse
import torch
import torch.nn as nn
from src.data.vocab import QuestionVocab
from src.utils.serialization import make_json_serializable
from src.utils.model_info import count_parameters, estimate_int8_weight_size_bytes


def forward_model(model: nn.Module, batch_dev: dict) -> torch.Tensor:
    """
    Polymorphic forward pass helper.
    Routes parameters based on whether the model is a student (GAPCNNVQAModel)
    or a text-aware baseline/teacher (BaselineVQAModel).
    """
    if hasattr(model, "type_proj") and not hasattr(model, "question_encoder"):
        return model(
            images=batch_dev["images"],
            type_onehot=batch_dev.get("type_onehot"),
            type_id=batch_dev["type_id"],
        )
    else:
        # Use teacher_images if available (e.g. deterministic size during KD), else images
        images = batch_dev.get("teacher_images", batch_dev["images"])
        return model(
            images=images,
            question_ids=batch_dev["question_ids"],
            question_len=batch_dev["question_len"],
            type_id=batch_dev["type_id"],
        )


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """
    Standardize moving a dataset batch to target device.
    Supports baseline, teacher, and student batches with optional teacher images.
    """
    out = {
        "images": batch["image"].to(device, non_blocking=True),
        "question_ids": batch["question_ids"].to(device, non_blocking=True),
        "question_len": batch["question_len"].to(device, non_blocking=True),
        "answer_id": batch["answer_id"].to(device, non_blocking=True),
        "type_id": batch["type_id"].to(device, non_blocking=True),
    }

    if "type_onehot" in batch:
        out["type_onehot"] = batch["type_onehot"].to(device, non_blocking=True)

    if "teacher_image" in batch:
        out["teacher_images"] = batch["teacher_image"].to(device, non_blocking=True)
    else:
        out["teacher_images"] = out["images"]

    return out


def build_config(
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
    model: nn.Module,
    question_vocab: QuestionVocab,
    answer_vocab: dict,
    answer_ids_by_type: dict[str, list[int]],
    teacher: nn.Module | None = None,
    model_family: str | None = None,
) -> dict:
    """
    Build structured JSON-serializable training run configurations.
    """
    config = make_json_serializable(vars(args))

    config["device"] = str(device)
    config["use_amp"] = bool(use_amp)

    if model_family is not None:
        config["model_family"] = model_family
    elif hasattr(args, "model_name"):
        config["model_family"] = args.model_name
    else:
        config["model_family"] = "vqa_model"

    config["num_model_parameters_trainable"] = int(
        count_parameters(model, trainable_only=True)
    )
    config["num_model_parameters_total"] = int(
        count_parameters(model, trainable_only=False)
    )
    config["estimated_int8_weight_size_bytes"] = int(
        estimate_int8_weight_size_bytes(model, trainable_only=False)
    )
    config["estimated_int8_weight_size_mb"] = (
        config["estimated_int8_weight_size_bytes"] / (1024 * 1024)
    )

    config["question_vocab_size"] = int(question_vocab.size)

    id_to_answer = answer_vocab.get("id_to_answer", answer_vocab)
    config["num_answer_classes"] = int(len(id_to_answer))

    if "object_answers" in answer_vocab:
        config["num_object_answers"] = int(len(answer_vocab["object_answers"]))
    if "color_answers" in answer_vocab:
        config["num_color_answers"] = int(len(answer_vocab["color_answers"]))

    config["answer_ids_by_type"] = make_json_serializable(answer_ids_by_type)
    config["uses_teacher"] = teacher is not None

    return config
