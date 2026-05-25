from pathlib import Path
from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.blocks import make_mlp
from src.models.backbones import build_image_encoder, GAPCNNSmallEncoder
from src.models.encoders import MeanPoolQuestionEncoder
from src.models.heads import TypeAwareClassifier


class BaselineVQAModel(nn.Module):
    """
    VQA model with configurable image encoder and classifier head.

    Inputs:
      images:       [B, 3, H, W]
      question_ids: [B, L]
      question_len: [B]
      type_id:      [B]

    Output:
      logits:       [B, num_answers]
    """

    def __init__(
        self,
        vocab_size: int,
        num_answers: int,
        num_types: int = 2,
        pad_id: int = 0,
        model_name: str = "cnn",
        pretrained: bool = True,
        freeze_image_encoder: bool = False,
        head_type: str = "shared",
        object_answer_ids: Sequence[int] | None = None,
        color_answer_ids: Sequence[int] | None = None,
        number_answer_ids: Sequence[int] | None = None,  # kept for old caller compatibility
        image_feature_dim: int = 256,
        question_embedding_dim: int = 128,
        question_feature_dim: int = 128,
        type_embedding_dim: int = 16,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.pretrained = pretrained
        self.freeze_image_encoder = freeze_image_encoder
        self.head_type = head_type

        self.image_encoder = build_image_encoder(
            model_name=model_name,
            image_feature_dim=image_feature_dim,
            pretrained=pretrained,
            freeze_image_encoder=freeze_image_encoder,
        )

        self.question_encoder = MeanPoolQuestionEncoder(
            vocab_size=vocab_size,
            embedding_dim=question_embedding_dim,
            question_feature_dim=question_feature_dim,
            pad_id=pad_id,
            dropout=dropout,
        )

        self.type_embedding = nn.Embedding(
            num_embeddings=num_types,
            embedding_dim=type_embedding_dim,
        )

        fusion_dim = image_feature_dim + question_feature_dim + type_embedding_dim

        self.classifier = TypeAwareClassifier(
            fusion_dim=fusion_dim,
            hidden_dim=hidden_dim,
            num_answers=num_answers,
            dropout=dropout,
            head_type=head_type,
            object_answer_ids=object_answer_ids,
            color_answer_ids=color_answer_ids,
            number_answer_ids=number_answer_ids,
        )

    def forward(
        self,
        images: torch.Tensor,
        question_ids: torch.Tensor,
        question_len: torch.Tensor,
        type_id: torch.Tensor,
    ) -> torch.Tensor:
        image_features = self.image_encoder(images)
        question_features = self.question_encoder(question_ids, question_len)
        type_features = self.type_embedding(type_id)

        fused = torch.cat(
            [image_features, question_features, type_features],
            dim=1,
        )

        logits = self.classifier(fused, type_id)
        return logits


class GAPCNNVQAModel(nn.Module):
    """
    GAPCNN student model for object/color VQA.

    Deployment-oriented interface:
      image + type_onehot -> object_head logits + color_head logits

    Training/evaluation convenience:
      If type_id is provided, the model can also assemble global logits
      over answer_vocab.json IDs.

    Inputs:
      images:      [B, 3, H, W]
      type_onehot: [B, 2], optional
      type_id:     [B], optional

    Output by default:
      global_logits: [B, num_answers]

    Output with return_dict=True:
      {
        "logits": global_logits,
        "object_logits": object_logits,
        "color_logits": color_logits,
        "type_features": type_features,
        "image_features": image_features,
      }
    """

    def __init__(
        self,
        num_answers: int,
        object_answer_ids: Sequence[int],
        color_answer_ids: Sequence[int],
        image_feature_dim: int = 160,
        type_feature_dim: int = 16,
        hidden_dim: int = 192,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        object_answer_ids = sorted(set(int(x) for x in object_answer_ids))
        color_answer_ids = sorted(set(int(x) for x in color_answer_ids))

        if len(object_answer_ids) == 0:
            raise ValueError("object_answer_ids cannot be empty.")
        if len(color_answer_ids) == 0:
            raise ValueError("color_answer_ids cannot be empty.")

        self.num_answers = int(num_answers)
        self.num_types = 2
        self.num_object_answers = len(object_answer_ids)
        self.num_color_answers = len(color_answer_ids)

        self.register_buffer(
            "object_answer_ids",
            torch.tensor(object_answer_ids, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "color_answer_ids",
            torch.tensor(color_answer_ids, dtype=torch.long),
            persistent=True,
        )

        self.image_encoder = GAPCNNSmallEncoder(
            image_feature_dim=image_feature_dim,
        )

        # Deployment-friendly type conditioning:
        # pass one-hot [object, color] into a Linear layer.
        self.type_proj = nn.Sequential(
            nn.Linear(self.num_types, type_feature_dim),
            nn.ReLU(inplace=True),
        )

        fusion_dim = image_feature_dim + type_feature_dim

        self.object_head = make_mlp(
            in_dim=fusion_dim,
            hidden_dim=hidden_dim,
            out_dim=self.num_object_answers,
            dropout=dropout,
        )

        self.color_head = make_mlp(
            in_dim=fusion_dim,
            hidden_dim=hidden_dim,
            out_dim=self.num_color_answers,
            dropout=dropout,
        )

    def _make_type_onehot(
        self,
        type_id: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return F.one_hot(type_id, num_classes=self.num_types).to(dtype=dtype)

    def _make_global_logits(
        self,
        object_logits: torch.Tensor,
        color_logits: torch.Tensor,
        type_id: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = type_id.shape[0]

        logits = object_logits.new_full(
            (batch_size, self.num_answers),
            fill_value=-1e4,
        )

        object_rows = torch.nonzero(type_id == 0, as_tuple=False).flatten()
        color_rows = torch.nonzero(type_id == 1, as_tuple=False).flatten()

        if object_rows.numel() > 0:
            row_index = object_rows.unsqueeze(1)
            col_index = self.object_answer_ids.to(logits.device).unsqueeze(0)
            logits[row_index, col_index] = object_logits[object_rows].to(logits.dtype)

        if color_rows.numel() > 0:
            row_index = color_rows.unsqueeze(1)
            col_index = self.color_answer_ids.to(logits.device).unsqueeze(0)
            logits[row_index, col_index] = color_logits[color_rows].to(logits.dtype)

        return logits

    def forward_heads(
        self,
        images: torch.Tensor,
        type_onehot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Clean deployment-style forward path.

        Returns:
          object_logits: [B, num_object_answers]
          color_logits:  [B, num_color_answers]
        """
        image_features = self.image_encoder(images)
        type_features = self.type_proj(type_onehot.to(dtype=image_features.dtype))

        fused = torch.cat([image_features, type_features], dim=1)

        object_logits = self.object_head(fused)
        color_logits = self.color_head(fused)

        return object_logits, color_logits

    def forward(
        self,
        images: torch.Tensor,
        type_onehot: torch.Tensor | None = None,
        type_id: torch.Tensor | None = None,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if type_onehot is None:
            if type_id is None:
                raise ValueError("GAPCNNVQAModel requires type_onehot or type_id.")
            type_onehot = self._make_type_onehot(type_id, dtype=images.dtype)

        if type_id is None:
            # Training/eval convenience.
            # For deployment/export, prefer passing type_onehot and using forward_heads.
            type_id = torch.argmax(type_onehot, dim=1)

        image_features = self.image_encoder(images)
        type_features = self.type_proj(type_onehot.to(dtype=image_features.dtype))

        fused = torch.cat([image_features, type_features], dim=1)

        object_logits = self.object_head(fused)
        color_logits = self.color_head(fused)

        global_logits = self._make_global_logits(
            object_logits=object_logits,
            color_logits=color_logits,
            type_id=type_id,
        )

        if return_dict:
            return {
                "logits": global_logits,
                "object_logits": object_logits,
                "color_logits": color_logits,
                "type_features": type_features,
                "image_features": image_features,
            }

        return global_logits


def build_baseline_vqa_model(
    vocab_size: int,
    num_answers: int,
    pad_id: int = 0,
    model_name: str = "cnn",
    pretrained: bool = True,
    freeze_image_encoder: bool = False,
    head_type: str = "shared",
    object_answer_ids: Sequence[int] | None = None,
    color_answer_ids: Sequence[int] | None = None,
    number_answer_ids: Sequence[int] | None = None,
) -> BaselineVQAModel:
    """
    Convenience builder.

    Defaults reproduce a shared-head CNN baseline.
    For the current object+color task, separate heads use:
      type_id 0 = object
      type_id 1 = color
    """
    return BaselineVQAModel(
        vocab_size=vocab_size,
        num_answers=num_answers,
        pad_id=pad_id,
        model_name=model_name,
        pretrained=pretrained,
        freeze_image_encoder=freeze_image_encoder,
        head_type=head_type,
        object_answer_ids=object_answer_ids,
        color_answer_ids=color_answer_ids,
        number_answer_ids=number_answer_ids,
    )


def build_gapcnn_s_vqa_model(
    num_answers: int,
    object_answer_ids: Sequence[int],
    color_answer_ids: Sequence[int],
    image_feature_dim: int = 160,
    type_feature_dim: int = 16,
    hidden_dim: int = 192,
    dropout: float = 0.1,
) -> GAPCNNVQAModel:
    """
    Build GAPCNN-S object/color VQA student.
    """
    return GAPCNNVQAModel(
        num_answers=num_answers,
        object_answer_ids=object_answer_ids,
        color_answer_ids=color_answer_ids,
        image_feature_dim=image_feature_dim,
        type_feature_dim=type_feature_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


class QuestionVQAModel(nn.Module):
    """
    Question-aware VQA classifier with separate object and color heads.
    Supports both standard linear classifier heads and text-embedding prototype cosine similarity heads.

    Inputs:
      images:       [B, 3, H, W]
      question_ids: [B, L]
      question_len: [B]

    Output:
      object_logits: [B, 40]
      color_logits:  [B, 10]
    """

    def __init__(
        self,
        vocab_size: int,
        num_object_classes: int = 40,
        num_color_classes: int = 10,
        image_encoder_name: str = "gapcnn_s",
        image_feature_dim: int = 160,
        question_embedding_dim: int = 64,
        question_feature_dim: int = 128,
        pad_id: int = 0,
        hidden_dim: int = 192,
        dropout: float = 0.1,
        pretrained: bool = True,
        freeze_image_encoder: bool = False,
        head_type: str = "classifier",
        answer_embed_dim: int = 128,
        logit_scale_init: float = 10.0,
        learn_logit_scale: bool = False,
    ) -> None:
        super().__init__()

        self.head_type = head_type.lower()
        if self.head_type not in {"classifier", "prototype"}:
            raise ValueError(f"Unknown head_type: {head_type}. Supported: classifier, prototype")

        self.answer_embed_dim = answer_embed_dim

        self.image_encoder = build_image_encoder(
            model_name=image_encoder_name,
            image_feature_dim=image_feature_dim,
            pretrained=pretrained,
            freeze_image_encoder=freeze_image_encoder,
        )

        self.question_encoder = MeanPoolQuestionEncoder(
            vocab_size=vocab_size,
            embedding_dim=question_embedding_dim,
            question_feature_dim=question_feature_dim,
            pad_id=pad_id,
            dropout=dropout,
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(image_feature_dim + question_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Standard classifier heads
        self.object_head = nn.Linear(hidden_dim, num_object_classes)
        self.color_head = nn.Linear(hidden_dim, num_color_classes)

        # Prototype cosine heads setup
        if self.head_type == "prototype":
            self.embedding_proj = nn.Linear(hidden_dim, answer_embed_dim)

            # Logit scale setup
            if learn_logit_scale:
                self.logit_scale = nn.Parameter(torch.tensor(logit_scale_init))
            else:
                self.register_buffer("logit_scale", torch.tensor(logit_scale_init))

            # Initialize mock/fallback prototype tables with random unit vectors
            obj_proto = torch.randn(num_object_classes, answer_embed_dim)
            obj_proto = F.normalize(obj_proto, p=2, dim=1)
            self.register_buffer("object_prototypes", obj_proto)

            col_proto = torch.randn(num_color_classes, answer_embed_dim)
            col_proto = F.normalize(col_proto, p=2, dim=1)
            self.register_buffer("color_prototypes", col_proto)

    def load_prototypes(self, path: str | Path) -> None:
        """
        Loads pre-generated prototype tables into registered buffers.
        """
        if self.head_type != "prototype":
            print("[WARNING] load_prototypes called but head_type is not 'prototype'. Skipping.")
            return

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Prototypes file not found: {path}")

        data = torch.load(path, map_location="cpu")
        obj_proto = data["object_prototypes"]
        col_proto = data["color_prototypes"]

        if obj_proto.shape[1] != self.answer_embed_dim:
            raise ValueError(f"Loaded object prototypes dim {obj_proto.shape[1]} != model answer_embed_dim {self.answer_embed_dim}")
        if col_proto.shape[1] != self.answer_embed_dim:
            raise ValueError(f"Loaded color prototypes dim {col_proto.shape[1]} != model answer_embed_dim {self.answer_embed_dim}")

        # Copy to the registered buffers, ensuring L2-normalization
        self.object_prototypes.copy_(F.normalize(obj_proto.to(self.object_prototypes.device), p=2, dim=1))
        self.color_prototypes.copy_(F.normalize(col_proto.to(self.color_prototypes.device), p=2, dim=1))
        print(f"Successfully loaded prototypes from {path} (embed_dim: {self.answer_embed_dim})")

    def forward(
        self,
        images: torch.Tensor,
        question_ids: torch.Tensor,
        question_len: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_features = self.image_encoder(images)
        question_features = self.question_encoder(question_ids, question_len)

        fused = torch.cat([image_features, question_features], dim=1)
        fused_features = self.fusion_mlp(fused)

        if self.head_type == "prototype":
            # Map fusion dim to answer_embed_dim and L2-normalize
            fused_emb = self.embedding_proj(fused_features)
            fused_emb = F.normalize(fused_emb, p=2, dim=1)

            # Compute scaled cosine logits using matrix dot product with prototype buffers
            object_logits = self.logit_scale * torch.matmul(fused_emb, self.object_prototypes.t())
            color_logits = self.logit_scale * torch.matmul(fused_emb, self.color_prototypes.t())
        else:
            object_logits = self.object_head(fused_features)
            color_logits = self.color_head(fused_features)

        return object_logits, color_logits

    def inference(
        self,
        images: torch.Tensor,
        questions: list[str] | str,
        question_ids: torch.Tensor,
        question_len: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Runs model forward and selects predictions from the correct head
        based on the question text (checks if the question contains "color").

        Returns a tensor of predicted class IDs (either in [0, 39] for object,
        or in [0, 9] for color).
        """
        if isinstance(questions, str):
            questions = [questions]

        object_logits, color_logits = self.forward(images, question_ids, question_len)

        object_preds = object_logits.argmax(dim=-1)  # [B]
        color_preds = color_logits.argmax(dim=-1)    # [B]

        preds = []
        for idx, q in enumerate(questions):
            if "color" in q.lower():
                preds.append(color_preds[idx].item())
            else:
                preds.append(object_preds[idx].item())

        return torch.tensor(preds, dtype=torch.long, device=images.device)


def compute_type_aware_loss(
    object_logits: torch.Tensor,
    color_logits: torch.Tensor,
    object_answer_id: torch.Tensor,
    color_answer_id: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Computes type-aware loss where object samples use object_head and color samples use color_head.
    object_answer_id contains -1 for color samples, and color_answer_id contains -1 for object samples.
    """
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    object_mask = object_answer_id != -1
    color_mask = color_answer_id != -1

    loss = torch.tensor(0.0, device=object_logits.device)
    num_object_samples = object_mask.sum().item()
    num_color_samples = color_mask.sum().item()
    total_samples = num_object_samples + num_color_samples

    object_loss = torch.tensor(0.0, device=object_logits.device)
    color_loss = torch.tensor(0.0, device=color_logits.device)

    if num_object_samples > 0:
        object_loss = loss_fn(object_logits[object_mask], object_answer_id[object_mask])
        loss += object_loss

    if num_color_samples > 0:
        color_loss = loss_fn(color_logits[color_mask], color_answer_id[color_mask])
        loss += color_loss

    if total_samples > 0:
        loss = loss / total_samples

    # Accuracy calculations
    object_acc = torch.tensor(0.0, device=object_logits.device)
    if num_object_samples > 0:
        object_preds = object_logits[object_mask].argmax(dim=-1)
        object_acc = (object_preds == object_answer_id[object_mask]).float().mean()

    color_acc = torch.tensor(0.0, device=color_logits.device)
    if num_color_samples > 0:
        color_preds = color_logits[color_mask].argmax(dim=-1)
        color_acc = (color_preds == color_answer_id[color_mask]).float().mean()

    # Total accuracy is the average over all samples
    total_correct = 0.0
    if num_object_samples > 0:
        total_correct += (object_logits[object_mask].argmax(dim=-1) == object_answer_id[object_mask]).sum().item()
    if num_color_samples > 0:
        total_correct += (color_logits[color_mask].argmax(dim=-1) == color_answer_id[color_mask]).sum().item()

    total_acc = torch.tensor(total_correct / max(total_samples, 1), device=object_logits.device)

    return {
        "loss": loss,
        "object_loss": object_loss / max(num_object_samples, 1) if num_object_samples > 0 else torch.tensor(0.0, device=object_logits.device),
        "color_loss": color_loss / max(num_color_samples, 1) if num_color_samples > 0 else torch.tensor(0.0, device=color_logits.device),
        "object_acc": object_acc,
        "color_acc": color_acc,
        "total_acc": total_acc,
        "num_object": torch.tensor(num_object_samples, dtype=torch.float32, device=object_logits.device),
        "num_color": torch.tensor(num_color_samples, dtype=torch.float32, device=object_logits.device),
    }
