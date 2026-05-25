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
