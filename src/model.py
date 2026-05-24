"""
VQA models for pruned COCO-QA.

Task:
  image + question + question type -> 70-class answer prediction

Available image encoders:
  - cnn: small CNN trained from scratch
  - mobilenet_v2: ImageNet-pretrained MobileNetV2 backbone

Available classifier heads:
  - shared: one classifier over all answer classes
  - separate: type-aware classifier heads for object/color/number answers

Architecture:
  image encoder
  + word embedding / masked mean pooling question encoder
  + question type embedding
  + classifier head(s)
"""

from typing import Sequence

import torch
import torch.nn as nn

from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class SmallCNNEncoder(nn.Module):
    """
    Small CNN image encoder for 128x128 RGB images.

    Input:
      image: [B, 3, H, W]

    Output:
      image_features: [B, image_feature_dim]
    """

    def __init__(self, image_feature_dim: int = 256) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 16x16 -> 8x8
            nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            # 8x8 -> 4x4
            nn.Conv2d(192, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, image_feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.conv(images)
        x = self.proj(x)
        return x


class MobileNetV2Encoder(nn.Module):
    """
    MobileNetV2 image encoder.

    Uses ImageNet-pretrained MobileNetV2 by default.

    Input:
      image: [B, 3, H, W]

    Output:
      image_features: [B, image_feature_dim]
    """

    def __init__(
        self,
        image_feature_dim: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v2(weights=weights)

        self.features = backbone.features
        mobilenet_feature_dim = 1280

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(mobilenet_feature_dim, image_feature_dim),
            nn.ReLU(inplace=True),
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.features.parameters():
            param.requires_grad = True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)
        x = self.pool(x)
        x = self.proj(x)
        return x


class MeanPoolQuestionEncoder(nn.Module):
    """
    Question encoder using word embeddings + masked mean pooling.

    Input:
      question_ids: [B, L]
      question_len: [B]

    Output:
      question_features: [B, question_feature_dim]
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        question_feature_dim: int = 128,
        pad_id: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.pad_id = pad_id

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_id,
        )

        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, question_feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        question_ids: torch.Tensor,
        question_len: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings = self.embedding(question_ids)  # [B, L, D]

        mask = (question_ids != self.pad_id).float().unsqueeze(-1)  # [B, L, 1]

        summed = (embeddings * mask).sum(dim=1)  # [B, D]
        denom = mask.sum(dim=1).clamp(min=1.0)   # [B, 1]

        pooled = summed / denom
        features = self.proj(pooled)

        return features


def make_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    dropout: float,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class TypeAwareClassifier(nn.Module):
    """
    Classifier over global answer IDs.

    head_type="shared":
      One classifier produces [B, num_answers].

    head_type="separate":
      Three classifiers produce logits only for valid answer IDs:
        type_id 0 = object
        type_id 1 = number
        type_id 2 = color

      Output is still [B, num_answers].
      Invalid answer logits are set to a large negative value.
    """

    def __init__(
        self,
        fusion_dim: int,
        hidden_dim: int,
        num_answers: int,
        dropout: float,
        head_type: str = "shared",
        object_answer_ids: Sequence[int] | None = None,
        color_answer_ids: Sequence[int] | None = None,
        number_answer_ids: Sequence[int] | None = None,
    ) -> None:
        super().__init__()

        self.head_type = head_type.lower()
        self.num_answers = int(num_answers)

        if self.head_type == "shared":
            self.shared_head = make_mlp(
                in_dim=fusion_dim,
                hidden_dim=hidden_dim,
                out_dim=num_answers,
                dropout=dropout,
            )
            return

        if self.head_type != "separate":
            raise ValueError(
                f"Unknown head_type='{head_type}'. "
                f"Supported: shared, separate"
            )

        if object_answer_ids is None or color_answer_ids is None or number_answer_ids is None:
            raise ValueError(
                "Separate heads require object_answer_ids, color_answer_ids, "
                "and number_answer_ids."
            )

        object_answer_ids = sorted(set(int(x) for x in object_answer_ids))
        color_answer_ids = sorted(set(int(x) for x in color_answer_ids))
        number_answer_ids = sorted(set(int(x) for x in number_answer_ids))

        if len(object_answer_ids) == 0:
            raise ValueError("object_answer_ids cannot be empty.")
        if len(color_answer_ids) == 0:
            raise ValueError("color_answer_ids cannot be empty.")
        if len(number_answer_ids) == 0:
            raise ValueError("number_answer_ids cannot be empty.")

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
        self.register_buffer(
            "number_answer_ids",
            torch.tensor(number_answer_ids, dtype=torch.long),
            persistent=True,
        )

        self.object_head = make_mlp(
            in_dim=fusion_dim,
            hidden_dim=hidden_dim,
            out_dim=len(object_answer_ids),
            dropout=dropout,
        )

        self.color_head = make_mlp(
            in_dim=fusion_dim,
            hidden_dim=hidden_dim,
            out_dim=len(color_answer_ids),
            dropout=dropout,
        )

        self.number_head = make_mlp(
            in_dim=fusion_dim,
            hidden_dim=hidden_dim,
            out_dim=len(number_answer_ids),
            dropout=dropout,
        )

    def _scatter_head_logits(
        self,
        global_logits: torch.Tensor,
        fused: torch.Tensor,
        type_id: torch.Tensor,
        target_type_id: int,
        head: nn.Module,
        answer_ids: torch.Tensor,
    ) -> None:
        rows = torch.nonzero(type_id == target_type_id, as_tuple=False).flatten()

        if rows.numel() == 0:
            return

        local_logits = head(fused[rows])  # [N_type, num_type_answers]

        # AMP/autocast can make local_logits float16 while global_logits is float32.
        # Index assignment requires matching dtypes.
        local_logits = local_logits.to(dtype=global_logits.dtype)

        row_index = rows.unsqueeze(1)
        col_index = answer_ids.to(global_logits.device).unsqueeze(0)

        global_logits[row_index, col_index] = local_logits

    def forward(self, fused: torch.Tensor, type_id: torch.Tensor) -> torch.Tensor:
        if self.head_type == "shared":
            return self.shared_head(fused)

        # Large negative value, safe for AMP/fp16.
        logits = fused.new_full(
            (fused.shape[0], self.num_answers),
            fill_value=-1e4,
        )

        # Dataset type IDs:
        # 0 = object
        # 1 = number
        # 2 = color
        self._scatter_head_logits(
            global_logits=logits,
            fused=fused,
            type_id=type_id,
            target_type_id=0,
            head=self.object_head,
            answer_ids=self.object_answer_ids,
        )

        self._scatter_head_logits(
            global_logits=logits,
            fused=fused,
            type_id=type_id,
            target_type_id=1,
            head=self.number_head,
            answer_ids=self.number_answer_ids,
        )

        self._scatter_head_logits(
            global_logits=logits,
            fused=fused,
            type_id=type_id,
            target_type_id=2,
            head=self.color_head,
            answer_ids=self.color_answer_ids,
        )

        return logits


def build_image_encoder(
    model_name: str,
    image_feature_dim: int,
    pretrained: bool = True,
    freeze_image_encoder: bool = False,
) -> nn.Module:
    """
    Build image encoder by name.

    Supported:
      - cnn
      - mobilenet_v2
    """
    model_name = model_name.lower()

    if model_name in {"cnn", "small_cnn", "baseline_cnn"}:
        encoder = SmallCNNEncoder(
            image_feature_dim=image_feature_dim,
        )

        if freeze_image_encoder:
            for param in encoder.parameters():
                param.requires_grad = False

        return encoder

    if model_name in {"mobilenet_v2", "mobilenetv2"}:
        return MobileNetV2Encoder(
            image_feature_dim=image_feature_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_image_encoder,
        )

    raise ValueError(
        f"Unknown model_name='{model_name}'. "
        f"Supported: cnn, mobilenet_v2"
    )


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
        num_types: int = 3,
        pad_id: int = 0,
        model_name: str = "cnn",
        pretrained: bool = True,
        freeze_image_encoder: bool = False,
        head_type: str = "shared",
        object_answer_ids: Sequence[int] | None = None,
        color_answer_ids: Sequence[int] | None = None,
        number_answer_ids: Sequence[int] | None = None,
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


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    return sum(p.numel() for p in model.parameters())


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

    Defaults reproduce the original shared-head CNN baseline.
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