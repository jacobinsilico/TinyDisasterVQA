"""
VQA models for pruned COCO-QA.

Current final task:
  image + question type -> object/color answer prediction

Supported image encoders:
  - cnn: small CNN trained from scratch
  - mobilenet_v2: ImageNet-pretrained MobileNetV2 backbone
  - mobilenet_v3_large: ImageNet-pretrained MobileNetV3-Large backbone
  - gapcnn_s: GAP9/NE16-friendly small CNN encoder

Supported model families:
  - BaselineVQAModel:
      image + tokenized question + type embedding -> global logits
      useful for CNN/MobileNet baselines and teachers

  - GAPCNNVQAModel:
      image + type one-hot / type id -> object/color heads
      hardware-aware student model for GAP9 experiments

Important:
  GAPCNN-S intentionally avoids residuals, attention, dynamic control flow,
  depthwise convolutions, LayerNorm, GELU/SiLU, and fancy indexing in the
  architecture itself. During training/evaluation we may still assemble
  global logits for convenience, but deployment should export the clean
  object/color heads directly.
"""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    MobileNet_V2_Weights,
    MobileNet_V3_Large_Weights,
    mobilenet_v2,
    mobilenet_v3_large,
)


# Object/color-only type mapping.
TYPE_TO_ID = {
    "object": 0,
    "color": 1,
}

ID_TO_TYPE = {
    0: "object",
    1: "color",
}


# ---------------------------------------------------------------------
# Generic building blocks
# ---------------------------------------------------------------------


class ConvBNReLU(nn.Module):
    """
    GAP9-friendly Conv-BN-ReLU block.

    During deployment, BatchNorm should be folded into Conv.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
    ) -> None:
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


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


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    return sum(p.numel() for p in model.parameters())


def estimate_int8_weight_size_bytes(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Rough INT8 weight memory estimate.

    1 parameter ~= 1 byte after INT8 weight quantization.
    This does not include activation memory, code, buffers, or metadata.
    """
    return count_parameters(model, trainable_only=trainable_only)


# ---------------------------------------------------------------------
# Image encoders
# ---------------------------------------------------------------------


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
            ConvBNReLU(3, 32, kernel_size=3, stride=2),

            # 64x64 -> 32x32
            ConvBNReLU(32, 64, kernel_size=3, stride=2),

            # 32x32 -> 16x16
            ConvBNReLU(64, 128, kernel_size=3, stride=2),

            # 16x16 -> 8x8
            ConvBNReLU(128, 192, kernel_size=3, stride=2),

            # 8x8 -> 4x4
            ConvBNReLU(192, 256, kernel_size=3, stride=2),

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


class GAPCNNSmallEncoder(nn.Module):
    """
    GAPCNN-S image encoder.

    Design goals:
      - deployment-friendly
      - mostly 3x3 convolutions
      - no residuals in v1
      - no depthwise conv in v1
      - no attention / normalization tricks
      - target roughly <1 MB INT8 for full model

    Input:
      image: [B, 3, 128, 128]

    Output:
      image_features: [B, image_feature_dim]
    """

    def __init__(self, image_feature_dim: int = 160) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # 128x128 -> 64x64
            ConvBNReLU(3, 24, kernel_size=3, stride=2),
            ConvBNReLU(24, 32, kernel_size=3, stride=1),

            # 64x64 -> 32x32
            ConvBNReLU(32, 48, kernel_size=3, stride=2),
            ConvBNReLU(48, 48, kernel_size=3, stride=1),

            # 32x32 -> 16x16
            ConvBNReLU(48, 72, kernel_size=3, stride=2),
            ConvBNReLU(72, 72, kernel_size=3, stride=1),

            # 16x16 -> 8x8
            ConvBNReLU(72, 96, kernel_size=3, stride=2),
            ConvBNReLU(96, 96, kernel_size=3, stride=1),

            # 8x8 -> 4x4
            ConvBNReLU(96, 128, kernel_size=3, stride=2),
            ConvBNReLU(128, 128, kernel_size=3, stride=1),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, image_feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.features(images)
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

class MobileNetV3LargeEncoder(nn.Module):
    """
    MobileNetV3-Large image encoder.

    Uses ImageNet-pretrained MobileNetV3-Large by default.

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

        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_large(weights=weights)

        self.features = backbone.features

        # In torchvision MobileNetV3-Large, classifier[0] maps from 960 features.
        # This is more robust than hardcoding 960.
        mobilenet_feature_dim = backbone.classifier[0].in_features

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
      - small_cnn
      - gapcnn_s
      - mobilenet_v2
      - mobilenet_v3_large
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

    if model_name in {"gapcnn_s", "gapcnn-small", "gapcnn_small"}:
        encoder = GAPCNNSmallEncoder(
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

    if model_name in {
        "mobilenet_v3_large",
        "mobilenetv3_large",
        "mobilenet_v3",
        "mobilenetv3",
    }:
        return MobileNetV3LargeEncoder(
            image_feature_dim=image_feature_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_image_encoder,
        )

    raise ValueError(
        f"Unknown model_name='{model_name}'. "
        f"Supported: cnn, gapcnn_s, mobilenet_v2, mobilenet_v3_large"
    )


# ---------------------------------------------------------------------
# Baseline / teacher VQA model
# ---------------------------------------------------------------------


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


class TypeAwareClassifier(nn.Module):
    """
    Object/color classifier over global answer IDs.

    head_type="shared":
      One classifier produces [B, num_answers].

    head_type="separate":
      Two classifiers produce logits only for valid answer IDs:
        type_id 0 = object
        type_id 1 = color

      Output is still [B, num_answers].
      Invalid answer logits are set to a large negative value.

    Note:
      The global scatter is convenient for training/evaluation.
      For deployment, prefer exporting separate object/color heads directly.
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
        number_answer_ids: Sequence[int] | None = None,  # kept for old caller compatibility
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

        if object_answer_ids is None or color_answer_ids is None:
            raise ValueError(
                "Separate object/color heads require object_answer_ids "
                "and color_answer_ids."
            )

        object_answer_ids = sorted(set(int(x) for x in object_answer_ids))
        color_answer_ids = sorted(set(int(x) for x in color_answer_ids))

        if len(object_answer_ids) == 0:
            raise ValueError("object_answer_ids cannot be empty.")
        if len(color_answer_ids) == 0:
            raise ValueError("color_answer_ids cannot be empty.")

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
        local_logits = local_logits.to(dtype=global_logits.dtype)

        row_index = rows.unsqueeze(1)
        col_index = answer_ids.to(global_logits.device).unsqueeze(0)

        global_logits[row_index, col_index] = local_logits

    def forward(self, fused: torch.Tensor, type_id: torch.Tensor) -> torch.Tensor:
        if self.head_type == "shared":
            return self.shared_head(fused)

        logits = fused.new_full(
            (fused.shape[0], self.num_answers),
            fill_value=-1e4,
        )

        # Object/color dataset type IDs:
        # 0 = object
        # 1 = color
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
            head=self.color_head,
            answer_ids=self.color_answer_ids,
        )

        return logits


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


# ---------------------------------------------------------------------
# GAPCNN student model
# ---------------------------------------------------------------------


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