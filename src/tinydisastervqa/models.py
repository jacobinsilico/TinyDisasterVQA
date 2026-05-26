"""
models.py

Model definitions for TinyDisasterVQA.

Current models:
  - TeacherVQA: strong image encoder + LSTM question encoder + MLP classifier
  - TDMSVQA: TDM-S, smallest TinyDisasterVQA student model
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from torchvision import models


ImageBackboneName = Literal[
    "convnext_tiny",
    "efficientnet_b0",
    "efficientnet_b1",
    "resnet18",
    "resnet50",
]


# =============================================================================
# Teacher model
# =============================================================================


@dataclass
class TeacherConfig:
    image_backbone: ImageBackboneName = "convnext_tiny"
    pretrained: bool = True

    vocab_size: int = 50
    pad_id: int = 0
    question_embed_dim: int = 128
    question_hidden_dim: int = 256
    question_num_layers: int = 1
    question_bidirectional: bool = False
    question_dropout: float = 0.0

    fusion_hidden_dim: int = 512
    fusion_dropout: float = 0.3

    num_classes: int = 19
    freeze_image_encoder: bool = False


class TorchvisionImageEncoder(nn.Module):
    """
    Wraps torchvision image backbones and returns one feature vector per image.
    """

    def __init__(
        self,
        backbone_name: ImageBackboneName = "convnext_tiny",
        pretrained: bool = True,
        freeze: bool = False,
    ) -> None:
        super().__init__()

        self.backbone_name = backbone_name
        self.pretrained = pretrained

        if backbone_name == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            model = models.convnext_tiny(weights=weights)

            self.feature_dim = model.classifier[2].in_features
            self.encoder = nn.Sequential(
                model.features,
                model.avgpool,
                model.classifier[0],
                model.classifier[1],
            )

        elif backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)

            self.feature_dim = model.classifier[1].in_features
            self.encoder = nn.Sequential(
                model.features,
                model.avgpool,
                nn.Flatten(1),
            )

        elif backbone_name == "efficientnet_b1":
            weights = models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b1(weights=weights)

            self.feature_dim = model.classifier[1].in_features
            self.encoder = nn.Sequential(
                model.features,
                model.avgpool,
                nn.Flatten(1),
            )

        elif backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)

            self.feature_dim = model.fc.in_features
            self.encoder = nn.Sequential(
                *list(model.children())[:-1],
                nn.Flatten(1),
            )

        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)

            self.feature_dim = model.fc.in_features
            self.encoder = nn.Sequential(
                *list(model.children())[:-1],
                nn.Flatten(1),
            )

        else:
            raise ValueError(f"Unknown image backbone: {backbone_name}")

        if freeze:
            self.freeze()

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)


class LSTMQuestionEncoder(nn.Module):
    """
    TinyVQA-style question encoder:

      token ids -> embedding -> LSTM -> final hidden state
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int = 0,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_id,
        )

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout,
        )

        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(
        self,
        question_tokens: torch.Tensor,
        question_lengths: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.embedding(question_tokens)

        lengths_cpu = question_lengths.detach().cpu().clamp(min=1)

        packed = pack_padded_sequence(
            embedded,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        )

        _, (hidden, _) = self.lstm(packed)

        if self.bidirectional:
            forward_hidden = hidden[-2]
            backward_hidden = hidden[-1]
            question_features = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            question_features = hidden[-1]

        return question_features


class TeacherVQA(nn.Module):
    """
    Strong teacher model:

      image -> ConvNeXt/EfficientNet/ResNet -> image feature
      question tokens -> Embedding + LSTM -> question feature
      concat(image, question) -> MLP -> edge_global logits
    """

    def __init__(self, config: TeacherConfig) -> None:
        super().__init__()

        self.config = config

        self.image_encoder = TorchvisionImageEncoder(
            backbone_name=config.image_backbone,
            pretrained=config.pretrained,
            freeze=config.freeze_image_encoder,
        )

        self.question_encoder = LSTMQuestionEncoder(
            vocab_size=config.vocab_size,
            pad_id=config.pad_id,
            embed_dim=config.question_embed_dim,
            hidden_dim=config.question_hidden_dim,
            num_layers=config.question_num_layers,
            bidirectional=config.question_bidirectional,
            dropout=config.question_dropout,
        )

        fusion_dim = self.image_encoder.feature_dim + self.question_encoder.output_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, config.fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.fusion_hidden_dim // 2, config.num_classes),
        )

    def forward(
        self,
        images: torch.Tensor,
        question_tokens: torch.Tensor,
        question_lengths: torch.Tensor,
    ) -> torch.Tensor:
        image_features = self.image_encoder(images)
        question_features = self.question_encoder(question_tokens, question_lengths)

        fused = torch.cat([image_features, question_features], dim=1)
        logits = self.classifier(fused)

        return logits

    def freeze_image_encoder(self) -> None:
        self.image_encoder.freeze()

    def unfreeze_image_encoder(self) -> None:
        self.image_encoder.unfreeze()


def build_teacher_from_metadata(
    metadata: dict,
    image_backbone: ImageBackboneName = "convnext_tiny",
    pretrained: bool = True,
    num_classes: int = 19,
    freeze_image_encoder: bool = False,
    question_embed_dim: int = 128,
    question_hidden_dim: int = 256,
    fusion_hidden_dim: int = 512,
    fusion_dropout: float = 0.3,
) -> TeacherVQA:
    """
    Convenience builder using outputs/training_data/metadata.json.
    """
    vocab_size = int(metadata["vocab_size_with_pad"])
    pad_id = int(metadata["pad_id"])

    config = TeacherConfig(
        image_backbone=image_backbone,
        pretrained=pretrained,
        vocab_size=vocab_size,
        pad_id=pad_id,
        question_embed_dim=question_embed_dim,
        question_hidden_dim=question_hidden_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        fusion_dropout=fusion_dropout,
        num_classes=num_classes,
        freeze_image_encoder=freeze_image_encoder,
    )

    return TeacherVQA(config)


# =============================================================================
# TDM-S student model
# =============================================================================


@dataclass
class TDMSConfig:
    """
    TDM-S = TinyDisasterModel Small.

    Design goal:
      - smallest useful student
      - no LSTM
      - question_template_id embedding instead of token sequence encoding
      - depthwise-separable CNN image encoder
      - 19-class edge_global output for simple teacher distillation

    Later:
      - TDM-M can increase channels/hidden_dim
      - TDM-L can add richer fusion or token encoder
    """

    num_question_templates: int = 31
    question_template_embed_dim: int = 32

    image_channels: tuple[int, int, int, int, int] = (24, 48, 96, 128, 160)
    image_feature_dim: int = 160

    fusion_hidden_dim: int = 192
    fusion_dropout: float = 0.1

    num_classes: int = 19


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
    ) -> None:
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyCNNImageEncoder(nn.Module):
    """
    Small depthwise-separable CNN for TDM-S.

    Input:
      image [B, 3, 224, 224]

    Output:
      image feature [B, image_feature_dim]
    """

    def __init__(
        self,
        channels: tuple[int, int, int, int, int] = (24, 48, 96, 128, 160),
    ) -> None:
        super().__init__()

        c1, c2, c3, c4, c5 = channels

        self.feature_dim = c5

        self.encoder = nn.Sequential(
            ConvBNAct(3, c1, kernel_size=3, stride=2),          # 224 -> 112
            DepthwiseSeparableConv(c1, c2, stride=2),           # 112 -> 56
            DepthwiseSeparableConv(c2, c3, stride=2),           # 56 -> 28
            DepthwiseSeparableConv(c3, c4, stride=2),           # 28 -> 14
            DepthwiseSeparableConv(c4, c5, stride=2),           # 14 -> 7
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)


class TemplateQuestionEncoder(nn.Module):
    """
    Question encoder for TDM-S.

    FloodNet has only 31 question templates, so for the smallest student
    we use question_template_id directly instead of an LSTM.
    """

    def __init__(
        self,
        num_question_templates: int = 31,
        embed_dim: int = 32,
    ) -> None:
        super().__init__()

        self.num_question_templates = num_question_templates
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(
            num_embeddings=num_question_templates,
            embedding_dim=embed_dim,
        )

        self.output_dim = embed_dim

    def forward(self, question_template_ids: torch.Tensor) -> torch.Tensor:
        if question_template_ids.ndim > 1:
            question_template_ids = question_template_ids.squeeze(-1)

        return self.embedding(question_template_ids.long())


class TDMSVQA(nn.Module):
    """
    TDM-S student model.

    Forward signature intentionally accepts question_tokens/question_lengths too,
    but it only uses question_template_ids. This keeps script code flexible.

    Output:
      logits [B, 19] for edge_global classification.
    """

    def __init__(self, config: TDMSConfig) -> None:
        super().__init__()

        self.config = config

        self.image_encoder = TinyCNNImageEncoder(
            channels=config.image_channels,
        )

        self.question_encoder = TemplateQuestionEncoder(
            num_question_templates=config.num_question_templates,
            embed_dim=config.question_template_embed_dim,
        )

        fusion_dim = self.image_encoder.feature_dim + self.question_encoder.output_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, config.fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.fusion_hidden_dim, config.num_classes),
        )

    def forward(
        self,
        images: torch.Tensor,
        question_tokens: torch.Tensor | None = None,
        question_lengths: torch.Tensor | None = None,
        question_template_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if question_template_ids is None:
            raise ValueError("TDMSVQA requires question_template_ids.")

        image_features = self.image_encoder(images)
        question_features = self.question_encoder(question_template_ids)

        fused = torch.cat([image_features, question_features], dim=1)
        logits = self.classifier(fused)

        return logits

@dataclass
class TDMMConfig:
    """
    TDM-M = TinyDisasterModel Medium.

    Design goal:
      - medium student
      - still no LSTM
      - question_template_id embedding
      - larger depthwise-separable CNN than TDM-S
      - 19-class edge_global output

    Target:
      - better count/density performance than TDM-S
      - still small enough for embedded deployment
    """

    num_question_templates: int = 31
    question_template_embed_dim: int = 64

    image_channels: tuple[int, int, int, int, int] = (32, 64, 128, 192, 256)
    image_feature_dim: int = 256

    fusion_hidden_dim: int = 384
    fusion_dropout: float = 0.15

    num_classes: int = 19


class TDMMVQA(nn.Module):
    """
    TDM-M student model.

    Same basic design as TDM-S, but wider:
      - larger CNN channels
      - larger template embedding
      - larger fusion MLP
    """

    def __init__(self, config: TDMMConfig) -> None:
        super().__init__()

        self.config = config

        self.image_encoder = TinyCNNImageEncoder(
            channels=config.image_channels,
        )

        self.question_encoder = TemplateQuestionEncoder(
            num_question_templates=config.num_question_templates,
            embed_dim=config.question_template_embed_dim,
        )

        fusion_dim = self.image_encoder.feature_dim + self.question_encoder.output_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, config.fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.fusion_hidden_dim // 2, config.num_classes),
        )

    def forward(
        self,
        images: torch.Tensor,
        question_tokens: torch.Tensor | None = None,
        question_lengths: torch.Tensor | None = None,
        question_template_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if question_template_ids is None:
            raise ValueError("TDMMVQA requires question_template_ids.")

        image_features = self.image_encoder(images)
        question_features = self.question_encoder(question_template_ids)

        fused = torch.cat([image_features, question_features], dim=1)
        logits = self.classifier(fused)

        return logits

def build_tdm_s_from_metadata(
    metadata: dict,
    num_classes: int = 19,
    num_question_templates: int = 31,
    question_template_embed_dim: int = 32,
    fusion_hidden_dim: int = 192,
    fusion_dropout: float = 0.1,
) -> TDMSVQA:
    """
    Convenience builder for TDM-S.

    metadata is accepted for API symmetry with teacher builder.
    """
    _ = metadata

    config = TDMSConfig(
        num_question_templates=num_question_templates,
        question_template_embed_dim=question_template_embed_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        fusion_dropout=fusion_dropout,
        num_classes=num_classes,
    )

    return TDMSVQA(config)

def build_tdm_m_from_metadata(
    metadata: dict,
    num_classes: int = 19,
    num_question_templates: int = 31,
    question_template_embed_dim: int = 64,
    fusion_hidden_dim: int = 384,
    fusion_dropout: float = 0.15,
) -> TDMMVQA:
    """
    Convenience builder for TDM-M.
    """
    _ = metadata

    config = TDMMConfig(
        num_question_templates=num_question_templates,
        question_template_embed_dim=question_template_embed_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        fusion_dropout=fusion_dropout,
        num_classes=num_classes,
    )

    return TDMMVQA(config)

# =============================================================================
# Shared model utilities
# =============================================================================


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    return sum(p.numel() for p in model.parameters())


def estimate_model_size_mb(model: nn.Module) -> float:
    """
    Rough fp32 parameter size estimate, not including activations.
    """
    num_params = count_parameters(model, trainable_only=False)
    return num_params * 4 / (1024 ** 2)


def estimate_int8_model_size_kb(model: nn.Module) -> float:
    """
    Rough int8 parameter size estimate, not including activations.
    """
    num_params = count_parameters(model, trainable_only=False)
    return num_params / 1024


def describe_model(model: nn.Module) -> str:
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    size_mb = estimate_model_size_mb(model)
    int8_size_kb = estimate_int8_model_size_kb(model)

    lines = []
    lines.append("Model summary")
    lines.append("=" * 80)
    lines.append(f"Class:             {model.__class__.__name__}")
    lines.append(f"Total params:      {total_params:,}")
    lines.append(f"Trainable params:  {trainable_params:,}")
    lines.append(f"FP32 param size:   {size_mb:.2f} MB")
    lines.append(f"Int8 param size:   {int8_size_kb:.1f} KB")

    if isinstance(model, TeacherVQA):
        lines.append(f"Image backbone:    {model.config.image_backbone}")
        lines.append(f"Pretrained:        {model.config.pretrained}")
        lines.append(f"Image feature dim: {model.image_encoder.feature_dim}")
        lines.append(f"Question dim:      {model.question_encoder.output_dim}")
        lines.append(f"Num classes:       {model.config.num_classes}")

    if isinstance(model, TDMSVQA):
        lines.append("Student variant:   TDM-S")
        lines.append(f"Image channels:    {model.config.image_channels}")
        lines.append(f"Image feature dim: {model.image_encoder.feature_dim}")
        lines.append(f"Template emb dim:  {model.question_encoder.output_dim}")
        lines.append(f"Fusion hidden dim: {model.config.fusion_hidden_dim}")
        lines.append(f"Num classes:       {model.config.num_classes}")

    if isinstance(model, TDMMVQA):
        lines.append("Student variant:   TDM-M")
        lines.append(f"Image channels:    {model.config.image_channels}")
        lines.append(f"Image feature dim: {model.image_encoder.feature_dim}")
        lines.append(f"Template emb dim:  {model.question_encoder.output_dim}")
        lines.append(f"Fusion hidden dim: {model.config.fusion_hidden_dim}")
        lines.append(f"Num classes:       {model.config.num_classes}")

    return "\n".join(lines)