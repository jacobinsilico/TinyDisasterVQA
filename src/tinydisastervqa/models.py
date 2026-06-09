"""
models.py

Model definitions for TinyDisasterVQA.

Current models:
  - TeacherVQA: strong image encoder + configurable question encoder + MLP classifier
  - TDMVQA: tiny CNN + template embedding student model

Teacher supports:
  - LSTM question encoder
  - question_template_id embedding encoder
  - optional count auxiliary head for count-aware teacher ablation
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
    "swin_tiny",
    "efficientnet_b0",
    "efficientnet_b1",
    "resnet18",
    "resnet50",
]

TeacherQuestionEncoderName = Literal["lstm", "template"]


# =============================================================================
# Shared encoders
# =============================================================================


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

        elif backbone_name == "swin_tiny":
            weights = models.Swin_T_Weights.DEFAULT if pretrained else None
            model = models.swin_t(weights=weights)

            self.feature_dim = model.head.in_features
            self.encoder = nn.Sequential(
                model.features,
                model.norm,
                model.permute,
                model.avgpool,
                nn.Flatten(1),
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
                nn.FlatTensor(1) if False else nn.Flatten(1),
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


class TemplateQuestionEncoder(nn.Module):
    """
    Template-ID question encoder.

    FloodNet-VQA has a small fixed set of question templates, so template IDs are
    a useful lightweight alternative to a full language encoder.
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


def _make_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    dropout: float,
    layers: int = 1,
) -> nn.Sequential:
    if layers <= 1:
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim // 2, out_dim),
    )


# =============================================================================
# Teacher model
# =============================================================================


@dataclass
class TeacherConfig:
    image_backbone: ImageBackboneName = "convnext_tiny"
    pretrained: bool = True

    question_encoder: TeacherQuestionEncoderName = "lstm"

    vocab_size: int = 50
    pad_id: int = 0
    question_embed_dim: int = 128
    question_hidden_dim: int = 256
    question_num_layers: int = 1
    question_bidirectional: bool = False
    question_dropout: float = 0.0

    num_question_templates: int = 31
    template_embed_dim: int = 128

    fusion_hidden_dim: int = 512
    fusion_dropout: float = 0.3

    num_classes: int = 14
    freeze_image_encoder: bool = False

    use_count_aux: bool = False
    num_count_classes: int = 6


class TeacherVQA(nn.Module):
    """
    Strong teacher model.

    Main output:
      logits [B, num_classes] for compact edge_global classification.

    Optional auxiliary output:
      count_logits [B, num_count_classes] for count-only auxiliary training.

    The auxiliary head is training-only. The main deployment/student target stays
    single-head edge_global.
    """

    def __init__(self, config: TeacherConfig) -> None:
        super().__init__()

        self.config = config

        self.image_encoder = TorchvisionImageEncoder(
            backbone_name=config.image_backbone,
            pretrained=config.pretrained,
            freeze=config.freeze_image_encoder,
        )

        if config.question_encoder == "lstm":
            self.question_encoder = LSTMQuestionEncoder(
                vocab_size=config.vocab_size,
                pad_id=config.pad_id,
                embed_dim=config.question_embed_dim,
                hidden_dim=config.question_hidden_dim,
                num_layers=config.question_num_layers,
                bidirectional=config.question_bidirectional,
                dropout=config.question_dropout,
            )
        elif config.question_encoder == "template":
            self.question_encoder = TemplateQuestionEncoder(
                num_question_templates=config.num_question_templates,
                embed_dim=config.template_embed_dim,
            )
        else:
            raise ValueError(f"Unknown teacher question_encoder: {config.question_encoder}")

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

        if config.use_count_aux:
            self.count_aux_classifier = nn.Sequential(
                nn.Linear(fusion_dim, config.fusion_hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(config.fusion_dropout),
                nn.Linear(config.fusion_hidden_dim // 2, config.num_count_classes),
            )
        else:
            self.count_aux_classifier = None

    def encode_question(
        self,
        question_tokens: torch.Tensor | None = None,
        question_lengths: torch.Tensor | None = None,
        question_template_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.config.question_encoder == "lstm":
            if question_tokens is None or question_lengths is None:
                raise ValueError(
                    "LSTM teacher requires question_tokens and question_lengths."
                )

            return self.question_encoder(
                question_tokens=question_tokens,
                question_lengths=question_lengths,
            )

        if self.config.question_encoder == "template":
            if question_template_ids is None:
                raise ValueError("Template teacher requires question_template_ids.")

            return self.question_encoder(question_template_ids)

        raise ValueError(f"Unknown teacher question_encoder: {self.config.question_encoder}")

    def forward(
        self,
        images: torch.Tensor,
        question_tokens: torch.Tensor | None = None,
        question_lengths: torch.Tensor | None = None,
        question_template_ids: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        image_features = self.image_encoder(images)

        question_features = self.encode_question(
            question_tokens=question_tokens,
            question_lengths=question_lengths,
            question_template_ids=question_template_ids,
        )

        fused = torch.cat([image_features, question_features], dim=1)

        logits = self.classifier(fused)

        if not return_aux:
            return logits

        outputs: dict[str, torch.Tensor] = {
            "logits": logits,
        }

        if self.count_aux_classifier is not None:
            outputs["count_logits"] = self.count_aux_classifier(fused)

        return outputs

    def freeze_image_encoder(self) -> None:
        self.image_encoder.freeze()

    def unfreeze_image_encoder(self) -> None:
        self.image_encoder.unfreeze()


def _infer_num_classes_from_metadata(metadata: dict, fallback: int | None = None) -> int:
    for key in ["num_classes", "num_edge_global_classes"]:
        if key in metadata:
            return int(metadata[key])

    try:
        return int(metadata["answer_space"]["target_modes"]["edge_global"]["num_classes"])
    except KeyError:
        pass

    if fallback is not None:
        return int(fallback)

    raise KeyError(
        "Could not infer num_classes from metadata. Expected one of: "
        "num_classes, num_edge_global_classes, or answer_space.target_modes.edge_global.num_classes."
    )


def _infer_num_question_templates_from_metadata(metadata: dict, fallback: int = 31) -> int:
    if "num_question_templates" in metadata:
        return int(metadata["num_question_templates"])

    return int(fallback)


def _infer_num_count_classes_from_metadata(metadata: dict, fallback: int = 6) -> int:
    if "head_label_maps" in metadata and "count" in metadata["head_label_maps"]:
        return len(metadata["head_label_maps"]["count"])

    try:
        return len(
            metadata["answer_space"]["target_modes"]["edge_head_local"]["head_label_maps"]["count"]
        )
    except KeyError:
        pass

    try:
        return len(
            metadata["answer_space"]["target_modes"]["edge_multihead"]["head_label_maps"]["count"]
        )
    except KeyError:
        pass

    return int(fallback)


def build_teacher_from_metadata(
    metadata: dict,
    image_backbone: ImageBackboneName = "convnext_tiny",
    pretrained: bool = True,
    num_classes: int | None = None,
    freeze_image_encoder: bool = False,
    question_encoder: TeacherQuestionEncoderName = "lstm",
    question_embed_dim: int = 128,
    question_hidden_dim: int = 256,
    template_embed_dim: int = 128,
    fusion_hidden_dim: int = 512,
    fusion_dropout: float = 0.3,
    use_count_aux: bool = False,
    num_count_classes: int | None = None,
) -> TeacherVQA:
    """
    Convenience builder using outputs/training_data_*/metadata.json.
    """
    vocab_size = int(metadata["vocab_size_with_pad"])
    pad_id = int(metadata["pad_id"])

    inferred_num_classes = _infer_num_classes_from_metadata(
        metadata=metadata,
        fallback=num_classes,
    )

    inferred_num_templates = _infer_num_question_templates_from_metadata(metadata)

    inferred_num_count_classes = (
        _infer_num_count_classes_from_metadata(metadata)
        if num_count_classes is None
        else int(num_count_classes)
    )

    config = TeacherConfig(
        image_backbone=image_backbone,
        pretrained=pretrained,
        question_encoder=question_encoder,
        vocab_size=vocab_size,
        pad_id=pad_id,
        question_embed_dim=question_embed_dim,
        question_hidden_dim=question_hidden_dim,
        num_question_templates=inferred_num_templates,
        template_embed_dim=template_embed_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        fusion_dropout=fusion_dropout,
        num_classes=inferred_num_classes,
        freeze_image_encoder=freeze_image_encoder,
        use_count_aux=use_count_aux,
        num_count_classes=inferred_num_count_classes,
    )

    return TeacherVQA(config)


# =============================================================================
# TDM student models
# =============================================================================


StudentHeadType = Literal["single", "multihead"]
StudentVariant = Literal["tdm_xxs", "tdm_xs", "tdm_s", "tdm_m"]

EDGE_HEADS: tuple[str, str, str, str] = ("binary", "condition", "density", "count")
EDGE_HEAD_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(EDGE_HEADS)}


@dataclass
class TDMConfig:
    """
    Generic TinyDisasterModel student configuration.

    Used for:
      - TDM-XXS: ultra-small student
      - TDM-XS:  very small student
      - TDM-S:   current small baseline
      - TDM-M:   wider reference model

    Preferred final deployment head_type:
      - single: one global edge_clean classifier.

    multihead is kept for backward compatibility only.
    """

    variant: StudentVariant = "tdm_s"

    num_question_templates: int = 31
    question_template_embed_dim: int = 32

    image_channels: tuple[int, int, int, int, int] = (24, 48, 96, 128, 160)

    fusion_hidden_dim: int = 192
    fusion_dropout: float = 0.1
    fusion_layers: int = 1

    num_classes: int = 14

    head_type: StudentHeadType = "single"
    edge_head_names: tuple[str, ...] = EDGE_HEADS


@dataclass
class TDMSConfig(TDMConfig):
    variant: StudentVariant = "tdm_s"
    num_question_templates: int = 31
    question_template_embed_dim: int = 32
    image_channels: tuple[int, int, int, int, int] = (24, 48, 96, 128, 160)
    fusion_hidden_dim: int = 192
    fusion_dropout: float = 0.1
    fusion_layers: int = 1
    num_classes: int = 14
    head_type: StudentHeadType = "single"


@dataclass
class TDMMConfig(TDMConfig):
    variant: StudentVariant = "tdm_m"
    num_question_templates: int = 31
    question_template_embed_dim: int = 64
    image_channels: tuple[int, int, int, int, int] = (32, 64, 128, 192, 256)
    fusion_hidden_dim: int = 384
    fusion_dropout: float = 0.15
    fusion_layers: int = 2
    num_classes: int = 14
    head_type: StudentHeadType = "single"


TDM_VARIANT_DEFAULTS: dict[StudentVariant, dict[str, object]] = {
    "tdm_xxs": {
        "question_template_embed_dim": 16,
        "image_channels": (8, 16, 32, 48, 64),
        "fusion_hidden_dim": 64,
        "fusion_dropout": 0.05,
        "fusion_layers": 1,
    },
    "tdm_xs": {
        "question_template_embed_dim": 24,
        "image_channels": (16, 32, 64, 96, 128),
        "fusion_hidden_dim": 128,
        "fusion_dropout": 0.08,
        "fusion_layers": 1,
    },
    "tdm_s": {
        "question_template_embed_dim": 32,
        "image_channels": (24, 48, 96, 128, 160),
        "fusion_hidden_dim": 192,
        "fusion_dropout": 0.10,
        "fusion_layers": 1,
    },
    "tdm_m": {
        "question_template_embed_dim": 64,
        "image_channels": (32, 64, 128, 192, 256),
        "fusion_hidden_dim": 384,
        "fusion_dropout": 0.15,
        "fusion_layers": 2,
    },
}


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
    Small depthwise-separable CNN for TDM students.

    Input:
      image [B, 3, H, W]

    Output:
      image feature [B, image_channels[-1]]
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


class MultiHeadGlobalClassifier(nn.Module):
    """
    Backward-compatible multi-head classifier.

    Not recommended for the final GAP9 formulation, but kept so old experiments
    do not immediately break.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        edge_head_names: tuple[str, ...] = EDGE_HEADS,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.edge_head_names = tuple(edge_head_names)

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.heads = nn.ModuleDict(
            {
                head_name: nn.Linear(hidden_dim, num_classes)
                for head_name in self.edge_head_names
            }
        )

    def _mask_for_head(
        self,
        edge_heads: list[str] | tuple[str, ...] | torch.Tensor,
        head_name: str,
        head_id: int,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(edge_heads, torch.Tensor):
            edge_heads = edge_heads.to(device)

            if edge_heads.ndim > 1:
                edge_heads = edge_heads.squeeze(-1)

            return edge_heads.long() == int(head_id)

        normalized = []

        for value in edge_heads:
            if isinstance(value, bytes):
                value = value.decode("utf-8")

            normalized.append(str(value))

        return torch.tensor(
            [value == head_name for value in normalized],
            dtype=torch.bool,
            device=device,
        )

    def forward(
        self,
        fused: torch.Tensor,
        edge_heads: list[str] | tuple[str, ...] | torch.Tensor | None = None,
        edge_head_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_head_ids is not None:
            edge_heads = edge_head_ids

        if edge_heads is None:
            raise ValueError(
                "MultiHeadGlobalClassifier requires edge_heads or edge_head_ids."
            )

        hidden = self.trunk(fused)

        logits = hidden.new_full(
            (hidden.size(0), self.num_classes),
            fill_value=-1.0e4,
        )

        for head_id, head_name in enumerate(self.edge_head_names):
            mask = self._mask_for_head(
                edge_heads=edge_heads,
                head_name=head_name,
                head_id=head_id,
                device=fused.device,
            )

            if bool(mask.any()):
                logits[mask] = self.heads[head_name](hidden[mask])

        return logits


class TDMVQA(nn.Module):
    """
    Generic TDM student model.

    Forward signature accepts question_tokens/question_lengths too, but the
    student only uses question_template_ids.

    Output:
      logits [B, num_classes] for edge_global classification.
    """

    def __init__(self, config: TDMConfig) -> None:
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

        if config.head_type == "single":
            self.classifier = _make_mlp(
                in_dim=fusion_dim,
                hidden_dim=config.fusion_hidden_dim,
                out_dim=config.num_classes,
                dropout=config.fusion_dropout,
                layers=config.fusion_layers,
            )
        elif config.head_type == "multihead":
            self.classifier = MultiHeadGlobalClassifier(
                in_dim=fusion_dim,
                hidden_dim=config.fusion_hidden_dim,
                num_classes=config.num_classes,
                dropout=config.fusion_dropout,
                edge_head_names=config.edge_head_names,
            )
        else:
            raise ValueError(f"Unknown student head_type: {config.head_type}")

    def forward(
        self,
        images: torch.Tensor,
        question_tokens: torch.Tensor | None = None,
        question_lengths: torch.Tensor | None = None,
        question_template_ids: torch.Tensor | None = None,
        edge_heads: list[str] | tuple[str, ...] | torch.Tensor | None = None,
        edge_head_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = question_tokens, question_lengths

        if question_template_ids is None:
            raise ValueError("TDMVQA requires question_template_ids.")

        image_features = self.image_encoder(images)
        question_features = self.question_encoder(question_template_ids)

        fused = torch.cat([image_features, question_features], dim=1)

        if self.config.head_type == "multihead":
            logits = self.classifier(
                fused,
                edge_heads=edge_heads,
                edge_head_ids=edge_head_ids,
            )
        else:
            logits = self.classifier(fused)

        return logits


class TDMSVQA(TDMVQA):
    pass


class TDMMVQA(TDMVQA):
    pass


def make_tdm_config(
    variant: StudentVariant = "tdm_s",
    num_classes: int = 14,
    num_question_templates: int = 31,
    question_template_embed_dim: int | None = None,
    image_channels: tuple[int, int, int, int, int] | None = None,
    fusion_hidden_dim: int | None = None,
    fusion_dropout: float | None = None,
    fusion_layers: int | None = None,
    head_type: StudentHeadType = "single",
) -> TDMConfig:
    defaults = dict(TDM_VARIANT_DEFAULTS[variant])

    config = TDMConfig(
        variant=variant,
        num_question_templates=num_question_templates,
        question_template_embed_dim=int(
            defaults["question_template_embed_dim"]
            if question_template_embed_dim is None
            else question_template_embed_dim
        ),
        image_channels=(
            defaults["image_channels"]
            if image_channels is None
            else image_channels
        ),  # type: ignore[arg-type]
        fusion_hidden_dim=int(
            defaults["fusion_hidden_dim"]
            if fusion_hidden_dim is None
            else fusion_hidden_dim
        ),
        fusion_dropout=float(
            defaults["fusion_dropout"]
            if fusion_dropout is None
            else fusion_dropout
        ),
        fusion_layers=int(
            defaults["fusion_layers"]
            if fusion_layers is None
            else fusion_layers
        ),
        num_classes=num_classes,
        head_type=head_type,
    )

    return config


def _infer_tdm_num_classes(metadata: dict, fallback: int = 14) -> int:
    try:
        return int(metadata["num_classes"])
    except KeyError:
        pass

    try:
        return int(metadata["num_edge_global_classes"])
    except KeyError:
        pass

    try:
        return int(metadata["answer_space"]["target_modes"]["edge_global"]["num_classes"])
    except KeyError:
        return int(fallback)


def _infer_tdm_num_question_templates(metadata: dict, fallback: int = 31) -> int:
    try:
        return int(metadata["num_question_templates"])
    except KeyError:
        return int(fallback)


def build_tdm_from_metadata(
    metadata: dict,
    variant: StudentVariant = "tdm_s",
    num_classes: int | None = None,
    num_question_templates: int | None = None,
    question_template_embed_dim: int | None = None,
    image_channels: tuple[int, int, int, int, int] | None = None,
    fusion_hidden_dim: int | None = None,
    fusion_dropout: float | None = None,
    fusion_layers: int | None = None,
    head_type: StudentHeadType = "single",
) -> TDMVQA:
    """
    Generic convenience builder for all TDM student variants.
    """
    inferred_num_classes = (
        _infer_tdm_num_classes(metadata)
        if num_classes is None
        else int(num_classes)
    )

    inferred_num_templates = (
        _infer_tdm_num_question_templates(metadata)
        if num_question_templates is None
        else int(num_question_templates)
    )

    config = make_tdm_config(
        variant=variant,
        num_classes=inferred_num_classes,
        num_question_templates=inferred_num_templates,
        question_template_embed_dim=question_template_embed_dim,
        image_channels=image_channels,
        fusion_hidden_dim=fusion_hidden_dim,
        fusion_dropout=fusion_dropout,
        fusion_layers=fusion_layers,
        head_type=head_type,
    )

    if variant == "tdm_m":
        return TDMMVQA(config)

    return TDMSVQA(config)


def build_tdm_xxs_from_metadata(
    metadata: dict,
    num_classes: int | None = None,
    num_question_templates: int | None = None,
    head_type: StudentHeadType = "single",
) -> TDMVQA:
    return build_tdm_from_metadata(
        metadata=metadata,
        variant="tdm_xxs",
        num_classes=num_classes,
        num_question_templates=num_question_templates,
        head_type=head_type,
    )


def build_tdm_xs_from_metadata(
    metadata: dict,
    num_classes: int | None = None,
    num_question_templates: int | None = None,
    head_type: StudentHeadType = "single",
) -> TDMVQA:
    return build_tdm_from_metadata(
        metadata=metadata,
        variant="tdm_xs",
        num_classes=num_classes,
        num_question_templates=num_question_templates,
        head_type=head_type,
    )


def build_tdm_s_from_metadata(
    metadata: dict,
    num_classes: int | None = None,
    num_question_templates: int | None = None,
    question_template_embed_dim: int = 32,
    fusion_hidden_dim: int = 192,
    fusion_dropout: float = 0.1,
    head_type: StudentHeadType = "single",
) -> TDMSVQA:
    return build_tdm_from_metadata(
        metadata=metadata,
        variant="tdm_s",
        num_classes=num_classes,
        num_question_templates=num_question_templates,
        question_template_embed_dim=question_template_embed_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        fusion_dropout=fusion_dropout,
        head_type=head_type,
    )  # type: ignore[return-value]


def build_tdm_m_from_metadata(
    metadata: dict,
    num_classes: int | None = None,
    num_question_templates: int | None = None,
    question_template_embed_dim: int = 64,
    fusion_hidden_dim: int = 384,
    fusion_dropout: float = 0.15,
    head_type: StudentHeadType = "single",
) -> TDMMVQA:
    return build_tdm_from_metadata(
        metadata=metadata,
        variant="tdm_m",
        num_classes=num_classes,
        num_question_templates=num_question_templates,
        question_template_embed_dim=question_template_embed_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        fusion_dropout=fusion_dropout,
        head_type=head_type,
    )  # type: ignore[return-value]


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
        lines.append(f"Question encoder:  {model.config.question_encoder}")
        lines.append(f"Question dim:      {model.question_encoder.output_dim}")
        lines.append(f"Num classes:       {model.config.num_classes}")
        lines.append(f"Count aux:         {model.config.use_count_aux}")

        if model.config.use_count_aux:
            lines.append(f"Count classes:     {model.config.num_count_classes}")

    if isinstance(model, TDMVQA):
        lines.append(f"Student variant:   {model.config.variant}")
        lines.append(f"Head type:         {model.config.head_type}")
        lines.append(f"Image channels:    {model.config.image_channels}")
        lines.append(f"Image feature dim: {model.image_encoder.feature_dim}")
        lines.append(f"Template emb dim:  {model.question_encoder.output_dim}")
        lines.append(f"Fusion hidden dim: {model.config.fusion_hidden_dim}")
        lines.append(f"Fusion layers:     {model.config.fusion_layers}")
        lines.append(f"Num classes:       {model.config.num_classes}")

    return "\n".join(lines)