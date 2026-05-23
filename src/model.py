"""
Baseline VQA model for pruned COCO-QA.

Task:
  image + question + question type -> 70-class answer prediction

Architecture:
  - small CNN image encoder
  - word embedding + masked mean pooling question encoder
  - question type embedding
  - MLP classifier

This is intentionally simple for the first baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class BaselineVQAModel(nn.Module):
    """
    Simple baseline model.

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
        image_feature_dim: int = 256,
        question_embedding_dim: int = 128,
        question_feature_dim: int = 128,
        type_embedding_dim: int = 16,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.image_encoder = SmallCNNEncoder(
            image_feature_dim=image_feature_dim,
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

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_answers),
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

        logits = self.classifier(fused)
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
) -> BaselineVQAModel:
    """
    Convenience builder with default baseline hyperparameters.
    """
    return BaselineVQAModel(
        vocab_size=vocab_size,
        num_answers=num_answers,
        pad_id=pad_id,
    )