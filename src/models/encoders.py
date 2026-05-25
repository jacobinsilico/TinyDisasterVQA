import torch
import torch.nn as nn


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
