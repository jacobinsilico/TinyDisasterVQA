from typing import Sequence
import torch
import torch.nn as nn
from src.models.blocks import make_mlp


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
