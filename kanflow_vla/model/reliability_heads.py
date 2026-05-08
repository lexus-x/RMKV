"""
Reliability behaviour heads for RCAR-VLA.

Small classification / regression heads that attach to the fused condition
vector produced by ``KANFlowVLA._encode_condition()``.  These heads predict:

- **mode**:     act (0) | ask (1) | abstain (2) | recover (3)
- **failure**:  none (0) | ambiguity (1) | wrong_object (2) | grasp_miss (3)
                | occlusion (4) | unreachable (5) | unsafe (6) | contradiction (7)
                | unknown (8)
- **progress**: normalised task progress in [0, 1]

The heads are deliberately kept in a separate module so that they can be
ablated independently from the flow-matching action decoder.
"""

from __future__ import annotations

from enum import IntEnum

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Label enums ──────────────────────────────────────────────────────────────

class Mode(IntEnum):
    ACT = 0
    ASK = 1
    ABSTAIN = 2
    RECOVER = 3

    @classmethod
    def n_classes(cls) -> int:
        return len(cls)


class FailureType(IntEnum):
    NONE = 0
    AMBIGUITY = 1
    WRONG_OBJECT = 2
    GRASP_MISS = 3
    OCCLUSION = 4
    UNREACHABLE = 5
    UNSAFE = 6
    CONTRADICTION = 7
    UNKNOWN = 8

    @classmethod
    def n_classes(cls) -> int:
        return len(cls)


# ── Head module ──────────────────────────────────────────────────────────────

class ReliabilityHeads(nn.Module):
    """Lightweight behaviour prediction heads.

    Args:
        d_model:   Dimension of the fused condition vector.
        d_hidden:  Hidden dimension of the shared projection layer.
        dropout:   Dropout probability for the shared layer.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Shared projection (keeps parameter count low)
        self.shared = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Per-task output heads
        self.mode_head = nn.Linear(d_hidden, Mode.n_classes())
        self.failure_head = nn.Linear(d_hidden, FailureType.n_classes())
        self.progress_head = nn.Linear(d_hidden, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in [self.shared[0], self.mode_head, self.failure_head, self.progress_head]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        condition: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through all behaviour heads.

        Args:
            condition: ``(B, d_model)`` fused condition vector.

        Returns:
            Dictionary with keys:
                - ``mode_logits``:    ``(B, 4)``
                - ``failure_logits``: ``(B, 9)``
                - ``progress``:       ``(B,)``  in [0, 1]
        """
        h = self.shared(condition)

        return {
            "mode_logits": self.mode_head(h),
            "failure_logits": self.failure_head(h),
            "progress": self.progress_head(h).squeeze(-1).sigmoid(),
        }


# ── Inference helpers ────────────────────────────────────────────────────────

def decode_behavior(
    head_outputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor | int]:
    """Convert raw logits to structured decisions (greedy argmax).

    Intended for single-sample rollout inference (B=1).

    Returns:
        Dictionary with keys:
            - ``mode``:     int  (0-3)
            - ``failure``:  int  (0-8)
            - ``progress``: float (0-1)
            - ``mode_prob``: float — confidence of the chosen mode
    """
    mode_probs = F.softmax(head_outputs["mode_logits"], dim=-1)
    mode = int(mode_probs.argmax(dim=-1).item())

    failure = int(head_outputs["failure_logits"].argmax(dim=-1).item())
    progress = float(head_outputs["progress"].item())

    return {
        "mode": mode,
        "mode_name": Mode(mode).name.lower(),
        "failure": failure,
        "failure_name": FailureType(failure).name.lower(),
        "progress": progress,
        "mode_prob": float(mode_probs[0, mode].item()) if mode_probs.ndim == 2 else float(mode_probs[mode].item()),
    }
