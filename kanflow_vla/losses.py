"""
Combined loss functions for KANFlow-VLA training.

The total training loss is:
  L = L_CFM + λ_ACR · L_ACR
    + λ_mode · L_mode
    + λ_failure · L_failure
    + λ_progress · L_progress

where:
  - L_CFM: Multi-segment Consistency Flow Matching loss (endpoint + velocity)
  - L_ACR: Action Consistency Regularization (expert anchoring)
  - L_mode: Cross-entropy over act/ask/abstain/recover labels
  - L_failure: Cross-entropy over failure-type labels
  - L_progress: Smooth-L1 over normalised episode progress

Both L_CFM and L_ACR are computed inside ConsistencyFlowMatching.compute_loss().
This module adds logging utilities and optional auxiliary losses.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANFlowLoss(nn.Module):
    """Combined loss wrapper with logging and auxiliary regularization.

    The primary CFM + ACR loss is computed by the flow matching module.
    This wrapper adds optional:
      - Action smoothness regularization (temporal consistency)
      - Velocity field magnitude penalty (prevent exploding velocities)
      - RCAR behaviour head losses (mode, failure, progress)

    Args:
        lambda_smooth:       Weight for action smoothness loss.
        lambda_vel_reg:      Weight for velocity magnitude regularization.
        gripper_weight_mult: Extra weight multiplier for gripper dimension.
        gripper_dim:         Index of the gripper dimension in action space (default: -1).
        lambda_mode:         Weight for mode classification loss (0 = disabled).
        lambda_failure:      Weight for failure-type classification loss (0 = disabled).
        lambda_progress:     Weight for progress regression loss (0 = disabled).
    """

    def __init__(
        self,
        lambda_smooth: float = 0.0,
        lambda_vel_reg: float = 0.0,
        gripper_weight_mult: float = 3.0,
        gripper_dim: int = -1,
        # RCAR auxiliary loss weights
        lambda_mode: float = 0.0,
        lambda_failure: float = 0.0,
        lambda_progress: float = 0.0,
    ):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_vel_reg = lambda_vel_reg
        self.gripper_weight_mult = gripper_weight_mult
        self.gripper_dim = gripper_dim
        self.lambda_mode = lambda_mode
        self.lambda_failure = lambda_failure
        self.lambda_progress = lambda_progress

    def forward(
        self,
        cfm_loss_dict: dict[str, torch.Tensor],
        predicted_actions: torch.Tensor | None = None,
        expert_actions: torch.Tensor | None = None,
        # RCAR head outputs from model
        mode_logits: torch.Tensor | None = None,
        failure_logits: torch.Tensor | None = None,
        progress_pred: torch.Tensor | None = None,
        # RCAR ground-truth labels from batch
        mode_labels: torch.Tensor | None = None,
        failure_labels: torch.Tensor | None = None,
        progress_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute total loss with optional auxiliary terms.

        Args:
            cfm_loss_dict: Dictionary from ConsistencyFlowMatching.compute_loss()
                containing 'loss', 'cfm_loss', 'acr_loss', 'velocity_loss'.
            predicted_actions: (B, H, A) predicted actions (for smoothness).
            expert_actions: (B, H, A) expert actions (for gripper weighting).
            mode_logits: (B, 4) mode classification logits.
            failure_logits: (B, 9) failure type classification logits.
            progress_pred: (B,) predicted normalised progress in [0, 1].
            mode_labels: (B,) ground-truth mode class indices.
            failure_labels: (B,) ground-truth failure class indices.
            progress_labels: (B,) ground-truth normalised progress in [0, 1].

        Returns:
            Dictionary with all loss components and total loss.
        """
        total_loss = cfm_loss_dict["loss"]
        output = dict(cfm_loss_dict)

        # ── Action Smoothness Regularization ──
        if self.lambda_smooth > 0.0 and predicted_actions is not None:
            if predicted_actions.shape[1] > 1:
                diffs = predicted_actions[:, 1:, :] - predicted_actions[:, :-1, :]
                smooth_loss = diffs.pow(2).mean()
                total_loss = total_loss + self.lambda_smooth * smooth_loss
                output["smooth_loss"] = smooth_loss

        # ── RCAR: Mode Classification Loss ──
        if (
            self.lambda_mode > 0.0
            and mode_logits is not None
            and mode_labels is not None
        ):
            mode_loss = F.cross_entropy(mode_logits, mode_labels)
            total_loss = total_loss + self.lambda_mode * mode_loss
            output["mode_loss"] = mode_loss

        # ── RCAR: Failure Type Classification Loss ──
        if (
            self.lambda_failure > 0.0
            and failure_logits is not None
            and failure_labels is not None
        ):
            failure_loss = F.cross_entropy(failure_logits, failure_labels)
            total_loss = total_loss + self.lambda_failure * failure_loss
            output["failure_loss"] = failure_loss

        # ── RCAR: Progress Regression Loss ──
        if (
            self.lambda_progress > 0.0
            and progress_pred is not None
            and progress_labels is not None
        ):
            progress_labels_f = progress_labels.float().to(progress_pred.device)
            progress_loss = F.smooth_l1_loss(progress_pred, progress_labels_f)
            total_loss = total_loss + self.lambda_progress * progress_loss
            output["progress_loss"] = progress_loss

        output["total_loss"] = total_loss
        return output

