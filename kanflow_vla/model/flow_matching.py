"""
Consistency Flow Matching (CFM) + Action Consistency Regularization (ACR).

Implements the training and inference procedures from KAN-We-Flow
(arXiv:2602.01115v2, §III-C, §III-D, §III-E).

Key ideas:
  - Linear interpolation flow: a_t = (1-t)·a_src + t·a_tar
  - One-step decoder: f_θ(t, a_t, c) = a_t + (1-t)·v_θ(a_t, t, c)
  - CFM consistency loss: f_θ(t) ≈ f_{θ⁻}(t+Δt) (EMA teacher)
  - Multi-segment extension: K=2 segments for better quality
  - ACR: anchors decoded actions to expert demonstrations
"""

from __future__ import annotations

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from kanflow_vla.model.rwkv_kan_unet import RWKVKANUNet


class ConsistencyFlowMatching(nn.Module):
    """
    Multi-Segment Consistency Flow Matching with Action Consistency Regularization.

    Trains the RWKV-KAN UNet to predict a velocity field that enables
    one-step action generation via explicit Euler integration.

    The key insight is that enforcing consistency between decoded actions
    at nearby time points (via an EMA teacher) yields models that can
    generate high-quality actions in a single step at test time.

    Args:
        velocity_net: The velocity prediction network (RWKVKANUNet).
        action_dim: Dimension of action space.
        horizon: Action prediction horizon.
        num_segments: K segments for multi-segment CFM (default: 2).
        ema_decay: EMA decay for teacher network (default: 0.95).
        delta_t: Time step for consistency loss (default: 0.01).
        lambda_acr: Weight for ACR loss (default: 1.0).
        alpha_consistency: Weight for velocity consistency term (default: 1.0).
        obs_length: Number of observation steps (default: 2).
        inference_t: Time point for inference decode (default: 0.0).
    """

    def __init__(
        self,
        velocity_net: RWKVKANUNet,
        action_dim: int = 4,
        horizon: int = 4,
        num_segments: int = 2,
        ema_decay: float = 0.95,
        delta_t: float = 0.01,
        lambda_acr: float = 1.0,
        alpha_consistency: float = 1.0,
        obs_length: int = 2,
        inference_t: float = 0.0,
    ):
        super().__init__()
        self.velocity_net = velocity_net
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_segments = num_segments
        self.ema_decay = ema_decay
        self.delta_t = delta_t
        self.lambda_acr = lambda_acr
        self.alpha_consistency = alpha_consistency
        self.obs_length = obs_length
        self.inference_t = inference_t

        # EMA teacher network (θ⁻)
        self.teacher_net = deepcopy(velocity_net)
        for p in self.teacher_net.parameters():
            p.requires_grad = False

        # Action prediction window for ACR
        # W = {obs_length-1, ..., obs_length-1+horizon-1}
        self.action_window_start = obs_length - 1
        self.action_window_end = obs_length - 1 + horizon

    @torch.no_grad()
    def update_ema(self):
        """
        Update EMA teacher parameters: θ⁻ ← μ·θ⁻ + (1-μ)·θ

        Call after each optimizer step.
        """
        for p_teacher, p_student in zip(
            self.teacher_net.parameters(),
            self.velocity_net.parameters(),
        ):
            p_teacher.data.mul_(self.ema_decay).add_(
                p_student.data, alpha=1.0 - self.ema_decay
            )

    def _interpolate(
        self,
        a_src: torch.Tensor,
        a_tar: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Linear interpolation between source and target actions (Eq.15).

        a_t = (1-t)·a_src + t·a_tar

        Args:
            a_src: (B, horizon, action_dim) source (noise) actions.
            a_tar: (B, horizon, action_dim) target (expert) actions.
            t: (B, 1, 1) time values in [0, 1].

        Returns:
            a_t: (B, horizon, action_dim) interpolated actions.
        """
        return (1.0 - t) * a_src + t * a_tar

    def _one_step_decode(
        self,
        a_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        use_teacher: bool = False,
    ) -> torch.Tensor:
        """
        One-step Euler decode from time t to t=1 (Eq.16).

        f_θ(t, a_t, c) = a_t + (1-t)·v_θ(a_t, t, c)

        Args:
            a_t: (B, horizon, action_dim) current noisy actions.
            t: (B,) time values.
            condition: (B, cond_dim) condition vector.
            use_teacher: If True, use EMA teacher network.

        Returns:
            decoded: (B, horizon, action_dim) decoded actions at t=1.
        """
        net = self.teacher_net if use_teacher else self.velocity_net
        velocity = net(a_t, t, condition)  # (B, horizon, action_dim)

        # Reshape t for broadcasting
        t_expand = t.view(-1, 1, 1)  # (B, 1, 1)

        # Euler step: a_t + (1-t) * v_θ
        decoded = a_t + (1.0 - t_expand) * velocity

        return decoded

    def compute_loss(
        self,
        expert_actions: torch.Tensor,
        condition: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute the full training loss: L = L_CFM + λ_ACR · L_ACR

        Args:
            expert_actions: (B, horizon, action_dim) expert demonstration actions.
            condition: (B, cond_dim) fused VL condition vector.

        Returns:
            Dictionary with:
              - loss: total loss
              - cfm_loss: consistency flow matching loss
              - acr_loss: action consistency regularization loss
              - velocity_loss: instantaneous velocity alignment loss
        """
        B = expert_actions.shape[0]
        device = expert_actions.device

        # Sample source noise
        a_src = torch.randn_like(expert_actions)

        # ── Multi-Segment CFM Loss (Eq.19) ──
        cfm_loss = torch.tensor(0.0, device=device)
        velocity_loss = torch.tensor(0.0, device=device)

        for seg_idx in range(self.num_segments):
            # Segment boundaries
            t_low = seg_idx / self.num_segments
            t_high = (seg_idx + 1) / self.num_segments - self.delta_t

            if t_high <= t_low:
                t_high = t_low + self.delta_t

            # Sample time t uniformly within segment (Eq.19)
            t = torch.rand(B, device=device) * (t_high - t_low) + t_low
            t_next = t + self.delta_t

            # Clamp to valid range
            t = t.clamp(0.0, 1.0 - self.delta_t)
            t_next = t_next.clamp(self.delta_t, 1.0)

            # Interpolate at t and t+Δt
            t_expand = t.view(B, 1, 1)
            t_next_expand = t_next.view(B, 1, 1)

            a_t = self._interpolate(a_src, expert_actions, t_expand)
            a_t_next = self._interpolate(a_src, expert_actions, t_next_expand)

            # One-step decode from student at t
            f_student = self._one_step_decode(a_t, t, condition, use_teacher=False)

            # One-step decode from teacher at t+Δt
            with torch.no_grad():
                f_teacher = self._one_step_decode(
                    a_t_next, t_next, condition, use_teacher=True
                )

            # Endpoint consistency loss: ||f_θ(t) - f_{θ⁻}(t+Δt)||² (Eq.18)
            endpoint_loss = F.mse_loss(f_student, f_teacher)

            # Velocity consistency: align instantaneous velocities
            v_student = self.velocity_net(a_t, t, condition)
            with torch.no_grad():
                v_teacher = self.teacher_net(a_t_next, t_next, condition)
            vel_loss = F.mse_loss(v_student, v_teacher)

            # Segment weight (uniform for simplicity)
            seg_weight = 1.0 / self.num_segments
            cfm_loss = cfm_loss + seg_weight * (
                endpoint_loss + self.alpha_consistency * vel_loss
            )
            velocity_loss = velocity_loss + seg_weight * vel_loss

        # ── Action Consistency Regularization (Eq.24-25) ──
        # Sample a random time for ACR
        t_acr = torch.rand(B, device=device).clamp(0.01, 0.99)
        t_acr_expand = t_acr.view(B, 1, 1)

        a_t_acr = self._interpolate(a_src, expert_actions, t_acr_expand)

        # One-step decode to t=1
        f_acr = self._one_step_decode(a_t_acr, t_acr, condition, use_teacher=False)

        # ACR loss: MSE between decoded actions and expert actions (Eq.25)
        # Over the action prediction window W
        acr_loss = F.mse_loss(f_acr, expert_actions)

        # ── Total loss (Eq.26) ──
        total_loss = cfm_loss + self.lambda_acr * acr_loss

        return {
            "loss": total_loss,
            "cfm_loss": cfm_loss,
            "acr_loss": acr_loss,
            "velocity_loss": velocity_loss,
        }

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        batch_size: int | None = None,
        t: float | None = None,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """
        Generate actions via one-step decode (inference).

        Draw a₀ ~ N(0, I) and decode to t=1 via single Euler step.
        Starting from pure noise is only distributionally consistent at t=0,
        so the default inference path decodes from t=0 unless the caller
        explicitly overrides ``t``.

        Args:
            condition: (B, cond_dim) condition vector.
            batch_size: Override batch size (uses condition's B if None).
            t: Override inference time point (default: self.inference_t).
            use_ema: If True, decode with the EMA teacher network.

        Returns:
            actions: (B, horizon, action_dim) decoded action trajectory.
        """
        B = condition.shape[0] if batch_size is None else batch_size
        device = condition.device

        # Sample from source distribution: a₀ ~ N(0, I)
        a_0 = torch.randn(
            B, self.horizon, self.action_dim,
            device=device, dtype=condition.dtype,
        )

        # Pure-noise starts correspond to t=0 in training. Using a later
        # starting time here is off-distribution unless the caller also
        # constructs the matching a_t state.
        t_val = t if t is not None else self.inference_t
        t_tensor = torch.full((B,), t_val, device=device, dtype=condition.dtype)

        return self._one_step_decode(
            a_0, t_tensor, condition, use_teacher=use_ema
        )

    @torch.no_grad()
    def sample_multistep(
        self,
        condition: torch.Tensor,
        num_steps: int = 2,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """
        Multi-step inference for higher quality (optional).

        Divides [0, 1] into num_steps segments and performs Euler integration.

        Args:
            condition: (B, cond_dim) condition vector.
            num_steps: Number of integration steps.
            use_ema: If True, integrate with the EMA teacher network.

        Returns:
            actions: (B, horizon, action_dim) decoded action trajectory.
        """
        B = condition.shape[0]
        device = condition.device

        # Start from pure noise
        a = torch.randn(
            B, self.horizon, self.action_dim,
            device=device, dtype=condition.dtype,
        )

        dt = 1.0 / num_steps
        net = self.teacher_net if use_ema else self.velocity_net
        for step in range(num_steps):
            t_val = step * dt
            t_tensor = torch.full(
                (B,), t_val, device=device, dtype=condition.dtype,
            )
            velocity = net(a, t_tensor, condition)
            a = a + dt * velocity

        return a
