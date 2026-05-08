"""
RWKV (Receptance Weighted Key Value) modules for KANFlow-VLA.

Implements the RWKV time-mixing and channel-mixing operations from
KAN-We-Flow (arXiv:2602.01115v2, §III-B1).

RWKV achieves transformer-level sequence modeling with O(T) linear
complexity via receptance-gated exponential decay aggregation.
Bidirectional scanning is used (forward + backward) for full trajectory
context, following the paper's Eq.3.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RWKVTimeMixing(nn.Module):
    """
    RWKV Time-Mixing with bidirectional scan.

    Given token x_t ∈ ℝ^C, we:
      1. Compute shifted input x̃_t = Shift(x_t)  (temporal shift by 1)
      2. Project to r_t, k_t, v_t via learned W_r, W_k, W_v
      3. Aggregate values with per-channel exponential time-decay w ∈ ℝ^C
         and a separate current-token boost u ∈ ℝ^C
      4. Run both forward and backward scans, combine them
      5. Gate output by receptance σ(r_t) and project via W_o

    This replaces quadratic self-attention with O(T) linear complexity.

    Args:
        dim: Channel dimension C.
        shift_amount: Number of positions to shift for temporal mixing (default: 1).
    """

    def __init__(self, dim: int, shift_amount: int = 1):
        super().__init__()
        self.dim = dim
        self.shift_amount = shift_amount

        # Projections for receptance, key, value
        self.W_r = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.W_o = nn.Linear(dim, dim, bias=False)

        # Per-channel exponential time-decay (learnable log-space for stability)
        # Initialized such that w ∈ (0, 1) to ensure stable exponential decay
        self.w_log = nn.Parameter(torch.zeros(dim))
        nn.init.uniform_(self.w_log, -2.0, 0.0)

        # Current-token boost parameter
        self.u = nn.Parameter(torch.zeros(dim))
        nn.init.uniform_(self.u, -1.0, 1.0)

        # Mixing coefficients for shifted vs current token
        self.mix_r = nn.Parameter(torch.ones(dim) * 0.5)
        self.mix_k = nn.Parameter(torch.ones(dim) * 0.5)
        self.mix_v = nn.Parameter(torch.ones(dim) * 0.5)

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for projection matrices."""
        for module in [self.W_r, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)

    def _temporal_shift(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shift sequence by `shift_amount` positions along time axis.

        Args:
            x: (B, T, C) input tensor.

        Returns:
            Shifted tensor (B, T, C) with zero-padding at the start.
        """
        if self.shift_amount <= 0:
            return x
        # Pad beginning with zeros, drop the last `shift_amount` steps
        padding = torch.zeros(
            x.shape[0], self.shift_amount, x.shape[2],
            device=x.device, dtype=x.dtype
        )
        return torch.cat([padding, x[:, :-self.shift_amount, :]], dim=1)

    def _mix_tokens(self, x: torch.Tensor, x_shifted: torch.Tensor) -> tuple:
        """
        Linearly mix current and shifted tokens for r, k, v inputs.

        This is the "token shift" mechanism that provides RWKV with
        local temporal context at negligible cost.
        """
        r_in = x * self.mix_r + x_shifted * (1.0 - self.mix_r)
        k_in = x * self.mix_k + x_shifted * (1.0 - self.mix_k)
        v_in = x * self.mix_v + x_shifted * (1.0 - self.mix_v)
        return r_in, k_in, v_in

    def _wkv_scan(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        reverse: bool = False,
    ) -> torch.Tensor:
        """
        WKV aggregation via sequential scan (linear complexity).

        Implements Eq.2 from the paper:
          ṽ_t = [Σ_{i<t} exp(-(t-1-i)w + k_i) ⊙ v_i + exp(u + k_t) ⊙ v_t]
                 / [Σ_{i<t} exp(-(t-1-i)w + k_i) + exp(u + k_t)]

        Args:
            k: (B, T, C) key tensor.
            v: (B, T, C) value tensor.
            w: (C,) per-channel decay rates (positive, from exp of w_log).
            u: (C,) current-token boost.
            reverse: If True, scan from T→1 (backward pass).

        Returns:
            wkv: (B, T, C) aggregated values.
        """
        B, T, C = k.shape

        if reverse:
            k = k.flip(1)
            v = v.flip(1)

        # Initialize running numerator and denominator
        # Using log-space accumulation for numerical stability
        output = torch.zeros_like(v)

        # Running state: numerator and denominator in linear space
        num = torch.zeros(B, C, device=k.device, dtype=k.dtype)
        den = torch.zeros(B, C, device=k.device, dtype=k.dtype)

        for t in range(T):
            k_t = k[:, t, :]  # (B, C)
            v_t = v[:, t, :]  # (B, C)

            # Current-token contribution: exp(u + k_t) * v_t
            uk = torch.exp(torch.clamp(u + k_t, max=30.0))  # (B, C)
            current_num = uk * v_t
            current_den = uk

            # Combined aggregation
            total_num = num + current_num
            total_den = den + current_den + 1e-8  # avoid div by zero

            output[:, t, :] = total_num / total_den

            # Update running state with exponential decay
            # num_{t+1} = exp(-w) * num_t + exp(k_t) * v_t
            decay = torch.exp(-torch.clamp(w, min=0.0, max=10.0))
            ek = torch.exp(torch.clamp(k_t, max=30.0))
            num = decay * num + ek * v_t
            den = decay * den + ek

        if reverse:
            output = output.flip(1)

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        RWKV Time-Mixing forward pass.

        Args:
            x: (B, T, C) input sequence.

        Returns:
            output: (B, T, C) time-mixed output.
        """
        # Temporal shift
        x_shifted = self._temporal_shift(x)

        # Mix current and shifted tokens
        r_in, k_in, v_in = self._mix_tokens(x, x_shifted)

        # Project to r, k, v
        r = self.W_r(r_in)  # receptance
        k = self.W_k(k_in)  # key
        v = self.W_v(v_in)  # value

        # Per-channel decay (ensure positive via exp)
        w = torch.exp(self.w_log)

        # Bidirectional WKV aggregation (Eq.3 from paper)
        wkv_fwd = self._wkv_scan(k, v, w, self.u, reverse=False)
        wkv_bwd = self._wkv_scan(k, v, w, self.u, reverse=True)
        wkv = wkv_fwd + wkv_bwd  # combine forward and backward scans

        # Gate by receptance and project
        output = self.W_o(torch.sigmoid(r) * wkv)

        return output


class RWKVChannelMixing(nn.Module):
    """
    RWKV Channel-Mixing with squared-ReLU gating.

    Implements Eq.5 from the paper:
      r'_t = W'_r · S(x_t)
      k'_t = W'_k · S(x_t)
      CM(x_t) = σ(r'_t) ⊙ (W'_v · max(k'_t, 0)²)

    This is a token-wise gated MLP that processes each position independently,
    providing per-channel nonlinear mixing complementary to time-mixing.

    Args:
        dim: Channel dimension C.
        expand_factor: Hidden dimension multiplier for the MLP (default: 4).
    """

    def __init__(self, dim: int, expand_factor: int = 4):
        super().__init__()
        hidden = dim * expand_factor

        self.W_r = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, hidden, bias=False)
        self.W_v = nn.Linear(hidden, dim, bias=False)

        # Mixing coefficients for shifted vs current token
        self.mix_r = nn.Parameter(torch.ones(dim) * 0.5)
        self.mix_k = nn.Parameter(torch.ones(dim) * 0.5)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_r.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        # Zero-init output projection for clean residual at start
        nn.init.zeros_(self.W_v.weight)

    def _temporal_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Shift by 1 position along time."""
        padding = torch.zeros(
            x.shape[0], 1, x.shape[2],
            device=x.device, dtype=x.dtype
        )
        return torch.cat([padding, x[:, :-1, :]], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Channel-Mixing forward pass.

        Args:
            x: (B, T, C) input sequence.

        Returns:
            output: (B, T, C) channel-mixed output.
        """
        x_shifted = self._temporal_shift(x)

        # Mix for receptance and key
        r_in = x * self.mix_r + x_shifted * (1.0 - self.mix_r)
        k_in = x * self.mix_k + x_shifted * (1.0 - self.mix_k)

        # Receptance gate
        r = torch.sigmoid(self.W_r(r_in))

        # Squared ReLU (key nonlinearity from paper)
        k = F.relu(self.W_k(k_in)).square()

        # Gated output
        output = r * self.W_v(k)

        return output


class RWKVBlock(nn.Module):
    """
    Full RWKV block: pre-norm residual wrapping Time-Mixing + Channel-Mixing.

    Implements Eq.6 from the paper:
      z_t = x_t + CM(LN₂(x_t + TM(LN₁(x_t))))

    Args:
        dim: Channel dimension C.
        expand_factor: Hidden dimension multiplier for channel-mixing MLP.
        drop_path: Stochastic depth rate (default: 0.0).
    """

    def __init__(
        self,
        dim: int,
        expand_factor: int = 4,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.time_mixing = RWKVTimeMixing(dim)
        self.channel_mixing = RWKVChannelMixing(dim, expand_factor)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) input sequence.

        Returns:
            output: (B, T, C) processed sequence.
        """
        # Time-mixing with pre-norm residual
        x = x + self.drop_path(self.time_mixing(self.ln1(x)))
        # Channel-mixing with pre-norm residual
        x = x + self.drop_path(self.channel_mixing(self.ln2(x)))
        return x


class DropPath(nn.Module):
    """
    Stochastic Depth (DropPath) for regularization.

    Drops entire residual branches during training with probability `p`.
    """

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        # Random tensor with shape (B, 1, 1, ...) for broadcasting
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor

    def extra_repr(self) -> str:
        return f"p={self.p}"
