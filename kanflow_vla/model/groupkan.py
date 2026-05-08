"""
GroupKAN (Grouped Kolmogorov-Arnold Network) with Channel Affinity Modulation.

Implements the GroupKAN module from KAN-We-Flow (arXiv:2602.01115v2, §III-B2).

KAN replaces fixed activation functions (ReLU, GELU) with learnable univariate
spline functions placed on edges of the computation graph. GroupKAN partitions
channels into G groups, processes each with an independent KAN operator sharing
parameters across time, then applies Channel Affinity Modulation (CAM) for
adaptive per-channel gating.

This achieves extremely compact multivariate function approximation with
drastically fewer parameters than standard MLPs.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BSplineBasis(nn.Module):
    """
    B-spline basis function evaluator.

    Computes B-spline basis values for input x on a uniform grid,
    used as the foundation for learnable edge functions in KAN.

    Args:
        num_knots: Number of internal grid knots (G_grid in paper).
        spline_order: Order of B-spline (k). Cubic = 3.
        grid_range: (min, max) range for the uniform knot grid.
    """

    def __init__(
        self,
        num_knots: int = 8,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        self.num_knots = num_knots
        self.spline_order = spline_order
        self.num_bases = num_knots + spline_order  # total basis functions

        # Create uniform knot vector with boundary extension
        # Extended grid: add `spline_order` knots on each side
        h = (grid_range[1] - grid_range[0]) / num_knots
        extended_grid = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            num_knots + 2 * spline_order + 1,
        )
        self.register_buffer("grid", extended_grid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate B-spline basis functions at points x.

        Args:
            x: (...) arbitrary shape tensor of evaluation points.

        Returns:
            bases: (..., num_bases) B-spline basis values.
        """
        x = x.unsqueeze(-1)  # (..., 1)
        grid = self.grid        # (num_knots + 2k + 1,)

        # Order-0 basis (piecewise constant)
        # B_i^0(x) = 1 if grid[i] <= x < grid[i+1], else 0
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()  # (..., num_bases)

        # Recursively build higher-order bases via Cox-de Boor recursion
        for k in range(1, self.spline_order + 1):
            # Left term: (x - grid[i]) / (grid[i+k] - grid[i]) * B_i^{k-1}
            left_num = x - grid[:-(k + 1)]
            left_den = grid[k:-1] - grid[:-(k + 1)]
            left_den = left_den.clamp(min=1e-8)
            left = (left_num / left_den) * bases[..., :-1]

            # Right term: (grid[i+k+1] - x) / (grid[i+k+1] - grid[i+1]) * B_{i+1}^{k-1}
            right_num = grid[k + 1:] - x
            right_den = grid[k + 1:] - grid[1:-(k)]
            right_den = right_den.clamp(min=1e-8)
            right = (right_num / right_den) * bases[..., 1:]

            bases = left + right

        return bases  # (..., num_knots + spline_order)


class KANLayer(nn.Module):
    """
    Single KAN (Kolmogorov-Arnold Network) layer.

    Replaces a standard linear layer `y = Wx + b` with learned univariate
    spline functions on each edge:
      y_j = Σ_i ϕ_{j,i}(x_i)

    where ϕ_{j,i} is a B-spline parameterized by learnable control points.

    Following the paper (§III-B2, Eq.9), each edge function is:
      ϕ(x) = Σ_b c_b · B_b(x)  (B-spline expansion)

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        num_knots: Number of grid knots for B-spline basis.
        spline_order: B-spline order (3 = cubic).
        grid_range: Range for the knot grid.
        residual_weight: Weight for linear residual connection (SiLU baseline).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_knots: int = 8,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        residual_weight: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.residual_weight = residual_weight

        # B-spline basis evaluator
        self.basis = BSplineBasis(num_knots, spline_order, grid_range)
        num_bases = self.basis.num_bases

        # Learnable coefficients for each edge's spline function
        # Shape: (out_features, in_features, num_bases)
        self.coefficients = nn.Parameter(
            torch.empty(out_features, in_features, num_bases)
        )

        # Linear residual (base function + scale)
        self.residual_linear = nn.Linear(in_features, out_features, bias=False)
        self.scale = nn.Parameter(torch.ones(out_features))

        self._init_weights()

    def _init_weights(self):
        """Initialize spline coefficients with small random values."""
        # Small init so splines start near-linear
        nn.init.normal_(self.coefficients, std=0.1 / math.sqrt(self.in_features))
        nn.init.xavier_uniform_(self.residual_linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., in_features) input tensor.

        Returns:
            y: (..., out_features) output tensor.
        """
        batch_shape = x.shape[:-1]
        in_dim = x.shape[-1]

        # Evaluate B-spline basis for each input feature
        # x: (..., in_features) → bases: (..., in_features, num_bases)
        bases = self.basis(x)

        # Compute spline outputs via Einstein summation
        # coefficients: (out, in, num_bases)
        # bases: (..., in, num_bases)
        # result: (..., out)
        x_flat = bases.reshape(-1, in_dim, bases.shape[-1])
        spline_out = torch.einsum(
            "oin,bin->bo", self.coefficients, x_flat
        )
        spline_out = spline_out.reshape(*batch_shape, self.out_features)

        # Add scaled linear residual (SiLU-based base function)
        residual = self.residual_linear(F.silu(x))
        output = self.scale * (spline_out + self.residual_weight * residual)

        return output


class ChannelAffinityModulation(nn.Module):
    """
    Channel Affinity Modulation (CAM).

    Implements Eq.12 from the paper:
      X̄ = (1/T) Σ_t X[:,t,:]          (temporal mean pooling)
      a = σ(W₂ · φ(W₁ · X̄))           (gating vector)
      output = Y ⊙ A                   (broadcast element-wise)

    where φ = SiLU and σ = sigmoid.

    CAM adaptively reweights channels based on sequence-level statistics,
    highlighting task-relevant features.

    Args:
        dim: Channel dimension C.
        reduction: Bottleneck reduction factor for the gating MLP.
    """

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(dim // reduction, 16)
        self.gate = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, dim, bias=False),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.gate[0].weight)
        # Initialize last layer near-identity (sigmoid(0) = 0.5)
        nn.init.zeros_(self.gate[2].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) input sequence.

        Returns:
            gate: (B, T, C) per-channel gating weights (broadcast over T).
        """
        # Temporal mean pooling: (B, T, C) → (B, C)
        x_pooled = x.mean(dim=1)

        # Compute gating vector: (B, C)
        a = self.gate(x_pooled)

        # Broadcast to (B, T, C)
        return a.unsqueeze(1).expand_as(x)


class GroupKAN(nn.Module):
    """
    GroupKAN: Grouped Kolmogorov-Arnold Network with Channel Affinity Modulation.

    Implements §III-B2 from KAN-We-Flow:
      1. Partition channels into G equal groups
      2. Process each group with an independent KAN operator
      3. Concatenate outputs
      4. Apply Channel Affinity Modulation (CAM) for adaptive gating

    Using groups reduces parameters from O(C²) to O(G·(C/G)²) = O(C²/G)
    while CAM compensates for the lost cross-group interaction.

    Args:
        dim: Channel dimension C (must be divisible by num_groups).
        num_groups: Number of groups G (default: 4, as in paper).
        num_knots: Knot count for B-spline basis.
        spline_order: B-spline order (cubic = 3).
        cam_reduction: CAM bottleneck reduction ratio.
        drop_path: Stochastic depth rate.
    """

    def __init__(
        self,
        dim: int,
        num_groups: int = 4,
        num_knots: int = 8,
        spline_order: int = 3,
        cam_reduction: int = 4,
        drop_path: float = 0.0,
    ):
        super().__init__()
        assert dim % num_groups == 0, (
            f"dim={dim} must be divisible by num_groups={num_groups}"
        )
        self.dim = dim
        self.num_groups = num_groups
        self.group_dim = dim // num_groups

        # Independent KAN operator per group (Eq.10)
        self.group_kans = nn.ModuleList([
            KANLayer(
                in_features=self.group_dim,
                out_features=self.group_dim,
                num_knots=num_knots,
                spline_order=spline_order,
            )
            for _ in range(num_groups)
        ])

        # Channel Affinity Modulation (Eq.12-13)
        self.cam = ChannelAffinityModulation(dim, reduction=cam_reduction)

        # Layer norm for post-KAN normalization
        self.norm = nn.LayerNorm(dim)

        # Drop path for regularization
        self.drop_path = _make_drop_path(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) input sequence.

        Returns:
            output: (B, T, C) GroupKAN output with CAM gating.
        """
        B, T, C = x.shape

        # Split into G groups along channel dim (Eq.10)
        groups = torch.chunk(x, self.num_groups, dim=-1)  # list of (B, T, C/G)

        # Process each group with independent KAN (Eq.10)
        y_groups = []
        for kan, group in zip(self.group_kans, groups):
            # Flatten B,T for KAN processing, then reshape back
            flat = group.reshape(B * T, self.group_dim)
            out = kan(flat)
            y_groups.append(out.reshape(B, T, self.group_dim))

        # Concatenate group outputs (Eq.11)
        y = torch.cat(y_groups, dim=-1)  # (B, T, C)

        # Apply CAM gating (Eq.13)
        cam_weights = self.cam(x)  # (B, T, C)
        y = cam_weights * y

        # Residual connection with normalization and drop path (Eq.14)
        output = x + self.drop_path(self.norm(y))

        return output


def _make_drop_path(p: float) -> nn.Module:
    """Create DropPath or Identity based on probability."""
    if p > 0.0:
        from kanflow_vla.model.rwkv import DropPath
        return DropPath(p)
    return nn.Identity()
