"""
RWKV-KAN U-Net: 3-stage encoder-decoder backbone for velocity field prediction.

Implements the RWKV-KAN UNet from KAN-We-Flow (arXiv:2602.01115v2, §III-B).

Architecture: 3-stage encoder-decoder with skip connections.
Each stage stacks RWKV-KAN blocks (RWKV for temporal mixing, GroupKAN for
per-channel nonlinear calibration). Conditioned on fused visual-language-state
embeddings via FiLM (Feature-wise Linear Modulation).

The UNet predicts a velocity field v_θ(a_t, t, c) for consistency flow matching.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from kanflow_vla.model.rwkv import RWKVBlock
from kanflow_vla.model.groupkan import GroupKAN


class StandardTransformerBlock(nn.Module):
    """Tiny fallback standard transformer block for 'standard_transformer' ablation."""
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return x + attn_out + self.ff(x + attn_out)


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for the flow time variable t ∈ [0, 1].

    Maps scalar time t to a d_model-dimensional vector using sine/cosine
    frequencies, following the standard diffusion/flow-matching convention.

    Args:
        dim: Output embedding dimension.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) or (B, 1) time values in [0, 1].

        Returns:
            emb: (B, dim) time embedding.
        """
        if t.ndim == 0:
            t = t.unsqueeze(0)
        if t.ndim == 2:
            t = t.squeeze(-1)

        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=t.device, dtype=t.dtype) / half_dim
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half_dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return self.mlp(emb)


class FiLMConditioner(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) for conditioning.

    Modulates features x with condition c:
      output = γ(c) * x + β(c)

    where γ and β are learned affine transforms of the condition vector.

    Args:
        feature_dim: Dimension of the features to modulate.
        cond_dim: Dimension of the condition vector.
    """

    def __init__(self, feature_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.proj = nn.Linear(cond_dim, feature_dim * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        # Initialize so that γ≈1, β≈0 (identity modulation at start)
        self.proj.bias.data[feature_dim:] = 1.0  # gamma bias

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) features to modulate.
            cond: (B, cond_dim) condition vector.

        Returns:
            modulated: (B, T, C) conditioned features.
        """
        x = self.norm(x)
        gamma_beta = self.proj(cond)  # (B, 2*C)
        beta, gamma = gamma_beta.chunk(2, dim=-1)  # each (B, C)
        gamma = gamma.unsqueeze(1)  # (B, 1, C)
        beta = beta.unsqueeze(1)    # (B, 1, C)
        return gamma * x + beta


class RWKVKANBlock(nn.Module):
    """
    Combined RWKV-KAN block with conditioning.

    Sequentially applies:
      1. RWKV block (time-mixing + channel-mixing)
      2. GroupKAN (grouped spline calibration + CAM)
      3. FiLM conditioning from task context

    This is the fundamental building block of the RWKV-KAN UNet.

    Args:
        dim: Channel dimension.
        num_groups: Number of groups for GroupKAN.
        num_knots: Knot count for B-spline basis.
        cond_dim: Condition vector dimension for FiLM.
        drop_path: Stochastic depth rate.
    """

    def __init__(
        self,
        dim: int,
        num_groups: int = 4,
        num_knots: int = 8,
        cond_dim: int = 256,
        drop_path: float = 0.0,
        ablation: str | None = None,
    ):
        super().__init__()
        self.ablation = ablation
        
        if ablation == 'standard_transformer':
            self.transformer = StandardTransformerBlock(dim)
        elif ablation == 'disable_groupkan':
            self.rwkv = RWKVBlock(dim=dim, drop_path=drop_path)
        elif ablation == 'disable_rwkv':
            self.groupkan = GroupKAN(
                dim=dim,
                num_groups=num_groups,
                num_knots=num_knots,
                drop_path=drop_path,
            )
        else:
            self.rwkv = RWKVBlock(dim=dim, drop_path=drop_path)
            self.groupkan = GroupKAN(
                dim=dim,
                num_groups=num_groups,
                num_knots=num_knots,
                drop_path=drop_path,
            )
            
        self.film = FiLMConditioner(dim, cond_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) input sequence.
            cond: (B, cond_dim) condition vector (VL features + time embedding).

        Returns:
            output: (B, T, C) processed sequence.
        """
        if self.ablation == 'standard_transformer':
            x = self.transformer(x)
        elif self.ablation == 'disable_groupkan':
            x = self.rwkv(x)
        elif self.ablation == 'disable_rwkv':
            x = self.groupkan(x)
        else:
            x = self.rwkv(x)
            x = self.groupkan(x)
            
        # FiLM: condition on task context
        x = self.film(x, cond)
        return x


class Downsample1D(nn.Module):
    """1D downsampling via strided convolution (T → T/2)."""

    def __init__(self, dim: int, dim_out: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim_out, kernel_size=3, stride=2, padding=1)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → (B, T/2, C_out)"""
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.conv(x)       # (B, C_out, T/2)
        return x.transpose(1, 2)


class Upsample1D(nn.Module):
    """1D upsampling via transposed convolution (T → 2T)."""

    def __init__(self, dim: int, dim_out: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            dim, dim_out, kernel_size=4, stride=2, padding=1
        )
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → (B, 2T, C_out)"""
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.conv(x)       # (B, C_out, 2T)
        return x.transpose(1, 2)


class EncoderStage(nn.Module):
    """
    Single encoder stage: N RWKV-KAN blocks + downsample.

    Args:
        dim_in: Input channel dimension.
        dim_out: Output channel dimension (after downsample).
        num_blocks: Number of RWKV-KAN blocks in this stage.
        num_groups: GroupKAN groups.
        cond_dim: Condition dimension.
        drop_path: Base stochastic depth rate.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        num_blocks: int = 2,
        num_groups: int = 4,
        cond_dim: int = 256,
        drop_path: float = 0.0,
        ablation: str | None = None,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            RWKVKANBlock(
                dim=dim_in,
                num_groups=num_groups,
                cond_dim=cond_dim,
                drop_path=drop_path,
                ablation=ablation,
            )
            for _ in range(num_blocks)
        ])
        self.downsample = Downsample1D(dim_in, dim_out)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            down: (B, T/2, dim_out) downsampled output.
            skip: (B, T, dim_in) pre-downsample features for skip connection.
        """
        for block in self.blocks:
            x = block(x, cond)
        skip = x
        down = self.downsample(x)
        return down, skip


class DecoderStage(nn.Module):
    """
    Single decoder stage: upsample + skip concat + N RWKV-KAN blocks.

    Args:
        dim_in: Input channel dimension (from previous decoder stage).
        dim_out: Output channel dimension.
        skip_dim: Dimension of skip connection from encoder.
        num_blocks: Number of RWKV-KAN blocks.
        num_groups: GroupKAN groups.
        cond_dim: Condition dimension.
        drop_path: Stochastic depth rate.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        skip_dim: int,
        num_blocks: int = 2,
        num_groups: int = 4,
        cond_dim: int = 256,
        drop_path: float = 0.0,
        ablation: str | None = None,
    ):
        super().__init__()
        self.upsample = Upsample1D(dim_in, dim_out)
        # After concat with skip, dim = dim_out + skip_dim
        self.skip_proj = nn.Linear(dim_out + skip_dim, dim_out)
        nn.init.xavier_uniform_(self.skip_proj.weight)
        nn.init.zeros_(self.skip_proj.bias)

        self.blocks = nn.ModuleList([
            RWKVKANBlock(
                dim=dim_out,
                num_groups=num_groups,
                cond_dim=cond_dim,
                drop_path=drop_path,
                ablation=ablation,
            )
            for _ in range(num_blocks)
        ])

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T_in, dim_in) input from previous decoder stage.
            skip: (B, T_skip, skip_dim) skip connection from encoder.
            cond: (B, cond_dim) condition vector.

        Returns:
            output: (B, T_skip, dim_out) decoded features.
        """
        # Upsample
        x = self.upsample(x)  # (B, 2*T_in, dim_out)

        # Match temporal dimensions with skip (in case of rounding)
        if x.shape[1] != skip.shape[1]:
            x = x[:, :skip.shape[1], :]

        # Concatenate skip connection and project
        x = torch.cat([x, skip], dim=-1)  # (B, T, dim_out + skip_dim)
        x = self.skip_proj(x)             # (B, T, dim_out)

        # Process through RWKV-KAN blocks
        for block in self.blocks:
            x = block(x, cond)

        return x


class RWKVKANUNet(nn.Module):
    """
    RWKV-KAN UNet: 3-stage encoder-decoder for velocity field prediction.

    Architecture from KAN-We-Flow §III-B:
      Encoder: 3 stages (C → 2C → 4C) with RWKV-KAN blocks + downsample
      Bottleneck: 2 RWKV-KAN blocks at 4C
      Decoder: 3 stages (4C → 2C → C) with upsample + skip + RWKV-KAN blocks
      Output: Linear projection to action_dim

    The UNet is conditioned on:
      - Flow time t (via sinusoidal embedding)
      - Visual-language condition vector (from VL encoder)

    These are combined and injected via FiLM at every RWKV-KAN block.

    Args:
        action_dim: Dimension of the action space.
        horizon: Prediction horizon (number of action steps to predict).
        base_dim: Base channel dimension C (default: 128).
        cond_dim: Dimension of the external condition vector.
        num_blocks_per_stage: RWKV-KAN blocks per encoder/decoder stage.
        num_groups: GroupKAN groups G.
        num_knots: B-spline knot count.
        drop_path: Stochastic depth rate.
    """

    def __init__(
        self,
        action_dim: int = 4,
        horizon: int = 4,
        base_dim: int = 128,
        cond_dim: int = 256,
        num_blocks_per_stage: int = 2,
        num_groups: int = 4,
        num_knots: int = 8,
        drop_path: float = 0.0,
        ablation: str | None = None,
        domain_randomize: bool = False,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.base_dim = base_dim

        # Channel dimensions per stage
        dims = [base_dim, base_dim * 2, base_dim * 4]  # [128, 256, 512]

        # Combined condition dimension (external cond + time embedding)
        combined_cond_dim = cond_dim + base_dim  # external + time emb
        self.cond_proj = nn.Linear(cond_dim, cond_dim)
        nn.init.xavier_uniform_(self.cond_proj.weight)

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(base_dim)

        # Condition combiner
        self.cond_combine = nn.Sequential(
            nn.Linear(combined_cond_dim, combined_cond_dim),
            nn.SiLU(),
            nn.Linear(combined_cond_dim, combined_cond_dim),
        )

        # Input projection: action_dim → base_dim
        self.input_proj = nn.Linear(action_dim, base_dim)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        # ── Encoder ──
        self.encoder_stages = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.encoder_stages.append(
                EncoderStage(
                    dim_in=dims[i],
                    dim_out=dims[i + 1],
                    num_blocks=num_blocks_per_stage,
                    num_groups=num_groups,
                    cond_dim=combined_cond_dim,
                    drop_path=drop_path,
                    ablation=ablation,
                )
            )

        # ── Bottleneck ──
        self.bottleneck = nn.ModuleList([
            RWKVKANBlock(
                dim=dims[-1],
                num_groups=num_groups,
                cond_dim=combined_cond_dim,
                drop_path=drop_path,
                ablation=ablation,
            )
            for _ in range(num_blocks_per_stage)
        ])

        # ── Decoder ──
        self.decoder_stages = nn.ModuleList()
        for i in range(len(dims) - 2, -1, -1):
            self.decoder_stages.append(
                DecoderStage(
                    dim_in=dims[i + 1],
                    dim_out=dims[i],
                    skip_dim=dims[i],
                    num_blocks=num_blocks_per_stage,
                    num_groups=num_groups,
                    cond_dim=combined_cond_dim,
                    drop_path=drop_path,
                    ablation=ablation,
                )
            )

        # Output projection: base_dim → action_dim
        self.output_norm = nn.LayerNorm(base_dim)
        self.output_proj = nn.Linear(base_dim, action_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict velocity field v_θ(a_t, t, c).

        Args:
            noisy_actions: (B, horizon, action_dim) noisy action trajectory a_t.
            t: (B,) or (B, 1) flow time in [0, 1].
            condition: (B, cond_dim) fused visual-language-state condition.

        Returns:
            velocity: (B, horizon, action_dim) predicted velocity field.
        """
        B = noisy_actions.shape[0]

        # ── Build combined condition ──
        time_emb = self.time_embed(t)              # (B, base_dim)
        cond = self.cond_proj(condition)            # (B, cond_dim)
        combined_cond = torch.cat([cond, time_emb], dim=-1)  # (B, cond_dim + base_dim)
        combined_cond = self.cond_combine(combined_cond)

        # ── Input projection ──
        x = self.input_proj(noisy_actions)  # (B, horizon, base_dim)

        # ── Encoder with skip connections ──
        skips = []
        for stage in self.encoder_stages:
            x, skip = stage(x, combined_cond)
            skips.append(skip)

        # ── Bottleneck ──
        for block in self.bottleneck:
            x = block(x, combined_cond)

        # ── Decoder with skip connections ──
        for stage, skip in zip(self.decoder_stages, reversed(skips)):
            x = stage(x, skip, combined_cond)

        # ── Output projection ──
        velocity = self.output_proj(self.output_norm(x))  # (B, horizon, action_dim)

        return velocity

    def count_parameters(self) -> dict:
        """Count parameters by component."""
        counts = {}
        for name, child in self.named_children():
            total = sum(p.numel() for p in child.parameters())
            trainable = sum(p.numel() for p in child.parameters() if p.requires_grad)
            counts[name] = {"total": total, "trainable": trainable}
        grand_total = sum(p.numel() for p in self.parameters())
        grand_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        counts["_total"] = {"total": grand_total, "trainable": grand_trainable}
        return counts
