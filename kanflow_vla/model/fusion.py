"""
Cross-attention fusion for KANFlow-VLA.

Language tokens attend to visual tokens to extract task-relevant spatial
features. Multi-layer cross-attention with pre-norm and FFN.

Adapted from Janus VLA's fusion module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionLayer(nn.Module):
    """Single cross-attention layer with pre-norm and FFN."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Cross-attention
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        # FFN
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

        self._init_weights()

    def _init_weights(self):
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.ffn[0].weight)
        nn.init.zeros_(self.ffn[0].bias)
        nn.init.xavier_uniform_(self.ffn[3].weight)
        nn.init.zeros_(self.ffn[3].bias)

    def forward(
        self, lang_tokens: torch.Tensor, vis_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            lang_tokens: (B, L, d_model) queries
            vis_tokens:  (B, V, d_model) keys/values

        Returns:
            updated lang_tokens: (B, L, d_model)
        """
        B, L, _ = lang_tokens.shape
        V = vis_tokens.shape[1]

        q = self.norm_q(lang_tokens)
        kv = self.norm_kv(vis_tokens)

        q = self.W_q(q).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(kv).view(B, V, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(kv).view(B, V, self.num_heads, self.d_k).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=False,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        attn_out = self.W_o(attn_out)

        lang_tokens = lang_tokens + attn_out
        lang_tokens = lang_tokens + self.ffn(self.norm_ffn(lang_tokens))

        return lang_tokens


class CrossAttentionFusion(nn.Module):
    """
    Multi-layer cross-attention fusion.

    Language tokens attend to visual tokens across `num_layers` layers,
    producing task-conditioned visual-language representations.

    Args:
        d_model: Shared embedding dimension.
        num_layers: Number of cross-attention layers.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self, lang_tokens: torch.Tensor, vis_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            lang_tokens: (B, L, d_model)
            vis_tokens:  (B, V, d_model)

        Returns:
            fused: (B, L, d_model) language tokens enriched with visual info
        """
        for layer in self.layers:
            lang_tokens = layer(lang_tokens, vis_tokens)
        return self.final_norm(lang_tokens)
