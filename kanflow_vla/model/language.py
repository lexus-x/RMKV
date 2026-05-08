"""
Language encoder for KANFlow-VLA.

Uses a small pre-trained language model (SmolLM-135M by default) to encode
task instructions into token-level representations for cross-attention
fusion with visual features.

Adapted from Janus VLA's language encoder.
"""

import torch
import torch.nn as nn

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel = None
    AutoTokenizer = None


class LanguageEncoder(nn.Module):
    """
    Pre-trained LM encoder with projection to shared d_model.

    The LM backbone is frozen by default — only the projection layer
    is trainable, keeping the parameter budget minimal.

    Args:
        model_name: HuggingFace model identifier.
        d_model: Output projection dimension.
        d_lang: Language model hidden dimension.
        max_tokens: Maximum sequence length for tokenization.
        freeze: Freeze LM backbone weights.
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM-135M",
        d_model: int = 256,
        d_lang: int = 576,
        max_tokens: int = 32,
        freeze: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_lang = d_lang
        self.max_tokens = max_tokens
        self.model_name = model_name

        if AutoModel is not None:
            try:
                self.backbone = AutoModel.from_pretrained(model_name)
                actual_dim = self.backbone.config.hidden_size
                if actual_dim != d_lang:
                    print(
                        f"[LanguageEncoder] LM hidden_size={actual_dim}, "
                        f"expected {d_lang}. Adjusting."
                    )
                    self.d_lang = actual_dim
            except Exception as e:
                print(f"[LanguageEncoder] Failed to load {model_name}: {e}")
                print("[LanguageEncoder] Using embedding fallback.")
                self.backbone = None
                self._build_fallback()
        else:
            self.backbone = None
            self._build_fallback()

        self.projection = nn.Linear(self.d_lang, d_model)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

        if freeze and self.backbone is not None:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _build_fallback(self):
        """Simple embedding lookup fallback when HF models unavailable."""
        self.backbone = None
        self.embed = nn.Embedding(32000, self.d_lang)
        self.pos_embed = nn.Embedding(self.max_tokens, self.d_lang)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L) token IDs.
            attention_mask: (B, L) optional attention mask.

        Returns:
            lang_tokens: (B, L, d_model) language representations.
        """
        if self.backbone is not None:
            with torch.no_grad():
                outputs = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                hidden = outputs.last_hidden_state
        else:
            # Fallback path
            positions = torch.arange(
                input_ids.shape[1], device=input_ids.device
            ).unsqueeze(0)
            hidden = self.embed(input_ids) + self.pos_embed(positions)

        lang_tokens = self.projection(hidden)
        return lang_tokens
