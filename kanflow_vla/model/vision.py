"""
Visual encoder for KANFlow-VLA.

SigLIP-base/16 with linear projection to d_model.
Produces spatial visual tokens (196 tokens of dim d_model) for cross-attention
fusion with language tokens.

Adapted from Janus VLA's vision encoder.
"""

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    timm = None

try:
    from transformers import SiglipVisionModel
except ImportError:
    SiglipVisionModel = None


class VisionEncoder(nn.Module):
    """
    SigLIP-base/16 visual encoder.

    Produces spatial tokens: (B, num_patches, d_model).
    Frozen by default for parameter efficiency.

    Args:
        d_model: Output projection dimension.
        d_vision: Vision backbone hidden dimension.
        img_size: Input image size (square).
        patch_size: ViT patch size.
        pretrained: Load pretrained weights.
        freeze: Freeze backbone weights (recommended for <400M budget).
    """

    def __init__(
        self,
        d_model: int = 256,
        d_vision: int = 768,
        img_size: int = 224,
        patch_size: int = 16,
        pretrained: bool = True,
        freeze: bool = True,
    ):
        super().__init__()
        self.d_vision = d_vision
        self.d_model = d_model
        self.num_patches = (img_size // patch_size) ** 2  # 196

        # Load backbone
        self.backbone = self._load_backbone(pretrained, img_size)

        # Projection to shared dim
        self.projection = nn.Linear(self.d_vision, d_model)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

        if freeze:
            self._freeze_backbone()

    def _load_backbone(self, pretrained: bool, img_size: int) -> nn.Module:
        """Load SigLIP-base. Fallback to timm ViT."""
        if SiglipVisionModel is not None and pretrained:
            try:
                model = SiglipVisionModel.from_pretrained(
                    "google/siglip-base-patch16-224"
                )
                if model.config.hidden_size != self.d_vision:
                    print(
                        f"[VisionEncoder] SigLIP hidden_size={model.config.hidden_size}, "
                        f"expected {self.d_vision}. Adjusting."
                    )
                    self.d_vision = model.config.hidden_size
                return model
            except Exception as e:
                print(f"[VisionEncoder] SigLIP failed: {e}. Trying timm.")

        if timm is not None:
            model = timm.create_model(
                "vit_base_patch16_224",
                pretrained=pretrained,
                img_size=img_size,
                num_classes=0,
            )
            self.d_vision = model.embed_dim
            return model

        raise RuntimeError(
            "Neither transformers (SigLIP) nor timm available. Install one."
        )

    def _freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W) or (B, V, 3, H, W) normalized RGB images.

        Returns:
            visual_tokens: (B, num_patch_sequences, d_model)
        """
        if images.ndim == 5:
            B, V, C, H, W = images.shape
            images = images.reshape(B * V, C, H, W)
            is_multiview = True
        else:
            is_multiview = False

        if isinstance(self.backbone, nn.Module) and hasattr(
            self.backbone, "forward_features"
        ):
            features = self.backbone.forward_features(images)
            if features.shape[1] == self.num_patches + 1:
                features = features[:, 1:, :]  # remove CLS
        elif SiglipVisionModel is not None and isinstance(
            self.backbone, SiglipVisionModel
        ):
            outputs = self.backbone(pixel_values=images)
            features = outputs.last_hidden_state
        else:
            features = self.backbone(images)

        visual_tokens = self.projection(features)  # (B[*V], 196, d_model)
        
        if is_multiview:
            # (B*V, 196, d_model) -> (B, V, 196, d_model) -> (B, V*196, d_model)
            _, N, D = visual_tokens.shape
            visual_tokens = visual_tokens.view(B, V, N, D).reshape(B, V * N, D)
            
        return visual_tokens
