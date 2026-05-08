"""
KANFlow-VLA: Top-level Vision-Language-Action model.

Combines:
  1. VisionEncoder (SigLIP-base, frozen) → visual tokens
  2. LanguageEncoder (SmolLM-135M, frozen) → language tokens
  3. CrossAttentionFusion → fused VL representations
  4. ProprioMLP → robot state embedding
  5. RWKVKANUNet → velocity field prediction
  6. ConsistencyFlowMatching → one-step action generation

Total parameters: ~150-320M (depending on encoder choice).
Action head (RWKV-KAN UNet): ~33M trainable parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from kanflow_vla.model.vision import VisionEncoder
from kanflow_vla.model.language import LanguageEncoder
from kanflow_vla.model.fusion import CrossAttentionFusion
from kanflow_vla.model.rwkv_kan_unet import RWKVKANUNet
from kanflow_vla.model.flow_matching import ConsistencyFlowMatching
from kanflow_vla.model.reliability_heads import ReliabilityHeads
from kanflow_vla.model.octo_adapter import OctoConditionEncoder


@dataclass
class KANFlowVLAOutput:
    """Structured output from KANFlowVLA forward pass."""
    actions: torch.Tensor              # (B, horizon, action_dim) decoded actions
    loss: torch.Tensor | None = None   # total training loss
    cfm_loss: torch.Tensor | None = None
    acr_loss: torch.Tensor | None = None
    velocity_loss: torch.Tensor | None = None
    condition: torch.Tensor | None = None  # fused condition vector
    # RCAR behaviour head outputs
    mode_logits: torch.Tensor | None = None    # (B, 4)
    failure_logits: torch.Tensor | None = None # (B, 9)
    progress: torch.Tensor | None = None       # (B,)


class KANFlowVLA(nn.Module):
    """
    KANFlow-VLA: RWKV-GroupKAN Flow-Matching Vision-Language-Action Model.

    Architecture:
      Vision + Language Encoder (SigLIP + SmolLM, frozen)
               ↓ (fused embeddings + language tokens)
      CrossAttentionFusion (2-layer)
               ↓
      Condition Encoder (pool + proprio + time)
               ↓
      RWKV-KAN U-Net (3-stage encoder-decoder, ~33M params)
               ↓ (velocity field)
      Consistency Flow Matching (one-step decode)
               ↓
      Action output (horizon=4, action_dim=4 for MetaWorld)

    Args:
        action_dim: Action space dimension (4 for MetaWorld: 3D delta + gripper).
        horizon: Action prediction horizon (default: 4).
        d_model: Shared embedding dimension.
        proprio_dim: Proprioception input dimension.
        unet_base_dim: Base channel dim for RWKV-KAN UNet.
        num_groups: GroupKAN groups (G=4).
        num_knots: B-spline knot count.
        num_segments: CFM segments (K=2).
        ema_decay: EMA decay for teacher network.
        lambda_acr: ACR loss weight.
        vision_config: Vision encoder config dict.
        language_config: Language encoder config dict.
        fusion_config: Fusion config dict.
        freeze_encoder: Freeze VL encoder weights.
    """

    def __init__(
        self,
        action_dim: int = 4,
        horizon: int = 4,
        d_model: int = 256,
        proprio_dim: int = 15,
        unet_base_dim: int = 128,
        num_groups: int = 4,
        num_knots: int = 8,
        num_segments: int = 2,
        ema_decay: float = 0.95,
        lambda_acr: float = 1.0,
        delta_t: float = 0.01,
        alpha_consistency: float = 1.0,
        inference_t: float = 0.0,
        vision_config: dict | None = None,
        language_config: dict | None = None,
        fusion_config: dict | None = None,
        reliability_config: dict | None = None,
        freeze_encoder: bool = True,
        ablation: str | None = None,
        domain_randomize: bool = False,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.d_model = d_model
        self.use_octo = False

        # ── Vision Encoder ──
        v_cfg = vision_config or {}
        v_name = str(v_cfg.get("name", "")).lower()
        if v_name.startswith("octo"):
            self.use_octo = True
            self.octo = OctoConditionEncoder(
                checkpoint_path=v_cfg.get("pretrained_path", "hf://rail-berkeley/octo-small-1.5"),
                output_dim=d_model,
                platform=v_cfg.get("platform", "cpu"),
            )
            self.vision = nn.Identity()
            self.language = nn.Identity()
            self.fusion = nn.Identity()
        else:
            self.octo = None
            self.vision = VisionEncoder(
                d_model=d_model,
                d_vision=v_cfg.get("d_vision", 768),
                img_size=v_cfg.get("img_size", 224),
                patch_size=v_cfg.get("patch_size", 16),
                pretrained=v_cfg.get("pretrained", True),
                freeze=freeze_encoder,
            )

            # ── Language Encoder ──
            l_cfg = language_config or {}
            self.language = LanguageEncoder(
                model_name=l_cfg.get("model_name", "HuggingFaceTB/SmolLM-135M"),
                d_model=d_model,
                d_lang=l_cfg.get("d_lang", 576),
                max_tokens=l_cfg.get("max_tokens", 32),
                freeze=freeze_encoder,
            )

            # ── Cross-Attention Fusion ──
            f_cfg = fusion_config or {}
            self.fusion = CrossAttentionFusion(
                d_model=d_model,
                num_layers=f_cfg.get("num_layers", 2),
                num_heads=f_cfg.get("num_heads", 4),
                dropout=f_cfg.get("dropout", 0.1),
            )

        # ── Proprioception MLP ──
        self.proprio_mlp = nn.Sequential(
            nn.Linear(proprio_dim, 128),
            nn.GELU(),
            nn.Linear(128, d_model),
        )
        nn.init.kaiming_normal_(self.proprio_mlp[0].weight, nonlinearity="relu")
        nn.init.zeros_(self.proprio_mlp[0].bias)
        nn.init.xavier_uniform_(self.proprio_mlp[2].weight)
        nn.init.zeros_(self.proprio_mlp[2].bias)

        # ── RWKV-KAN UNet (Action Decoder) ──
        self.unet = RWKVKANUNet(
            action_dim=action_dim,
            horizon=horizon,
            base_dim=unet_base_dim,
            cond_dim=d_model,
            num_blocks_per_stage=2,
            num_groups=num_groups,
            num_knots=num_knots,
            drop_path=0.0,
            ablation=ablation,
            domain_randomize=domain_randomize,
        )

        # ── Consistency Flow Matching ──
        self.cfm = ConsistencyFlowMatching(
            velocity_net=self.unet,
            action_dim=action_dim,
            horizon=horizon,
            num_segments=num_segments,
            ema_decay=ema_decay,
            delta_t=delta_t,
            lambda_acr=lambda_acr,
            alpha_consistency=alpha_consistency,
            inference_t=inference_t,
        )

        # ── Reliability Heads (RCAR) ──
        r_cfg = reliability_config or {}
        self.reliability = ReliabilityHeads(
            d_model=d_model,
            d_hidden=r_cfg.get("d_hidden", d_model),
            dropout=r_cfg.get("dropout", 0.1),
        )

        self._print_param_summary()

    def _print_param_summary(self) -> None:
        """Print parameter counts by component."""
        modules = {
            "proprio_mlp": self.proprio_mlp,
            "unet (action head)": self.unet,
            "cfm (teacher)": self.cfm.teacher_net,
            "reliability heads": self.reliability,
        }
        if self.use_octo:
            modules = {"octo encoder": self.octo, **modules}
        else:
            modules = {
                "vision": self.vision,
                "language": self.language,
                "fusion": self.fusion,
                **modules,
            }

        total = 0
        trainable = 0
        print("\n" + "=" * 65)
        print("KANFlow-VLA Parameter Summary")
        print("=" * 65)
        for name, module in modules.items():
            total_count = sum(p.numel() for p in module.parameters())
            train_count = sum(
                p.numel() for p in module.parameters() if p.requires_grad
            )
            total += total_count
            trainable += train_count
            frozen_str = " (frozen)" if train_count == 0 else ""
            print(
                f"  {name:25s}: {self._fmt(total_count):>10s} total, "
                f"{self._fmt(train_count):>10s} trainable{frozen_str}"
            )
        print("-" * 65)
        print(
            f"  {'TOTAL':25s}: {self._fmt(total):>10s} total, "
            f"{self._fmt(trainable):>10s} trainable"
        )
        print("=" * 65 + "\n")

    @staticmethod
    def _fmt(n: int) -> str:
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n / 1_000:.1f}K"
        return str(n)

    def _encode_condition(
        self,
        images: torch.Tensor,
        lang_ids: torch.Tensor,
        proprio: torch.Tensor,
        lang_mask: torch.Tensor | None = None,
        task_texts: list[str] | tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        """
        Build fused condition vector from visual, language, and proprioceptive inputs.

        Args:
            images: (B, 3, H, W) or (B, T, 3, H, W) or (B, T, V, 3, H, W) RGB images.
            lang_ids: (B, L) language token IDs.
            proprio: (B, proprio_dim) or (B, T, proprio_dim) robot state.
            lang_mask: (B, L) optional attention mask.

        Returns:
            condition: (B, d_model) fused condition vector.
        """
        target_proprio_dim = self.proprio_mlp[0].in_features
        if proprio.ndim == 3:
            B, T, D = proprio.shape
            flat_proprio = proprio.reshape(B, T * D)
            if flat_proprio.shape[1] == target_proprio_dim:
                proprio = flat_proprio
            else:
                proprio = proprio[:, -1, :]

        proprio_emb = self.proprio_mlp(proprio)  # (B, d_model)

        if self.use_octo:
            octo_cond = self.octo(images, task_texts=task_texts)
            return octo_cond + proprio_emb

        # Handle temporal dimension: fuse into views/spatial dimension
        if images.ndim == 6:
            # (B, T, V, C, H, W) -> (B, T * V, C, H, W)
            B, T, V, C, H, W = images.shape
            images = images.reshape(B, T * V, C, H, W)
        elif images.ndim == 5:
            # (B, T, C, H, W) -> treated as (B, V, C, H, W)
            pass

        # Vision: (B, V, 3, H, W) -> (B, V * num_patches, d_model)
        vis_tokens = self.vision(images)

        # Language: (B, L) → (B, L, d_model)
        lang_tokens = self.language(lang_ids, lang_mask)

        # Cross-attention fusion: (B, L, d_model)
        fused = self.fusion(lang_tokens, vis_tokens)

        # Pool language tokens + add proprioception
        fused_pooled = fused.mean(dim=1)  # (B, d_model)

        condition = fused_pooled + proprio_emb  # (B, d_model)

        return condition

    def forward(
        self,
        images: torch.Tensor,
        lang_ids: torch.Tensor,
        proprio: torch.Tensor,
        expert_actions: torch.Tensor | None = None,
        lang_mask: torch.Tensor | None = None,
        mode_labels: torch.Tensor | None = None,
        failure_labels: torch.Tensor | None = None,
        progress_labels: torch.Tensor | None = None,
        task_texts: list[str] | tuple[str, ...] | None = None,
    ) -> KANFlowVLAOutput:
        """
        Forward pass: training (with expert_actions) or inference (without).

        Args:
            images: (B, 3, H, W) or (B, T, 3, H, W) input images.
            lang_ids: (B, L) language token IDs.
            proprio: (B, proprio_dim) or (B, T, proprio_dim) robot state.
            expert_actions: (B, horizon, action_dim) expert actions for training.
            lang_mask: (B, L) optional attention mask.

        Returns:
            KANFlowVLAOutput with decoded actions and (optionally) losses.
                """
        # Build condition vector
        condition = self._encode_condition(
            images, lang_ids, proprio, lang_mask, task_texts=task_texts
        )

        # Behaviour heads — always forward (cheap; used for monitoring  even without labels)
        rel_out = self.reliability(condition)

        if expert_actions is not None:
            # Training mode: compute CFM + ACR losses
            loss_dict = self.cfm.compute_loss(expert_actions, condition)

            # Also generate sample for monitoring
            with torch.no_grad():
                actions = self.cfm.sample(condition)

            return KANFlowVLAOutput(
                actions=actions,
                loss=loss_dict["loss"],
                cfm_loss=loss_dict["cfm_loss"],
                acr_loss=loss_dict["acr_loss"],
                velocity_loss=loss_dict["velocity_loss"],
                condition=condition,
                mode_logits=rel_out["mode_logits"],
                failure_logits=rel_out["failure_logits"],
                progress=rel_out["progress"],
            )
        else:
            # Inference mode: one-step decode
            actions = self.cfm.sample(condition)

            return KANFlowVLAOutput(
                actions=actions,
                condition=condition,
                mode_logits=rel_out["mode_logits"],
                failure_logits=rel_out["failure_logits"],
                progress=rel_out["progress"],
            )

    @torch.no_grad()
    def predict_action(
        self,
        images: torch.Tensor,
        lang_ids: torch.Tensor,
        proprio: torch.Tensor,
        lang_mask: torch.Tensor | None = None,
        num_steps: int = 1,
        use_ema: bool = True,
        task_texts: list[str] | tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        """
        Predict actions for deployment (inference only).

        Args:
            images: (B, 3, H, W) input image.
            lang_ids: (B, L) language token IDs.
            proprio: (B, proprio_dim) robot state.
            lang_mask: (B, L) optional attention mask.
            num_steps: Number of flow integration steps (1=optimized one-step).
            use_ema: If True, decode with EMA teacher weights.

        Returns:
            actions: (B, horizon, action_dim) predicted action trajectory.
        """
        condition = self._encode_condition(
            images, lang_ids, proprio, lang_mask, task_texts=task_texts
        )

        if num_steps == 1:
            return self.cfm.sample(condition, use_ema=use_ema)
        else:
            return self.cfm.sample_multistep(
                condition, num_steps=num_steps, use_ema=use_ema
            )

    def update_ema(self):
        """Update EMA teacher network. Call after each optimizer step."""
        self.cfm.update_ema()

    def get_param_groups(self, base_lr: float, weight_decay: float = 0.05) -> list[dict]:
        """
        Get parameter groups with per-module learning rate multipliers.

        Returns parameter groups suitable for AdamW optimizer:
          - Encoder (fusion + proprio): base_lr
          - UNet (action head): base_lr (main trainable component)
          - Vision/Language: skipped (frozen)
        """
        groups = []
        seen = set()

        def add_group(name: str, module: nn.Module, lr_mult: float):
            params = [p for p in module.parameters() if p.requires_grad]
            new_params = [p for p in params if id(p) not in seen]
            seen.update(id(p) for p in params)
            if new_params:
                groups.append({
                    "params": new_params,
                    "lr": base_lr * lr_mult,
                    "weight_decay": weight_decay,
                    "name": name,
                })

        if not self.use_octo:
            add_group("vision", self.vision, 0.0)  # frozen
            add_group("language", self.language, 0.1)  # projection only
            add_group("fusion", self.fusion, 1.0)
        add_group("proprio_mlp", self.proprio_mlp, 1.0)
        add_group("unet", self.unet, 1.0)  # main trainable component

        # Catch any remaining params
        remaining = [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in seen
        ]
        if remaining:
            groups.append({
                "params": remaining,
                "lr": base_lr,
                "weight_decay": weight_decay,
                "name": "other",
            })

        return groups
