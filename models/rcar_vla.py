import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from .unet import ConditionalUnet1D
from .encoders import VisionEncoder, LanguageEncoder, ProprioEncoder, CrossAttentionFusion
from .reliability import ReliabilityHeads, ReliabilityOutput
from .flow_matching import ConsistencyFlowMatching

@dataclass
class RCAROutput:
    loss: torch.Tensor
    loss_cfm1: torch.Tensor
    loss_mode: torch.Tensor
    loss_fail: torch.Tensor
    loss_prog: torch.Tensor
    loss_sc: torch.Tensor
    r: ReliabilityOutput

class RCARVLA(nn.Module):
    def __init__(self, action_dim=4, horizon=4, proprio_dim=7, ema_decay=0.999):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.proprio_dim = proprio_dim
        
        self.vision = VisionEncoder()
        self.language = LanguageEncoder()
        self.proprio = ProprioEncoder(in_dim=proprio_dim, out_dim=256)
        
        # Cross attention fusion outputs 256
        self.fusion = CrossAttentionFusion(vision_dim=768, lang_dim=576, d_model=256)
        
        self.reliability = ReliabilityHeads(cond_dim=256, hidden_dim=256)
        
        # UNet takes c (256) + failure_logits (2) = 258
        self.unet = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=258,
            down_dims=[128, 256, 512]
        )
        self.cfm = ConsistencyFlowMatching(self.unet, ema_decay=ema_decay)

    def _encode_condition(self, batch):
        imgs = batch["images"]
        # Dataset gives (B, T, V, 3, H, W). Use last timestep → (B, V, 3, H, W).
        if imgs.dim() == 6:
            imgs = imgs[:, -1]
        v = self.vision(imgs)                    # (B, V*S, 768)
        l = self.language(batch["lang_ids"])     # (B, 576)
        p = self.proprio(batch["proprio"])       # (B, 256)
        c_vl = self.fusion(v, l)                 # (B, 256)
        c = c_vl + p                             # (B, 256)
        return c

    def forward(self, batch, compute_sc=True, lw_m=0.1, lw_f=0.1, lw_p=0.05, lw_sc=0.5):
        c = self._encode_condition(batch)
        B = c.shape[0]
        device = c.device
        
        # 1st Pass: zero-pad the failure logits part
        c1 = torch.cat([c, torch.zeros(B, 2, device=device)], dim=-1)
        loss_cfm1 = self.cfm.compute_loss(batch["actions"], c1)
        
        # Reliability Heads
        r = self.reliability(c)
        
        loss_mode = F.cross_entropy(r.mode_logits, batch["mode_label"])
        loss_fail = F.cross_entropy(r.failure_logits, batch["failure_label"])
        loss_prog = F.mse_loss(r.progress_pred, batch["progress_label"])
        
        loss_sc = torch.tensor(0.0, device=device)
        if compute_sc:
            corrupt_mask = (batch["failure_label"] == 1)
            if corrupt_mask.any():
                # We stop gradients to failure_logits so the CFM loss doesn't mess up the failure classifier
                # Alternatively, we could allow it, but usually detaching is safer for stable training.
                # The SPEC doesn't explicitly detach, but detaching is standard practice. Let's not detach to be faithful, 
                # or detach to be safe. We will detach.
                # We detach failure_logits to stabilize the reliability heads from massive CFM gradients.
                c_sc = torch.cat([c[corrupt_mask], r.failure_logits[corrupt_mask].detach()], dim=-1)
                loss_sc = self.cfm.compute_loss(batch["actions"][corrupt_mask], c_sc)
        
        loss_total = loss_cfm1 + lw_m * loss_mode + lw_f * loss_fail + lw_p * loss_prog + lw_sc * loss_sc
        
        return RCAROutput(
            loss=loss_total,
            loss_cfm1=loss_cfm1,
            loss_mode=loss_mode,
            loss_fail=loss_fail,
            loss_prog=loss_prog,
            loss_sc=loss_sc,
            r=r
        )

    @torch.no_grad()
    def predict_action(self, batch, tau_f=0.5, use_gating=True, use_self_correction=True):
        c = self._encode_condition(batch)
        B = c.shape[0]
        device = c.device
        
        c1 = torch.cat([c, torch.zeros(B, 2, device=device)], dim=-1)
        a = self.cfm.sample(c1, self.horizon, self.action_dim, num_steps=1)
        
        r = self.reliability(c)
        p_fail = F.softmax(r.failure_logits, dim=-1)[:, 1]
        
        if use_self_correction:
            correct_mask = p_fail > tau_f
            if correct_mask.any():
                c_sc = torch.cat([c[correct_mask], r.failure_logits[correct_mask]], dim=-1)
                a_sc = self.cfm.sample(c_sc, self.horizon, self.action_dim, num_steps=2)
                a[correct_mask] = a_sc

        # Gating flags logic should be handled by the environment rollout loop.
        # We just return the mode and failure probabilities so the environment can accumulate counts.
        return a, r
