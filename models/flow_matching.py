"""Flow Matching with EMA teacher (Lipman et al. 2023, arXiv:2210.02747).

NOT consistency flow matching — kept as plain rectified-flow MSE for v1
simplicity. EMA teacher used for stable inference.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowMatching(nn.Module):
    def __init__(self, unet, ema_decay: float = 0.999):
        super().__init__()
        self.unet = unet
        self.ema_decay = ema_decay
        # EMA copy of unet for stable inference
        self.ema_unet = copy.deepcopy(unet)
        for p in self.ema_unet.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_ema(self):
        d = self.ema_decay
        for ema_p, p in zip(self.ema_unet.parameters(), self.unet.parameters()):
            ema_p.data.mul_(d).add_(p.data, alpha=1 - d)
        for ema_b, b in zip(self.ema_unet.buffers(), self.unet.buffers()):
            ema_b.data.copy_(b.data)

    def compute_loss(self, actions: torch.Tensor, condition_vector: torch.Tensor) -> torch.Tensor:
        """actions: (B, H, D); condition_vector: (B, cond_dim)."""
        B = actions.shape[0]
        device = actions.device
        t = torch.rand(B, device=device)
        x_1 = torch.randn_like(actions)               # noise at t=0
        x_0 = actions                                  # data at t=1
        t_e = t.view(B, 1, 1)
        x_t = (1 - t_e) * x_1 + t_e * x_0
        v_target = x_0 - x_1
        v_pred = self.unet(x_t, t, global_cond=condition_vector)
        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def sample(
        self,
        condition_vector: torch.Tensor,
        horizon: int,
        action_dim: int,
        num_steps: int = 1,
        use_ema: bool = True,
    ) -> torch.Tensor:
        B = condition_vector.shape[0]
        device = condition_vector.device
        net = self.ema_unet if use_ema else self.unet
        x = torch.randn(B, horizon, action_dim, device=device)
        dt = 1.0 / num_steps
        t_val = 0.0
        for _ in range(num_steps):
            t = torch.full((B,), t_val, device=device)
            v = net(x, t, global_cond=condition_vector)
            x = x + v * dt
            t_val += dt
        return x


# Back-compat alias for existing imports
ConsistencyFlowMatching = FlowMatching
