import torch
import torch.nn as nn
from typing import NamedTuple

class ReliabilityOutput(NamedTuple):
    mode_logits: torch.Tensor
    failure_logits: torch.Tensor
    progress_pred: torch.Tensor

class ReliabilityHeads(nn.Module):
    def __init__(self, cond_dim=256, hidden_dim=256):
        super().__init__()
        # Shared MLP base
        self.shared = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish()
        )
        
        # 2-class Mode: ACT (0) vs STOP (1)
        self.mode_head = nn.Linear(hidden_dim, 2)
        
        # 2-class Failure: NONE (0) vs CORRUPT (1)
        self.failure_head = nn.Linear(hidden_dim, 2)
        
        # 1D Progress regression
        self.progress_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, c: torch.Tensor) -> ReliabilityOutput:
        h = self.shared(c)
        return ReliabilityOutput(
            mode_logits=self.mode_head(h),
            failure_logits=self.failure_head(h),
            progress_pred=self.progress_head(h).squeeze(-1)
        )
