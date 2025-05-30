"""BasicBlock with configurable skip projection."""
from __future__ import annotations
import torch.nn as nn
import torch
from typing import Optional, Callable

class OrthogonalSkip(nn.Module):
    def __init__(self, channels: int, projection: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        super().__init__()
        self.channels = channels
        if projection is None:
            self.proj = nn.Identity()
        else:
            self.register_buffer('dummy', torch.empty(0))  # placeholder
            self.proj_fn = projection

    def forward(self, x):
        if hasattr(self, 'proj_fn'):
            B, C, H, W = x.shape
            x_flat = x.view(B, C, -1)
            x_proj = self.proj_fn(x_flat)  # expects linear projection per channel dim
            return x_proj.view(B, C, H, W)
        return self.proj(x)
