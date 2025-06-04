"""Skip‑connection primitives with optional orthogonality constraints."""
from __future__ import annotations

import torch
import torch.nn as nn
from orthogonal_skip_connections.utils.orthogonal_ops import (
    cayley_transform,
    qr_reorthogonal,
    svd_sharp,
    steepest_descent_update,
)

class BaseSkip(nn.Module):
    """Abstract base class; subclasses implement `forward` and (optionally) `orth_update`."""
    def orth_update(self, eta: float = 1e-3):
        """Optional hook called in the training loop after every optimiser step."""
        pass

class IdentitySkip(BaseSkip):
    def forward(self, x):
        return x

class FixedOrthogonalSkip(BaseSkip):
    """Frozen orthogonal projection implemented as 1×1 conv with orthogonal matrix W."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        # initialise to identity (which is orthogonal) and freeze
        nn.init.eye_(self.conv.weight.data.view(channels, channels))
        self.conv.weight.requires_grad_(False)

    def forward(self, x):
        return self.conv(x)

class LearnableOrthogonalSkip(BaseSkip):
    def __init__(self, channels: int, update_rule: str = "qr"):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        nn.init.orthogonal_(self.conv.weight.data.view(channels, channels))
        self.update_rule = update_rule

    def _get_W_matrix(self):
        return self.conv.weight.view(self.channels, self.channels)

    def orth_update(self, eta: float = 1e-3):
        W = self._get_W_matrix()
        if self.update_rule == "qr":
            self.conv.weight.data.copy_(qr_reorthogonal(W).view_as(self.conv.weight))
        elif self.update_rule == "svd":
            self.conv.weight.data.copy_(svd_sharp(W).view_as(self.conv.weight))
        elif self.update_rule == "cayley":
            # treat current grad as G in steepest descent
            G = self.conv.weight.grad.view(self.channels, self.channels)
            self.conv.weight.data.copy_(steepest_descent_update(W, G, eta).view_as(self.conv.weight))
        else:
            raise ValueError(self.update_rule)

    def forward(self, x):
        return self.conv(x)

class RandomSkip(BaseSkip):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)

    def forward(self, x):
        return self.conv(x)

# Registry --------------------------------------------------------------------
_REGISTRY = {
    "identity": IdentitySkip,
    "fixed_orth": FixedOrthogonalSkip,
    "learnable_orth": LearnableOrthogonalSkip,
    "random": RandomSkip,
}

def get_skip(kind: str, channels: int, **kwargs):
    return _REGISTRY[kind](channels, **kwargs) if kind in _REGISTRY else None