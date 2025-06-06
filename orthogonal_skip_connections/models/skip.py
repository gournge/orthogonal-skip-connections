"""Skip-connection primitives with optional orthogonality constraints."""

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
    def __init__(self, channels: int, **kwargs):
        super().__init__()
        # channels is unused, but required for interface compatibility

    def forward(self, x):
        return x


class FixedOrthogonalSkip(BaseSkip):
    """Frozen orthogonal projection implemented as 1×1 conv with orthogonal matrix W."""

    def __init__(self, channels: int):
        super().__init__()
        # Orthogonal transformation implemented as a frozen random orthogonal matrix
        w = torch.empty(channels, channels)
        nn.init.orthogonal_(w)
        self.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        N, C, H, W = x.shape
        out = x.permute(0, 2, 3, 1).reshape(-1, C) @ self.weight.t()
        return out.view(N, H, W, C).permute(0, 3, 1, 2)


class LearnableOrthogonalSkip(BaseSkip):
    def __init__(
        self, channels: int, update_rule: str = "steepest", update_rule_iters: int = 5
    ):
        super().__init__()
        self.channels = channels
        self.weight = nn.Parameter(torch.empty(channels, channels))
        nn.init.orthogonal_(self.weight)
        self.update_rule = update_rule
        self.update_rule_iters = update_rule_iters

    def _get_W_matrix(self):
        return self.weight

    def orth_update(self, eta: float = 1e-3):
        W = self._get_W_matrix()
        if self.update_rule == "qr":
            self.weight.data.copy_(qr_reorthogonal(W))
        elif self.update_rule == "svd":
            self.weight.data.copy_(svd_sharp(W))
        elif self.update_rule == "steepest":
            # treat current grad as G in steepest descent
            G = self.weight.grad
            self.weight.data.copy_(
                steepest_descent_update(W, G, eta, self.update_rule_iters)
            )
        else:
            raise ValueError(self.update_rule)

    def forward(self, x):
        N, C, H, W = x.shape
        out = x.permute(0, 2, 3, 1).reshape(-1, C) @ self.weight.t()
        return out.view(N, H, W, C).permute(0, 3, 1, 2)


class RandomSkip(BaseSkip):
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, channels))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x):
        N, C, H, W = x.shape
        out = x.permute(0, 2, 3, 1).reshape(-1, C) @ self.weight.t()
        return out.view(N, H, W, C).permute(0, 3, 1, 2)


# Registry --------------------------------------------------------------------
_REGISTRY = {
    "identity": IdentitySkip,
    "fixed_orth": FixedOrthogonalSkip,
    "learnable_orth": LearnableOrthogonalSkip,
    "random": RandomSkip,
}


def if_model_needs_update_rule(skip_kind: str) -> bool:
    return skip_kind in ["learnable_orth"]


def get_skip_names():
    """Return a list of available skip connection names."""
    return list(_REGISTRY.keys())


def get_skip(kind: str, channels: int, **kwargs):
    return _REGISTRY[kind](channels, **kwargs) if kind in _REGISTRY else None
