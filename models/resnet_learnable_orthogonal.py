from __future__ import annotations

# for patching one argument
from functools import partial
import torch
import torch.nn as nn
from typing import Optional
from .resnet_baseline import ResNetBaseline
from utils.orthogonal import get_retraction, orth_error, requires_sharp_iters


class LearnableOrthogonalProjection(nn.Module):
    def __init__(
        self,
        dim: Optional[int] = None,
        retraction: str = "cayley",
        sharp_iters: int = 8,
        eta: float = 1e-1,
    ):
        super().__init__()
        self.initial_dim = dim
        if dim is not None:
            self.W = nn.Parameter(torch.eye(dim))
        else:
            self.W = None
        self.sharp_iters = sharp_iters
        self.retraction = get_retraction(retraction)
        if requires_sharp_iters(retraction):
            self.retraction = partial(self.retraction, sharp_iters=sharp_iters)
        self.eta = eta

    def forward(self, x):
        # Lazy initialization of W if not provided during __init__
        if self.W is None:
            dim = x.shape[-1]
            self.W = nn.Parameter(torch.eye(dim, device=x.device, dtype=x.dtype))
        return x @ self.W

    def step(self, grad: torch.Tensor):
        with torch.no_grad():
            self.W.copy_(self.retraction(self.W, grad, self.eta))


class ResNetLearnableOrthogonal(ResNetBaseline):
    """Learnable orthogonal matrices with retraction after each update."""

    def __init__(
        self, num_classes: int = 10, retraction: str = "cayley", sharp_iters: int = 8
    ):
        super().__init__(num_classes)
        self.ortho_modules = []
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Identity):
                # Use lazy initialization by setting dim to None.
                op = LearnableOrthogonalProjection(
                    dim=None, retraction=retraction, sharp_iters=sharp_iters
                )
                setattr(self.backbone, name, op)
                self.ortho_modules.append(op)

    def orth_error(self):
        return sum(
            orth_error(m.W) for m in self.ortho_modules if m.W is not None
        ) / len(self.ortho_modules)
