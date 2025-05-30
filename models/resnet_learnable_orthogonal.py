from __future__ import annotations

# for patching one argument
from functools import partial
import torch
import torch.nn as nn
from typing import Optional
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

        # hook to convert Euclid grad → Riemannian grad
        self.W.register_hook(self._orthogonalize_grad)

    def forward(self, x):
        # case A: feature map 4-D ⇒ apply Wᵀ over channels
        if x.dim() == 4:
            B, C, H, W = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
            out = x_flat @ self.W.T
            return out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        # case B: already flattened 2-D feature vectors
        elif x.dim() == 2:
            return x @ self.W.T
        else:
            raise ValueError(f"OrthProj: expected 2D or 4D tensor, got {x.dim()}D")

    def _orthogonalize_grad(self, grad_W):
        # G_riem = G - W sym(Wᵀ G)
        WTg = self.W.t() @ grad_W
        sym = (WTg + WTg.t()) * 0.5
        return grad_W - self.W @ sym

    def step(self, grad: torch.Tensor):
        with torch.no_grad():
            self.W.copy_(self.retraction(self.W, grad, self.eta))


from .patch import patch_resnet_skips
from .resnet_baseline import ResNetBaseline


class ResNetLearnableOrthogonal(ResNetBaseline):
    def __init__(self, num_classes=10, retraction="cayley", sharp_iters=8):
        super().__init__(num_classes)
        # patch every plain skip with its own W of the correct size
        self.ortho_modules = patch_resnet_skips(
            self.backbone, retraction=retraction, sharp_iters=sharp_iters
        )
