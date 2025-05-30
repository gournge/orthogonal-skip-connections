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

        # hook to convert Euclid grad → Riemannian grad
        self.W.register_hook(self._orthogonalize_grad)

    def forward(self, x):
        B, C, H, W = x.shape
        flat = x.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        out = flat @ self.W.t()  # use Wᵀ as skip
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)

    def _orthogonalize_grad(self, grad_W):
        # G_riem = G - W sym(Wᵀ G)
        WTg = self.W.t() @ grad_W
        sym = (WTg + WTg.t()) * 0.5
        return grad_W - self.W @ sym

    def step(self, grad: torch.Tensor):
        with torch.no_grad():
            self.W.copy_(self.retraction(self.W, grad, self.eta))


from .patch import patch_resnet_skips
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck


class ResNetLearnableOrthogonal(ResNet):
    def __init__(
        self,
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=1000,
        retraction="cayley",
        sharp_iters=8,
    ):
        super().__init__(block, layers, num_classes=num_classes)
        # attach a learnable orthogonal W to every plain skip
        self.ortho_modules = patch_resnet_skips(
            self, mode="learnable", retraction=retraction, sharp_iters=sharp_iters
        )

    def orth_error(self):
        return sum(
            orth_error(m.W) for m in self.ortho_modules if m.W is not None
        ) / len(self.ortho_modules)
