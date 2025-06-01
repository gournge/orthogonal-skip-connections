"""ResNet variants where the identity skip becomes *W x* with W ≈ orthogonal.

Variants implemented:
    • FullyOrthResNet  – every residual path uses an OrthogonalProjection.
    • LearnableOrthResNet – W is *learned* + regularised (may deviate slightly).
    • PartialOrthResNet – choose which stages use W.
    • RandomisedSkipResNet – W fixed, non-orthogonal (control).
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Sequence, Type
from .resnet_baseline import ResNet, Block, Bottleneck
from .orthogonal_layers import OrthogonalProjection

def _make_skip(dim: int, mode: str, method: str = "cayley") -> nn.Module | None:
    if mode == "identity":
        return None
    if mode == "orth":
        return OrthogonalProjection(dim, method)
    if mode == "learnable_orth":
        proj = OrthogonalProjection(dim, method)
        proj.W_raw.requires_grad = True
        return proj
    if mode == "random":
        proj = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(proj.weight, 0, 0.02)
        return proj
    raise ValueError(mode)

class OrthBlock(Block):
    """Extends the basic *Block* with a configurable skip."""

    def __init__(self, in_c: int, out_c: int, skip_mode: str = "identity", ortho_method: str = "cayley", **kw):
        super().__init__(in_c, out_c, **kw)
        self.custom_skip = _make_skip(out_c * self.expansion, skip_mode, ortho_method)

    def forward(self, x):
        identity = x
        out = super().forward(x)  # internal conv path
        if self.custom_skip is not None:
            identity = self.custom_skip(identity)
        return self.relu(out + identity)

class OrthResNet(ResNet):
    """A drop-in replacement where _make_layer injects *OrthBlocks* with the
    chosen skip_mode profile.
    """

    def __init__(self, layers: Sequence[int], num_classes: int = 10, skip_profile: Sequence[str] | str = "orth", ortho_method: str = "cayley"):
        self.skip_profile = skip_profile
        self.ortho_method = ortho_method
        super().__init__(OrthBlock, layers, num_classes)

    def _make_layer(self, ResBlock, planes, blocks, stride=1):
        layers = []
        for i in range(blocks):
            skip_mode = self.skip_profile if isinstance(self.skip_profile, str) else self.skip_profile[i]
            layers.append(OrthBlock(self.in_channels, planes, skip_mode=skip_mode, ortho_method=self.ortho_method, stride=stride if i == 0 else 1))
            self.in_channels = planes * ResBlock.expansion
        return nn.Sequential(*layers)
