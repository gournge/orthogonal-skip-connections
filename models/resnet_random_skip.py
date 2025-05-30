from __future__ import annotations
import torch
import torch.nn as nn
from .resnet_baseline import ResNetBaseline

class ResNetRandomSkip(ResNetBaseline):
    """Skip projections are random full‑rank matrices without orthogonality constraint."""
    def __init__(self, num_classes: int = 10, **kwargs):
        super().__init__(num_classes)
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Identity):
                proj = nn.Parameter(torch.randn(64, 64), requires_grad=False)
                setattr(self.backbone, name, proj)
