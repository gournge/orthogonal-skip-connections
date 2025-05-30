from __future__ import annotations
import torch
import torch.nn as nn
from .resnet_baseline import ResNetBaseline
from utils.orthogonal import sharp_operator

class ResNetOrthogonal(ResNetBaseline):
    """Every skip uses a fixed, randomly initialised orthogonal projection."""
    def __init__(self, num_classes: int = 10, **kwargs):
        super().__init__(num_classes)
        # iterate through backbone modules and replace identity skips (if any) – placeholder
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Identity):
                proj = nn.Parameter(sharp_operator(torch.randn(64, 64)), requires_grad=False)
                setattr(self.backbone, name, proj)
