from __future__ import annotations
import torch.nn as nn
from .resnet_learnable_orthogonal import LearnableOrthogonalProjection
from .resnet_baseline import ResNetBaseline
from utils.orthogonal import orth_error

class ResNetPartialOrthogonal(ResNetBaseline):
    """Mask controls which stages use orthogonal skip projections."""
    def __init__(self, num_classes: int = 10, mask=(True, False, True, False), retraction: str = "qr"):
        super().__init__(num_classes)
        self.ortho_modules = []
        idx = 0
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Identity):
                if idx < len(mask) and mask[idx]:
                    op = LearnableOrthogonalProjection(64, retraction=retraction)
                    setattr(self.backbone, name, op)
                    self.ortho_modules.append(op)
                idx += 1

    def orth_error(self):
        return sum(orth_error(m.W) for m in self.ortho_modules) / max(1, len(self.ortho_modules))
