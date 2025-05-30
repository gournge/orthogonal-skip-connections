from .resnet_baseline import ResNetBaseline
from .patch import patch_resnet_skips
import torch.nn as nn


class ResNetOrthogonal(ResNetBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # fixed (randomly initialised) orthogonal skip
        self.ortho_modules = patch_resnet_skips(self.backbone, mode="fixed")
