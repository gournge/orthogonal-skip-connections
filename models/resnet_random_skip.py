from .resnet_baseline import ResNetBaseline
from .patch import patch_resnet_skips
import torch.nn as nn


class ResNetRandomSkip(ResNetBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # purely random (Gaussian) skip, no learning
        patch_resnet_skips(self.backbone, mode="random")
