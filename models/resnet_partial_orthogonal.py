from .resnet_baseline import ResNetBaseline
from .patch import patch_resnet_skips
import torch.nn as nn


class ResNetPartialOrthogonal(ResNetBaseline):
    def __init__(self, mask, *args, retraction="cayley", sharp_iters=8, **kwargs):
        """
        mask: iterable of booleans, one per BasicBlock skip.
        True = learnable orthogonal, False = identity.
        """
        super().__init__(*args, **kwargs)
        mask_iter = iter(mask)

        def choose_mode():
            return "learnable" if next(mask_iter, False) else "identity"

        self.ortho_modules = patch_resnet_skips(
            self.backbone,
            mode=choose_mode(),
            retraction=retraction,
            sharp_iters=sharp_iters,
        )
