# models/patch.py

import torch.nn as nn
from torchvision.models.resnet import BasicBlock
from .resnet_learnable_orthogonal import LearnableOrthogonalProjection


class _Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


def patch_resnet_skips(net: nn.Module, *, retraction="cayley", sharp_iters=8):
    ortho_modules = []
    for m in net.modules():
        if isinstance(m, BasicBlock) and m.downsample is None:
            C = m.conv1.in_channels
            proj = LearnableOrthogonalProjection(
                dim=C, retraction=retraction, sharp_iters=sharp_iters
            )
            ortho_modules.append(proj)

            # replace “identity” skip by x → Wᵀx on channels
            m.i_downsample = nn.Sequential(
                _Permute(0, 2, 3, 1),  # B,C,H,W → B,H,W,C
                proj,  # applies Wᵀ on last dim
                _Permute(0, 3, 1, 2),  # back to B,C,H,W
            )
    return ortho_modules
