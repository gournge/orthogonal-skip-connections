import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock
from .resnet_learnable_orthogonal import LearnableOrthogonalProjection
from utils.orthogonal import sharp_operator


def _make_W(ch, *, fixed=False, retraction="cayley", sharp_iters=8):
    if fixed:
        # random orthogonal once
        W = sharp_operator(torch.randn(ch, ch))
        proj = nn.Linear(ch, ch, bias=False)
        proj.weight.data.copy_(W)
        proj.weight.requires_grad_(False)
        return proj
    else:
        # learnable on the Stiefel
        return LearnableOrthogonalProjection(
            dim=ch, retraction=retraction, sharp_iters=sharp_iters
        )


def patch_resnet_skips(net, *, mode="learnable", **kw):
    """
    mode in {"learnable", "fixed", "random", "identity"}
    Returns list of learnable orthogonal modules (for retraction after each step).
    """
    ortho_modules = []
    for m in net.modules():
        if isinstance(m, BasicBlock):
            # only replace the identity-skip (no downsample) when in_channels == out_channels
            if m.downsample is None and m.conv1.in_channels == m.conv2.out_channels:
                ch = m.conv1.in_channels

                if mode == "identity":
                    proj = nn.Identity()
                elif mode == "random":
                    proj = nn.Linear(ch, ch, bias=False)
                    nn.init.normal_(proj.weight, std=1 / ch**0.5)
                    proj.weight.requires_grad_(False)
                else:
                    proj = _make_W(ch, fixed=(mode == "fixed"), **kw)
                    if mode == "learnable":
                        ortho_modules.append(proj)

                # wrap so shapes match: (B,C,H,W) → flatten channels → apply → restore
                m.downsample = nn.Sequential(
                    nn.Flatten(start_dim=2),  # (B, C, H*W)
                    proj,  # (B, C, H*W)
                    nn.Unflatten(2, (ch, -1)),  # back to (B, C, H*W)
                )
    return ortho_modules
