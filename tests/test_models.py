import torch
from models import resnet_baseline, orthogonal_resnet


def test_forward_shapes():
    x = torch.randn(2, 3, 32, 32)
    baseline = resnet_baseline.resnet18()
    orth = orthogonal_resnet.OrthResNet([2, 2, 2, 2])
    for m in (baseline, orth):
        y = m(x)
        assert y.shape == (2, 10)
