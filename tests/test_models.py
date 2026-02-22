import torch

from orthogonal_skip_connections.models import get_model
from orthogonal_skip_connections.models.skip import get_skip_names


def test_zero_skip_is_registered():
    assert "zero" in get_skip_names()


def test_resnet56_identity_forward_shape():
    model = get_model(num_classes=10, model_type="resnet56", skip_kind="identity")
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (2, 10)


def test_resnet56_no_skip_forward_shape():
    model = get_model(num_classes=100, model_type="resnet56", skip_kind="zero")
    x = torch.randn(4, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (4, 100)
