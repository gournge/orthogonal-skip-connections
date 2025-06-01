import torch
from utils.orth_utils import (
    cayley_project,
    qr_retraction,
    svd_retraction,
    sharp_operator,
)


@torch.no_grad()
def _is_orth(M):
    I = torch.eye(M.shape[0])
    return torch.allclose(M.T @ M, I, atol=1e-5)


def test_cayley():
    M = torch.randn(16, 16)
    assert _is_orth(cayley_project(M))


def test_qr():
    M = torch.randn(16, 16)
    assert _is_orth(qr_retraction(M))


def test_svd():
    M = torch.randn(16, 16)
    assert _is_orth(svd_retraction(M))


def test_sharp():
    M = torch.randn(16, 16)
    assert _is_orth(sharp_operator(M))
