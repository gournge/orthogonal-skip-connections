import torch
from orthogonal_skip_connections.utils.orthogonal_ops import cayley_transform, qr_reorthogonal, svd_sharp

def is_orthogonal(Q, eps=1e-5):
    I = torch.eye(Q.size(0))
    return torch.allclose(Q.T @ Q, I, atol=eps)

def test_cayley():
    A = torch.randn(4,4)
    A = A - A.T  # skew-sym.
    Q = cayley_transform(A)
    assert is_orthogonal(Q)

def test_qr():
    W = torch.randn(8, 8)
    Q = qr_reorthogonal(W)
    assert is_orthogonal(Q)

def test_sharp():
    W = torch.randn(16, 16)
    Q = svd_sharp(W)
    assert is_orthogonal(Q)