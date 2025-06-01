"""Matrix helpers for orthogonality enforcement & diagnostics."""
from __future__ import annotations
import torch

def skew(M: torch.Tensor) -> torch.Tensor:
    return 0.5 * (M - M.T)

def cayley_project(M: torch.Tensor) -> torch.Tensor:
    """Project via Cayley transform (first-order)."""
    A = skew(M)
    I = torch.eye(A.shape[0], device=A.device)
    return torch.linalg.solve(I + 0.5 * A, I - 0.5 * A)

def qr_retraction(M):
    Q, _ = torch.linalg.qr(M)
    return Q

def svd_retraction(M):
    U, _, Vt = torch.linalg.svd(M, full_matrices=False)
    return U @ Vt

def sharp_operator(M: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """Newton-Schulz iteration to *polarise* a matrix (modula docs)."""
    norm = M.norm(p="fro")
    Y = M / norm
    I = torch.eye(M.shape[0], device=M.device)
    for _ in range(iters):
        Y = 1.5 * Y - 0.5 * Y @ Y.T @ Y
    return Y
