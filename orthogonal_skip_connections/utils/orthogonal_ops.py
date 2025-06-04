"""Orthogonality-related helper functions."""
from __future__ import annotations

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Basic transforms
# ---------------------------------------------------------------------------

def cayley_transform(A: torch.Tensor) -> torch.Tensor:
    """Cayley transform to map skew-symmetric A to orthogonal Q = (I - A)(I + A)^{-1}."""
    I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
    return torch.linalg.solve(I + A, I - A)

def qr_reorthogonal(W: torch.Tensor) -> torch.Tensor:
    Q, _ = torch.linalg.qr(W)
    return Q

def svd_sharp(W: torch.Tensor) -> torch.Tensor:
    U, _, Vh = torch.linalg.svd(W, full_matrices=False)
    return U @ Vh

# ---------------------------------------------------------------------------
# Custom steepest descent on St(O(n)) – from Modula Docs 2025
# ---------------------------------------------------------------------------

def steepest_descent_update(W: torch.Tensor, G: torch.Tensor, eta: float) -> torch.Tensor:
    """One step of the steepest descent update that *stays* on the orthogonal manifold."""
    skew = W.T @ G - G.T @ W
    X = svd_sharp(skew)
    if torch.linalg.matrix_rank(skew) == W.size(0):
        return (W @ (torch.eye(W.size(0), device=W.device) - eta * X)) / torch.sqrt(torch.tensor(1 + eta**2, device=W.device))
    return svd_sharp(W @ (torch.eye(W.size(0), device=W.device) - eta * X))

# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------

def reorthogonalize_model(model: torch.nn.Module, eta: float = 1e-3):
    """Call `orth_update` on every module that implements it."""
    for m in model.modules():
        if hasattr(m, "orth_update"):
            m.orth_update(eta)