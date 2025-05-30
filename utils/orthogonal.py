"""Orthogonal matrix utilities and retractions.

Implements:
- Cayley transform parametrisation
- QR and SVD retractions
- Sharp operator
- Custom steepest-descent update on the orthogonal manifold
"""

from __future__ import annotations
import torch
import math
from typing import Callable, Dict

_RETRACTIONS: Dict[str, Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]] = (
    {}
)


def get_retraction_names() -> list[str]:
    """Get names of all registered retractions."""
    return list(_RETRACTIONS.keys())


def register_retraction(
    name: str, fn: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]
):
    _RETRACTIONS[name] = fn


def get_retraction(name: str):
    if name not in _RETRACTIONS:
        raise ValueError(f"Unknown retraction {name}")
    return _RETRACTIONS[name]


# ---------- Helper ops ----------
def orth_error(W: torch.Tensor) -> torch.Tensor:
    """Frobenius norm of W^T W - I."""
    I = torch.eye(W.size(0), device=W.device, dtype=W.dtype)
    return torch.norm(W.T @ W - I, p="fro")


# ---------- Sharp operator ----------
def sharp_operator(M: torch.Tensor, iters: int) -> torch.Tensor:
    """Newton‑Schulz iteration to project to nearest orthogonal matrix."""
    M = M / M.norm(p="fro")
    for _ in range(iters):
        M = 1.5 * M - 0.5 * M @ M.T @ M
    return M


# ---------- Retractions ----------
def retraction_qr(W: torch.Tensor, G: torch.Tensor, eta: float) -> torch.Tensor:
    Q, R = torch.linalg.qr(W - eta * G, mode="reduced")
    # Ensure determinant = 1 to stay in SO(n)
    if torch.det(Q) < 0:
        Q[..., -1] *= -1
    return Q


def retraction_svd(W: torch.Tensor, G: torch.Tensor, eta: float) -> torch.Tensor:
    U, _, Vh = torch.linalg.svd(W - eta * G, full_matrices=False)
    return U @ Vh


def retraction_cayley(W: torch.Tensor, G: torch.Tensor, eta: float) -> torch.Tensor:
    A = G @ W.T - W @ G.T  # skew‑sym.
    I = torch.eye(W.size(0), device=W.device, dtype=W.dtype)
    return torch.linalg.solve(I + 0.5 * eta * A, I - 0.5 * eta * A) @ W


# ---------- Custom steepest descent (Bernstein, 2025) ----------
def _compute_X(W: torch.Tensor, G: torch.Tensor, sharp_iters: int) -> torch.Tensor:
    return sharp_operator(W.T @ G - G.T @ W, sharp_iters)


def custom_steepest_descent_update(
    W: torch.Tensor,
    G: torch.Tensor,
    eta: float,
    sharp_iters: int = 8,
) -> torch.Tensor:
    X = _compute_X(W, G, sharp_iters)
    n = W.size(0)
    I = torch.eye(n, device=W.device, dtype=W.dtype)
    if torch.matrix_rank(W.T @ G - G.T @ W) == n:
        W_new = W @ (I - eta * X) / math.sqrt(1 + eta**2)
    else:
        W_new = sharp_operator(W @ (I - eta * X), sharp_iters)
    return W_new


def requires_sharp_iters(retraction: str) -> bool:
    """Check if the retraction requires sharp iterations."""
    return retraction in ["cayley", "steepest_manifold"]


# Register defaults
register_retraction("qr", retraction_qr)
register_retraction("svd", retraction_svd)
register_retraction("cayley", retraction_cayley)
register_retraction("steepest_manifold", custom_steepest_descent_update)
