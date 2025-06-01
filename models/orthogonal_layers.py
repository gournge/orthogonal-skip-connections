"""Light-weight wrappers and utilities for orthogonal (or near-orthogonal) skip
connections.  Any block wanting an *evolving* orthogonal W can import these.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from ..utils.orth_utils import (
    cayley_project,
    qr_retraction,
    svd_retraction,
    sharp_operator,
    skew,
)

class OrthogonalProjection(nn.Module):
    """Stores a learnable square matrix *W* constrained to remain orthogonal via
    a chosen *method* (string).  The forward pass multiplies the input by *W*.
    """

    def __init__(self, dim: int, method: str = "cayley"):
        super().__init__()
        self.W_raw = nn.Parameter(torch.eye(dim) + 0.01 * torch.randn(dim, dim))
        self.method = method

    def orthogonal(self) -> torch.Tensor:
        """Return the orthogonal projection of *W_raw* without altering the param.
        """
        if self.method == "cayley":
            return cayley_project(self.W_raw)
        if self.method == "qr":
            return qr_retraction(self.W_raw)
        if self.method == "svd":
            return svd_retraction(self.W_raw)
        if self.method == "sharp":
            return sharp_operator(self.W_raw)
        raise ValueError(f"Unknown orthogonal method {self.method}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.orthogonal().to(x)
        return x @ W.T  # (B, C) or treat 4-D via einops

    @property
    def orth_error(self) -> torch.Tensor:
        W = self.orthogonal()
        return torch.linalg.norm(W.T @ W - torch.eye(W.shape[0], device=W.device))
