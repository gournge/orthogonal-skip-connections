import torch

@torch.no_grad()
def spectral_norm(matrix: torch.Tensor):
    return torch.linalg.svdvals(matrix)[0]

@torch.no_grad()
def compute_orthogonality_deviation(model):
    """Return a scalar measuring \sum \|W^T W - I\|_F^2 over all learnable orth skips."""
    dev = 0.0
    for m in model.modules():
        if m.__class__.__name__.endswith("OrthogonalSkip") and hasattr(m, "_get_W_matrix"):
            W = m._get_W_matrix()
            dev += torch.norm(W.T @ W - torch.eye(W.size(0), device=W.device), p="fro").item()
    return dev