import torch

def spectral_norm(W: torch.Tensor):
    return torch.linalg.svdvals(W)[0]

def orth_error(W):
    return torch.linalg.norm(W.T @ W - torch.eye(W.shape[0], device=W.device))
