"""Construction Validity Sweep
================================
Runs a sweep over (data-dimension, output-classes, #blocks) to quantify how
well a *trained* ResNet with *learnable orthogonal skip connections* matches the
theoretical construction

    h = h_m \circ … \circ h_1,  with  h_i(x) = W_i x,

on a synthetic blob-classification task.

For every trio (d, k, B)
* a classification dataset in **d** dimensions and **k** classes is created,
* a depth-`B` residual network is trained,
* each residual block is fitted with the *best* orthogonal matrix (Procrustes),
  and the relative MSE ‖f_i(x)–Ŵ_i x‖²/‖f_i(x)‖² is recorded.

Two outputs are saved in the working directory Hydra places us in (``${hydra:runtime.output_dir}``):

* **alignment_grid.png** – a |dim|×|classes| grid; every subplot shows the
  per-block alignment curves for the different B values.
* **alignment_heatmap.png** – a heat-map with *rows = #blocks*, *cols =
  dimension*, colour = mean alignment error (averaged over blocks & classes).

Run with e.g.::

    python construction_validity_sweep.py \
        dimension_list="[2,4,8,16]" \
        output_classes_list="[2,4]" \
        n_blocks_list="[2,4,6]"

All arguments have reasonable defaults and can be overridden from the CLI or
through a YAML config.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.datasets import make_blobs
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from orthogonal_skip_connections.models.skip import (
    steepest_descent_update,
)


# -----------------------------------------------------------------------------
#  Hyper-parameter / experiment configuration handled by Hydra
# -----------------------------------------------------------------------------


@dataclass
class DataCfg:
    seed: int = 0
    n_samples: int = 1_000
    noise: float = 0.01
    batch_size: int = 256
    n_blobs: int = 50


@dataclass
class TrainCfg:
    epochs: int = 50
    lr: float = 1e-3
    eta: float = 1e-3  # orthogonal retraction step size


@dataclass
class SweepCfg:
    dimension_list: List[int] = field(default_factory=lambda: [2, 4, 8])
    output_classes_list: List[int] = field(default_factory=lambda: [2])
    n_blocks_list: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 6])


@dataclass
class Config:
    data: DataCfg = DataCfg()
    train: TrainCfg = TrainCfg()
    sweep: SweepCfg = SweepCfg()


# -----------------------------------------------------------------------------
#  Data utilities
# -----------------------------------------------------------------------------


def build_dataset(dim: int, n_classes: int, cfg: DataCfg) -> DataLoader:
    """Synthetic Gaussian blobs for classification."""

    rng = np.random.default_rng(cfg.seed)
    X, y = make_blobs(
        n_samples=cfg.n_samples,
        n_features=dim,
        centers=cfg.n_blobs,
        cluster_std=cfg.noise * dim,
        random_state=cfg.seed,
    )

    # make y have exactly n_classes classes
    # use modulo to ensure we have exactly n_classes
    y = np.array([i % n_classes for i in y])

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    return loader


# -----------------------------------------------------------------------------
#  Model building blocks
# -----------------------------------------------------------------------------


class OrthLinear(nn.Module):
    """Linear layer with a *strictly orthogonal* weight matrix (square)."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dim))
        nn.init.orthogonal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, d)
        return x @ self.weight.T  # (N, d)

    # ------------------------------------------------------------------
    #  Retraction onto the Stiefel manifold using steepest-descent update
    # ------------------------------------------------------------------
    def orth_update(self, eta: float = 1e-3):
        if self.weight.grad is None:
            return
        self.weight.data = steepest_descent_update(
            self.weight.data, self.weight.grad, eta, update_rule_iters=5
        )


class ResidualBlock(nn.Module):
    """Residual block ``y = W x + f(x)`` with *learnable orthogonal* ``W``."""

    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or 2 * dim
        self.W = OrthLinear(dim)
        self.f = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.f[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W(x) + self.f(x)

    # Hydra expects hooks to exist on the *module*, so we expose this.
    def orth_update(self, eta: float = 1e-3):  # noqa: D401
        self.W.orth_update(eta)


class ResidualClassifier(nn.Module):
    """A stack of residual blocks followed by a linear head."""

    def __init__(self, dim: int, n_blocks: int, n_classes: int):
        super().__init__()
        self.blocks = nn.ModuleList([ResidualBlock(dim) for _ in range(n_blocks)])
        self.head = nn.Linear(dim, n_classes)
        nn.init.kaiming_normal_(self.head.weight, nonlinearity="linear")
        nn.init.zeros_(self.head.bias)

    # ------------------------------------------------------------------
    #  Forward (optionally returning per-block intermediates)
    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, *, return_intermediates: bool = False
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        intermediates: List[Tuple[torch.Tensor, torch.Tensor]] = []
        z = x
        for blk in self.blocks:
            z_in = z
            z = blk(z)
            if return_intermediates:
                intermediates.append((z_in, z))
        logits = self.head(z)
        if return_intermediates:
            return logits, intermediates
        return logits, []


# -----------------------------------------------------------------------------
#  Training & evaluation helpers
# -----------------------------------------------------------------------------


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if logits.size(1) == 1:  # binary – use sigmoid
        preds = (torch.sigmoid(logits.squeeze(1)) > 0.5).long()
    else:  # multi-class – use argmax
        preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


@torch.no_grad()
def procrustes_alignment_error(x: torch.Tensor, y: torch.Tensor) -> float:
    """Return relative MSE of the best orthogonal fit ``Ŵ`` such that ``y≈Ŵx``."""

    # solve argmin_W ‖y - Wx‖² s.t. W^⊤W = I  (orthogonal Procrustes)
    m = y.T @ x  # (d,d)
    u, _, vT = torch.linalg.svd(m, full_matrices=True)
    W_hat = u @ vT
    y_hat = x @ W_hat.T  # (N,d)
    mse = torch.mean(torch.sum((y - y_hat) ** 2, dim=1))
    denom = torch.mean(torch.sum(y**2, dim=1)) + 1e-8
    return (mse / denom).cpu().item()


def train_one_model(
    dim: int,
    n_classes: int,
    n_blocks: int,
    loader: DataLoader,
    train_cfg: TrainCfg,
) -> Tuple[ResidualClassifier, List[float]]:
    """Train a single network and return per-block alignment errors."""

    torch.manual_seed(train_cfg.epochs + dim * 17 + n_blocks * 31 + n_classes * 13)
    model = ResidualClassifier(dim, n_blocks, n_classes).train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    opt = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)

    # -------------------- optimisation loop ---------------------------
    for ep in range(train_cfg.epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            # retraction onto the manifold after *every* optimiser step
            for m in model.modules():
                if hasattr(m, "orth_update"):
                    m.orth_update(train_cfg.eta)
        if ep % 10 == 0 or ep == train_cfg.epochs - 1:
            acc = _accuracy(logits, yb)
            print(
                f"Epoch {ep:4d} | Loss: {loss.item():.4f} | "
                f"Accuracy: {acc:.4f} | Blocks: {n_blocks}, Dim: {dim}, Classes: {n_classes}"
            )

    return model.cpu(), _gather_alignment_errors(model, loader, device)


def _gather_alignment_errors(
    model: ResidualClassifier,
    loader: DataLoader,
    device: torch.device,
) -> List[float]:
    """
    We have a theoretical model of what residual blocks are (written in terms of the ideal h we know )
    We want to see how well the trained model matches it.
    """

    """Return per-block alignment errors for the trained model on the dataset."""

    model.to(device)  # Ensure model is on the correct device
    model.eval()
    errors = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            _, intermediates = model(xb, return_intermediates=True)
            for z_in, z_out in intermediates:
                error = procrustes_alignment_error(z_in, z_out)
                errors.append(error)

    # average over all blocks
    n_blocks = len(model.blocks)
    errors = [math.sqrt(e) for e in np.array(errors).reshape(-1, n_blocks).mean(axis=0)]
    return errors


# -----------------------------------------------------------------------------
#  Plotting utilities
# -----------------------------------------------------------------------------


def plot_alignment_grid(
    errors_dict: Dict[Tuple[int, int, int], List[float]],
    cfg: Config,
    workdir: Path,
):
    """Save the |dim|x|classes| grid with per-block alignment curves."""

    dims = cfg.sweep.dimension_list
    classes = cfg.sweep.output_classes_list
    n_blocks_list = cfg.sweep.n_blocks_list

    n_rows, n_cols = len(dims), len(classes)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharey=True
    )

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[None, :]
    elif n_cols == 1:
        axes = axes[:, None]

    for r, dim in enumerate(dims):
        for c, k in enumerate(classes):
            ax = axes[r, c]
            for n_blocks in n_blocks_list:
                key = (dim, k, n_blocks)
                errs = errors_dict[key]
                ax.plot(
                    np.arange(1, n_blocks + 1),
                    errs,
                    marker="o",
                    label=f"B={n_blocks}",
                )
            if r == 0:
                ax.set_title(f"k={k}")
            if c == 0:
                ax.set_ylabel(f"d={dim}\nrelative MSE")
            ax.set_xlabel("block index i")
            ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=len(n_blocks_list), fontsize="small"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = workdir / "alignment_grid.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved per-block alignment grid → {out_path}")


def plot_heatmap(
    errors_dict: Dict[Tuple[int, int, int], List[float]],
    cfg: Config,
    workdir: Path,
):
    """Save a heat-map of *mean* alignment error vs (#blocks, dimension)."""

    dims = cfg.sweep.dimension_list
    n_blocks_list = cfg.sweep.n_blocks_list
    classes = cfg.sweep.output_classes_list

    heat = np.zeros((len(n_blocks_list), len(dims)))
    for i_b, B in enumerate(n_blocks_list):
        for j_d, d in enumerate(dims):
            errs = []
            for k in classes:
                errs.extend(errors_dict[(d, k, B)])
            heat[i_b, j_d] = np.mean(errs)

    fig, ax = plt.subplots(figsize=(1 + 0.9 * len(dims), 1 + 0.7 * len(n_blocks_list)))
    im = ax.imshow(heat, origin="lower", aspect="auto", cmap="plasma")
    ax.set_xticks(np.arange(len(dims)), dims)
    ax.set_yticks(np.arange(len(n_blocks_list)), n_blocks_list)
    ax.set_xlabel("dimension d")
    ax.set_ylabel("# blocks B")
    ax.set_title("Mean relative MSE (lower = better)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out_path = workdir / "alignment_heatmap.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved alignment heat-map → {out_path}")


# -----------------------------------------------------------------------------
#  Main driver
# -----------------------------------------------------------------------------


@hydra.main(version_base="1.3", config_path=None, config_name=None)
def main(cfg: DictConfig):  # noqa: D401  – Hydra passes DictConfig
    cfg = OmegaConf.merge(OmegaConf.structured(Config()), cfg)
    print("Loaded configuration:\n" + OmegaConf.to_yaml(cfg))

    # deterministic-ish
    torch.manual_seed(cfg.data.seed)
    random.seed(cfg.data.seed)
    np.random.seed(cfg.data.seed)

    workdir = Path(os.getcwd())  # Hydra changes cwd → output dir

    errors_dict: Dict[Tuple[int, int, int], List[float]] = {}

    for dim in cfg.sweep.dimension_list:
        for k in cfg.sweep.output_classes_list:
            loader = build_dataset(dim, k, cfg.data)
            for B in cfg.sweep.n_blocks_list:
                print(f"→ Training model: d={dim}, k={k}, B={B}…")
                _, errors = train_one_model(dim, k, B, loader, cfg.train)
                errors_dict[(dim, k, B)] = errors

    # plotting −−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−
    plot_alignment_grid(errors_dict, cfg, workdir)
    plot_heatmap(errors_dict, cfg, workdir)


if __name__ == "__main__":
    main()
