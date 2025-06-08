from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from sklearn.datasets import make_blobs
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------------------------------------------------------
#  Try to import existing orthogonal-skip abstraction + helpers
# -----------------------------------------------------------------------------
try:
    from orthogonal_skip_connections.models.skip import (
        LearnableOrthogonalConnection as _OrthConn,
        steepest_descent_update as _steepest_descent_update,
    )
except ImportError:

    def _steepest_descent_update(
        W: torch.Tensor, G: torch.Tensor, eta: float, update_rule_iters: int = 5
    ):
        """Simplified steepest-descent re-projection onto O(n)."""
        with torch.no_grad():
            A = W - eta * G
            Q, _ = torch.linalg.qr(A, mode="reduced")
            return Q

    class _OrthConn(nn.Module):
        """Fallback minimal learnable orthogonal connection"""

        def __init__(self, dim: int, eta: float = 1e-3, update_rule_iters: int = 5):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(dim, dim))
            nn.init.orthogonal_(self.weight)
            self.eta = eta
            self.iters = update_rule_iters

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x @ self.weight.T

        @torch.no_grad()
        def orth_update(self):
            if self.weight.grad is None:
                return
            self.weight.data.copy_(
                _steepest_descent_update(
                    self.weight.data, self.weight.grad.data, self.eta, self.iters
                )
            )


LearnableOrthogonalConnection = _OrthConn


# -----------------------------------------------------------------------------
#  Configuration
# -----------------------------------------------------------------------------
@dataclass
class DataCfg:
    seed: int = 0
    n_samples: int = 2000
    noise: float = 0.25
    batch_size: int = 256
    n_blobs: int = 50


@dataclass
class TrainCfg:
    epochs: int = 300
    lr: float = 1e-3
    eta: float = 1e-3
    update_rule_iters: int = 5


@dataclass
class SweepCfg:
    dimension_list: List[int] = field(default_factory=lambda: [2, 4, 8, 16])
    output_classes_list: List[int] = field(
        default_factory=lambda: [4]
    )  # currently unused
    n_blocks_list: List[int] = field(default_factory=lambda: [3, 4, 5, 6])


@dataclass
class Config:
    data: DataCfg = DataCfg()
    train: TrainCfg = TrainCfg()
    sweep: SweepCfg = SweepCfg()
    out_file: str = "output/orth_to_id"


cs = ConfigStore.instance()
cs.store(name="orth_to_id_cfg", node=Config)


# -----------------------------------------------------------------------------
#  Utilities
# -----------------------------------------------------------------------------
def set_seeds(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lipschitz_constant(f: nn.Module) -> float:
    L = 1.0
    for layer in f.modules():
        if isinstance(layer, nn.Linear):
            svals = torch.linalg.svdvals(layer.weight.detach())
            L *= svals.max().item()
    return L


# -----------------------------------------------------------------------------
#  Model definitions
# -----------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, dim: int, hidden_mult: int, train_cfg: TrainCfg):
        super().__init__()
        self.skip = LearnableOrthogonalConnection(
            dim, eta=train_cfg.eta, update_rule_iters=train_cfg.update_rule_iters
        )
        hidden = hidden_mult * dim
        self.f = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.f[-1].weight)
        nn.init.zeros_(self.f[-1].bias)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.f(x)

    def metrics(self) -> Tuple[float, float]:
        with torch.no_grad():
            I = torch.eye(self.dim, device=self.skip.weight.device)
            orth_to_id = torch.linalg.norm(self.skip.weight - I, ord="fro").item()
            lip_f = lipschitz_constant(self.f)
        return orth_to_id, lip_f

    def orth_update(self):
        if hasattr(self.skip, "orth_update"):
            self.skip.orth_update()


class OrthResNet(nn.Module):
    def __init__(self, dim: int, n_blocks: int, n_classes: int, cfg: Config):
        super().__init__()
        hidden_mult = 1  # you can parameterize this if desired
        self.blocks = nn.ModuleList(
            [ResBlock(dim, hidden_mult, cfg.train) for _ in range(n_blocks)]
        )
        # ← new classification head:
        self.classifier = nn.Linear(dim, n_classes)
        # good practice: initialize head (optional)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for blk in self.blocks:
            out = blk(out)
        return self.classifier(out)  # now returns [batch, n_classes]

    def orth_update(self):
        for blk in self.blocks:
            blk.orth_update()


# -----------------------------------------------------------------------------
#  Data & training
# -----------------------------------------------------------------------------
def make_dataloader(
    dim: int, n_classes: int, cfg: Config, verbose: bool = True
) -> DataLoader:
    rng = np.random.default_rng(cfg.data.seed)
    X, y = make_blobs(
        n_samples=cfg.data.n_samples,
        n_features=dim,
        centers=cfg.data.n_blobs,
        cluster_std=cfg.data.noise,
        random_state=cfg.data.seed,
    )

    y = y % n_classes

    if verbose:
        # project data to 2D for visualization and save a plot
        if dim > 2:
            # use pca to reduce X
            from sklearn.decomposition import PCA

            pca = PCA(n_components=2)
            X_reduced = pca.fit_transform(X)

        # plot the data
        plt.figure(figsize=(8, 6))
        plt.scatter(
            X_reduced[:, 0], X_reduced[:, 1], c=y, cmap="tab10", s=10, alpha=0.6
        )
        plt.title(f"Data Distribution (dim={dim}, n_classes={n_classes})")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.colorbar(label="Class Label")
        out_path = Path(cfg.out_file + f"_dim{dim}_nc{n_classes}").with_suffix(".png")
        plt.savefig(out_path, dpi=300)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=0,  # adjust as needed
    )
    return loader


def accuracy(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            _, predicted = torch.max(out, 1)
            total += yb.size(0)
            correct += (predicted == yb).sum().item()
    return correct / total if total > 0 else 0.0


def run_single_experiment(dim: int, n_blocks: int, n_classes: int, cfg: Config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = make_dataloader(dim, n_classes, cfg)
    model = OrthResNet(dim, n_blocks, n_classes, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    trajectories: List[List[Tuple[float, float]]] = [[] for _ in range(n_blocks)]
    for b_idx, blk in enumerate(model.blocks):
        trajectories[b_idx].append(blk.metrics())

    for epoch in range(cfg.train.epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
            model.orth_update()
            for b_idx, blk in enumerate(model.blocks):
                trajectories[b_idx].append(blk.metrics())
        if (epoch + 1) % 10 == 0 or epoch == cfg.train.epochs - 1:
            acc = accuracy(model, loader, device)
            print(
                f"\tEpoch {epoch + 1}/{cfg.train.epochs}, Loss: {loss.item():.4f}, Acc: {acc:.4f}"
            )
            for b_idx, blk in enumerate(model.blocks):
                orth_to_id, lip_f = blk.metrics()
                # print(
                #     f"  Block {b_idx + 1}: ||W-I||_F={orth_to_id:.4f}, L(f)={lip_f:.4f}"
                # )
                trajectories[b_idx].append((orth_to_id, lip_f))
    # Convert trajectories to tensor format
    all_traj = torch.zeros((n_blocks, len(trajectories[0]), 2), device=device)
    for b_idx, traj in enumerate(trajectories):
        for t_idx, (orth_to_id, lip_f) in enumerate(traj):
            all_traj[b_idx, t_idx, 0] = orth_to_id
            all_traj[b_idx, t_idx, 1] = lip_f

    # Convert each traj to a numpy array for easier plotting
    all_traj = all_traj.cpu().numpy()
    return all_traj


# -----------------------------------------------------------------------------
#  Plotting
# -----------------------------------------------------------------------------
def plot_grid(cfg: Config, all_runs: dict[Tuple[int, int], torch.Tensor]):
    dims = cfg.sweep.dimension_list
    n_blocks_list = cfg.sweep.n_blocks_list
    n_rows, n_cols = len(dims), len(n_blocks_list)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    cmap = plt.cm.get_cmap("tab10")

    for r, dim in enumerate(dims):
        for c, n_blocks in enumerate(n_blocks_list):
            ax = axes[r, c] if n_rows > 1 or n_cols > 1 else axes
            traj = all_runs[(dim, n_blocks)]
            for b in range(n_blocks):
                col = cmap(b % 10)
                x = traj[b, :, 0]
                y = traj[b, :, 1]
                ax.plot(x, y, label=f"blk {b+1}", color=col, linewidth=2)
                ax.scatter(
                    x[0], y[0], marker="o", s=40, color=col, label=f"start blk {b+1}"
                )
                ax.scatter(
                    x[-1], y[-1], marker="X", s=60, color=col, label=f"end blk {b+1}"
                )
            ax.set_xlabel(r"$||W-I||_F$")
            ax.set_ylabel(r"$L(f)$")
            ax.set_title(f"d={dim}, blocks={n_blocks}")
            ax.grid(True, linestyle=":", linewidth=0.5)
    handles, labels = axes[-1, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = Path(cfg.out_file).with_suffix(".png")
    fig.savefig(out_path, dpi=300)
    print(f"Saved figure to {out_path.resolve()}")


# -----------------------------------------------------------------------------
#  Entry point
# -----------------------------------------------------------------------------
@hydra.main(version_base="1.3", config_name="orth_to_id_cfg")
def main(cfg: Config):
    print("==== Orth-to-Identity Sweep ====")
    print(cfg)
    set_seeds(cfg.data.seed)
    all_runs = {}
    for dim in cfg.sweep.dimension_list:
        for n_blocks in cfg.sweep.n_blocks_list:
            for n_classes in cfg.sweep.output_classes_list:
                print(f"Running d={dim}, n_blocks={n_blocks}, n_classes={n_classes}")
                try:
                    traj = run_single_experiment(dim, n_blocks, n_classes, cfg)
                except Exception as e:
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        f"Failed for d={dim}, n_blocks={n_blocks}, n_classes={n_classes}: {e}"
                    )
                all_runs[(dim, n_blocks)] = traj

    plot_grid(cfg, all_runs)


if __name__ == "__main__":
    main()
