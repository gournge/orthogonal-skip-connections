from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from sklearn.decomposition import PCA
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, Subset
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from sklearn.datasets import make_blobs
from torch.utils.data import DataLoader, TensorDataset

from orthogonal_skip_connections.models.skip import (
    LearnableOrthogonalSkip,
)


# -----------------------------------------------------------------------------
#  Configuration
# -----------------------------------------------------------------------------
@dataclass
class DataCfg:
    seed: int = 0  # random seed
    batch_size: int = 256  # batch size
    n_samples_per_class: int = 1000  # samples per class
    noise: float = 0.5  # blob std deviation
    sep_factor: float = 5.0  # separation factor “s”
    n_blobs_per_dim: float = 20.0  # blobs-per-class = n_blobs_per_dim * dim
    dataset_type: str = "blobs"  # either "blobs" or "parity"
    num_workers: int = 0  # optional for DataLoader
    test_split: float = 0.2


@dataclass
class TrainCfg:
    epochs: int = 200
    lr: float = 1e-2
    eta: float = 1e-3


@dataclass
class SweepCfg:
    dimension_list: List[int] = field(default_factory=lambda: [2, 3])
    n_blocks_list: List[int] = field(default_factory=lambda: [3, 4])
    seeds_list: List[int] = field(default_factory=lambda: [0])


@dataclass
class Config:
    data: DataCfg = DataCfg()
    train: TrainCfg = TrainCfg()
    sweep: SweepCfg = SweepCfg()
    n_classes: int = 2
    out_file: str = "orth_to_id"


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
    def __init__(self, dim: int, hidden_mult: int):
        super().__init__()
        self.skip = LearnableOrthogonalSkip(dim)
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
        # if we're being fed a “flat” [N, C] tensor, do the 2-D skip by hand
        if x.ndim == 2:
            skip_out = x @ self.skip.weight.T
        # otherwise assume [N, C, H, W] and call the skip module as-is
        elif x.ndim == 4:
            skip_out = self.skip(x)
        else:
            raise ValueError(f"Unexpected input dims {x.ndim}")

        return skip_out + self.f(x)

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
        hidden_mult = 2
        self.blocks = nn.ModuleList(
            [ResBlock(dim, hidden_mult) for _ in range(n_blocks)]
        )
        self.classifier = nn.Linear(dim, n_classes)
        # initialize classification head with random orthogonal weights
        nn.init.orthogonal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for blk in self.blocks:
            out = blk(out)
        return self.classifier(out)

    def orth_update(self):
        for blk in self.blocks:
            blk.orth_update()


# -----------------------------------------------------------------------------
#  Data & training
# -----------------------------------------------------------------------------
def make_dataloaders(
    dim: int, n_classes: int, cfg: Config
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns a DataLoader for either:
      - constant‐difficulty Gaussian blobs (cfg.data.dataset_type=="blobs")
      - d‐dimensional parity (cfg.data.dataset_type=="parity")
    """
    seed = cfg.data.seed
    batch_size = cfg.data.batch_size
    rng = np.random.default_rng(seed)

    if cfg.data.dataset_type == "parity":
        # --- parity task: X in {0,1}^dim, label = sum(X) mod n_classes
        n = cfg.data.n_samples_per_class * n_classes
        # sample uniform bits
        X = rng.integers(0, 2, size=(n, dim))
        # compute parity (or general mod-n_classes)
        y = X.sum(axis=1) % n_classes

    else:
        # --- Gaussian-mixture blobs with RANDOM centers at fixed Bayes-error
        n = cfg.data.n_samples_per_class * n_classes
        blobs_per_class = cfg.data.n_blobs_per_dim * dim
        total_blobs = int(blobs_per_class)

        # 1) Draw centers ~ N(0, I)
        raw_centers = rng.standard_normal((total_blobs, dim))

        # 2) Compute expected distance of two random Gaussians in dim-D:
        #    E||c1 − c2|| = sqrt(2) * Γ((dim+1)/2) / Γ(dim/2)
        E_dist = math.sqrt(2) * math.gamma((dim + 1) / 2) / math.gamma(dim / 2)

        # 3) Scale so that average pairwise distance = sep_factor*sqrt(dim)*sigma
        sigma = cfg.data.noise
        target = cfg.data.sep_factor * math.sqrt(dim) * sigma
        scale = target / E_dist
        centers = raw_centers * scale

        # 4) Generate blobs and fold into n_classes
        X, y_blob = make_blobs(
            n_samples=n,
            n_features=dim,
            centers=centers,
            cluster_std=sigma,
            random_state=seed,
        )
        y = (y_blob % n_classes).astype(int)

    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()
    ds = TensorDataset(X_tensor, y_tensor)

    # split into train / test
    n_test = int(len(ds) * cfg.data.test_split)
    n_train = len(ds) - n_test
    train_ds, test_ds = random_split(
        ds, [n_train, n_test], generator=torch.Generator().manual_seed(cfg.data.seed)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )
    return train_loader, test_loader


def accuracy(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            _, pred = out.max(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return correct / total


def run_single_experiment(
    dim: int, n_blocks: int, n_classes: int, cfg: Config, plot=True
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = make_dataloaders(dim, n_classes, cfg)
    model = OrthResNet(dim, n_blocks, n_classes, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    trajectories: List[List[Tuple[float, float]]] = [[] for _ in range(n_blocks)]
    for b_idx, blk in enumerate(model.blocks):
        trajectories[b_idx].append(blk.metrics())

    for epoch in range(cfg.train.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            model.orth_update()
            for b_idx, blk in enumerate(model.blocks):
                orth_to_id, lip_f = blk.metrics()
                trajectories[b_idx].append((orth_to_id, lip_f))

        if (epoch + 1) % 10 == 0 or epoch == cfg.train.epochs - 1:
            train_acc = accuracy(model, train_loader, device)
            test_acc = accuracy(model, test_loader, device)
            print(
                f"Epoch {epoch+1}/{cfg.train.epochs}  "
                f"Train Acc: {train_acc:.4f}  Test Acc: {test_acc:.4f}"
            )

    # Plot the data distribution and in the background plot what the model classifies
    if plot:
        plot_data_distribution(train_loader, model, dim, n_classes)

    all_traj = torch.zeros((n_blocks, len(trajectories[0]), 2), device=device)
    for b_idx, traj in enumerate(trajectories):
        for t_idx, (orth_to_id, lip_f) in enumerate(traj):
            all_traj[b_idx, t_idx, 0] = orth_to_id
            all_traj[b_idx, t_idx, 1] = lip_f

    return all_traj.cpu().numpy()


# -----------------------------------------------------------------------------
#  Plotting (averaged over seeds with error bands)
# -----------------------------------------------------------------------------
def plot_data_distribution(
    loader: DataLoader, model: nn.Module, dim: int, n_classes: int
):
    # unpack X_all, y_all whether loader.dataset is a Subset or a TensorDataset
    ds = loader.dataset
    if isinstance(ds, Subset):
        # Subset(dataset, indices)
        orig_ds = ds.dataset
        idxs = ds.indices
        X_tensor = orig_ds.tensors[0][idxs]
        y_tensor = orig_ds.tensors[1][idxs]
    else:
        # plain TensorDataset
        X_tensor, y_tensor = ds.tensors

    X_all = X_tensor.cpu().numpy()
    y_all = y_tensor.cpu().numpy()

    # define a grid over feature space
    # project data to 2D
    pca = PCA(n_components=2)
    X_proj = pca.fit_transform(X_all)

    # build the grid in PCA space
    x_min, x_max = X_proj[:, 0].min() - 1.0, X_proj[:, 0].max() + 1.0
    y_min, y_max = X_proj[:, 1].min() - 1.0, X_proj[:, 1].max() + 1.0
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )

    # map grid back to original feature space
    grid_proj = np.stack([xx.ravel(), yy.ravel()], axis=1)
    # map grid back to original feature space
    grid = pca.inverse_transform(grid_proj)

    # run model on grid
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(torch.from_numpy(grid).float().to(device))
        Z = logits.argmax(dim=1).cpu().numpy().reshape(xx.shape)

    # plot decision regions
    plt.figure(figsize=(8, 6))
    plt.contourf(
        xx, yy, Z, levels=np.arange(n_classes + 1) - 0.5, cmap="tab10", alpha=0.3
    )
    # overlay the true samples
    plt.scatter(
        X_all[:, 0],
        X_all[:, 1],
        c=y_all,
        cmap="tab10",
        s=15,
        edgecolor="k",
        linewidth=0.2,
        alpha=0.6,
    )
    plt.title(f"Decision Boundary (dim={dim}, classes={n_classes})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    # save to file
    out_path = Path(f"pred_dim{dim}_nc{n_classes}").with_suffix(".png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved decision boundary plot to {out_path.resolve()}")


def plot_grid(cfg: Config, all_runs: dict[Tuple[int, int], np.ndarray]):
    dims = cfg.sweep.dimension_list
    n_blocks_list = cfg.sweep.n_blocks_list
    fig, axes = plt.subplots(
        len(dims), len(n_blocks_list), figsize=(4 * len(n_blocks_list), 3.5 * len(dims))
    )
    cmap = plt.cm.get_cmap("tab10")

    for r, dim in enumerate(dims):
        for c, n_blocks in enumerate(n_blocks_list):
            ax = axes[r, c] if axes.ndim > 1 else axes[max(r, c)]
            runs = all_runs[(dim, n_blocks)]  # shape: (n_seeds, n_blocks, T, 2)
            mean_traj = np.mean(runs, axis=0)  # (n_blocks, T, 2)
            std_traj = np.std(runs, axis=0)  # (n_blocks, T, 2)
            for b in range(n_blocks):
                col = cmap(b % 10)
                mx = mean_traj[b, :, 0]
                my = mean_traj[b, :, 1]
                sx = std_traj[b, :, 0]
                sy = std_traj[b, :, 1]
                ax.plot(mx, my, label=f"blk {b+1}", color=col, linewidth=2)
                ax.fill_between(mx, my - sy, my + sy, alpha=0.2, color=col)
                ax.fill_betweenx(my, mx - sx, mx + sx, alpha=0.2, color=col)
                ax.scatter(mx[0], my[0], marker="o", s=40, color=col)
                ax.scatter(mx[-1], my[-1], marker="X", s=60, color=col)
            ax.set_xlabel(r"$\|W-I\|_F$")
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
    print("==== Orth-to-Identity Sweep (averaged over seeds) ====")
    print(cfg)
    all_runs: dict[Tuple[int, int], np.ndarray] = {}
    for dim in cfg.sweep.dimension_list:
        for n_blocks in cfg.sweep.n_blocks_list:
            traj_list = []
            for seed in cfg.sweep.seeds_list:
                print(
                    f"Running seed={seed}, dim={dim}, blocks={n_blocks}, classes={cfg.n_classes}"
                )
                cfg.data.seed = seed
                set_seeds(seed)
                traj = run_single_experiment(dim, n_blocks, cfg.n_classes, cfg)
                traj_list.append(traj)
            all_runs[(dim, n_blocks)] = np.stack(traj_list, axis=0)

    print(all_runs)

    plot_grid(cfg, all_runs)


if __name__ == "__main__":
    main()
