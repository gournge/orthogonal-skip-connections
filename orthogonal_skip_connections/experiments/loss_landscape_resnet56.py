"""Compare 3D loss landscapes for ResNet56 variants on CIFAR datasets.

Methodology:
- Pick converged weights theta* for each model variant.
- Sample two random Gaussian directions and apply filter-wise normalization.
- Keep batch-norm parameters fixed while perturbing the model.
- Evaluate loss on a 2D grid around theta* and render a 3D surface.
"""

from __future__ import annotations

import random
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

from orthogonal_skip_connections.models import get_model
from orthogonal_skip_connections.utils.orthogonal_ops import reorthogonalize_model


_DATASETS = {
    "cifar10": (torchvision.datasets.CIFAR10, 10),
    "cifar100": (torchvision.datasets.CIFAR100, 100),
}

_VARIANTS = [
    ("ResNet56 (no skip)", "zero"),
    ("ResNet56 (identity skip)", "identity"),
    ("ResNet56 (learnable orth skip)", "learnable_orth"),
]


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return torch.device(device)


def get_loaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    ds_cls, num_classes = _DATASETS[cfg.dataset]
    data_root = str(Path(cfg.data_root).expanduser())

    train_transform = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    eval_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )

    train_set = ds_cls(root=data_root, train=True, download=True, transform=train_transform)
    test_set = ds_cls(root=data_root, train=False, download=True, transform=eval_transform)

    if cfg.landscape.eval_subset_size > 0 and cfg.landscape.eval_subset_size < len(test_set):
        rng = np.random.default_rng(cfg.seed)
        indices = rng.choice(
            len(test_set), size=cfg.landscape.eval_subset_size, replace=False
        ).tolist()
        test_subset = Subset(test_set, indices)
    else:
        test_subset = test_set

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    landscape_loader = DataLoader(
        test_subset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    return train_loader, test_loader, landscape_loader, num_classes


def evaluate_loss(model: torch.nn.Module, loader: DataLoader, device: torch.device, max_batches: int) -> float:
    model.eval()
    loss_sum = 0.0
    n_items = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
            n_items += y.size(0)
    return loss_sum / max(1, n_items)


def evaluate_accuracy(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / max(1, total)


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    cfg: DictConfig,
    device: torch.device,
    variant_name: str,
) -> None:
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.optimizer.lr,
        momentum=cfg.optimizer.momentum,
        weight_decay=cfg.optimizer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.training.epochs)
    )

    for epoch in range(cfg.training.epochs):
        model.train()
        running_loss = 0.0
        n_items = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            if cfg.training.max_train_batches > 0 and batch_idx >= cfg.training.max_train_batches:
                break
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=cfg.training.label_smoothing)
            loss.backward()
            optimizer.step()

            # Keep learnable orthogonal skip matrices on manifold.
            reorthogonalize_model(model)

            running_loss += loss.item() * y.size(0)
            n_items += y.size(0)

        scheduler.step()
        mean_loss = running_loss / max(1, n_items)
        print(f"[{variant_name}] epoch {epoch + 1}/{cfg.training.epochs} loss={mean_loss:.4f}")


def _filter_normalize(direction: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if direction.ndim < 2:
        return torch.zeros_like(direction)

    d_view = direction.reshape(direction.shape[0], -1)
    w_view = weight.reshape(weight.shape[0], -1)
    d_norm = torch.norm(d_view, dim=1, keepdim=True).clamp_min(1e-12)
    w_norm = torch.norm(w_view, dim=1, keepdim=True)
    d_view = d_view / d_norm * w_norm
    return d_view.reshape_as(direction)


def build_filterwise_directions(
    model: torch.nn.Module, seed: int
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    set_seeds(seed)
    dir1: dict[str, torch.Tensor] = {}
    dir2: dict[str, torch.Tensor] = {}

    for name, param in model.named_parameters():
        if "bn" in name or param.ndim < 2:
            dir1[name] = torch.zeros_like(param)
            dir2[name] = torch.zeros_like(param)
            continue
        d1 = _filter_normalize(torch.randn_like(param), param)
        d2 = _filter_normalize(torch.randn_like(param), param)
        dir1[name] = d1
        dir2[name] = d2

    return dir1, dir2


def compute_loss_surface(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: DictConfig,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alphas = np.linspace(-cfg.landscape.radius, cfg.landscape.radius, cfg.landscape.steps)
    betas = np.linspace(-cfg.landscape.radius, cfg.landscape.radius, cfg.landscape.steps)
    losses = np.zeros((cfg.landscape.steps, cfg.landscape.steps), dtype=np.float32)

    base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    dir1, dir2 = build_filterwise_directions(model, seed=seed)

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            perturbed = {}
            for name, value in base_state.items():
                if name in dir1:
                    perturbed[name] = value + alpha * dir1[name] + beta * dir2[name]
                else:
                    perturbed[name] = value
            model.load_state_dict(perturbed, strict=True)
            losses[i, j] = evaluate_loss(
                model,
                loader,
                device=device,
                max_batches=cfg.landscape.max_eval_batches,
            )

        print(f"landscape row {i + 1}/{cfg.landscape.steps}")

    model.load_state_dict(base_state, strict=True)
    return alphas, betas, losses


def plot_surfaces(
    output_path: Path,
    surfaces: dict[str, np.ndarray],
    alphas: np.ndarray,
    betas: np.ndarray,
    cfg: DictConfig,
) -> None:
    clip_pct = float(cfg.plot.clip_top_percent)
    alpha_mesh, beta_mesh = np.meshgrid(alphas, betas, indexing="ij")
    fig = plt.figure(figsize=(6 * len(surfaces), 5))

    for idx, (name, loss_grid) in enumerate(surfaces.items(), start=1):
        display_grid = loss_grid
        cap = None
        if clip_pct > 0.0:
            cap = float(np.percentile(loss_grid, 100.0 - clip_pct))
            display_grid = np.minimum(loss_grid, cap)

        ax = fig.add_subplot(1, len(surfaces), idx, projection="3d")
        surface = ax.plot_surface(
            alpha_mesh,
            beta_mesh,
            display_grid,
            cmap=cfg.plot.cmap,
            linewidth=0,
            antialiased=True,
            alpha=0.95,
        )
        fig.colorbar(surface, ax=ax, shrink=0.6, pad=0.08)
        ax.set_title(name)
        ax.set_xlabel("alpha")
        ax.set_ylabel("beta")
        ax.set_zlabel("loss")
        ax.view_init(elev=cfg.plot.elev, azim=cfg.plot.azim)
        if cap is not None:
            ax.set_zlim(top=cap)

    fig.tight_layout()
    fig.savefig(output_path, dpi=cfg.plot.dpi, bbox_inches="tight")
    plt.close(fig)


def load_cached_surfaces(npz_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    data = np.load(npz_path)
    if "alphas" not in data or "betas" not in data:
        raise ValueError(f"Cached file is missing alphas/betas: {npz_path}")
    surfaces = {
        key: data[key]
        for key in data.files
        if key not in ("alphas", "betas")
    }
    return data["alphas"], data["betas"], surfaces


@hydra.main(
    version_base="1.3",
    config_path="../config",
    config_name="loss_landscape_resnet56",
)
def main(cfg: DictConfig) -> None:
    set_seeds(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    surface_png = output_dir / cfg.plot.output_filename

    if cfg.runtime.use_cached:
        cached_path = Path(cfg.runtime.cached_npz_path)
        if not cached_path.is_absolute():
            cached_path = Path.cwd() / cached_path
        if not cached_path.exists():
            raise FileNotFoundError(f"Cached landscape file not found: {cached_path}")

        alphas, betas, surfaces = load_cached_surfaces(cached_path)
        print(f"Loaded cached landscapes from {cached_path}")
        for name, grid in surfaces.items():
            print(f"{name}: min={float(grid.min()):.4f}, max={float(grid.max()):.4f}")
        plot_surfaces(surface_png, surfaces, alphas, betas, cfg)
        print(f"Saved 3D plot to {surface_png}")
        return

    device = resolve_device(cfg.device)
    train_loader, test_loader, landscape_loader, num_classes = get_loaders(cfg)

    surfaces: dict[str, np.ndarray] = {}
    metrics: list[tuple[str, float]] = []
    last_axes: tuple[np.ndarray, np.ndarray] | None = None

    for variant_idx, (label, skip_kind) in enumerate(_VARIANTS):
        kwargs = {
            "num_classes": num_classes,
            "model_type": "resnet56",
            "skip_kind": skip_kind,
        }
        if skip_kind == "learnable_orth":
            kwargs["update_rule"] = cfg.model.update_rule
            kwargs["update_rule_iters"] = cfg.model.update_rule_iters

        model = get_model(**kwargs).to(device)
        print(f"Training {label}")
        train_model(model, train_loader, cfg, device=device, variant_name=label)
        accuracy = evaluate_accuracy(model, test_loader, device=device)
        metrics.append((label, accuracy))
        print(f"{label} test accuracy: {accuracy:.2f}%")

        print(f"Computing loss landscape for {label}")
        alphas, betas, losses = compute_loss_surface(
            model,
            landscape_loader,
            cfg,
            device=device,
            seed=cfg.seed + variant_idx * 13,
        )
        surfaces[label] = losses
        last_axes = (alphas, betas)

    if last_axes is None:
        raise RuntimeError("No surfaces were generated.")

    alphas, betas = last_axes
    plot_surfaces(surface_png, surfaces, alphas, betas, cfg)

    npz_path = output_dir / "resnet56_loss_landscape_data.npz"
    np.savez_compressed(
        npz_path,
        alphas=alphas,
        betas=betas,
        **{name: grid for name, grid in surfaces.items()},
    )

    metrics_path = output_dir / "resnet56_metrics.txt"
    with metrics_path.open("w", encoding="utf-8") as f:
        for label, acc in metrics:
            f.write(f"{label}: {acc:.2f}%\n")

    print(f"Saved 3D plot to {surface_png}")
    print(f"Saved landscape data to {npz_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
