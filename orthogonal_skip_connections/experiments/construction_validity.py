"""Synthetic 2-D experiments comparing identity and orthogonal residual blocks."""

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.neighbors import KernelDensity
from sklearn.datasets import make_moons, make_circles, make_classification, make_blobs
import hydra
from omegaconf import DictConfig

# ---- Synthetic 2-D classification dataset ----


def build_dataset(cfg, seed):
    """Return tensors and dataloader for the chosen synthetic dataset."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if cfg.dataset_type == "moons":
        x, y = make_moons(n_samples=cfg.n_samples, noise=cfg.noise, random_state=seed)
    elif cfg.dataset_type == "circles":
        x, y = make_circles(
            n_samples=cfg.n_samples,
            noise=cfg.noise,
            factor=cfg.factor,
            random_state=seed,
        )
    elif cfg.dataset_type == "blobs":
        x, y = make_blobs(
            n_samples=cfg.n_samples,
            centers=cfg.n_blobs,
            cluster_std=100 * cfg.noise / cfg.n_blobs,
            random_state=seed,
        )
        y = (y >= (cfg.n_blobs // 2)).astype(np.float32)  # binary classification
    elif cfg.dataset_type == "classification":
        x, y = make_classification(
            n_samples=cfg.n_samples,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown dataset_type {cfg.dataset_type}")

    y = y.astype(np.float32)
    X_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    print("Dataset and DataLoader initialized with dataset_type =", cfg.dataset_type)
    return x, y, X_tensor, y_tensor, loader


# ---- Building blocks ----
class SmallMLP(nn.Module):
    def __init__(self, in_dim=2, hidden=32, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),  # quicker to train than tanh
            nn.Linear(hidden, out_dim),  # no final tanh; let the block
        )  # add whatever it wants
        nn.init.kaiming_normal_(self.net[0].weight, nonlinearity="relu")
        nn.init.zeros_(self.net[0].bias)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x):
        return self.net(x)


class IdentityBlock(nn.Module):
    """y = x + f(x)"""

    def __init__(self):
        super().__init__()
        self.f = SmallMLP()
        print("IdentityBlock initialized.")

    def forward(self, x):
        return x + self.f(x)


class OrthBlock(nn.Module):
    """y = W x + f(x) with trainable orthogonal W (parameterised as rotation)"""

    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.randn(()) * 0.01)  # near-identity rotation
        self.f = SmallMLP()
        print("OrthBlock initialized.")

    @property
    def W(self):
        c, s = torch.cos(self.theta), torch.sin(self.theta)
        return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])

    def forward(self, x):
        rot = x @ self.W.t()
        return rot + self.f(x)


# ---- Classifier wrappers ----
class ResidualClassifier(nn.Module):
    def __init__(self, block_cls, n_blocks: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList([block_cls() for _ in range(n_blocks)])
        self.head = nn.Linear(2, 1)
        print(f"ResidualClassifier initialized with {n_blocks} blocks.")

    def forward(self, x):
        z = x
        for i, blk in enumerate(self.blocks):
            z = blk(z)
            # print(f"Block {i+1} output: {z.shape}")
        return self.head(z).squeeze(-1)


def train(model, loader, epochs, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    print(f"Training started for {epochs} epochs with learning rate {lr}.")
    for epoch in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    print("Training completed.")


def accuracy(model, X_tensor, y_tensor):
    with torch.no_grad():
        preds = (torch.sigmoid(model(X_tensor)) > 0.5).float()
        acc = (preds == y_tensor).float().mean().item()
        return acc


def model_avg_weights_magnitude(model):
    """Compute the average magnitude of weights in the model."""
    total_magnitude = 0.0
    total_params = 0
    for param in model.parameters():
        if param.requires_grad:
            total_magnitude += param.abs().sum().item()
            total_params += param.numel()
    return total_magnitude / total_params if total_params > 0 else 0.0


# ---- Instantiate, train, and evaluate both classifiers ----
def run_experiment(cfg, X_tensor, y_tensor, loader):
    n_blocks = cfg.n_blocks
    epochs = cfg.epochs
    lr = cfg.lr

    id_model = ResidualClassifier(IdentityBlock, n_blocks=n_blocks)
    orth_model = ResidualClassifier(OrthBlock, n_blocks=n_blocks)

    print("Training IdentityBlock model...")
    train(id_model, loader, epochs, lr)
    print("Training OrthBlock model...")
    train(orth_model, loader, epochs, lr)

    acc_id = accuracy(id_model, X_tensor, y_tensor)
    weights_magnitude_id = model_avg_weights_magnitude(id_model)
    acc_orth = accuracy(orth_model, X_tensor, y_tensor)
    weights_magnitude_orth = model_avg_weights_magnitude(orth_model)
    print(
        f"Identity model accuracy: {acc_id:.4f}, Avg weights magnitude: {weights_magnitude_id:.4f}"
    )
    print(
        f"Orthogonal model accuracy: {acc_orth:.4f}, Avg weights magnitude: {weights_magnitude_orth:.4f}"
    )

    return id_model, orth_model


# ---- Grid morphing visualisation utilities ----
def morph_grid(ax, transform, lim=2, size=30, n_samples=100):
    """Draw morphed grid lines after applying `transform`."""
    xs = np.arange(-lim, lim + 1e-9, 2 * lim / size)
    horizontal_gradient = np.linspace(0, 1, n_samples)
    ys = np.arange(-lim, lim + 1e-9, 2 * lim / size)
    vertical_gradient = np.linspace(0, 1, n_samples)
    for x0 in xs:
        y_line = np.linspace(-lim, lim, n_samples)
        pts = np.stack([np.full_like(y_line, x0), y_line], axis=1)
        mapped = transform(pts)
        ax.plot(
            mapped[:, 0],
            mapped[:, 1],
            linewidth=1.2,
            color=plt.cm.Oranges(
                horizontal_gradient[int((x0 + lim) / (2 * lim) * (n_samples - 1))]
            ),
        )
    for y0 in ys:
        x_line = np.linspace(-lim, lim, n_samples)
        pts = np.stack([x_line, np.full_like(x_line, y0)], axis=1)
        mapped = transform(pts)
        ax.plot(
            mapped[:, 0],
            mapped[:, 1],
            linewidth=1.2,
            color=plt.cm.Blues(
                vertical_gradient[int((y0 + lim) / (2 * lim) * (n_samples - 1))]
            ),
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def plot_models(id_model, orth_model, x, y, n_blocks, out_file="orth_residuals.png"):
    """Visualize learned transformations and classification regions."""
    fig, axes = plt.subplots(3, (n_blocks + 1), figsize=(3 * (n_blocks + 1), 9))
    x_min, x_max = x[:, 0].min(), x[:, 0].max()
    y_min, y_max = x[:, 1].min(), x[:, 1].max()

    lim = max(x_max - x_min, y_max - y_min) / 2

    for j in range(n_blocks):
        blk = id_model.blocks[j]

        def t_id(points, b=blk):
            with torch.no_grad():
                pts = torch.tensor(points, dtype=torch.float32)
                return (pts + b.f(pts)).numpy()

        morph_grid(axes[0, j], t_id, lim=lim)
        axes[0, j].set_title(f"Identity block {j+1}")

    for j in range(n_blocks):
        blk = orth_model.blocks[j]

        with torch.no_grad():
            W = blk.W.detach().numpy()

        def t_W(points, W=W):
            return points @ W.T

        morph_grid(axes[1, j], t_W, lim=lim)
        axes[1, j].set_title(f"Orth W {j+1}")

        def t_f(points, b=blk):
            with torch.no_grad():
                pts = torch.tensor(points, dtype=torch.float32)
                return (pts + b.f(pts)).numpy()

        morph_grid(axes[2, j], t_f, lim=lim)
        axes[2, j].set_title(f"Orth f {j+1}")

    # Prepare grid for background classification visualization
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32)

    # Identity model classification background
    with torch.no_grad():
        preds_id = (
            (torch.sigmoid(id_model(grid_tensor)) > 0.5).numpy().reshape(xx.shape)
        )
    ax_top = axes[0, -1]
    ax_top.cla()
    ax_top.imshow(
        preds_id,
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        origin="lower",
        cmap="bwr",
        alpha=0.2,
        aspect="auto",
    )
    ax_top.scatter(
        x[:, 0], x[:, 1], c=y, cmap="bwr", alpha=0.5, s=10, edgecolor="k", linewidth=0.2
    )
    ax_top.set_title("Identity: Classification regions")

    # Orthogonal model classification background
    with torch.no_grad():
        preds_orth = (
            (torch.sigmoid(orth_model(grid_tensor)) > 0.5).numpy().reshape(xx.shape)
        )
    ax_mid = axes[1, -1]
    ax_mid.cla()
    ax_mid.imshow(
        preds_orth,
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        origin="lower",
        cmap="bwr",
        alpha=0.2,
        aspect="auto",
    )
    ax_mid.scatter(
        x[:, 0], x[:, 1], c=y, cmap="bwr", alpha=0.5, s=10, edgecolor="k", linewidth=0.2
    )
    ax_mid.set_title("Orthogonal: Classification regions")

    axes[-1, -1].axis("off")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    print(f"Visualization saved as '{out_file}'.")


@hydra.main(version_base="1.3", config_path="../config", config_name="validity")
def main(cfg: DictConfig):
    x, y, X_tensor, y_tensor, loader = build_dataset(cfg.dataset, cfg.seed)
    id_model, orth_model = run_experiment(cfg, X_tensor, y_tensor, loader)
    plot_models(id_model, orth_model, x, y, cfg.n_blocks)


if __name__ == "__main__":
    main()
