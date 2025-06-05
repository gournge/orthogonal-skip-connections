import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KernelDensity
import numpy as np
import torch
from sklearn.datasets import make_moons, make_circles, make_classification, make_blobs

# ---- Synthetic 2-D classification dataset ----

torch.manual_seed(0)
np.random.seed(0)

n_blocks = 2
N = 1000
EPOCHS = 200

# Choose the type of synthetic dataset: "moons", "circles", "blobs", or "classification"
dataset_type = "circles"

if dataset_type == "moons":
    x, y = make_moons(n_samples=N, noise=0.1, random_state=0)
elif dataset_type == "circles":
    x, y = make_circles(n_samples=N, noise=0.1, factor=0.5, random_state=0)
elif dataset_type == "blobs":
    x, y = make_blobs(n_samples=N, centers=5, cluster_std=1.0, random_state=0)
elif dataset_type == "classification":
    x, y = make_classification(
        n_samples=N,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=0,
    )
else:
    raise ValueError("Unknown dataset_type selected.")

y = y.astype(np.float32)

X_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

print("Dataset and DataLoader initialized with dataset_type =", dataset_type)


# ---- Building blocks ----
class SmallMLP(nn.Module):
    def __init__(self, in_dim=2, hidden=16, out_dim=2, scale=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        # near-identity initialisation
        nn.init.normal_(self.fc1.weight, std=scale)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=scale)
        nn.init.zeros_(self.fc2.bias)
        print("SmallMLP initialized.")

    def forward(self, x):
        return torch.tanh(self.fc2(torch.tanh(self.fc1(x))))


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
    def __init__(self, block_cls, n_blocks=3):
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


def train(model, epochs, lr=1e-3):
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


def accuracy(model):
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
id_model = ResidualClassifier(IdentityBlock, n_blocks=n_blocks)
orth_model = ResidualClassifier(OrthBlock, n_blocks=n_blocks)

print("Training IdentityBlock model...")
train(id_model, EPOCHS)
print("Training OrthBlock model...")
train(orth_model, EPOCHS)

acc_id = accuracy(id_model)
weights_magnitude_id = model_avg_weights_magnitude(id_model)
acc_orth = accuracy(orth_model)
weights_magnitude_orth = model_avg_weights_magnitude(orth_model)
print(
    f"Identity model accuracy: {acc_id:.4f}, Avg weights magnitude: {weights_magnitude_id:.4f}"
)
print(
    f"Orthogonal model accuracy: {acc_orth:.4f}, Avg weights magnitude: {weights_magnitude_orth:.4f}"
)


# ---- Grid morphing visualisation utilities ----
def morph_grid(ax, transform, lim=2, step=0.5, n_samples=100):
    """Draw morphed grid lines after applying `transform`."""
    xs = np.arange(-lim, lim + 1e-9, step)
    horizontal_gradient = np.linspace(0, 1, n_samples)
    ys = np.arange(-lim, lim + 1e-9, step)
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


# ---- Build the (3, n_blocks) figure ----
fig, axes = plt.subplots(3, (n_blocks + 1), figsize=(3 * (n_blocks + 1), 9))

# Row 0: near-identity residual blocks (x + f(x))
for j in range(n_blocks):
    blk = id_model.blocks[j]

    def t_id(points, b=blk):
        with torch.no_grad():
            pts = torch.tensor(points, dtype=torch.float32)
            return (pts + b.f(pts)).numpy()

    morph_grid(axes[0, j], t_id)
    axes[0, j].set_title(f"Identity block {j+1}")

# Rows 1 & 2: near-orthogonal blocks (W and f separately)
for j in range(n_blocks):
    blk = orth_model.blocks[j]
    # W only
    with torch.no_grad():
        W = blk.W.detach().numpy()

    def t_W(points, W=W):
        return points @ W.T

    morph_grid(axes[1, j], t_W)
    axes[1, j].set_title(f"Orth W {j+1}")

    # f(x) only (shown as x + f(x) to keep geometry)
    def t_f(points, b=blk):
        with torch.no_grad():
            pts = torch.tensor(points, dtype=torch.float32)
            return (pts + b.f(pts)).numpy()

    morph_grid(axes[2, j], t_f)
    axes[2, j].set_title(f"Orth f {j+1}")

# ---- Additional plots for decision boundaries ----
# We'll overlay scatterplots of the datapoints with the bayes optimal and learned boundaries.
# Here, we use a logistic regression model as an approximation to the Bayes optimal classifier.

# Estimate class conditional densities using KDE, which gives the Bayes optimal classifier
x0 = x[y == 0]
x1 = x[y == 1]

# Fit KDE for each class (bandwidth can be tuned)
kde0 = KernelDensity(bandwidth=0.2).fit(x0)
kde1 = KernelDensity(bandwidth=0.2).fit(x1)

# Generate a grid over the input space
xx, yy = np.meshgrid(np.linspace(-2.5, 3, 200), np.linspace(-2.5, 3, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

# Compute log densities for both classes on the grid
log_dens0 = kde0.score_samples(grid)
log_dens1 = kde1.score_samples(grid)

# Bayes optimal decision function: log likelihood ratio (threshold at 0)
Z_bayes = (log_dens1 - log_dens0).reshape(xx.shape)

# Top right subplot: Identity block predictions vs Bayes optimal boundary
ax_top = axes[0, -1]
ax_top.cla()  # clear previous plot
ax_top.scatter(x[:, 0], x[:, 1], c=y, cmap="bwr", alpha=0.2, s=10)
ax_top.contour(
    xx, yy, Z_bayes, levels=[0], colors="green", linewidths=2, linestyles="--"
)

# Compute the learned identity boundary: id_model outputs logits, so 0 is the threshold.
with torch.no_grad():
    grid_tensor = torch.tensor(grid, dtype=torch.float32)
    Z_id = id_model(grid_tensor).numpy().reshape(xx.shape)
ax_top.contour(xx, yy, Z_id, levels=[0], colors="black", linewidths=2)
ax_top.set_title("Identity: Bayes vs Learned")

# Just below the top right (second row left): Orthogonal block predictions vs Bayes optimal boundary
ax_mid = axes[1, -1]
ax_mid.cla()  # clear previous plot
ax_mid.scatter(x[:, 0], x[:, 1], c=y, cmap="bwr", alpha=0.2, s=10)
with torch.no_grad():
    Z_orth = orth_model(grid_tensor).numpy().reshape(xx.shape)
ax_mid.contour(
    xx, yy, Z_bayes, levels=[0], colors="green", linewidths=2, linestyles="--"
)
ax_mid.contour(xx, yy, Z_orth, levels=[0], colors="black", linewidths=2)
ax_mid.set_title("Orthogonal: Bayes vs Learned")

axes[-1, -1].axis("off")

plt.tight_layout()
plt.savefig("orth_residuals.png", dpi=300)
print("Visualization saved as 'orth_residuals.png'.")
