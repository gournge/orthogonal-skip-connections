# Orthogonal-ResNet Research Prototype

**Purpose**  
Explore the effect of *evolving orthogonal matrices* used as skip-connection projections in ResNet-style architectures.  

---

## Variants

| Key                   | Description                                                               |
|-----------------------|---------------------------------------------------------------------------|
| **baseline**          | Standard ResNet-18/34/50/... identity skips                              |
| **orthogonal**        | Fixed orthogonal projections at every skip                                |
| **learnable_ortho**   | On-manifold learned projections (Cayley/QR/SVD/steepest)                  |
| **partial_ortho**     | Maskable stages for orthogonal skips                                      |
| **random_skip**       | Random full-rank (non-orthogonal) projections                             |
| **resnet18/34/50/101/152** | Standard ResNet depths using custom `Block`/`Bottleneck` definitions |

---

## Datasets

- **CIFAR-10 / CIFAR-100** (default)  
- **STL-10 / ImageNet-Lite** (optional; see GPU memory presets in `README.md`)

---

## Installation with **uv**

[`uv`](https://astral.sh/uv/) is a fast, Rust-powered replacement for pip/venv:

```bash
# 1. install uv (Linux/macOS)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. from repo root, create & activate env
uv venv              # creates .venv
source .venv/bin/activate

# 3a. install deps via requirements.txt
uv pip install -r requirements.txt

# 3b. — or — generate & sync a lock:
uv lock
uv pip sync

# 4. run training
uv run python -m train.train --variant baseline --dataset cifar10
```

*Tip*: use `uv add <package>` to pin new libs, and `uv lock`/`uv pip sync` for fully‑reproducible environments.
