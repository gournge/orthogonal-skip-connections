# Orthogonal-ResNet

**Research question**  Can enforcing (or *learning*) orthogonality in the skip connections of a ResNet improve gradient flow, spectral stability, or generalisation?

## Installation
```bash
uv venv orthogonal-resnet-dev && source .venv/bin/activate
uv pip install .
```

> **Python ≥3.10**

## Quick start (CIFAR-10, baseline vs fully-orthogonal)
```bash
# vanilla ResNet18
python -m train.train model=baseline dataset=cifar10

# fully-orthogonal variant with Cayley parameterisation
python -m train.train model=orth dataset=cifar10 ortho.method=cayley
```

The first run creates a `wandb` project called `orthogonal-resnet` and logs all hyper-parameters, gradients, and special orthogonality diagnostics.

## Repo design
* **models** – drop-in variants.  Every model exposes `forward`, `loss`, and a `orth_loss` property (deviation from perfect orthogonality).
* **utils.orth_utils** – matrix ops (Cayley transform, SVD/QR retractions, Newton-Schulz sharp).
* **train** – YAML-driven training loop (⚡ Lightning-free by design for full control).
* **experiments** – reproducible scripts: ablations, hyper-sweeps.
* **tests** – `pytest` ensuring gradients stay finite & orthogonality error ≤1e-5 when expected.

See `docs/` for the maths notes, including the sharp-operator steepest descent update.
