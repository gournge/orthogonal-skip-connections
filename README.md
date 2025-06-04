# Orth-ResNet 🌀

A modular research codebase to **investigate orthogonal skip-connections** in ResNet-style architectures.  Supports *fixed*, *learnable*, *partially orthogonal* and *random* projections, with out-of-the-box CIFAR-10/100 experiments and Weights-and-Biases tracking.

## Quick-start
```bash
# 1. create an isolated environment & install deps
uv venv && source .venv/bin/activate
uv pip install -e .[experiments]

# 2. baseline ResNet on CIFAR-10
python -m orthogonal_skip_connections.experiments.run_experiments resnet_variant=baseline dataset=cifar10

# 3. learnable orthogonal variant
python -m orthogonal_skip_connections.experiments.run_experiments resnet_variant=learnable_orth dataset=cifar10
```