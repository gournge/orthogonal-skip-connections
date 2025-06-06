# Orth-ResNet 🌀

A modular research codebase to **investigate orthogonal skip-connections** in ResNet-style architectures.  Supports *fixed*, *learnable*, *partially orthogonal* and *random* projections, with out-of-the-box CIFAR-10/100 experiments and Weights-and-Biases tracking.

## Quick-start
```bash
# 1. create an isolated environment & install deps
uv venv && source .venv/bin/activate
uv pip install -e .[experiments]

# 2. learnable orthogonal variant
python -m orthogonal_skip_connections.experiments.run_experiments model.skip_kind=learnable_orth

# 2. baseline ResNet on CIFAR-100
python -m orthogonal_skip_connections.experiments.run_experiments model.skip_kind=identity dataset=cifar100
```

## Experiments

### CIFAR Training

```bash 
python -m orthogonal_skip_connections.experiments.run_experiments model.skip_kind=identity dataset=cifar100
```

## Comparison of weight transformations on synthetic data

```bash
python -m orthogonal_skip_connections.experiments.orth_residual_synthetic_data dataset_type=circles
```

## TODO

- [x] Modularize synthetic data experiments and use hydra
- [ ] How valid is the construction introduced by Bartlett et al. (2018)? How often do models converge to it?
- [ ] Explore ViT architectures
- [ ] Explore the scheme of training where we first train the model with identity skip connections, then approximate each non-linear connection with an orthogonal connection and adjust the weights of non-linear connection.
- [ ] Literature review of related papers

## Citation
If you use this codebase in your research, please cite it:

```bibtex
@misc{orthogonal_skip_connections,
  title = {Orthogonal Skip Connections},
  author = {Filip Morawiec},
  year = {2025},
    publisher = {GitHub},
    journal = {GitHub repository},
}
```