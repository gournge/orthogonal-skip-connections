"""Convenience wrapper that trains *all* five variants on CIFAR-10 so you can
push a single button in VS Code and walk away."""

import subprocess, pathlib

variants = ["baseline", "orth", "learnable_orth", "random"]
logdir = pathlib.Path("runs")
logdir.mkdir(exist_ok=True)

for v in variants:
    cmd = [
        "python",
        "-m",
        "train.train",
        f"model={v}",
        "dataset=cifar10",
    ]
    subprocess.run(cmd, check=True)
