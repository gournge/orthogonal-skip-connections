#!/bin/bash
# Setup script for Orth-ResNet 🌀

set -e

echo "=== [Orth-ResNet] Environment Setup ==="

# 1. Create isolated Python environment using uv
if ! command -v uv &> /dev/null; then
    echo "uv not found. Please install uv: https://github.com/astral-sh/uv"
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with uv..."
    uv venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

# 2. Install dependencies (including experiments/optional extras)
echo "Installing dependencies..."
uv pip install -e .[dev]

echo "Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To run a baseline experiment:"
echo "  python -m experiments.run_experiments resnet_variant=baseline dataset=cifar10"