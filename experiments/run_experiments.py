"""Convenience script to sweep variants and datasets."""
import itertools, subprocess, yaml, pathlib

variants = ['baseline', 'orthogonal', 'learnable_ortho', 'partial_ortho', 'random_skip']
datasets = ['cifar10', 'cifar100']

for variant, dataset in itertools.product(variants, datasets):
    cmd = [
        'python', '-m', 'train.train',
        '--variant', variant,
        '--dataset', dataset
    ]
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd)
