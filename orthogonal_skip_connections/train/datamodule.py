from __future__ import annotations

import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

_DATASETS = {
    "cifar10": (torchvision.datasets.CIFAR10, 10),
    "cifar100": (torchvision.datasets.CIFAR100, 100),
    "stl10": (torchvision.datasets.STL10, 10),
}


def get_dataloaders(name: str, batch_size: int, num_workers: int):
    ds_cls, num_classes = _DATASETS[name]
    transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    train_set = ds_cls(root="~/.data", train=True, download=True, transform=transform)
    test_set = ds_cls(root="~/.data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader, num_classes