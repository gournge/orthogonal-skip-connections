"""Dataset loaders with standard augmentation pipelines."""
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader

def cifar10(root: str = "./data", train: bool = True, batch_size: int = 128):
    transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    ds = datasets.CIFAR10(root, train=train, download=True, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=4)

def cifar100(root: str = "./data", train: bool = True, batch_size: int = 128):
    transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    ds = datasets.CIFAR100(root, train=train, download=True, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=4)
