from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def cifar(root: str = "~/data", batch_size: int = 128, cifar100: bool = False):
    T = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    ds_cls = datasets.CIFAR100 if cifar100 else datasets.CIFAR10
    train_ds = ds_cls(root, train=True, download=True, transform=T)
    test_ds = ds_cls(root, train=False, download=True, transform=T)
    return (
        DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True),
        DataLoader(test_ds, batch_size, shuffle=False, num_workers=4, pin_memory=True),
    )
