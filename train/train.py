import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import os

from utils.data import cifar10, cifar100
from utils.metrics import AverageMeter, accuracy
from utils.orthogonal import orth_error
from models import get_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--retraction", default="cayley")
    parser.add_argument(
        "--sharp_iters",
        type=int,
        default=8,
        help="Number of iterations for sharp_operator",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    wandb.init(project="orthogonal-resnet", config=vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    # Get number of classes
    num_classes = 10 if args.dataset == "cifar10" else 100
    model = get_model(
        args.variant,
        num_classes=num_classes,
        retraction=args.retraction,
        sharp_iters=args.sharp_iters,
    ).to(device)

    print(f"Using model: {args.variant} with {num_classes} classes")
    # print number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params / 1e6:.2f}M trainable parameters")

    # Watch model parameters and gradients
    wandb.watch(model, log="all", log_freq=args.logging_steps)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Data loaders
    print("Loading dataset...")
    train_loader = (
        cifar10(batch_size=args.batch_size)
        if args.dataset == "cifar10"
        else cifar100(batch_size=args.batch_size)
    )
    print(f"Loaded {len(train_loader.dataset)} training samples.")
    print("Creating validation loader...")
    val_loader = (
        cifar10(train=False, batch_size=args.batch_size)
        if args.dataset == "cifar10"
        else cifar100(train=False, batch_size=args.batch_size)
    )
    print(f"Loaded {len(val_loader.dataset)} validation samples.")

    best_acc = 0.0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        print(f"\tTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print("\tValidating...")
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"\tVal Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        scheduler.step()

        # Log metrics to W&B
        logs = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
        }
        # Log orthogonality error if available
        if hasattr(model, "orth_error"):
            logs["orth_error"] = model.orth_error()
        wandb.log(logs)

        # Save best model
        if val_acc > best_acc:
            print("\tNew best model found! Saving...")
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, "best.pt"))

    print("Best Acc:", best_acc)


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for i, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if hasattr(model, "ortho_modules"):
            for mod in model.ortho_modules:
                # only LearnableOrthogonalProjection has .step()
                if hasattr(mod, "step"):
                    mod.step()  # uses stored grad to retract W back onto O(n)

        # Update meters
        loss_meter.update(loss.item(), images.size(0))
        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        acc_meter.update(acc1.item(), images.size(0))

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc1.item(), images.size(0))

    return loss_meter.avg, acc_meter.avg


if __name__ == "__main__":
    main()
