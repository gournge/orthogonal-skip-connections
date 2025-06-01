"""YAML-first training script (hydra-lite doc)."""
import argparse, yaml, time, wandb, torch, torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from ..models import resnet_baseline, orthogonal_resnet
from ..utils.datasets import cifar
from ..utils.metrics import orth_error

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=str(Path(__file__).with_suffix(".yaml")))
    p.add_argument("model", default="baseline", choices=["baseline", "orth", "learnable_orth", "random"])
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    return p.parse_args()

def make_model(model: str, num_classes: int):
    if model == "baseline":
        return resnet_baseline.resnet18(num_classes)
    if model == "orth":
        return orthogonal_resnet.OrthResNet([2, 2, 2, 2], num_classes, skip_profile="orth")
    if model == "learnable_orth":
        return orthogonal_resnet.OrthResNet([2, 2, 2, 2], num_classes, skip_profile="learnable_orth")
    if model == "random":
        return orthogonal_resnet.OrthResNet([2, 2, 2, 2], num_classes, skip_profile="random")
    raise ValueError(model)

def train_epoch(model, loader, opt, device):
    model.train()
    total, correct, loss_accum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y, label_smoothing=0.1)
        loss.backward()
        opt.step()
        total += y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        loss_accum += loss.item() * y.size(0)
    return loss_accum / total, correct / total

def evaluate(model, loader, device):
    model.eval()
    total, correct, loss_accum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            total += y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            loss_accum += loss.item() * y.size(0)
    return loss_accum / total, correct / total

def main():
    cfg = yaml.safe_load(Path(parse().config).read_text())
    args = parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dl, test_dl = cifar(cifar100=args.dataset == "cifar100")
    model = make_model(args.model, num_classes=100 if args.dataset == "cifar100" else 10).to(device)

    opt = SGD(model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=5e-4)
    sched = CosineAnnealingLR(opt, T_max=cfg["epochs"])

    wandb.init(project="orthagonal-resnet", config={**cfg, **vars(args)})

    for epoch in range(cfg["epochs"]):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_dl, opt, device)
        test_loss, test_acc = evaluate(model, test_dl, device)
        ortho_dev = sum(p.orth_error.item() for p in model.modules() if hasattr(p, "orth_error"))
        wandb.log({
            "train/loss": train_loss,
            "train/acc": train_acc,
            "test/loss": test_loss,
            "test/acc": test_acc,
            "orth/deviation": ortho_dev,
            "lr": sched.get_last_lr()[0],
            "epoch": epoch,
        })
        sched.step()
        print(f"[E{epoch:03d}] acc={test_acc*100:.2f}% • ortho={ortho_dev:.3e} • {time.time()-t0:.1f}s")

    wandb.finish()

if __name__ == "__main__":
    main()
