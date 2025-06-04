from __future__ import annotations

import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from orthogonal_skip_connections.utils.orthogonal_ops import reorthogonalize_model
from orthogonal_skip_connections.utils.metrics import compute_orthogonality_deviation
from .scheduler import cosine_scheduler

class Trainer:
    def __init__(self, model, train_loader, test_loader, cfg):
        self.model = model.to(cfg.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cfg = cfg

        self.opt = torch.optim.SGD(model.parameters(), lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay)
        self.sched = cosine_scheduler(self.opt, cfg.scheduler.epochs, cfg.scheduler.warmup_epochs)
        self.scaler = torch.cuda.amp.GradScaler()
        wandb.watch(model, log_freq=100)

    def _step(self, batch):
        x, y = batch
        x, y = x.to(self.cfg.device), y.to(self.cfg.device)
        with torch.cuda.amp.autocast():
            logits = self.model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=self.cfg.label_smoothing)
        self.scaler.scale(loss).backward()
        return loss, logits

    def fit(self):
        for epoch in range(self.cfg.scheduler.epochs):
            self.model.train()
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            for batch in pbar:
                self.opt.zero_grad(set_to_none=True)
                loss, logits = self._step(batch)
                self.scaler.step(self.opt)
                self.scaler.update()
                # maintain orthogonality on the manifold
                reorthogonalize_model(self.model)
                pbar.set_postfix(loss=loss.item())
                wandb.log({"train/loss": loss.item(), "epoch": epoch})
            self.sched.step()
            if epoch % 5 == 0:
                self.evaluate(epoch)

    @torch.no_grad()
    def evaluate(self, epoch):
        self.model.eval()
        correct = total = 0
        for x, y in self.test_loader:
            x, y = x.to(self.cfg.device), y.to(self.cfg.device)
            logits = self.model(x)
            pred = logits.argmax(1)
            total += y.size(0)
            correct += (pred == y).sum().item()
        acc = 100 * correct / total
        orth_dev = compute_orthogonality_deviation(self.model)
        wandb.log({"val/acc": acc, "orth_dev": orth_dev, "epoch": epoch})
        print(f"\nValidation @epoch {epoch}: acc={acc:.2f}%  orth_dev={orth_dev:.3f}")