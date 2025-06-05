from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from orthogonal_skip_connections.models.skip import if_model_needs_update_rule
from orthogonal_skip_connections.utils.orthogonal_ops import reorthogonalize_model
from orthogonal_skip_connections.utils.wandb_process import (
    data_which_histogram_looks_like_an_arr,
)
from .scheduler import cosine_scheduler


class Trainer:
    def __init__(self, model, train_loader, test_loader, cfg):
        self.model = model.to(cfg.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cfg = cfg

        self.opt = torch.optim.SGD(
            model.parameters(),
            lr=cfg.optimizer.lr,
            momentum=cfg.optimizer.momentum,
            weight_decay=cfg.optimizer.weight_decay,
        )
        self.sched = cosine_scheduler(
            self.opt, cfg.scheduler.epochs, cfg.scheduler.warmup_epochs
        )
        self.scaler = torch.cuda.amp.GradScaler()
        self.epoch_eval_freq = cfg.scheduler.epoch_eval_freq
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
            if epoch % self.epoch_eval_freq == 0:
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

        log_dict = {
            "val/acc": acc,
        }
        orthogonality_deviations = []
        identity_deviations = []
        if if_model_needs_update_rule(self.model.skip_kind):
            # Compute Frobenius deviation from identity for square weight matrices
            orthogonality_deviations = []
            identity_deviations = []
            for m in self.model.modules():
                if hasattr(m, "weight"):
                    W = m.weight
                    if W.dim() == 2 and W.size(0) == W.size(1):
                        I = torch.eye(W.size(0), device=W.device, dtype=W.dtype)
                        deviation = torch.norm(W.T @ W - I, p="fro").item()
                        orthogonality_deviations.append(deviation)
                        identity_deviations.append(torch.norm(W - I, p="fro").item())
            log_dict.update(
                {
                    "orth_dev_hist": wandb.Histogram(
                        data_which_histogram_looks_like_an_arr(
                            orthogonality_deviations
                        )[0]
                    ),
                    "identity_dev_hist": wandb.Histogram(
                        data_which_histogram_looks_like_an_arr(identity_deviations)[0]
                    ),
                    "orth_dev_mean": np.mean(orthogonality_deviations),
                    "orth_dev_min": np.min(orthogonality_deviations),
                    "orth_dev_max": np.max(orthogonality_deviations),
                    "identity_dev_mean": np.mean(identity_deviations),
                    "identity_dev_min": np.min(identity_deviations),
                    "identity_dev_max": np.max(identity_deviations),
                }
            )

        wandb.log(log_dict)
        print(
            f"Epoch {epoch}: Validation accuracy: {acc:.2f}%"
            + (
                f" | Orthogonality deviation: {np.mean(orthogonality_deviations):.4f}"
                if orthogonality_deviations
                else ""
            )
            + (
                "" f" | Identity deviation: {np.mean(identity_deviations):.4f}"
                if identity_deviations
                else ""
            )
        )
