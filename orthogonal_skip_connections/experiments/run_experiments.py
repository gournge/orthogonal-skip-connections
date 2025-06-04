"""Hydra‑powered CLI to launch a single training run."""
import hydra
from omegaconf import DictConfig
import wandb

from orthogonal_skip_connections.models.variants import get_model
from orthogonal_skip_connections.train.datamodule import get_dataloaders
from orthogonal_skip_connections.train.trainer import Trainer

@hydra.main(version_base="1.3", config_path="../config", config_name="default")
def main(cfg: DictConfig):
    wandb.init(project=cfg.wandb.project, mode=cfg.wandb.mode, config=dict(cfg))
    train_loader, test_loader, num_classes = get_dataloaders(cfg.dataset, cfg.batch_size, cfg.num_workers)
    model = get_model(cfg.resnet_variant, depth=20, num_classes=num_classes)
    trainer = Trainer(model, train_loader, test_loader, cfg)
    trainer.fit()

if __name__ == "__main__":
    main()