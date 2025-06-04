"""Hydra-powered CLI to launch a single training run."""

import hydra
from omegaconf import DictConfig
import wandb
import torch

from orthogonal_skip_connections.models import get_model, if_model_needs_update_rule
from orthogonal_skip_connections.train.datamodule import get_dataloaders
from orthogonal_skip_connections.train.trainer import Trainer


@hydra.main(version_base="1.3", config_path="../config", config_name="default")
def main(cfg: DictConfig):
    # Log whether we are using cuda
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but 'device' is set to 'cuda'.")
    elif cfg.device == "cpu":
        print("Using CPU for training.")
    else:
        print(f"Using {cfg.device} for training.")
    wandb.init(project=cfg.wandb.project, mode=cfg.wandb.mode, config=dict(cfg))
    train_loader, test_loader, num_classes = get_dataloaders(
        cfg.dataset, cfg.batch_size, cfg.num_workers
    )

    model_kwargs = {
        "num_classes": num_classes,
        "model_type": cfg.model.model_type,
        "skip_kind": cfg.model.skip_kind,
    }
    if if_model_needs_update_rule(cfg.model.skip_kind):
        model_kwargs["update_rule"] = cfg.model.update_rule

    model = get_model(**model_kwargs)

    trainer = Trainer(model, train_loader, test_loader, cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
