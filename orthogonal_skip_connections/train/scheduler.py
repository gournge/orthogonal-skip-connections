import torch.optim as optim
import math
from torch.optim.lr_scheduler import LambdaLR

def cosine_scheduler(optimizer: optim.Optimizer, total_epochs: int, warmup_epochs: int):
    def _lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda=_lr_lambda)