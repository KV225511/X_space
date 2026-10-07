"""
AstroFoundation — Learning Rate Schedulers
===========================================
Implements learning rate schedulers for multimodal foundation models,
including Cosine Annealing with linear warmup.
"""

import math
import torch
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_epochs: int,
    num_total_epochs: int,
    min_lr_ratio: float = 1e-3,
) -> LambdaLR:
    """
    Creates a learning rate scheduler with linear warmup followed by
    cosine annealing down to min_lr_ratio * base_lr.

    Args:
        optimizer: PyTorch optimizer
        num_warmup_epochs: Epochs for linear warmup from 0 to base_lr
        num_total_epochs: Total number of training epochs
        min_lr_ratio: Minimum LR floor ratio (relative to initial LR)

    Returns:
        LambdaLR scheduler instance
    """
    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < num_warmup_epochs:
            # Linear warmup: from 0.05 to 1.0
            return max(0.05, float(current_epoch) / float(max(1, num_warmup_epochs)))
        
        # Cosine annealing
        progress = float(current_epoch - num_warmup_epochs) / float(
            max(1, num_total_epochs - num_warmup_epochs)
        )
        progress = min(1.0, max(0.0, progress))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)
