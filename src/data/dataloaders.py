import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from typing import Tuple, Dict, Any, Optional

from .dataset import LightCurveDataset, FITSDataset, SpectralDataset


def create_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int = 16,
    is_training: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Creates an optimized PyTorch DataLoader configured for AstroFoundation v4.0.

    Args:
        dataset: Instantiated PyTorch Dataset (LightCurveDataset, FITSDataset, or SpectralDataset).
        batch_size: Physical batch size (default 16 for RTX 4060 8GB VRAM footprint).
        is_training: If True, uses RandomSampler and drops partial trailing batches.
        num_workers: Number of background CPU worker subprocesses for parallel data loading.
        pin_memory: Locks host RAM memory pages to enable fast, asynchronous DMA transfer to GPU.

    Returns:
        Configured PyTorch DataLoader instance.
    """
    sampler = RandomSampler(dataset) if is_training else SequentialSampler(dataset)

    # Performance optimizations for multi-core CPU preprocessing
    kwargs = {}
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
        kwargs["persistent_workers"] = True

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_training,  # Prevents incomplete micro-batches during gradient accumulation
        **kwargs
    )


def get_dataloaders(
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    test_dataset: Optional[torch.utils.data.Dataset] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """
    DataLoader factory returning a dictionary containing train, validation, and test DataLoaders.

    Args:
        train_dataset: Dataset instance for training split.
        val_dataset: Dataset instance for validation split.
        test_dataset: Optional Dataset instance for evaluation split.
        batch_size: Physical batch size per GPU step (default: 16).
        num_workers: CPU background worker subprocesses (default: 4).
        pin_memory: Page-locked memory allocation for GPU transfer speed (default: True).

    Returns:
        Dictionary mapping keys ('train', 'val', 'test') to their respective DataLoaders.
    """
    dataloaders = {
        "train": create_dataloader(
            dataset=train_dataset,
            batch_size=batch_size,
            is_training=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": create_dataloader(
            dataset=val_dataset,
            batch_size=batch_size,
            is_training=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    if test_dataset is not None:
        dataloaders["test"] = create_dataloader(
            dataset=test_dataset,
            batch_size=batch_size,
            is_training=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return dataloaders