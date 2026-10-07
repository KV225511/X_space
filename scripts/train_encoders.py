"""
AstroFoundation — Encoder Training Script
==========================================
Trains each of the three modality encoders independently using the
pre-downloaded light_curves_2000.npz dataset.

Usage:
    python scripts/train_encoders.py --encoder light_curve --epochs 30
    python scripts/train_encoders.py --encoder light_curve --epochs 30 --seed 123
"""

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Data
from src.data.dataset import NPZLightCurveDataset, NPZSpectralDataset
from src.data.dataloaders import create_dataloader

# Models
from src.models.transformer_encoder import LightCurveEncoder
from src.models.image_encoder import FITSEncoder
from src.models.spectral_encoder import SpectralEncoder

# Training
from src.training.trainer import Trainer
from src.training.losses import FocalLoss

# ── Constants ──────────────────────────────────────────────────────────────
NPZ_PATH         = "data/processed/light_curves_2000.npz"
SPECTRA_NPZ_PATH = "data/processed/spectra_2000.npz"
SPLITS_DIR       = Path("data/splits")
NUM_CLASSES      = 6
BATCH_SIZE       = 16


# ── Helper: Classification head ────────────────────────────────────────────
class EncoderWithHead(nn.Module):
    """
    Wraps any encoder (outputs (B, 256) L2-normalized embedding) with a
    linear classification head that produces (B, NUM_CLASSES) logits.
    The FocalLoss operates on raw logits, not normalized embeddings.
    """
    def __init__(self, encoder: nn.Module, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(256, num_classes)

    def forward(self, x, mask=None):
        if mask is not None:
            embedding = self.encoder(x, mask)
        else:
            embedding = self.encoder(x)
        logits = self.head(embedding)  # (B, num_classes)
        return logits


# ── Helper: build NPZ splits ───────────────────────────────────────────────
def build_lc_dataloaders(npz_path: str, batch_size: int, num_workers: int = 0):
    """
    Loads the pre-built NPZ archive and returns train/val DataLoaders.
    Uses a fixed 70/15/15 split by index (reproducible regardless of seed).
    """
    data = np.load(npz_path, allow_pickle=True)
    N = len(data['labels'])

    # Deterministic split indices matching the CSV splits
    rng = np.random.default_rng(seed=42)
    idx = rng.permutation(N)
    train_end = int(0.70 * N)
    val_end   = int(0.85 * N)

    train_idx = idx[:train_end]
    val_idx   = idx[train_end:val_end]

    train_ds = NPZLightCurveDataset(npz_path, indices=train_idx)
    val_ds   = NPZLightCurveDataset(npz_path, indices=val_idx)

    train_loader = create_dataloader(train_ds, batch_size=batch_size, is_training=True,  num_workers=num_workers)
    val_loader   = create_dataloader(val_ds,   batch_size=batch_size, is_training=False, num_workers=num_workers)

    print(f"  Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")
    return train_loader, val_loader


def build_spectral_dataloaders(npz_path: str, batch_size: int, num_workers: int = 0, augment: bool = False):
    """
    Loads the pre-built spectral NPZ archive and returns train/val DataLoaders.
    Uses the identical fixed 70/15/15 split by index matching the master manifest.
    """
    data = np.load(npz_path, allow_pickle=True)
    N = len(data['labels'])

    rng = np.random.default_rng(seed=42)
    idx = rng.permutation(N)
    train_end = int(0.70 * N)
    val_end   = int(0.85 * N)

    train_idx = idx[:train_end]
    val_idx   = idx[train_end:val_end]

    train_ds = NPZSpectralDataset(npz_path, indices=train_idx, augment=augment)
    val_ds   = NPZSpectralDataset(npz_path, indices=val_idx, augment=False)

    train_loader = create_dataloader(train_ds, batch_size=batch_size, is_training=True,  num_workers=num_workers)
    val_loader   = create_dataloader(val_ds,   batch_size=batch_size, is_training=False, num_workers=num_workers)

    print(f"  Train: {len(train_ds)} samples (augment={augment})  |  Val: {len(val_ds)} samples")
    return train_loader, val_loader


# ── Helper: set random seed ─────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train AstroFoundation Encoders")
    parser.add_argument(
        '--encoder', type=str, required=True,
        choices=['light_curve', 'image', 'spectral'],
        help="Which encoder to train"
    )
    parser.add_argument('--epochs',  type=int,   default=30,   help="Number of training epochs")
    parser.add_argument('--seed',    type=int,   default=42,   help="Random seed (use 42, 123, 456 for 3-seed runs)")
    parser.add_argument('--lr',      type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--workers', type=int,   default=0,    help="DataLoader workers (0 = main process, safe on Windows)")
    parser.add_argument('--dataset-path', type=str, default=None, help="Custom dataset path (e.g. data/processed/spectra_sdss_600.npz)")
    parser.add_argument('--pretrained-ckpt', type=str, default=None, help="Path to checkpoint to initialize model weights from")
    parser.add_argument('--use-scheduler', action='store_true', help="Use cosine annealing learning rate scheduler with warmup")
    parser.add_argument('--warmup-epochs', type=int, default=3, help="Number of linear warmup epochs")
    parser.add_argument('--augment', action='store_true', help="Apply data augmentation during training")
    parser.add_argument('--save-prefix', type=str, default=None, help="Prefix for saved checkpoint name")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  AstroFoundation Encoder Training")
    print(f"  Encoder    : {args.encoder}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Seed       : {args.seed}")
    print(f"  Device     : {device}")
    print(f"  LR         : {args.lr}")
    print(f"  Scheduler  : {'Cosine with Warmup' if args.use_scheduler else 'Constant'}")
    print(f"  Augment    : {args.augment}")
    print(f"  Pretrained : {args.pretrained_ckpt}")
    print(f"{'='*60}\n")

    # Load config
    with open("config/model_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # ── Light Curve Encoder ──────────────────────────────────────────────
    if args.encoder == 'light_curve':
        print("Loading pre-built NPZ dataset...")
        train_loader, val_loader = build_lc_dataloaders(
            NPZ_PATH, batch_size=BATCH_SIZE, num_workers=args.workers
        )
        base_encoder = LightCurveEncoder()
        model = EncoderWithHead(base_encoder, num_classes=NUM_CLASSES)

    elif args.encoder == 'image':
        print("NOTE: FITSDataset requires local FITS files. Using dummy data for shape validation.")
        print("Wire your FITSDataset here once local files are available.")
        from torch.utils.data import TensorDataset
        dummy_X = torch.randn(100, 3, 224, 224)  # 3-channel (single-band FITS)
        dummy_y = torch.randint(0, NUM_CLASSES, (100,))
        dummy_ds = TensorDataset(dummy_X, dummy_y)
        train_loader = create_dataloader(dummy_ds, batch_size=BATCH_SIZE, is_training=True,  num_workers=0)
        val_loader   = create_dataloader(dummy_ds, batch_size=BATCH_SIZE, is_training=False, num_workers=0)
        base_encoder = FITSEncoder(in_channels=3)
        model = EncoderWithHead(base_encoder, num_classes=NUM_CLASSES)

    elif args.encoder == 'spectral':
        dataset_path = args.dataset_path or (
            "data/processed/spectra_sdss_600.npz" if Path("data/processed/spectra_sdss_600.npz").exists()
            else ("data/processed/spectra_sdss_real.npz" if Path("data/processed/spectra_sdss_real.npz").exists()
            else SPECTRA_NPZ_PATH)
        )
        if Path(dataset_path).exists():
            print(f"Loading spectral dataset from {dataset_path}...")
            train_loader, val_loader = build_spectral_dataloaders(
                dataset_path, batch_size=BATCH_SIZE, num_workers=args.workers, augment=args.augment
            )
        else:
            print(f"WARNING: {dataset_path} not found. Using fallback dummy data.")
            from torch.utils.data import TensorDataset
            dummy_X = torch.randn(100, 4096)
            dummy_y = torch.randint(0, NUM_CLASSES, (100,))
            dummy_ds = TensorDataset(dummy_X, dummy_y)
            train_loader = create_dataloader(dummy_ds, batch_size=BATCH_SIZE, is_training=True,  num_workers=0)
            val_loader   = create_dataloader(dummy_ds, batch_size=BATCH_SIZE, is_training=False, num_workers=0)
        base_encoder = SpectralEncoder()
        model = EncoderWithHead(base_encoder, num_classes=NUM_CLASSES)

    # ── Pretrained checkpoint loading (Transfer Learning) ───────────────
    if args.pretrained_ckpt:
        ckpt_p = Path(args.pretrained_ckpt)
        if ckpt_p.exists():
            print(f"Loading pretrained weights from {ckpt_p}...")
            ckpt = torch.load(ckpt_p, map_location=device)
            state_dict = ckpt.get('model_state_dict', ckpt)
            model_dict = model.state_dict()
            compatible = {
                k: v for k, v in state_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            model_dict.update(compatible)
            model.load_state_dict(model_dict)
            print(f"Successfully loaded {len(compatible)}/{len(model_dict)} tensors into model.")
        else:
            print(f"WARNING: Pretrained checkpoint {ckpt_p} not found! Initializing randomly.")

    # ── Training setup ───────────────────────────────────────────────────
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = None
    if args.use_scheduler:
        from src.training.schedulers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_epochs=args.warmup_epochs,
            num_total_epochs=args.epochs,
        )

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device, scheduler=scheduler)
    trainer.train(num_epochs=args.epochs)

    # ── Save checkpoint ──────────────────────────────────────────────────
    out_dir = Path("outputs/checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.save_prefix or args.encoder
    ckpt_path = out_dir / f"{prefix}_seed{args.seed}_epoch{args.epochs}.pt"
    torch.save({
        'epoch': args.epochs,
        'seed': args.seed,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, ckpt_path)
    print(f"\nCheckpoint saved -> {ckpt_path}")


if __name__ == "__main__":
    main()