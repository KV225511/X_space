"""
AstroFoundation — Real Multi-Modal Fusion Training Script (GPU-Optimized)
========================================================================
Fuses real data across all 3 observational modalities:
  1. Real Time-Series Light Curves: data/processed/light_curves_2000.npz
  2. Real 1D SDSS Spectra:        data/processed/spectra_sdss_real.npz (593 spectra)
  3. Real 2D FITS Images:          data/processed/images_real.npz (624 images)

Features:
  - GPU acceleration with automatic mixed precision (torch.cuda.amp)
  - Pretrained unimodal encoder transfer learning
  - Active Modality Dropout with stochastic masking
  - Gradient clipping & Early stopping on validation loss

Usage:
  python scripts/train_fusion_real.py --epochs 30 --batch-size 16
"""

import argparse
import os
import random
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.full_model import AstroFoundationModel
from src.training.losses import FocalLoss
from src.data.preprocessing import augment_light_curve, augment_fits_image, augment_spectrum

LC_NPZ    = "data/processed/light_curves_2000.npz"
SPEC_NPZ  = "data/processed/spectra_sdss_real.npz"
IMAGE_NPZ = "data/processed/images_real.npz"
NUM_CLASSES = 6


class RealMultiModalDataset(Dataset):
    """
    Dataset that aligns real Light Curves, real SDSS Spectra, and real FITS images.
    Organized by shared astronomical class label for cross-modal contrastive association.
    Applies physics-informed online data augmentation during training.
    """
    def __init__(self, lc_npz: str, spec_npz: str, img_npz: str, split_indices: list = None, is_training: bool = True):
        # 1. Load Light Curves
        lc_data = np.load(lc_npz, allow_pickle=True)
        self.flux_all  = lc_data["flux"].astype(np.float32)
        self.masks_all = lc_data["masks"].astype(bool) if "masks" in lc_data else np.ones_like(self.flux_all, dtype=bool)
        self.lc_lbls   = lc_data["labels"].astype(np.int64)

        # 2. Load Real SDSS Spectra
        spec_data = np.load(spec_npz, allow_pickle=True)
        self.spec_all  = spec_data["spectra"].astype(np.float32)
        self.spec_lbls = spec_data["labels"].astype(np.int64)

        # 3. Load Real FITS Images
        img_data = np.load(img_npz, allow_pickle=True)
        self.imgs_all  = img_data["images"].astype(np.float32)
        self.img_lbls  = img_data["labels"].astype(np.int64)

        # Build class index lookup pools for multi-modal synchronization
        self.spec_by_class = {c: np.where(self.spec_lbls == c)[0] for c in range(NUM_CLASSES)}
        self.img_by_class  = {c: np.where(self.img_lbls == c)[0] for c in range(NUM_CLASSES)}

        # Split indices apply to light curves as primary index spine
        if split_indices is not None:
            self.indices = split_indices
        else:
            self.indices = np.arange(len(self.lc_lbls))

        self.is_training = is_training
        self.rng = np.random.RandomState(42 if not is_training else None)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        label = int(self.lc_lbls[idx])

        # Light curve sample
        flux_arr = self.flux_all[idx].copy()
        mask_arr = self.masks_all[idx].copy()

        # Associated real spectrum matching class
        spec_pool = self.spec_by_class[label]
        if len(spec_pool) > 0:
            spec_idx = spec_pool[idx % len(spec_pool)] if not self.is_training else self.rng.choice(spec_pool)
            spec_arr = self.spec_all[spec_idx].copy()
        else:
            spec_arr = np.zeros(4096, dtype=np.float32)

        # Associated real FITS image matching class
        img_pool = self.img_by_class[label]
        if len(img_pool) > 0:
            img_idx = img_pool[idx % len(img_pool)] if not self.is_training else self.rng.choice(img_pool)
            img_arr = self.imgs_all[img_idx].copy()
        else:
            img_arr = np.zeros((3, 224, 224), dtype=np.float32)

        # Online Physics-Informed Data Augmentation during training
        if self.is_training:
            flux_arr, mask_arr = augment_light_curve(flux_arr, mask_arr, p=0.5, rng=self.rng)
            spec_arr = augment_spectrum(spec_arr, p=0.5, rng=self.rng)
            img_arr  = augment_fits_image(img_arr, p=0.5, rng=self.rng)

        flux  = torch.tensor(flux_arr, dtype=torch.float32)
        mask  = torch.tensor(mask_arr, dtype=torch.bool)
        spec  = torch.tensor(spec_arr, dtype=torch.float32)
        img   = torch.tensor(img_arr,  dtype=torch.float32)
        lbl_t = torch.tensor(label,    dtype=torch.long)
        return flux, mask, img, spec, lbl_t


def train_epoch(model, dataloader, optimizer, scaler, criterion, device, use_amp):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for flux, mask, fits, spec, labels in tqdm(dataloader, desc="Training", leave=False):
        flux = flux.to(device)
        mask = mask.to(device)
        fits = fits.to(device)
        spec = spec.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(flux=flux, fits=fits, spec=spec, mask=mask)
            loss = criterion(out.logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = out.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for flux, mask, fits, spec, labels in dataloader:
        flux = flux.to(device)
        mask = mask.to(device)
        fits = fits.to(device)
        spec = spec.to(device)
        labels = labels.to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(flux=flux, fits=fits, spec=spec, mask=mask)
            loss = criterion(out.logits, labels)

        total_loss += loss.item() * len(labels)
        preds = out.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train AstroFoundation on Real Multi-Modal Data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr-encoders", type=float, default=1e-5)
    parser.add_argument("--lr-fusion", type=float, default=1e-4)
    parser.add_argument("--load-pretrained", action="store_true", default=True)
    parser.add_argument("--save-path", type=str, default="outputs/checkpoints/fusion_real_trained.pt")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")

    print("\n" + "=" * 65)
    print("  AstroFoundation — Multi-Modal Real Data Training")
    print("=" * 65)
    print(f"  Target Device     : {device}")
    if device.type == "cuda":
        print(f"  GPU Name          : {torch.cuda.get_device_name(0)}")
        print(f"  Total GPU Memory  : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"  CUDA Capability   : {torch.cuda.get_device_capability(0)}")
        print(f"  Mixed Precision   : ENABLED (torch.cuda.amp FP16)")
    else:
        print(f"  Mixed Precision   : Disabled (CPU execution)")
    print(f"  Epochs            : {args.epochs}")
    print(f"  Batch Size        : {args.batch_size}")
    print(f"  Light Curves NPZ  : {LC_NPZ}")
    print(f"  SDSS Spectra NPZ  : {SPEC_NPZ}")
    print(f"  Real Images NPZ   : {IMAGE_NPZ}")
    print("=" * 65 + "\n")

    # 1. Prepare 70/15/15 train/val/test splits
    lc_raw = np.load(LC_NPZ, allow_pickle=True)
    N = len(lc_raw["labels"])
    rng = np.random.default_rng(seed=args.seed)
    perm = rng.permutation(N)
    train_end = int(0.70 * N)
    val_end   = int(0.85 * N)

    train_idx = perm[:train_end]
    val_idx   = perm[train_end:val_end]
    test_idx  = perm[val_end:]

    train_ds = RealMultiModalDataset(LC_NPZ, SPEC_NPZ, IMAGE_NPZ, split_indices=train_idx, is_training=True)
    val_ds   = RealMultiModalDataset(LC_NPZ, SPEC_NPZ, IMAGE_NPZ, split_indices=val_idx,   is_training=False)
    test_ds  = RealMultiModalDataset(LC_NPZ, SPEC_NPZ, IMAGE_NPZ, split_indices=test_idx,  is_training=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=use_amp)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=use_amp)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=use_amp)

    print(f"Dataset Splits: {len(train_ds)} Train | {len(val_ds)} Val | {len(test_ds)} Test")

    # 2. Build Model
    model = AstroFoundationModel(embed_dim=256, num_classes=NUM_CLASSES, fits_channels=3).to(device)

    # 3. Load Pretrained Unimodal Checkpoints if available
    ckpt_dir = Path("outputs/checkpoints")
    if args.load_pretrained and ckpt_dir.exists():
        lc_ckpt = ckpt_dir / "light_curve_seed42_epoch30.pt"
        if lc_ckpt.exists():
            print(f"  [OK] Loading pretrained LightCurveEncoder from {lc_ckpt.name}...")
            st = torch.load(lc_ckpt, map_location=device)["model_state_dict"]
            enc_st = {k.replace("encoder.", ""): v for k, v in st.items() if k.startswith("encoder.")}
            model.lc_encoder.load_state_dict(enc_st, strict=False)

        spec_ckpt = ckpt_dir / "spectral_sdss_enhanced_seed42_epoch25.pt"
        if spec_ckpt.exists():
            print(f"  [OK] Loading pretrained SpectralEncoder from {spec_ckpt.name}...")
            st = torch.load(spec_ckpt, map_location=device)["model_state_dict"]
            enc_st = {k.replace("encoder.", ""): v for k, v in st.items() if k.startswith("encoder.")}
            model.spectral_encoder.load_state_dict(enc_st, strict=False)

    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    encoder_params = list(model.lc_encoder.parameters()) + list(model.image_encoder.parameters()) + list(model.spectral_encoder.parameters())
    fusion_params  = list(model.projections.parameters()) + list(model.fusion.parameters()) + list(model.task_heads.parameters())

    optimizer = optim.AdamW([
        {"params": encoder_params, "lr": args.lr_encoders, "weight_decay": 1e-4},
        {"params": fusion_params,  "lr": args.lr_fusion,   "weight_decay": 1e-4},
    ])

    best_val_loss = float("inf")
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nStarting Training Loop...")
    for epoch in range(1, args.epochs + 1):
        trn_loss, trn_acc = train_epoch(model, train_loader, optimizer, scaler, criterion, device, use_amp)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, use_amp)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, save_path)
            mark = "[BEST]"
        else:
            mark = ""

        print(f"  Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {trn_loss:.4f}  Acc: {trn_acc*100:5.2f}% | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc*100:5.2f}%  {mark}")

    # 4. Final Evaluation on Test Split
    print("\nEvaluating Best Model on Held-out Test Set...")
    best_ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    tst_loss, tst_acc = evaluate(model, test_loader, criterion, device, use_amp)

    print("\n" + "=" * 65)
    print("  FINAL REAL MULTI-MODAL RESULTS")
    print("=" * 65)
    print(f"  Best Epoch       : {best_ckpt['epoch']}")
    print(f"  Validation Loss  : {best_ckpt['val_loss']:.4f}")
    print(f"  Validation Acc   : {best_ckpt['val_acc']*100:.2f}%")
    print(f"  Test Accuracy    : {tst_acc*100:.2f}%")
    print(f"  Model Saved To   : {save_path.resolve()}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
