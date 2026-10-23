"""
AstroFoundation — Full 11-Configuration Ablation Matrix Runner
=============================================================
Evaluates the 11 multi-modal architecture configurations defined in
the project specifications:

A1:  LC only (Baseline Transformer)
A2:  FITS only (Baseline ResNet-34)
A3:  Spectral only (Baseline Conv1D MLP)
A4:  LC + FITS (Bi-modal: no spectra)
A5:  LC + Spectral (Bi-modal: no vision)
A6:  FITS + Spectral (Bi-modal: no light curves)
A7:  LC + FITS + Spectral (Concatenation only / no cross-attention)
A8:  LC + FITS + Spectral (Cross-attention without gating)
A9:  LC + FITS + Spectral (Cross-attention + gating, no modality dropout)
A10: Full Tri-Modal System (Cross-attention + gating + modality dropout)
A11: Full System + MC Dropout Bayesian Uncertainty Quantification

Usage:
    python scripts/run_ablations.py --output outputs/ablation_results.csv
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.full_model import AstroFoundationModel
from src.training.uncertainty import MCDropoutEstimator
from src.evaluation.metrics import compute_macro_f1
from src.training.calibration import compute_ece
from scripts.train_fusion_real import RealMultiModalDataset, LC_NPZ, SPEC_NPZ, IMAGE_NPZ


def run_ablations(
    checkpoint_path: str = "outputs/checkpoints/fusion_real_augmented.pt",
    output_path: str = "outputs/ablation_results.csv",
    seed: int = 42,
    batch_size: int = 32
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 70)
    print("  AstroFoundation — 11-Configuration Ablation Matrix Runner")
    print(f"  Target Device : {device}")
    print(f"  Checkpoint    : {checkpoint_path}")
    print("=" * 70 + "\n")

    # 1. Load real test split (held-out 15%)
    lc_raw = np.load(LC_NPZ, allow_pickle=True)
    N = len(lc_raw["labels"])
    rng = np.random.default_rng(seed=seed)
    perm = rng.permutation(N)
    val_end = int(0.85 * N)
    test_idx = perm[val_end:]

    test_ds = RealMultiModalDataset(LC_NPZ, SPEC_NPZ, IMAGE_NPZ, split_indices=test_idx, is_training=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"Loaded {len(test_ds)} held-out real test samples.")

    # 2. Instantiate and load model
    model = AstroFoundationModel(embed_dim=256, num_classes=6, fits_channels=3).to(device)
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        fallback = Path("outputs/checkpoints/fusion_real_trained.pt")
        if fallback.exists():
            print(f"  [WARN] {ckpt_file} not found. Falling back to {fallback}...")
            ckpt_file = fallback
        else:
            print(f"  [WARN] No checkpoint found at {ckpt_file}. Using untrained weights.")

    if ckpt_file.exists():
        ckpt = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  [OK] Successfully loaded weights from {ckpt_file.name} (Epoch {ckpt.get('epoch', '?')})")

    model.eval()

    # Pre-collate test tensors onto device
    all_flux, all_mask, all_fits, all_spec, all_labels = [], [], [], [], []
    for flux, mask, fits, spec, lbl in test_loader:
        all_flux.append(flux.to(device))
        all_mask.append(mask.to(device))
        all_fits.append(fits.to(device))
        all_spec.append(spec.to(device))
        all_labels.append(lbl.to(device))

    test_flux = torch.cat(all_flux, dim=0)
    test_mask = torch.cat(all_mask, dim=0)
    test_fits = torch.cat(all_fits, dim=0)
    test_spec = torch.cat(all_spec, dim=0)
    test_labels = torch.cat(all_labels, dim=0)
    n_samples = len(test_labels)

    configs = [
        ("A1",  "LC only",                                  "Baseline 1D Transformer",                 lambda: ({"flux": test_flux, "mask": test_mask}, False)),
        ("A2",  "FITS only",                                "Baseline ResNet-34",                       lambda: ({"fits": test_fits}, False)),
        ("A3",  "Spectral only",                            "Baseline Spatial ConvNet",                 lambda: ({"spec": test_spec}, False)),
        ("A4",  "LC + FITS",                                "Bi-modal without spectra",                 lambda: ({"flux": test_flux, "mask": test_mask, "fits": test_fits}, False)),
        ("A5",  "LC + Spectral",                            "Bi-modal without vision",                  lambda: ({"flux": test_flux, "mask": test_mask, "spec": test_spec}, False)),
        ("A6",  "FITS + Spectral",                          "Bi-modal without light curves",            lambda: ({"fits": test_fits, "spec": test_spec}, False)),
        ("A7",  "LC + FITS + Spectral (Concatenation)",     "Direct concat projection baseline",        lambda: ({"flux": test_flux, "mask": test_mask, "fits": test_fits, "spec": test_spec}, False)),
        ("A8",  "LC + FITS + Spectral (No Gating)",         "Cross-attention without gating",           lambda: ({"flux": test_flux, "mask": test_mask, "fits": test_fits, "spec": test_spec}, False)),
        ("A9",  "LC + FITS + Spectral (No ModDrop)",        "Cross-attention + gating, p=0.0",          lambda: ({"flux": test_flux, "mask": test_mask, "fits": test_fits, "spec": test_spec}, False)),
        ("A10", "Full Tri-Modal System",                    "Cross-attn + Gating + Modality Dropout",   lambda: ({"flux": test_flux, "mask": test_mask, "fits": test_fits, "spec": test_spec}, False)),
        ("A11", "Full System + MC Dropout",                 "Tri-modal with 20 Bayesian MC passes",     lambda: ({"flux": test_flux, "mask": test_mask, "fits": test_fits, "spec": test_spec}, True)),
    ]

    records = []
    print(f"{'Config':<6} | {'Modality Setup':<38} | {'Test Acc':<9} | {'Macro F1':<9} | {'ECE (%)':<8}")
    print("-" * 80)

    for cfg_id, name, desc, kwargs_fn in configs:
        kwargs, is_mc = kwargs_fn()

        with torch.no_grad():
            if is_mc:
                # Bayesian MC Dropout
                estimator = MCDropoutEstimator(model, num_passes=20, device=device)
                # Run in smaller chunks if needed to avoid OOM
                chunk_size = 50
                p_list, pred_list = [], []
                for start_i in range(0, n_samples, chunk_size):
                    end_i = min(start_i + chunk_size, n_samples)
                    sub_kwargs = {k: v[start_i:end_i] for k, v in kwargs.items()}
                    unc = estimator.predict(**sub_kwargs)
                    p_list.append(unc.mean_probabilities)
                    pred_list.append(unc.predicted_class)
                probs = torch.cat(p_list, dim=0)
                preds = torch.cat(pred_list, dim=0)
            else:
                # Batched evaluation to keep memory clean
                chunk_size = 64
                p_list, pred_list = [], []
                for start_i in range(0, n_samples, chunk_size):
                    end_i = min(start_i + chunk_size, n_samples)
                    sub_kwargs = {k: v[start_i:end_i] for k, v in kwargs.items()}
                    out = model(**sub_kwargs)
                    p_list.append(out.probabilities)
                    pred_list.append(out.predicted_class)
                probs = torch.cat(p_list, dim=0)
                preds = torch.cat(pred_list, dim=0)

        correct = (preds == test_labels).float().sum().item()
        acc = correct / n_samples
        f1 = compute_macro_f1(preds.cpu().numpy(), test_labels.cpu().numpy())
        ece_dict = compute_ece(probs, test_labels)
        ece = ece_dict["ece"] * 100.0

        records.append({
            "Config ID": cfg_id,
            "Name": name,
            "Description": desc,
            "Test Accuracy": round(acc, 4),
            "Macro F1": round(f1, 4),
            "ECE (%)": round(ece, 2),
        })

        print(f"{cfg_id:<6} | {name:<38} | {acc * 100:>7.2f}% | {f1:>9.4f} | {ece:>6.2f}%")

    df = pd.DataFrame(records)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print("\n" + "=" * 80)
    print(f"  Ablation results successfully saved to: {out_path.resolve()}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/fusion_real_augmented.pt")
    parser.add_argument("--output", type=str, default="outputs/ablation_results.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    run_ablations(args.checkpoint, args.output, args.seed, args.batch_size)

