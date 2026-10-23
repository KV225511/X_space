"""
AstroFoundation -- Overfitting Diagnostic Script
=================================================
Runs 5 systematic overfitting checks on your trained model checkpoints:

  1. Train vs Val vs Test Accuracy -- the primary overfitting signal
  2. Train-Test Gap computation    -- quantifies generalization
  3. Memorization detection        -- train acc near 100% = dangerous
  4. Weight Norm Analysis          -- very large weights = overfitting indicator
  5. MC Dropout Epistemic Variance -- uncertainty on training data

Usage:
    python scripts/check_overfitting.py
    python scripts/check_overfitting.py --checkpoint outputs/checkpoints/fusion_seed42_epoch5.pt
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.full_model import AstroFoundationModel


def load_splits(npz_path, seed=42):
    """Load and split the processed NPZ into train/val/test arrays."""
    data = np.load(npz_path, allow_pickle=True)
    flux   = data["flux"]    # (N, 8192)
    labels = data["labels"]  # (N,)
    N = len(labels)

    rng = np.random.default_rng(seed=seed)
    idx = rng.permutation(N)
    train_end = int(0.70 * N)
    val_end   = int(0.85 * N)

    return {
        "train": (flux[idx[:train_end]],          labels[idx[:train_end]]),
        "val":   (flux[idx[train_end:val_end]],   labels[idx[train_end:val_end]]),
        "test":  (flux[idx[val_end:]],            labels[idx[val_end:]]),
    }


@torch.no_grad()
def evaluate_accuracy(model, flux_np, labels_np, device, batch_size=32, desc=""):
    """Compute accuracy on a split without gradient computation."""
    model.eval()
    N = len(labels_np)
    correct = 0
    all_probs = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        flux_b  = torch.tensor(flux_np[start:end],   dtype=torch.float32, device=device)
        label_b = torch.tensor(labels_np[start:end], dtype=torch.long,    device=device)

        # Dummy FITS/spec tensors matching training setup
        fits_b = torch.zeros(end - start, 3, 224, 224, device=device)
        spec_b = torch.zeros(end - start, 4096, device=device)

        out   = model(flux=flux_b, fits=fits_b, spec=spec_b)
        probs = F.softmax(out.logits, dim=-1)
        preds = probs.argmax(dim=1)

        correct += (preds == label_b).sum().item()
        all_probs.append(probs.cpu())

    acc        = correct / N
    all_probs  = torch.cat(all_probs, dim=0)
    avg_conf   = all_probs.max(dim=1).values.mean().item()

    print(f"  {desc:25s} | Acc: {acc*100:6.2f}%  | Avg Conf: {avg_conf*100:5.2f}%  | N={N}")
    return acc, avg_conf, all_probs


def compute_weight_norms(model):
    """Check for suspiciously large weight norms."""
    print("\n" + "-" * 65)
    print("  CHECK 4: Weight Norm Analysis")
    print("-" * 65)
    total_sq   = 0.0
    flagged    = []

    for name, param in model.named_parameters():
        if param.requires_grad and param.data is not None:
            norm = param.data.norm(2).item()
            total_sq += norm ** 2
            if norm > 50.0:
                flagged.append((name, norm))

    total_norm = total_sq ** 0.5
    print(f"  Total L2 Norm (all weights): {total_norm:.2f}")

    if flagged:
        print(f"  WARNING: {len(flagged)} layers with L2 norm > 50 (suspicious):")
        for name, norm in sorted(flagged, key=lambda x: -x[1])[:5]:
            print(f"    {name:50s}  norm={norm:.2f}")
    else:
        print("  OK: All layer weight norms within healthy range (< 50).")

    verdict = "OVERFITTING SIGNAL" if total_norm > 500 else "HEALTHY"
    print(f"  Verdict: {verdict}")
    return total_norm


def compute_mc_dropout_variance(model, flux_np, labels_np, device, n_passes=20):
    """
    MC Dropout: run N stochastic passes on training data.
    High epistemic uncertainty on TRAINING data = model hasn't memorized it.
    Very low epistemic uncertainty on training data = could be overfitting.
    """
    print("\n" + "-" * 65)
    print("  CHECK 5: MC Dropout Epistemic Uncertainty (on training data)")
    print("-" * 65)

    n_samples = min(50, len(labels_np))
    flux_sub  = torch.tensor(flux_np[:n_samples], dtype=torch.float32, device=device)
    fits_sub  = torch.zeros(n_samples, 3, 224, 224, device=device)
    spec_sub  = torch.zeros(n_samples, 4096, device=device)

    model.train()  # Activate dropout for stochastic sampling
    all_pass_probs = []
    for _ in range(n_passes):
        with torch.no_grad():
            out   = model(flux=flux_sub, fits=fits_sub, spec=spec_sub)
            probs = F.softmax(out.logits, dim=-1)
            all_pass_probs.append(probs.cpu())
    model.eval()

    stacked    = torch.stack(all_pass_probs, dim=0)  # (n_passes, n_samples, 6)
    mean_probs = stacked.mean(dim=0)                 # (n_samples, 6)

    eps           = 1e-8
    pred_entropy  = -(mean_probs * (mean_probs + eps).log()).sum(dim=-1).mean().item()
    per_pass_ent  = -(stacked * (stacked + eps).log()).sum(dim=-1)
    aleatoric     = per_pass_ent.mean().item()
    epistemic     = pred_entropy - aleatoric

    print(f"  Total Uncertainty  (H_total):  {pred_entropy:.4f} nats")
    print(f"  Aleatoric Uncertainty:         {aleatoric:.4f} nats")
    print(f"  Epistemic Uncertainty:         {epistemic:.4f} nats")
    print()

    if epistemic < 0.005:
        print("  OK: Very low epistemic uncertainty -- model is stable and confident.")
        print("     (Consistent with a well-trained, non-memorizing model.)")
    elif epistemic < 0.02:
        print("  OK: Low epistemic uncertainty -- model is well-trained, not overfitting.")
    else:
        print("  WARNING: Moderate epistemic uncertainty on training data.")
        print("     Model may not have fully converged. Try more training epochs.")

    return pred_entropy, epistemic


def main():
    parser = argparse.ArgumentParser(description="AstroFoundation Overfitting Diagnostic")
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/checkpoints/fusion_seed42_epoch5.pt",
                        help="Path to trained model checkpoint .pt file")
    parser.add_argument("--npz", type=str,
                        default="data/processed/light_curves_2000.npz",
                        help="Path to processed light curves NPZ")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)

    print("\n" + "=" * 65)
    print("  AstroFoundation -- Overfitting Diagnostic Report")
    print(f"  Checkpoint : {ckpt_path.name}")
    print(f"  Device     : {device}")
    print("=" * 65)

    # --- Load Model ---
    model = AstroFoundationModel(embed_dim=256, num_classes=6, fits_channels=3).to(device)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"\n  Model loaded: {ckpt_path}")
        if "val_loss" in ckpt:
            print(f"  Saved Val Loss: {ckpt['val_loss']:.4f}")
    else:
        print(f"\n  [!] Checkpoint not found: {ckpt_path}")
        print("      Running on random weights. Train first with:")
        print("      python scripts/train_fusion.py --epochs 20\n")
    model.eval()

    # --- Load Data ---
    splits = load_splits(args.npz, seed=args.seed)
    train_flux, train_labels = splits["train"]
    val_flux,   val_labels   = splits["val"]
    test_flux,  test_labels  = splits["test"]

    # --- CHECK 1, 2, 3: Accuracy across splits ---
    print("\n" + "-" * 65)
    print("  CHECK 1/2/3: Accuracy Comparison Across All Splits")
    print("-" * 65)
    print("  KEY RULE: If Train >> Val/Test accuracy => OVERFITTING")
    print()

    train_acc, train_conf, _ = evaluate_accuracy(model, train_flux, train_labels, device, desc="Training Set  ")
    val_acc,   val_conf,   _ = evaluate_accuracy(model, val_flux,   val_labels,   device, desc="Validation Set")
    test_acc,  test_conf,  _ = evaluate_accuracy(model, test_flux,  test_labels,  device, desc="Test Set      ")

    gap_tv = abs(train_acc - val_acc)
    gap_tt = abs(train_acc - test_acc)

    print()
    print(f"  Train - Val  Gap : {gap_tv*100:.2f}%")
    print(f"  Train - Test Gap : {gap_tt*100:.2f}%")
    print()

    if gap_tt < 0.01:
        print("  VERDICT: HEALTHY GENERALIZATION (gap < 1%)")
        print("  Model generalizes well. No significant overfitting detected.")
    elif gap_tt < 0.05:
        print("  VERDICT: MILD OVERFITTING (gap 1-5%)")
        print("  Consider adding L2 regularization or more dropout.")
    else:
        print("  VERDICT: SIGNIFICANT OVERFITTING (gap > 5%)")
        print("  Apply fixes listed at the end of this report.")

    print()
    print("  --- Memorization Check ---")
    if train_acc > 0.999:
        print(f"  WARNING: Train Acc = {train_acc*100:.2f}% (~100%) -- potential memorization!")
        print(f"           Compare with Test Acc = {test_acc*100:.2f}% to confirm.")
    else:
        print(f"  OK: Train Acc = {train_acc*100:.2f}% -- not near-perfect, healthy.")

    # --- CHECK 4: Weight Norms ---
    compute_weight_norms(model)

    # --- CHECK 5: MC Dropout ---
    compute_mc_dropout_variance(model, train_flux, train_labels, device)

    # --- Summary ---
    print("\n" + "=" * 65)
    print("  DIAGNOSTIC SUMMARY TABLE")
    print("=" * 65)
    print(f"  {'Metric':<30} {'Value':<12} Status")
    print(f"  {'-'*30} {'-'*12} {'-'*10}")
    print(f"  {'Train Accuracy':<30} {train_acc*100:>6.2f}%      {'OK' if train_acc < 0.999 else 'CHECK (high)'}")
    print(f"  {'Val Accuracy':<30} {val_acc*100:>6.2f}%      {'OK' if val_acc > 0.90 else 'LOW'}")
    print(f"  {'Test Accuracy':<30} {test_acc*100:>6.2f}%      {'OK' if test_acc > 0.90 else 'LOW'}")
    print(f"  {'Train-Val Gap':<30} {gap_tv*100:>6.2f}%      {'OK' if gap_tv < 0.05 else 'OVERFIT'}")
    print(f"  {'Train-Test Gap':<30} {gap_tt*100:>6.2f}%      {'OK' if gap_tt < 0.05 else 'OVERFIT'}")
    print()
    print("  IF OVERFITTING IS FOUND -- Apply these fixes:")
    print("  1. Add L2 regularization:   optimizer weight_decay=1e-4")
    print("  2. More modality dropout:   p=0.15 -> p=0.25")
    print("  3. Light curve augmentation: add Gaussian noise during training")
    print("  4. Early stopping: halt when val_loss doesn't improve for 5 epochs")
    print("  5. Gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)")
    print()
    print("  IF MODEL IS UNDERFITTING (val/test acc still low) -- Try:")
    print("  1. Train more epochs: currently 5 -> try 30-50 epochs")
    print("  2. Increase fusion learning rate (--lr-fusion 3e-4)")
    print("  3. Load pretrained encoder weights (--load-pretrained)")
    print()


if __name__ == "__main__":
    main()
