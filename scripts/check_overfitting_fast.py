"""
AstroFoundation -- FAST Overfitting Diagnostic (LC-only, no FITS)
==================================================================
Runs 5 overfitting checks using ONLY the LightCurveEncoder + spectral
checkpoint -- skips the slow ResNet-34 FITS encoder entirely.
Completes in ~10 seconds on CPU.

Checks:
  1. Train vs Val vs Test Accuracy gap
  2. Memorization detection (train acc near 100%?)
  3. Per-class accuracy breakdown
  4. Weight L2 norm analysis
  5. MC Dropout epistemic uncertainty

Usage:
    python scripts/check_overfitting_fast.py
"""

import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.transformer_encoder import LightCurveEncoder

CLASS_NAMES = [
    "exoplanet_transit", "eclipsing_binary", "variable_star",
    "supernova",         "cataclysmic_variable", "gamma_ray_burst"
]


# ── Minimal classifier head ──────────────────────────────────────────────────
class LCClassifier(nn.Module):
    """Wraps LightCurveEncoder + a simple classification head for diagnostics."""
    def __init__(self, num_classes=6, dropout_p=0.15):
        super().__init__()
        self.encoder = LightCurveEncoder(
            input_dim=1, d_model=256, nhead=8, num_layers=4,
            patch_size=16, dropout=dropout_p
        )
        self.dropout = nn.Dropout(p=dropout_p)
        self.head    = nn.Linear(256, num_classes)

    def forward(self, flux, mask=None):
        emb    = self.encoder(flux, mask=mask)           # (B, 256)
        emb    = self.dropout(emb)
        logits = self.head(emb)                          # (B, 6)
        return logits


# ── Data loading ─────────────────────────────────────────────────────────────
def load_splits(npz_path="data/processed/light_curves_2000.npz", seed=42):
    data   = np.load(npz_path, allow_pickle=True)
    flux   = data["flux"].astype(np.float32)   # (N, 8192)
    labels = data["labels"].astype(np.int64)   # (N,)
    N      = len(labels)

    rng       = np.random.default_rng(seed=seed)
    idx       = rng.permutation(N)
    train_end = int(0.70 * N)
    val_end   = int(0.85 * N)

    return (
        (flux[idx[:train_end]],          labels[idx[:train_end]]),    # train
        (flux[idx[train_end:val_end]],   labels[idx[train_end:val_end]]),  # val
        (flux[idx[val_end:]],            labels[idx[val_end:]]),      # test
    )


# ── Evaluation ───────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, flux_np, labels_np, device, batch_size=64, desc=""):
    model.eval()
    N       = len(labels_np)
    correct = 0
    per_class_correct = np.zeros(6, dtype=int)
    per_class_total   = np.zeros(6, dtype=int)
    all_confs = []

    for s in range(0, N, batch_size):
        e      = min(s + batch_size, N)
        flux_b = torch.tensor(flux_np[s:e], dtype=torch.float32, device=device)
        lbl_b  = torch.tensor(labels_np[s:e], dtype=torch.long, device=device)
        # unsqueeze to (B, L, 1) for the encoder's input_dim=1
        flux_b = flux_b.unsqueeze(-1)

        logits = model(flux_b)
        probs  = F.softmax(logits, dim=-1)
        preds  = probs.argmax(dim=1)

        correct += (preds == lbl_b).sum().item()
        confs    = probs.max(dim=1).values
        all_confs.append(confs.cpu())

        for c in range(6):
            mask = (lbl_b == c)
            per_class_correct[c] += (preds[mask] == c).sum().item()
            per_class_total[c]   += mask.sum().item()

    acc      = correct / N
    avg_conf = torch.cat(all_confs).mean().item()
    print(f"  {desc:22s}  Acc: {acc*100:6.2f}%   AvgConf: {avg_conf*100:5.2f}%   N={N}")
    return acc, avg_conf, per_class_correct, per_class_total


# ── Weight norm check ─────────────────────────────────────────────────────────
def weight_norm_check(model):
    print("\n  " + "-"*60)
    print("  CHECK 4 -- Weight Norm Analysis")
    print("  " + "-"*60)
    total_sq = 0.0
    flagged  = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            n = p.data.norm(2).item()
            total_sq += n * n
            if n > 30.0:
                flagged.append((name, n))
    total = total_sq ** 0.5
    print(f"  Total L2 norm of all weights: {total:.2f}")
    if flagged:
        print(f"  WARNING -- {len(flagged)} layers with norm > 30:")
        for name, n in sorted(flagged, key=lambda x: -x[1])[:5]:
            print(f"    {name[:55]:55s}  {n:.2f}")
    else:
        print("  OK -- All norms < 30 (healthy range)")
    print(f"  Verdict: {'OVERFITTING SIGNAL' if total > 300 else 'HEALTHY'}")
    return total


# ── MC Dropout ────────────────────────────────────────────────────────────────
def mc_dropout_check(model, flux_np, labels_np, device, n_passes=20):
    print("\n  " + "-"*60)
    print("  CHECK 5 -- MC Dropout Epistemic Uncertainty (on train data)")
    print("  " + "-"*60)

    n  = min(80, len(labels_np))
    fx = torch.tensor(flux_np[:n], dtype=torch.float32, device=device).unsqueeze(-1)

    model.train()   # activate dropout for stochastic passes
    passes = []
    for _ in range(n_passes):
        with torch.no_grad():
            probs = F.softmax(model(fx), dim=-1)
            passes.append(probs.cpu())
    model.eval()

    stk        = torch.stack(passes)           # (n_passes, n, 6)
    mean_p     = stk.mean(0)                   # (n, 6)
    eps        = 1e-8
    H_total    = -(mean_p * (mean_p + eps).log()).sum(-1).mean().item()
    H_per_pass = -(stk * (stk + eps).log()).sum(-1)
    aleatoric  = H_per_pass.mean().item()
    epistemic  = H_total - aleatoric

    print(f"  Total Predictive Entropy  : {H_total:.4f} nats")
    print(f"  Aleatoric Uncertainty     : {aleatoric:.4f} nats")
    print(f"  Epistemic Uncertainty     : {epistemic:.4f} nats")
    print()
    if epistemic < 0.005:
        print("  OK -- Extremely low epistemic uncertainty.")
        print("       Model is stable & confident on training data.")
        print("       (If train acc >> test acc, that still means memorization.)")
    elif epistemic < 0.03:
        print("  OK -- Low epistemic uncertainty. Model is well-trained.")
    else:
        print("  NOTE -- Moderate epistemic uncertainty.")
        print("       Model may not be fully converged. Consider more epochs.")
    return epistemic


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    npz    = "data/processed/light_curves_2000.npz"
    ckpt   = "outputs/checkpoints/light_curve_seed42_epoch30.pt"

    print("\n" + "="*65)
    print("  AstroFoundation -- FAST Overfitting Diagnostic")
    print(f"  Using: LightCurveEncoder only  (no FITS ResNet -- fast!)")
    print(f"  Checkpoint: {ckpt}")
    print(f"  Device    : {device}")
    print("="*65)

    # Build model
    model = LCClassifier(num_classes=6, dropout_p=0.15).to(device)

    # Try to load unimodal LC checkpoint
    ckpt_path = Path(ckpt)
    loaded = False
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        full_state = state.get("model_state_dict", state)
        try:
            model.load_state_dict(full_state, strict=False)
            print(f"\n  Full model weights (encoder + head) loaded from {ckpt_path.name}")
            loaded = True
        except Exception as e:
            print(f"\n  Could not load full state dict directly: {e}")
            enc_state  = {k.replace("encoder.", "", 1): v
                          for k, v in full_state.items()
                          if k.startswith("encoder.")}
            if enc_state:
                model.encoder.load_state_dict(enc_state, strict=False)
                print(f"\n  Encoder weights loaded from {ckpt_path.name}")
                loaded = True
    if not loaded:
        print(f"\n  [!] Could not load checkpoint -- using random weights.")
        print(f"      Results will show random-chance accuracy (~16.7%).")
        print(f"      This still correctly tests the diagnostic logic.\n")
    model.eval()

    # Load data
    print(f"\n  Loading data: {npz}")
    (train_flux, train_lbl), (val_flux, val_lbl), (test_flux, test_lbl) = load_splits(npz)

    # CHECK 1/2/3: Accuracy across splits
    print("\n  " + "-"*60)
    print("  CHECK 1/2/3 -- Accuracy Across Train / Val / Test Splits")
    print("  " + "-"*60)
    print("  Rule: Train Acc >> Test Acc  ==>  OVERFITTING")
    print()

    train_acc, train_conf, trn_cc, trn_ct = evaluate(model, train_flux, train_lbl, device, desc="Training Set  ")
    val_acc,   val_conf,   val_cc, val_ct = evaluate(model, val_flux,   val_lbl,   device, desc="Validation Set")
    test_acc,  test_conf,  tst_cc, tst_ct = evaluate(model, test_flux,  test_lbl,  device, desc="Test Set      ")

    gap_tv = train_acc - val_acc
    gap_tt = train_acc - test_acc

    print(f"\n  Train - Val  Gap : {gap_tv*100:+.2f}%")
    print(f"  Train - Test Gap : {gap_tt*100:+.2f}%")

    print("\n  --- Overfitting Verdict ---")
    if abs(gap_tt) < 0.01:
        verdict = "HEALTHY GENERALIZATION"
        detail  = "Train-Test gap < 1%. Model generalizes well. NOT overfitting."
    elif abs(gap_tt) < 0.05:
        verdict = "MILD OVERFITTING"
        detail  = "Train-Test gap 1-5%. Add L2 regularization or more dropout."
    else:
        verdict = "SIGNIFICANT OVERFITTING"
        detail  = "Train-Test gap > 5%. Apply early stopping + weight decay."
    print(f"  >> {verdict}")
    print(f"     {detail}")

    print("\n  --- Memorization Check ---")
    if train_acc > 0.999:
        print(f"  WARNING: Train Acc = {train_acc*100:.2f}% (near 100%!)")
        print(f"           Test  Acc = {test_acc*100:.2f}%")
        print(f"           Gap       = {gap_tt*100:.2f}%")
        if abs(gap_tt) > 0.05:
            print("           => Model is MEMORIZING training data (overfitting confirmed).")
        else:
            print("           => High train AND test acc means genuine learning, not memorization.")
    else:
        print(f"  OK: Train Acc {train_acc*100:.2f}% -- not near 100%, no memorization risk.")

    # Per-class breakdown
    print("\n  --- Per-Class Accuracy (Test Set) ---")
    for i, cname in enumerate(CLASS_NAMES):
        if tst_ct[i] > 0:
            acc_c = tst_cc[i] / tst_ct[i]
            bar   = "#" * int(acc_c * 20)
            print(f"  {cname:25s}  [{bar:<20}] {acc_c*100:5.1f}%  ({tst_cc[i]}/{tst_ct[i]})")

    # CHECK 4: Weight norms
    weight_norm_check(model)

    # CHECK 5: MC Dropout
    mc_dropout_check(model, train_flux, train_lbl, device)

    # Final Summary
    print("\n" + "="*65)
    print("  DIAGNOSTIC SUMMARY")
    print("="*65)
    print(f"  {'Metric':<28} {'Value':>10}    Interpretation")
    print(f"  {'-'*28} {'-'*10}    {'-'*25}")
    print(f"  {'Train Accuracy':<28} {train_acc*100:>9.2f}%   {'MEMORIZED?' if train_acc > 0.999 else 'Normal'}")
    print(f"  {'Val Accuracy':<28} {val_acc*100:>9.2f}%   {'Good' if val_acc > 0.90 else 'LOW - underfit'}")
    print(f"  {'Test Accuracy':<28} {test_acc*100:>9.2f}%   {'Good' if test_acc > 0.90 else 'LOW - underfit'}")
    print(f"  {'Train-Val Gap':<28} {gap_tv*100:>+9.2f}%   {'OK' if abs(gap_tv) < 0.05 else 'OVERFIT'}")
    print(f"  {'Train-Test Gap':<28} {gap_tt*100:>+9.2f}%   {'OK' if abs(gap_tt) < 0.05 else 'OVERFIT'}")

    print()
    print("  FIXES IF OVERFITTING:")
    print("  1. optimizer = Adam(params, lr=1e-4, weight_decay=1e-4)")
    print("  2. Increase dropout: dropout_p=0.15 -> 0.30")
    print("  3. Early stopping: save best model at lowest val_loss")
    print("  4. torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)")
    print()
    print("  FIXES IF UNDERFITTING (train acc still low):")
    print("  1. Train more epochs (30 -> 50)")
    print("  2. Increase learning rate (1e-5 -> 3e-4 for fusion layers)")
    print("  3. Use pretrained encoder weights (--load-pretrained)")
    print()

    # Write to file too
    print("  Report also saved to: outputs/overfitting_report.txt")


if __name__ == "__main__":
    # Redirect stdout to file AND console
    import io, os
    report_path = Path("outputs/overfitting_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    class Tee:
        def __init__(self, *files): self.files = files
        def write(self, s):
            for f in self.files: f.write(s)
        def flush(self):
            for f in self.files: f.flush()

    with open(report_path, "w", encoding="utf-8") as fout:
        import sys as _sys
        orig = _sys.stdout
        _sys.stdout = Tee(orig, fout)
        try:
            main()
        finally:
            _sys.stdout = orig
