"""
AstroFoundation — Full Multi-Modal Evaluation Suite
===================================================
Computes exhaustive evaluation metrics on the held-out real astronomical test set:
- Accuracy & Balanced Accuracy
- Macro & Weighted Precision
- Macro & Weighted Recall (Sensitivity)
- Specificity (True Negative Rate)
- Macro & Weighted F1-Score
- Per-Class Precision, Recall, F1, Specificity, and Support
- Matthews Correlation Coefficient (MCC)
- Macro AUROC (One-vs-Rest)
- Expected Calibration Error (ECE)
- Multi-Class Brier Score
- Multi-Class Log Loss (Cross-Entropy)
- Full 6x6 Confusion Matrix
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.full_model import AstroFoundationModel
from src.evaluation.metrics import compute_full_metrics_suite, ASTRONOMICAL_CLASSES
from scripts.train_fusion_real import RealMultiModalDataset, LC_NPZ, SPEC_NPZ, IMAGE_NPZ


def run_evaluation(
    checkpoint_path: str = "outputs/checkpoints/fusion_real_augmented.pt",
    output_json: str = "outputs/final_test_metrics.json",
    seed: int = 42,
    batch_size: int = 32,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 75)
    print("  AstroFoundation — Comprehensive Astronomical Metrics Suite")
    print(f"  Target Device : {device}")
    print(f"  Checkpoint    : {checkpoint_path}")
    print("=" * 75 + "\n")

    # 1. Held-out 15% test split (300 samples)
    lc_raw = np.load(LC_NPZ, allow_pickle=True)
    N = len(lc_raw["labels"])
    rng = np.random.default_rng(seed=seed)
    perm = rng.permutation(N)
    val_end = int(0.85 * N)
    test_idx = perm[val_end:]

    test_ds = RealMultiModalDataset(LC_NPZ, SPEC_NPZ, IMAGE_NPZ, split_indices=test_idx, is_training=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"Loaded {len(test_ds)} held-out real test samples.")

    # 2. Load Model
    model = AstroFoundationModel(embed_dim=256, num_classes=6, fits_channels=3).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint trained to Epoch {ckpt.get('epoch', '?')} (Val Loss: {ckpt.get('val_loss', 0.0):.4f})")

    # 3. Collate Test Batches
    all_flux, all_mask, all_fits, all_spec, all_labels = [], [], [], [], []
    for flux, mask, fits, spec, lbl in test_loader:
        all_flux.append(flux.to(device))
        all_mask.append(mask.to(device))
        all_fits.append(fits.to(device))
        all_spec.append(spec.to(device))
        all_labels.append(lbl.to(device))

    flux_t = torch.cat(all_flux, dim=0)
    mask_t = torch.cat(all_mask, dim=0)
    fits_t = torch.cat(all_fits, dim=0)
    spec_t = torch.cat(all_spec, dim=0)
    labels_t = torch.cat(all_labels, dim=0)
    n_samples = len(labels_t)

    # Function to evaluate given modalities
    def evaluate_modality_dict(sub_kwargs):
        chunk_size = 64
        probs_list, preds_list = [], []
        with torch.no_grad():
            for i in range(0, n_samples, chunk_size):
                chunk_kwargs = {k: v[i:min(i + chunk_size, n_samples)] for k, v in sub_kwargs.items()}
                out = model(**chunk_kwargs)
                probs_list.append(out.probabilities.cpu().numpy())
                preds_list.append(out.predicted_class.cpu().numpy())
        probs = np.concatenate(probs_list, axis=0)
        preds = np.concatenate(preds_list, axis=0)
        labels = labels_t.cpu().numpy()
        return compute_full_metrics_suite(probs, preds, labels, class_names=ASTRONOMICAL_CLASSES)

    # 4. Evaluate Tri-Modal (Primary) and Unimodal baselines
    tri_modal_metrics = evaluate_modality_dict({
        "flux": flux_t,
        "mask": mask_t,
        "fits": fits_t,
        "spec": spec_t,
    })

    lc_only_metrics = evaluate_modality_dict({"flux": flux_t, "mask": mask_t})
    fits_only_metrics = evaluate_modality_dict({"fits": fits_t})
    spec_only_metrics = evaluate_modality_dict({"spec": spec_t})

    results = {
        "tri_modal": tri_modal_metrics,
        "unimodal_lc": lc_only_metrics,
        "unimodal_fits": fits_only_metrics,
        "unimodal_spectral": spec_only_metrics,
    }

    out_file = Path(output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    # 5. Print Exhaustive Publication-Grade Summary
    m = tri_modal_metrics
    print("\n" + "=" * 75)
    print("  FINAL TRI-MODAL ASTRONOMICAL METRICS (HELD-OUT TEST SET)")
    print("=" * 75)
    print(f"  Classification Accuracy       : {m['accuracy'] * 100:6.2f}%")
    print(f"  Balanced Accuracy             : {m['balanced_accuracy'] * 100:6.2f}%")
    print(f"  Macro Precision               : {m['macro_precision']:6.4f}")
    print(f"  Weighted Precision            : {m['weighted_precision']:6.4f}")
    print(f"  Macro Recall (Sensitivity)    : {m['macro_recall']:6.4f}")
    print(f"  Weighted Recall               : {m['weighted_recall']:6.4f}")
    print(f"  Macro Specificity (TNR)       : {m['macro_specificity']:6.4f}")
    print(f"  Macro F1-Score                : {m['macro_f1']:6.4f}")
    print(f"  Weighted F1-Score             : {m['weighted_f1']:6.4f}")
    print(f"  Matthews Corr Coef (MCC)      : {m['matthews_corrcoef']:6.4f}")
    print(f"  Macro AUROC (One-vs-Rest)     : {m['macro_auroc']:6.4f}")
    print(f"  Expected Calibration Error    : {m['expected_calibration_error'] * 100:6.2f}%")
    print(f"  Brier Score (Prob MSE)        : {m['brier_score']:6.5f}")
    print(f"  Log Loss (Cross-Entropy)      : {m['log_loss']:6.5f}")
    print("=" * 75)

    print("\n" + "-" * 75)
    print(f"  PER-CLASS BREAKDOWN (Precision, Recall, F1, Specificity, Support)")
    print("-" * 75)
    print(f"  {'Class Name':<24} | {'Prec':<7} | {'Recall':<7} | {'F1':<7} | {'Spec':<7} | {'Support':<7}")
    print("-" * 75)
    for cname, stats in m["per_class"].items():
        print(f"  {cname:<24} | {stats['precision']:<7.4f} | {stats['recall']:<7.4f} | {stats['f1_score']:<7.4f} | {stats['specificity']:<7.4f} | {stats['support']:<7d}")
    print("-" * 75)

    print("\n" + "-" * 75)
    print("  CONFUSION MATRIX (6x6)")
    print("-" * 75)
    cm = np.array(m["confusion_matrix"])
    header = "  " + "".join([f"{cname[:8]:>9}" for cname in ASTRONOMICAL_CLASSES])
    print(header)
    for i, row in enumerate(cm):
        row_str = f"  {ASTRONOMICAL_CLASSES[i][:8]:<8} " + "".join([f"{val:>8d} " for val in row])
        print(row_str)
    print("-" * 75)

    print("\n" + "=" * 75)
    print(f"  Full metrics JSON saved to: {out_file.resolve()}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/fusion_real_augmented.pt")
    parser.add_argument("--output", type=str, default="outputs/final_test_metrics.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    run_evaluation(args.checkpoint, args.output, args.seed, args.batch_size)
