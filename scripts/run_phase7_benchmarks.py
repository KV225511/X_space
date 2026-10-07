"""
AstroFoundation — Phase 7 Benchmark Demonstration
=================================================
Evaluates:
1. Monte Carlo Dropout Bayesian Uncertainty Decomposition
2. Temperature Scaling & Expected Calibration Error (ECE) Minimization
3. Active Learning Target Acquisition on real SDSS spectroscopic observations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import torch.nn.functional as F

from src.models.spectral_encoder import SpectralEncoder
from scripts.train_encoders import EncoderWithHead, NUM_CLASSES
from src.data.dataset import NPZSpectralDataset
from src.data.dataloaders import create_dataloader
from src.training.uncertainty import MCDropoutEstimator
from src.training.calibration import TemperatureScaler, compute_ece, compute_brier_score
from src.training.active_learning import ActiveLearningSelector


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*65)
    print("  AstroFoundation — Phase 7: Uncertainty & Calibration Benchmark")
    print(f"  Device: {device}")
    print("="*65 + "\n")

    # 1. Load validation dataset
    npz_path = "data/processed/spectra_sdss_real.npz"
    data = np.load(npz_path, allow_pickle=True)
    N = len(data['labels'])
    rng = np.random.default_rng(seed=42)
    idx = rng.permutation(N)
    val_idx = idx[int(0.70 * N):int(0.85 * N)]

    val_ds = NPZSpectralDataset(npz_path, indices=val_idx, augment=False)
    val_loader = create_dataloader(val_ds, batch_size=16, is_training=False, num_workers=0)

    # 2. Load trained model
    ckpt_path = Path("outputs/checkpoints/spectral_seed42_epoch25.pt")
    base_encoder = SpectralEncoder()
    model = EncoderWithHead(base_encoder, num_classes=NUM_CLASSES)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {ckpt_path.name}")
    print(f"Validation dataset size: {len(val_ds)} samples\n")

    # 3. Collect validation logits & ground truth
    all_logits = []
    all_labels = []
    all_spectra = []

    with torch.no_grad():
        for bx, by in val_loader:
            bx = bx.to(device)
            logits = model(bx)
            all_logits.append(logits.cpu())
            all_labels.append(by)
            all_spectra.append(bx.cpu())

    val_logits = torch.cat(all_logits, dim=0)
    val_labels = torch.cat(all_labels, dim=0)
    val_spectra = torch.cat(all_spectra, dim=0)

    # -------------------------------------------------------------
    # PART 1: Temperature Scaling Post-Hoc Calibration
    # -------------------------------------------------------------
    print("--- 1. POST-HOC TEMPERATURE SCALING CALIBRATION ---")
    orig_probs = F.softmax(val_logits, dim=-1)
    pre_ece_dict = compute_ece(orig_probs, val_labels, n_bins=10)
    pre_brier = compute_brier_score(orig_probs, val_labels)

    scaler = TemperatureScaler(initial_temperature=1.0)
    cal_results = scaler.fit(val_logits, val_labels, max_iter=50, lr=0.05)

    _, cal_probs = scaler.calibrate(val_logits)
    post_ece_dict = compute_ece(cal_probs, val_labels, n_bins=10)
    post_brier = compute_brier_score(cal_probs, val_labels)

    print(f"Optimal Temperature (T):        {cal_results['optimal_temperature']:.4f}")
    print(f"Expected Calibration Error:     {pre_ece_dict['ece']*100:.2f}% -> {post_ece_dict['ece']*100:.2f}%  ({cal_results['ece_reduction_percent']:.1f}% reduction)")
    print(f"Maximum Calibration Error:      {pre_ece_dict['mce']*100:.2f}% -> {post_ece_dict['mce']*100:.2f}%")
    print(f"Negative Log-Likelihood (NLL):  {cal_results['pre_nll']:.4f} -> {cal_results['post_nll']:.4f}")
    print(f"Multi-class Brier Score:        {pre_brier:.4f} -> {post_brier:.4f}")

    # Verify accuracy preserved
    pre_acc = (torch.argmax(val_logits, dim=-1) == val_labels).float().mean().item()
    post_acc = (torch.argmax(cal_probs, dim=-1) == val_labels).float().mean().item()
    print(f"Top-1 Accuracy Invariance:      {pre_acc*100:.2f}% == {post_acc*100:.2f}% (Strictly preserved)\n")

    # -------------------------------------------------------------
    # PART 2: Monte Carlo Dropout Uncertainty Quantification
    # -------------------------------------------------------------
    print("--- 2. MONTE CARLO DROPOUT BAYESIAN DECOMPOSITION ---")
    num_passes = 30
    estimator = MCDropoutEstimator(model, num_passes=num_passes, device=device)
    unc = estimator.predict(val_spectra.to(device))

    mean_entropy = float(unc.predictive_entropy.mean().item())
    mean_aleatoric = float(unc.aleatoric_uncertainty.mean().item())
    mean_epistemic = float(unc.epistemic_uncertainty.mean().item())
    mean_confidence = float(unc.confidence.mean().item())

    print(f"MC Dropout Passes:              {num_passes}")
    print(f"Mean Predictive Confidence:     {mean_confidence*100:.2f}%")
    print(f"Mean Predictive Entropy (Total):{mean_entropy:.4f} nats")
    print(f"Mean Aleatoric Uncertainty:     {mean_aleatoric:.4f} nats (Observational noise)")
    print(f"Mean Epistemic Uncertainty:     {mean_epistemic:.4f} nats (Model ignorance)")

    # Identify sample with highest and lowest epistemic uncertainty
    highest_epistemic_idx = int(torch.argmax(unc.epistemic_uncertainty).item())
    lowest_epistemic_idx = int(torch.argmin(unc.epistemic_uncertainty).item())

    class_names = ["exoplanet_host", "white_dwarf", "variable_m_star", "supernova_host", "cataclysmic_var", "qso_powerlaw"]
    print(f"\nSample with Highest Epistemic Uncertainty (Index {highest_epistemic_idx}):")
    print(f"  True Label:      {class_names[val_labels[highest_epistemic_idx]]}")
    print(f"  Predicted Label: {class_names[unc.predicted_class[highest_epistemic_idx]]}")
    print(f"  Epistemic Value: {unc.epistemic_uncertainty[highest_epistemic_idx].item():.4f} nats")
    print(f"  Confidence:      {unc.confidence[highest_epistemic_idx].item()*100:.1f}%")

    print(f"\nSample with Lowest Epistemic Uncertainty (Index {lowest_epistemic_idx}):")
    print(f"  True Label:      {class_names[val_labels[lowest_epistemic_idx]]}")
    print(f"  Predicted Label: {class_names[unc.predicted_class[lowest_epistemic_idx]]}")
    print(f"  Epistemic Value: {unc.epistemic_uncertainty[lowest_epistemic_idx].item():.4f} nats")
    print(f"  Confidence:      {unc.confidence[lowest_epistemic_idx].item()*100:.1f}%\n")

    # -------------------------------------------------------------
    # PART 3: Active Learning Acquisition
    # -------------------------------------------------------------
    print("--- 3. ACTIVE LEARNING TARGET ACQUISITION (Top-5 Targets) ---")
    selector = ActiveLearningSelector(random_state=42)

    # Compute energy anomaly scores using T=1.0
    energy_scores = -torch.logsumexp(val_logits, dim=-1)

    batch_hybrid = selector.select(
        k=5,
        probabilities=cal_probs,
        epistemic_uncertainty=unc.epistemic_uncertainty.cpu(),
        energy_scores=energy_scores.cpu(),
        strategy="hybrid",
        alpha=0.5
    )

    print(f"Acquisition Strategy: HYBRID (50% Epistemic + 50% Energy Anomaly)")
    print(f"Selected Target Indices: {batch_hybrid.selected_indices.tolist()}")
    for rank, (s_idx, score) in enumerate(zip(batch_hybrid.selected_indices, batch_hybrid.acquisition_scores), 1):
        true_cls = class_names[val_labels[s_idx]]
        pred_cls = class_names[unc.predicted_class[s_idx]]
        ep_val = unc.epistemic_uncertainty[s_idx].item()
        en_val = energy_scores[s_idx].item()
        print(f"  Rank #{rank} | Index {s_idx:>2} | Priority Score: {score:.4f} | Epistemic: {ep_val:.4f} | Energy: {en_val:.4f} | Pred: {pred_cls:<16} | True: {true_cls}")

    print("\n" + "="*65)
    print("  Phase 7 Benchmark Successfully Completed!")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
