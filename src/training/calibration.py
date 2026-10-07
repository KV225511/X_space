"""
AstroFoundation — Post-Hoc Calibration & Expected Calibration Error (ECE)
========================================================================
Implements Temperature Scaling (Guo et al., 2017) to produce calibrated posterior
probabilities for astronomical event classification.

Uncalibrated neural networks frequently output overconfident probabilities
(e.g., 99% confidence on 75% accuracy regions). Temperature scaling recalibrates
the output distribution without affecting top-1 classification accuracy.
"""

from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def compute_ece(
    probabilities: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    n_bins: int = 15
) -> Dict[str, Any]:
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Args:
        probabilities: (N, num_classes) Softmax probabilities.
        labels: (N,) Ground truth integer class indices.
        n_bins: Number of confidence bins between 0.0 and 1.0.

    Returns:
        Dict with keys:
            'ece': float Expected Calibration Error
            'mce': float Maximum Calibration Error
            'bin_accuracies': np.ndarray of shape (n_bins,)
            'bin_confidences': np.ndarray of shape (n_bins,)
            'bin_counts': np.ndarray of shape (n_bins,)
    """
    if isinstance(probabilities, np.ndarray):
        probabilities = torch.from_numpy(probabilities)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    confidences, predictions = torch.max(probabilities, dim=1)
    accuracies = (predictions == labels).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = torch.zeros(1, device=probabilities.device)
    mce = torch.zeros(1, device=probabilities.device)

    bin_accs = []
    bin_confs = []
    bin_counts = []

    total_samples = len(labels)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()
        bin_count = in_bin.sum().item()
        bin_counts.append(bin_count)

        if bin_count > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            abs_diff = torch.abs(avg_confidence_in_bin - accuracy_in_bin)

            ece += abs_diff * prop_in_bin
            mce = torch.max(mce, abs_diff)

            bin_accs.append(accuracy_in_bin.item())
            bin_confs.append(avg_confidence_in_bin.item())
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_lower.item() + bin_upper.item()) / 2.0)

    return {
        "ece": float(ece.item()),
        "mce": float(mce.item()),
        "bin_accuracies": np.array(bin_accs),
        "bin_confidences": np.array(bin_confs),
        "bin_counts": np.array(bin_counts),
    }


def compute_brier_score(
    probabilities: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    num_classes: int = 6
) -> float:
    """
    Computes multi-class Brier score: (1/N) * sum_i sum_c (p_ic - y_ic)^2.
    Lower Brier score indicates better probabilistic calibration.
    """
    if isinstance(probabilities, np.ndarray):
        probabilities = torch.from_numpy(probabilities)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    one_hot = F.one_hot(labels.long(), num_classes=num_classes).float()
    brier = torch.mean(torch.sum((probabilities - one_hot) ** 2, dim=-1))
    return float(brier.item())


class TemperatureScaler(nn.Module):
    """
    Learns a single temperature parameter T > 0 on validation logits
    to calibrate confidence estimates without altering classification rankings.
    """
    def __init__(self, initial_temperature: float = 1.5):
        super().__init__()
        # Parameterize temperature with log_temperature to guarantee T > 0 strictly
        self.log_temperature = nn.Parameter(
            torch.ones(1) * np.log(max(1e-4, initial_temperature))
        )

    @property
    def temperature(self) -> torch.Tensor:
        """Returns positive scalar temperature T."""
        return torch.exp(self.log_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Scales logits by 1 / T.
        """
        return logits / self.temperature

    def calibrate(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (scaled_logits, calibrated_probabilities).
        """
        scaled = self.forward(logits)
        probs = F.softmax(scaled, dim=-1)
        return scaled, probs

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        max_iter: int = 50,
        lr: float = 0.01,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Optimizes temperature T using L-BFGS on validation logits to minimize NLL.

        Args:
            val_logits: (N, num_classes) Uncalibrated logits on validation set.
            val_labels: (N,) Ground truth labels.
            max_iter: Maximum optimization iterations.
            lr: Learning rate for L-BFGS optimizer.
            verbose: Whether to print optimization progress.

        Returns:
            Dict containing pre/post ECE, pre/post NLL, and learned temperature.
        """
        val_logits = val_logits.detach()
        val_labels = val_labels.detach()
        nll_criterion = nn.CrossEntropyLoss()

        # Compute pre-calibration metrics
        with torch.no_grad():
            orig_probs = F.softmax(val_logits, dim=-1)
            pre_ece_dict = compute_ece(orig_probs, val_labels)
            pre_nll = float(nll_criterion(val_logits, val_labels).item())
            pre_brier = compute_brier_score(orig_probs, val_labels, num_classes=val_logits.size(1))

        # L-BFGS optimizer for fast, robust scalar optimization
        optimizer = optim.LBFGS([self.log_temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(val_logits)
            loss = nll_criterion(scaled_logits, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        # Compute post-calibration metrics
        with torch.no_grad():
            scaled_logits = self.forward(val_logits)
            post_probs = F.softmax(scaled_logits, dim=-1)
            post_ece_dict = compute_ece(post_probs, val_labels)
            post_nll = float(nll_criterion(scaled_logits, val_labels).item())
            post_brier = compute_brier_score(post_probs, val_labels, num_classes=val_logits.size(1))
            learned_temp = float(self.temperature.item())

        results = {
            "optimal_temperature": learned_temp,
            "pre_ece": pre_ece_dict["ece"],
            "post_ece": post_ece_dict["ece"],
            "pre_nll": pre_nll,
            "post_nll": post_nll,
            "pre_brier": pre_brier,
            "post_brier": post_brier,
            "ece_reduction_percent": max(0.0, (pre_ece_dict["ece"] - post_ece_dict["ece"]) / max(1e-6, pre_ece_dict["ece"]) * 100.0)
        }

        if verbose:
            print(f"Optimal Temperature: {learned_temp:.4f}")
            print(f"  ECE: {results['pre_ece']:.4f} -> {results['post_ece']:.4f} ({results['ece_reduction_percent']:.1f}% reduction)")
            print(f"  NLL: {results['pre_nll']:.4f} -> {results['post_nll']:.4f}")

        return results
