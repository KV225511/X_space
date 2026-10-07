"""
AstroFoundation — Monte Carlo Dropout & Uncertainty Quantification
===================================================================
Implements Bayesian approximation via Test-Time Monte Carlo (MC) Dropout
(Gal & Ghahramani, 2016) for multimodal astronomical foundation models.

Decomposes predictive uncertainty into:
1. Aleatoric Uncertainty: Inherent data/sensor noise (Expected Entropy).
2. Epistemic Uncertainty: Model ignorance due to sparse training regions (Mutual Information).
3. Predictive Entropy: Total system uncertainty.
4. Per-Class Empirical Variance & Credible Intervals.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclass
class UncertaintyEstimate:
    """
    Structured container for Bayesian uncertainty quantification.
    """
    mean_probabilities: torch.Tensor    # (B, num_classes) Mean predicted probability distribution
    predicted_class: torch.Tensor       # (B,) Argmax class index
    confidence: torch.Tensor            # (B,) Maximum mean probability max_c p_c
    predictive_entropy: torch.Tensor    # (B,) Total uncertainty H(p_mean)
    aleatoric_uncertainty: torch.Tensor # (B,) Expected entropy E[H(p)] (data noise)
    epistemic_uncertainty: torch.Tensor # (B,) Mutual information I(y, w | x) (model ignorance)
    std_per_class: torch.Tensor         # (B, num_classes) Empirical std across MC passes
    credible_interval_90: Tuple[torch.Tensor, torch.Tensor] # (5th percentile, 95th percentile)
    raw_samples: Optional[torch.Tensor] = None # (num_passes, B, num_classes) if requested


def enable_dropout_only(model: nn.Module) -> None:
    """
    Puts all dropout layers into train mode while strictly preserving
    eval mode for BatchNorm and LayerNorm to avoid corrupting running statistics.
    """
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            module.train()


class MCDropoutEstimator:
    """
    Monte Carlo Dropout estimator for PyTorch models.
    Supports single-modality encoders, fused classifiers, and full AstroFoundationModel.
    """
    def __init__(
        self,
        model: nn.Module,
        num_passes: int = 50,
        epsilon: float = 1e-8,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: PyTorch model with dropout layers.
            num_passes: Number of stochastic forward passes (default 50).
            epsilon: Numerical stability floor for log operations.
            device: Target torch device (CPU or CUDA).
        """
        self.model = model
        self.num_passes = num_passes
        self.epsilon = epsilon
        self.device = device or (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def _forward_pass(self, *args, **kwargs) -> torch.Tensor:
        """
        Executes a single forward pass and returns raw logits or probabilities.
        Handles both models returning raw tensors and those returning TaskHeadOutput.
        """
        out = self.model(*args, **kwargs)
        if hasattr(out, 'logits'):
            return out.logits
        if hasattr(out, 'probabilities'):
            return out.probabilities
        return out

    def predict(
        self,
        *args,
        return_samples: bool = False,
        **kwargs
    ) -> UncertaintyEstimate:
        """
        Runs N stochastic forward passes with dropout enabled and computes
        full Bayesian uncertainty decomposition.

        Args:
            *args, **kwargs: Model inputs (e.g. flux, fits, spec, mask).
            return_samples: Whether to attach raw (N, B, C) samples to output.

        Returns:
            UncertaintyEstimate dataclass with full statistical metrics.
        """
        self.model.eval()
        enable_dropout_only(self.model)

        pass_probs = []

        with torch.no_grad():
            for _ in range(self.num_passes):
                logits = self._forward_pass(*args, **kwargs)
                # If output is already softmaxed
                if torch.allclose(logits.sum(dim=-1), torch.ones(logits.size(0), device=logits.device), atol=1e-3):
                    probs = logits
                else:
                    probs = F.softmax(logits, dim=-1)
                pass_probs.append(probs)

        # Shape: (num_passes, B, C)
        stacked_probs = torch.stack(pass_probs, dim=0)

        # 1. Mean probability distribution: (B, C)
        mean_probs = torch.mean(stacked_probs, dim=0)
        predicted_class = torch.argmax(mean_probs, dim=-1)
        confidence, _ = torch.max(mean_probs, dim=-1)

        # 2. Predictive Entropy (Total Uncertainty): H(p_mean) = -sum(p * log(p))
        predictive_entropy = -torch.sum(
            mean_probs * torch.log(mean_probs + self.epsilon), dim=-1
        )

        # 3. Aleatoric Uncertainty (Expected Entropy): E[H(p)] = (1/N) * sum_i(H(p_i))
        per_pass_entropy = -torch.sum(
            stacked_probs * torch.log(stacked_probs + self.epsilon), dim=-1
        )  # (num_passes, B)
        aleatoric_uncertainty = torch.mean(per_pass_entropy, dim=0)  # (B,)

        # 4. Epistemic Uncertainty (Mutual Information): I = H(p_mean) - E[H(p)]
        # Mutual information is strictly non-negative; clamp to zero for float precision
        epistemic_uncertainty = torch.clamp(
            predictive_entropy - aleatoric_uncertainty, min=0.0
        )

        # 5. Standard deviation per class: (B, C)
        std_per_class = torch.std(stacked_probs, dim=0)

        # 6. Empirical Credible Intervals (5th & 95th percentiles)
        q_low = torch.quantile(stacked_probs, 0.05, dim=0)
        q_high = torch.quantile(stacked_probs, 0.95, dim=0)

        return UncertaintyEstimate(
            mean_probabilities=mean_probs,
            predicted_class=predicted_class,
            confidence=confidence,
            predictive_entropy=predictive_entropy,
            aleatoric_uncertainty=aleatoric_uncertainty,
            epistemic_uncertainty=epistemic_uncertainty,
            std_per_class=std_per_class,
            credible_interval_90=(q_low, q_high),
            raw_samples=stacked_probs if return_samples else None,
        )
