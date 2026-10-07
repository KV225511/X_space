"""
AstroFoundation — Active Learning Acquisition Policies
======================================================
Implements Bayesian and Energy-based active learning selection strategies
for telescope target prioritization and expert follow-up.

Supported Acquisition Strategies:
1. ENTROPY: Maximum predictive entropy across all classes.
2. MARGIN: Smallest margin between top-1 and top-2 predicted probabilities.
3. LEAST_CONFIDENCE: Lowest top-1 predicted probability.
4. EPISTEMIC: Maximum mutual information from Monte Carlo Dropout.
5. ENERGY_ANOMALY: Maximum free energy E(x) detecting rare/unseen astrophysical events.
6. HYBRID: Joint acquisition balancing epistemic model uncertainty and OOD discovery.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
import numpy as np
import torch


@dataclass
class ActiveLearningBatch:
    """
    Result container for an active learning acquisition round.
    """
    selected_indices: np.ndarray        # Global indices into the unlabelled pool
    acquisition_scores: np.ndarray      # Ranked acquisition scores (higher = higher priority)
    strategy: str                       # Acquisition strategy used
    metadata: Dict[str, Any]            # Summary statistics (mean score, score percentiles)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """
    Min-max normalizes scores onto [0.0, 1.0] for fair cross-metric combination.
    """
    s_min, s_max = np.min(scores), np.max(scores)
    if s_max - s_min < 1e-8:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


class ActiveLearningSelector:
    """
    Coordinates target acquisition from an unlabelled observation pool.
    """
    VALID_STRATEGIES = [
        "entropy",
        "margin",
        "least_confidence",
        "epistemic",
        "energy_anomaly",
        "hybrid"
    ]

    def __init__(self, random_state: Optional[int] = 42):
        self.rng = np.random.default_rng(random_state)

    def compute_scores(
        self,
        probabilities: Optional[Union[torch.Tensor, np.ndarray]] = None,
        epistemic_uncertainty: Optional[Union[torch.Tensor, np.ndarray]] = None,
        energy_scores: Optional[Union[torch.Tensor, np.ndarray]] = None,
        strategy: str = "hybrid",
        alpha: float = 0.5,
        epsilon: float = 1e-8,
    ) -> np.ndarray:
        """
        Computes raw scalar acquisition score per sample.

        Args:
            probabilities: (N, C) Softmax probabilities.
            epistemic_uncertainty: (N,) Mutual information from MC Dropout.
            energy_scores: (N,) Free energy from TaskHeads / AnomalyHead.
            strategy: One of VALID_STRATEGIES.
            alpha: Weight for epistemic uncertainty in hybrid mode (1 - alpha for energy).
            epsilon: Numerical floor for log calculations.

        Returns:
            np.ndarray of shape (N,) containing acquisition scores.
        """
        strategy = strategy.lower()
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from {self.VALID_STRATEGIES}")

        # Convert tensors to numpy if needed
        if isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.detach().cpu().numpy()
        if isinstance(epistemic_uncertainty, torch.Tensor):
            epistemic_uncertainty = epistemic_uncertainty.detach().cpu().numpy()
        if isinstance(energy_scores, torch.Tensor):
            energy_scores = energy_scores.detach().cpu().numpy()

        # Strategy 1: Entropy
        if strategy == "entropy":
            if probabilities is None:
                raise ValueError("Strategy 'entropy' requires probabilities.")
            scores = -np.sum(probabilities * np.log(probabilities + epsilon), axis=-1)
            return scores

        # Strategy 2: Margin (smallest gap between 1st and 2nd best class)
        elif strategy == "margin":
            if probabilities is None:
                raise ValueError("Strategy 'margin' requires probabilities.")
            sorted_probs = np.sort(probabilities, axis=-1)
            # score = 1.0 - (top1 - top2) -> higher when top1 and top2 are close
            margin = sorted_probs[:, -1] - sorted_probs[:, -2]
            scores = 1.0 - margin
            return scores

        # Strategy 3: Least Confidence
        elif strategy == "least_confidence":
            if probabilities is None:
                raise ValueError("Strategy 'least_confidence' requires probabilities.")
            max_probs = np.max(probabilities, axis=-1)
            scores = 1.0 - max_probs
            return scores

        # Strategy 4: Epistemic Uncertainty (from MC Dropout)
        elif strategy == "epistemic":
            if epistemic_uncertainty is None:
                raise ValueError("Strategy 'epistemic' requires epistemic_uncertainty tensor.")
            return epistemic_uncertainty

        # Strategy 5: Energy Anomaly (from AnomalyHead)
        elif strategy == "energy_anomaly":
            if energy_scores is None:
                raise ValueError("Strategy 'energy_anomaly' requires energy_scores tensor.")
            return energy_scores

        # Strategy 6: Hybrid (Epistemic Exploration + Energy-based Discovery)
        elif strategy == "hybrid":
            if epistemic_uncertainty is None or energy_scores is None:
                raise ValueError("Strategy 'hybrid' requires both epistemic_uncertainty and energy_scores.")
            norm_epistemic = normalize_scores(epistemic_uncertainty)
            norm_energy = normalize_scores(energy_scores)
            scores = alpha * norm_epistemic + (1.0 - alpha) * norm_energy
            return scores

        raise NotImplementedError(f"Strategy {strategy} not implemented.")

    def select(
        self,
        k: int,
        probabilities: Optional[Union[torch.Tensor, np.ndarray]] = None,
        epistemic_uncertainty: Optional[Union[torch.Tensor, np.ndarray]] = None,
        energy_scores: Optional[Union[torch.Tensor, np.ndarray]] = None,
        already_labeled_mask: Optional[np.ndarray] = None,
        strategy: str = "hybrid",
        alpha: float = 0.5,
    ) -> ActiveLearningBatch:
        """
        Selects top-k most informative samples from pool.

        Args:
            k: Number of targets to acquire.
            probabilities: (N, C) class probabilities.
            epistemic_uncertainty: (N,) mutual information.
            energy_scores: (N,) OOD free energy.
            already_labeled_mask: (N,) bool array indicating samples already acquired.
            strategy: Acquisition policy.
            alpha: Epistemic vs energy weighting for hybrid mode.

        Returns:
            ActiveLearningBatch containing selected indices and ranked scores.
        """
        scores = self.compute_scores(
            probabilities=probabilities,
            epistemic_uncertainty=epistemic_uncertainty,
            energy_scores=energy_scores,
            strategy=strategy,
            alpha=alpha
        )

        n_total = len(scores)
        available_mask = np.ones(n_total, dtype=bool)
        if already_labeled_mask is not None:
            available_mask = available_mask & (~already_labeled_mask)

        valid_indices = np.where(available_mask)[0]
        if len(valid_indices) == 0:
            raise ValueError("No unlabelled samples remaining to select.")

        k = min(k, len(valid_indices))

        # Sort available samples by descending acquisition score
        valid_scores = scores[valid_indices]
        sorted_order = np.argsort(-valid_scores)
        top_k_rel_indices = sorted_order[:k]

        selected_indices = valid_indices[top_k_rel_indices]
        selected_scores = valid_scores[top_k_rel_indices]

        metadata = {
            "total_pool_size": n_total,
            "unlabeled_remaining": len(valid_indices) - k,
            "mean_acquired_score": float(np.mean(selected_scores)),
            "min_acquired_score": float(np.min(selected_scores)),
            "max_acquired_score": float(np.max(selected_scores)),
        }

        return ActiveLearningBatch(
            selected_indices=selected_indices,
            acquisition_scores=selected_scores,
            strategy=strategy,
            metadata=metadata
        )
