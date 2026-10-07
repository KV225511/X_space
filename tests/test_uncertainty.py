"""
Unit tests for AstroFoundation Phase 7:
- Monte Carlo Dropout (MCDropoutEstimator)
- Post-hoc Calibration (TemperatureScaler, compute_ece)
- Active Learning Acquisition Policies (ActiveLearningSelector)
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.training.uncertainty import MCDropoutEstimator, UncertaintyEstimate, enable_dropout_only
from src.training.calibration import TemperatureScaler, compute_ece, compute_brier_score
from src.training.active_learning import ActiveLearningSelector, ActiveLearningBatch
from src.models.full_model import AstroFoundationModel


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dummy Model Fixture with Dropout
# ─────────────────────────────────────────────────────────────────────────────
class DummyDropoutNet(nn.Module):
    def __init__(self, in_features: int = 16, num_classes: int = 6):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        return self.fc2(h)


@pytest.fixture
def dummy_net():
    torch.manual_seed(42)
    return DummyDropoutNet()


# ─────────────────────────────────────────────────────────────────────────────
# 2. MC Dropout Tests
# ─────────────────────────────────────────────────────────────────────────────
def test_mc_dropout_shapes_and_bounds(dummy_net):
    batch_size = 8
    num_classes = 6
    x = torch.randn(batch_size, 16)

    estimator = MCDropoutEstimator(dummy_net, num_passes=20)
    unc: UncertaintyEstimate = estimator.predict(x, return_samples=True)

    # 1. Shape checks
    assert unc.mean_probabilities.shape == (batch_size, num_classes)
    assert unc.predicted_class.shape == (batch_size,)
    assert unc.confidence.shape == (batch_size,)
    assert unc.predictive_entropy.shape == (batch_size,)
    assert unc.aleatoric_uncertainty.shape == (batch_size,)
    assert unc.epistemic_uncertainty.shape == (batch_size,)
    assert unc.std_per_class.shape == (batch_size, num_classes)
    assert unc.credible_interval_90[0].shape == (batch_size, num_classes)
    assert unc.credible_interval_90[1].shape == (batch_size, num_classes)
    assert unc.raw_samples.shape == (20, batch_size, num_classes)

    # 2. Probability axioms
    prob_sums = unc.mean_probabilities.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-4)

    # 3. Non-negativity
    assert (unc.predictive_entropy >= 0.0).all()
    assert (unc.aleatoric_uncertainty >= 0.0).all()
    assert (unc.epistemic_uncertainty >= 0.0).all()

    # 4. Fundamental Bayesian inequality: Total Entropy >= Aleatoric Entropy
    # (Epistemic = Total - Aleatoric >= 0)
    assert (unc.predictive_entropy >= unc.aleatoric_uncertainty - 1e-6).all()


def test_mc_dropout_with_astrofoundation_model():
    batch_size = 4
    model = AstroFoundationModel(embed_dim=256, num_classes=6, fits_channels=3)

    flux = torch.randn(batch_size, 8192)
    fits = torch.randn(batch_size, 3, 224, 224)
    spec = torch.randn(batch_size, 4096)

    estimator = MCDropoutEstimator(model, num_passes=5)
    unc = estimator.predict(flux=flux, fits=fits, spec=spec)

    assert unc.mean_probabilities.shape == (batch_size, 6)
    assert unc.predicted_class.shape == (batch_size,)
    assert torch.allclose(unc.mean_probabilities.sum(dim=-1), torch.ones(batch_size), atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Calibration & ECE Tests
# ─────────────────────────────────────────────────────────────────────────────
def test_compute_ece():
    n = 200
    # Create perfectly accurate and perfectly confident predictions
    probs_perfect = torch.zeros(n, 6)
    labels = torch.randint(0, 6, (n,))
    for i in range(n):
        probs_perfect[i, labels[i]] = 1.0

    ece_perfect = compute_ece(probs_perfect, labels, n_bins=10)
    assert ece_perfect["ece"] < 1e-4
    assert ece_perfect["mce"] < 1e-4
    assert len(ece_perfect["bin_accuracies"]) == 10

    # Overconfident wrong predictions
    probs_wrong = torch.zeros(n, 6)
    for i in range(n):
        wrong_class = (labels[i] + 1) % 6
        probs_wrong[i, wrong_class] = 0.99
        probs_wrong[i, labels[i]] = 0.01

    ece_wrong = compute_ece(probs_wrong, labels, n_bins=10)
    assert ece_wrong["ece"] > 0.80  # Severe miscalibration


def test_temperature_scaler_optimization():
    torch.manual_seed(42)
    n = 300
    num_classes = 6
    labels = torch.randint(0, 6, (n,))

    # Synthesize overconfident uncalibrated logits
    # (true class gets moderate logit, but scaled by 4x to simulate overconfidence)
    raw_logits = torch.randn(n, num_classes) * 1.5
    for i in range(n):
        if np.random.rand() < 0.70:  # 70% empirical accuracy
            raw_logits[i, labels[i]] += 4.0
    overconfident_logits = raw_logits * 3.0  # Artificially sharp probabilities

    scaler = TemperatureScaler(initial_temperature=1.0)
    results = scaler.fit(overconfident_logits, labels, max_iter=40, lr=0.05)

    # 1. Learned temperature should be > 1.0 to soften overconfident logits
    assert scaler.temperature.item() > 1.0

    # 2. Post-ECE should be significantly lower than pre-ECE
    assert results["post_ece"] < results["pre_ece"]

    # 3. Invariant: top-1 predicted accuracy MUST remain 100% identical
    pre_preds = torch.argmax(overconfident_logits, dim=-1)
    _, post_probs = scaler.calibrate(overconfident_logits)
    post_preds = torch.argmax(post_probs, dim=-1)
    assert (pre_preds == post_preds).all()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Active Learning Acquisition Tests
# ─────────────────────────────────────────────────────────────────────────────
def test_active_learning_strategies():
    n = 50
    num_classes = 6
    np.random.seed(42)

    probs = np.random.dirichlet(np.ones(num_classes), size=n)
    epistemic = np.random.exponential(scale=0.5, size=n)
    energy = np.random.normal(loc=2.0, scale=1.0, size=n)

    selector = ActiveLearningSelector(random_state=42)

    # 1. Entropy Strategy
    batch_entropy = selector.select(k=10, probabilities=probs, strategy="entropy")
    assert len(batch_entropy.selected_indices) == 10
    # Top score must be >= subsequent scores
    assert batch_entropy.acquisition_scores[0] >= batch_entropy.acquisition_scores[-1]

    # 2. Margin Strategy
    batch_margin = selector.select(k=10, probabilities=probs, strategy="margin")
    assert len(batch_margin.selected_indices) == 10

    # 3. Energy Anomaly Strategy
    batch_energy = selector.select(k=5, energy_scores=energy, strategy="energy_anomaly")
    assert len(batch_energy.selected_indices) == 5

    # 4. Hybrid Strategy
    batch_hybrid = selector.select(
        k=15,
        epistemic_uncertainty=epistemic,
        energy_scores=energy,
        strategy="hybrid",
        alpha=0.6
    )
    assert len(batch_hybrid.selected_indices) == 15
    assert batch_hybrid.strategy == "hybrid"

    # 5. Already labeled exclusion test
    already_labeled = np.zeros(n, dtype=bool)
    already_labeled[:10] = True  # First 10 already acquired

    batch_filtered = selector.select(
        k=10,
        probabilities=probs,
        already_labeled_mask=already_labeled,
        strategy="entropy"
    )
    # Selected indices must NOT contain any index from 0 to 9
    assert not any(idx < 10 for idx in batch_filtered.selected_indices)
