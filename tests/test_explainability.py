"""
Unit tests for AstroFoundation Explainability & Metrics:
- 1D Grad-CAM (GradCAM1D)
- 2D FITS Grad-CAM (GradCAM2D)
- SmoothGrad Input Saliency (SaliencyExplainer)
- SHAP (ShapleyExplainer)
- MultiModal Gating Explainer
- AUROC Metric
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from src.models.spectral_encoder import SpectralEncoder
from src.models.transformer_encoder import LightCurveEncoder
from src.models.fusion import CrossAttentionFusion
from src.evaluation.explainability import (
    GradCAM1D,
    GradCAM2D,
    SaliencyExplainer,
    ShapleyExplainer,
    MultiModalGatingExplainer,
    Explanation1DOutput,
    CANONICAL_SPECTRAL_LINES
)
from src.evaluation.metrics import compute_auroc


class SpectralClassifier(nn.Module):
    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.encoder = SpectralEncoder()
        self.head = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)
        return self.head(emb)


class LightCurveClassifier(nn.Module):
    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.encoder = LightCurveEncoder()
        self.head = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(x)
        return self.head(emb)


class Conv2DImageClassifier(nn.Module):
    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv(x))
        h = self.pool(h).flatten(1)
        return self.fc(h)


def test_gradcam1d_on_spectral_encoder():
    torch.manual_seed(42)
    model = SpectralClassifier(num_classes=6)
    target_layer = model.encoder.conv_stack[2]

    cam = GradCAM1D(model, target_layer)

    x = torch.randn(1, 4096)
    wave_grid = np.linspace(3800.0, 9200.0, 4096)
    explanation: Explanation1DOutput = cam.generate(x, wavelength_grid=wave_grid)

    assert isinstance(explanation, Explanation1DOutput)
    assert explanation.saliency_map.shape == (4096,)
    assert explanation.saliency_map.min() >= 0.0
    assert explanation.saliency_map.max() <= 1.0
    assert 0 <= explanation.target_class < 6
    assert 0.0 <= explanation.confidence <= 1.0

    cam.remove_hooks()
    assert len(cam._handles) == 0


def test_gradcam2d_on_image_backbone():
    torch.manual_seed(42)
    model = Conv2DImageClassifier(num_classes=6)
    cam = GradCAM2D(model, model.conv)

    x = torch.randn(1, 3, 32, 32)
    heatmap = cam.generate(x, target_class=1)

    assert heatmap.shape == (32, 32)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0
    cam.remove_hooks()
    assert len(cam._handles) == 0


def test_saliency_explainer_on_light_curve():
    torch.manual_seed(42)
    model = LightCurveClassifier(num_classes=6)

    explainer = SaliencyExplainer(model)
    x = torch.randn(1, 8192)

    saliency = explainer.explain(x, num_smoothgrad_samples=3)

    assert saliency.shape == (8192,)
    assert saliency.min() >= 0.0
    assert saliency.max() <= 1.0
    assert not np.isnan(saliency).any()




def test_multimodal_gating_explainer():
    fusion = CrossAttentionFusion(embed_dim=256, num_heads=8)

    e_lc = torch.randn(1, 256)
    e_img = torch.randn(1, 256)
    e_spec = torch.randn(1, 256)

    contributions = MultiModalGatingExplainer.explain_contributions(
        fusion, (e_lc, e_img, e_spec)
    )

    assert "light_curve_pct" in contributions
    assert "fits_image_pct" in contributions
    assert "spectra_pct" in contributions

    total_pct = (
        contributions["light_curve_pct"] +
        contributions["fits_image_pct"] +
        contributions["spectra_pct"]
    )
    assert pytest.approx(total_pct, rel=1e-2) == 100.0


def test_compute_auroc():
    n = 100
    y_true = np.random.randint(0, 3, size=n)
    y_scores = np.zeros((n, 3))
    for i in range(n):
        y_scores[i, y_true[i]] = 2.0
    y_scores = np.exp(y_scores) / np.sum(np.exp(y_scores), axis=-1, keepdims=True)

    score = compute_auroc(y_scores, y_true)
    assert 0.5 <= score <= 1.0
