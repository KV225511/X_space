"""
AstroFoundation — Evaluation, Metrics & Explainability Modules
"""

from .metrics import compute_macro_f1, compute_per_class_f1, compute_auroc, plot_confusion_matrix, compute_ece
from .explainability import (
    GradCAM1D,
    SaliencyExplainer,
    MultiModalGatingExplainer,
    Explanation1DOutput,
    CANONICAL_SPECTRAL_LINES
)

__all__ = [
    "compute_macro_f1",
    "compute_per_class_f1",
    "compute_auroc",
    "compute_ece",
    "plot_confusion_matrix",
    "GradCAM1D",
    "SaliencyExplainer",
    "MultiModalGatingExplainer",
    "Explanation1DOutput",
    "CANONICAL_SPECTRAL_LINES",
]
