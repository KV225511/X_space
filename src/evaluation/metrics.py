from typing import Dict, List, Optional, Union, Any
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    matthews_corrcoef,
    log_loss,
    brier_score_loss,
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.training.calibration import compute_ece


ASTRONOMICAL_CLASSES = [
    "exoplanet_transit",      # 0
    "eclipsing_binary",       # 1
    "variable_star",          # 2
    "supernova",              # 3
    "cataclysmic_variable",   # 4
    "gamma_ray_burst",        # 5
]


def _to_numpy(x: Union[np.ndarray, torch.Tensor, List]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_accuracy(preds: Any, labels: Any) -> float:
    """Calculates overall classification accuracy."""
    p = _to_numpy(preds)
    y = _to_numpy(labels)
    return float(accuracy_score(y, p))


def compute_balanced_accuracy(preds: Any, labels: Any) -> float:
    """Calculates balanced accuracy (unweighted mean of recalls across classes)."""
    p = _to_numpy(preds)
    y = _to_numpy(labels)
    return float(balanced_accuracy_score(y, p))


def compute_precision(preds: Any, labels: Any, average: str = "macro") -> float:
    """Calculates precision (macro, weighted, or micro)."""
    p = _to_numpy(preds)
    y = _to_numpy(labels)
    return float(precision_score(y, p, average=average, zero_division=0))


def compute_per_class_precision(preds: Any, labels: Any) -> np.ndarray:
    """Returns precision array, one per class."""
    p = _to_numpy(preds)
    y = _to_numpy(labels)
    return precision_score(y, p, average=None, zero_division=0)


def compute_recall(preds: Any, labels: Any, average: str = "macro") -> float:
    """Calculates sensitivity/recall (macro, weighted, or micro)."""
    p = _to_numpy(preds)
    y = _to_numpy(labels)
    return float(recall_score(y, p, average=average, zero_division=0))


def compute_per_class_recall(preds: Any, labels: Any) -> np.ndarray:
    """Returns recall array, one per class."""
    p = _to_numpy(preds)
    y = _to_numpy(labels)
    return recall_score(y, p, average=None, zero_division=0)


def compute_macro_f1(preds: Any, labels: Any) -> float:
    """
    Calculates Macro F1 score treating all classes equally.
    """
    p = _to_numpy(preds)
    y = _to_numpy(labels)
    return float(f1_score(y, p, average="macro", zero_division=0))


def compute_weighted_f1(preds: Any, labels: Any) -> float:
    """Calculates Weighted F1 score proportional to class support."""
    p = _to_numpy(preds)
    y = _to_numpy(labels)
    return float(f1_score(y, p, average="weighted", zero_division=0))


def compute_per_class_f1(preds: Any, labels: Any) -> np.ndarray:
    """Returns an array of F1 scores, one for each specific astronomical class."""
    p = _to_numpy(preds)
    y = _to_numpy(labels)
    return f1_score(y, p, average=None, zero_division=0)


def compute_mcc(preds: Any, labels: Any) -> float:
    """Calculates Matthews Correlation Coefficient (MCC)."""
    p = _to_numpy(preds)
    y = _to_numpy(labels)
    return float(matthews_corrcoef(y, p))


def compute_specificity(preds: Any, labels: Any) -> Dict[str, float]:
    """
    Calculates macro and per-class specificity (True Negative Rate).
    TNR = TN / (TN + FP)
    """
    p = _to_numpy(preds)
    y = _to_numpy(labels)
    cm = confusion_matrix(y, p)
    num_classes = cm.shape[0]
    specificities = []

    for i in range(num_classes):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(cm[:, i]) - cm[i, i]
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0
        specificities.append(spec)

    return {
        "macro_specificity": float(np.mean(specificities)),
        "per_class_specificity": np.array(specificities, dtype=np.float32),
    }


def compute_auroc(y_scores: Any, y_true: Any, multi_class: str = "ovr") -> float:
    """
    Computes Area Under the Receiver Operating Characteristic curve (AUROC).
    Supports multi-class One-vs-Rest (OvR) or One-vs-One (OvO).
    """
    scores = _to_numpy(y_scores)
    labels = _to_numpy(y_true)

    try:
        return float(roc_auc_score(labels, scores, multi_class=multi_class, average="macro"))
    except ValueError:
        return 0.0


def compute_brier_score(probs: Any, labels: Any) -> float:
    """
    Computes multi-class Brier score (mean squared error of posterior probabilities).
    Lower is better (0.0 is perfect calibration and sharpness).
    """
    p = _to_numpy(probs)
    y = _to_numpy(labels)
    num_classes = p.shape[1]
    one_hot = np.eye(num_classes)[y]
    return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))


def compute_log_loss_score(probs: Any, labels: Any) -> float:
    """Computes cross-entropy logarithmic loss over probability distributions."""
    p = _to_numpy(probs)
    y = _to_numpy(labels)
    eps = 1e-15
    p_clipped = np.clip(p, eps, 1 - eps)
    p_normalized = p_clipped / np.sum(p_clipped, axis=1, keepdims=True)
    try:
        return float(log_loss(y, p_normalized))
    except Exception:
        return 0.0


def compute_full_metrics_suite(
    probs: Any,
    preds: Any,
    labels: Any,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes a comprehensive astronomical diagnostic metrics suite.
    Includes:
    - Accuracy, Balanced Accuracy
    - Macro/Weighted/Per-Class Precision
    - Macro/Weighted/Per-Class Recall (Sensitivity)
    - Macro/Weighted/Per-Class F1-Score
    - Specificity (True Negative Rate)
    - Matthews Correlation Coefficient (MCC)
    - AUROC (OvR Macro)
    - Expected Calibration Error (ECE)
    - Multi-Class Brier Score
    - Log Loss (Cross-Entropy)
    - Confusion Matrix
    """
    p = _to_numpy(preds)
    y = _to_numpy(labels)
    pr = _to_numpy(probs)

    if class_names is None:
        class_names = ASTRONOMICAL_CLASSES[:pr.shape[1]]

    cm = confusion_matrix(y, p)
    ece_dict = compute_ece(torch.tensor(pr, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    spec_dict = compute_specificity(p, y)

    per_class_prec = compute_per_class_precision(p, y)
    per_class_rec = compute_per_class_recall(p, y)
    per_class_f1 = compute_per_class_f1(p, y)

    per_class_breakdown = {}
    for i, cname in enumerate(class_names):
        if i < len(per_class_f1):
            per_class_breakdown[cname] = {
                "precision": float(per_class_prec[i]),
                "recall": float(per_class_rec[i]),
                "f1_score": float(per_class_f1[i]),
                "specificity": float(spec_dict["per_class_specificity"][i]),
                "support": int(np.sum(y == i)),
            }

    metrics = {
        "accuracy": compute_accuracy(p, y),
        "balanced_accuracy": compute_balanced_accuracy(p, y),
        "macro_precision": compute_precision(p, y, average="macro"),
        "weighted_precision": compute_precision(p, y, average="weighted"),
        "macro_recall": compute_recall(p, y, average="macro"),
        "weighted_recall": compute_recall(p, y, average="weighted"),
        "macro_f1": compute_macro_f1(p, y),
        "weighted_f1": compute_weighted_f1(p, y),
        "macro_specificity": spec_dict["macro_specificity"],
        "matthews_corrcoef": compute_mcc(p, y),
        "macro_auroc": compute_auroc(pr, y),
        "expected_calibration_error": float(ece_dict["ece"]),
        "brier_score": compute_brier_score(pr, y),
        "log_loss": compute_log_loss_score(pr, y),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class_breakdown,
    }
    return metrics


def plot_confusion_matrix(preds, labels, class_names, save_path: Optional[str] = None):
    """
    Creates a heatmap showing predictions vs. ground truth.
    """
    p = _to_numpy(preds)
    y = _to_numpy(labels)
    cm = confusion_matrix(y, p)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_ylabel('Ground Truth (Actual)')
    ax.set_xlabel('Predicted Event (Model Output)')
    ax.set_title('AstroFoundation: Multi-Modal Confusion Matrix')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()