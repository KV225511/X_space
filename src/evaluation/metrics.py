from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

from src.training.calibration import compute_ece

def compute_macro_f1(preds, labels):
    """
    Calculates the Macro F1 score. 
    Treats all classes equally, regardless of how many examples exist.
    """
    return f1_score(labels, preds, average='macro')


def compute_auroc(y_scores, y_true, multi_class: str = "ovr") -> float:
    """
    Computes Area Under the Receiver Operating Characteristic curve (AUROC).
    Supports multi-class One-vs-Rest (OvR) or One-vs-One (OvO).
    """
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    try:
        return float(roc_auc_score(y_true, y_scores, multi_class=multi_class, average="macro"))
    except ValueError:
        # Fallback if some classes not present in test split
        return 0.0

def compute_per_class_f1(preds, labels):
    """
    Returns an array of F1 scores, one for each specific astronomical event.
    """
    return f1_score(labels, preds, average=None)

def plot_confusion_matrix(preds, labels, class_names):
    """
    Creates a heatmap showing exactly what the model gets right and wrong.
    """
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual Event (Truth)')
    plt.xlabel('Predicted Event (Model Guess)')
    plt.title('X_space: Confusion Matrix')
    plt.show()