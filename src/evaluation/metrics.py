from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def compute_macro_f1(preds, labels):
    """
    Calculates the Macro F1 score. 
    Treats all classes equally, regardless of how many examples exist.
    """
    return f1_score(labels, preds, average='macro')

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