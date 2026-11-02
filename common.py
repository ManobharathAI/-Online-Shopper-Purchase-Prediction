"""Shared helpers: metric computation used by both training and the Streamlit app."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

METRIC_COLUMNS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def compute_metrics(y_true, y_pred, y_proba):
    """Return a dict with the six required metrics.

    Binary problems use the positive class for precision/recall/F1/AUC.
    Multi-class problems use weighted averaging and one-vs-rest AUC.
    """
    n_classes = len(np.unique(y_true))
    if n_classes == 2:
        avg = "binary"
        auc = roc_auc_score(y_true, y_proba[:, 1])
    else:
        avg = "weighted"
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
