from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, confusion_matrix
)

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_map: Optional[Dict[int, str]] = None
) -> Dict[str, Any]:
    """
    Compute full suite of classification metrics.
    Trained model outputs multiclass predictions (0=Benign, >0=Attacks).
    Binary metrics are derived by collapsing all attack classes (>0) into 1.
    """
    if len(y_true) == 0:
        return {}

    acc_multiclass = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    per_class_rec = recall_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    per_class_recall = {int(c): float(r) for c, r in zip(classes, per_class_rec)}

    y_true_bin = (y_true > 0).astype(np.int64)
    y_pred_bin = (y_pred > 0).astype(np.int64)

    bin_acc = float(accuracy_score(y_true_bin, y_pred_bin))
    bin_precision = float(precision_score(y_true_bin, y_pred_bin, zero_division=0))
    bin_recall = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))
    bin_f1 = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))
    bin_balanced_acc = float(balanced_accuracy_score(y_true_bin, y_pred_bin))
    
    try:
        bin_mcc = float(matthews_corrcoef(y_true_bin, y_pred_bin))
    except Exception:
        bin_mcc = 0.0

    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    
    # HIGH-1 FIX: Return None when denominator is zero rather than misleading 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else None
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else None
    benign_recall = float(tn / (tn + fp)) if (tn + fp) > 0 else None
    attack_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else None

    return {
        "accuracy": bin_acc,
        "precision": bin_precision,
        "recall": bin_recall,
        "f1": bin_f1,
        "balanced_accuracy": bin_balanced_acc,
        "mcc": bin_mcc,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "benign_recall": benign_recall,
        "attack_recall": attack_recall,
        "multiclass": {
            "accuracy": acc_multiclass,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "per_class_recall": per_class_recall
        }
    }
