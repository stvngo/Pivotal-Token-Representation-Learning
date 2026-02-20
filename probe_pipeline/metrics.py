"""Metric computation utilities for binary probes."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute confusion matrix, F1, AUROC, and related stats."""
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = np.asarray(y_prob).astype(float).reshape(-1)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fp_fn_ratio = float(fp / fn) if fn > 0 else (float("inf") if fp > 0 else 0.0)
    auroc = float(roc_auc_score(y_true, y_prob)) if np.unique(y_true).size > 1 else None

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc_score": auroc,
        "threshold": float(threshold),
        "confusion_matrix": cm.tolist(),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "fp_fn_ratio": fp_fn_ratio,
    }

