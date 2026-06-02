"""Evaluation metrics for imbalanced fraud detection.

Accuracy is misleading at 0.17% fraud rate — this module focuses on
precision-recall metrics that matter operationally.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Area under the precision-recall curve (average precision).

    Equal to the weighted mean of precisions at each threshold,
    using the increase in recall as weight. Preferred over ROC-AUC
    for heavily imbalanced data.

    Args:
        y_true: Binary ground-truth labels (0/1).
        y_prob: Predicted probabilities for the positive class.

    Returns:
        PR-AUC score in [0, 1]. A random classifier scores ~fraud_rate.
    """
    return float(average_precision_score(y_true, y_prob))


def recall_at_precision(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float = 0.9,
) -> float:
    """Maximum recall achievable at or above a precision threshold.

    Finds the highest recall value across all decision thresholds
    where precision >= min_precision. Returns 0.0 if no threshold
    achieves the target precision.

    Args:
        y_true: Binary ground-truth labels (0/1).
        y_prob: Predicted probabilities for the positive class.
        min_precision: Minimum precision required (default 0.90).

    Returns:
        Recall value in [0, 1].
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns arrays in decreasing-threshold order;
    # mask to thresholds where precision meets the bar
    mask = precision >= min_precision
    if not mask.any():
        return 0.0
    return float(recall[mask].max())


def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """ROC-AUC score.

    Included for completeness; PR-AUC is the primary metric for this
    imbalanced dataset.

    Args:
        y_true: Binary ground-truth labels.
        y_prob: Predicted probabilities for the positive class.

    Returns:
        ROC-AUC score in [0, 1].
    """
    return float(roc_auc_score(y_true, y_prob))


def threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Precision, recall, and F1 at a fixed decision threshold.

    Args:
        y_true: Binary ground-truth labels.
        y_prob: Predicted probabilities for the positive class.
        threshold: Decision boundary (default 0.5).

    Returns:
        Dict with keys: threshold, precision, recall, f1, support_fraud.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "threshold": threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "support_fraud": int(y_true.sum()),
    }


def summarize_cv_scores(
    scores: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Aggregate per-fold metric dicts into mean ± std.

    Args:
        scores: List of metric dicts, one per CV fold.

    Returns:
        Dict mapping metric_name → {"mean": ..., "std": ...}.
    """
    if not scores:
        return {}

    keys = list(scores[0].keys())
    result: dict[str, dict[str, float]] = {}
    for key in keys:
        vals = np.array([s[key] for s in scores], dtype=float)
        result[key] = {
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std()), 4),
        }
    return result
