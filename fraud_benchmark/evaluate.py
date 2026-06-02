"""Cross-validation evaluation loop for the fraud model bake-off.

Runs stratified k-fold CV for each pipeline, collecting PR-AUC and
recall-at-precision scores per fold. Optionally logs to MLflow.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold

from fraud_benchmark.metrics import pr_auc, recall_at_precision, summarize_cv_scores

logger = logging.getLogger(__name__)


def cross_validate_pipeline(
    pipeline: ImbPipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
    min_precision: float = 0.9,
) -> dict[str, Any]:
    """Run stratified k-fold CV on one pipeline and return aggregated metrics.

    SMOTE inside the pipeline is only applied to training folds; the
    held-out fold is never oversampled.

    Args:
        pipeline: An unfitted ImbPipeline from build_pipeline().
        X: Feature DataFrame.
        y: Binary label Series.
        n_splits: Number of CV folds.
        random_state: Seed for StratifiedKFold.
        min_precision: Precision floor for recall_at_precision metric.

    Returns:
        Dict with keys:
          "fold_scores": list of per-fold metric dicts
          "summary": aggregated mean/std per metric
    """
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    fold_scores: list[dict[str, float]] = []

    X_arr = X.values if hasattr(X, "values") else np.array(X)
    y_arr = y.values if hasattr(y, "values") else np.array(y)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_arr)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_val)[:, 1]

        fold_pr_auc = pr_auc(y_val.values, y_prob)
        fold_recall = recall_at_precision(
            y_val.values, y_prob, min_precision=min_precision
        )

        fold_scores.append(
            {
                "pr_auc": fold_pr_auc,
                f"recall_at_p{int(min_precision * 100)}": fold_recall,
            }
        )
        logger.info(
            "fold %d/%d  pr_auc=%.4f  recall@p%.0f=%.4f",
            fold_idx + 1,
            n_splits,
            fold_pr_auc,
            min_precision * 100,
            fold_recall,
        )

    return {
        "fold_scores": fold_scores,
        "summary": summarize_cv_scores(fold_scores),
    }


def bake_off(
    pipelines: dict[str, ImbPipeline],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
    min_precision: float = 0.9,
) -> dict[str, dict[str, Any]]:
    """Evaluate all pipelines and return results keyed by model name.

    Args:
        pipelines: Dict mapping model name → ImbPipeline (unfitted).
        X: Feature DataFrame.
        y: Binary label Series.
        n_splits: Number of CV folds.
        random_state: Seed for reproducibility.
        min_precision: Precision floor for recall metric.

    Returns:
        Dict mapping model_name → result dict from cross_validate_pipeline().
    """
    results: dict[str, dict[str, Any]] = {}
    for name, pipeline in pipelines.items():
        logger.info("evaluating %s", name)
        results[name] = cross_validate_pipeline(
            pipeline,
            X,
            y,
            n_splits=n_splits,
            random_state=random_state,
            min_precision=min_precision,
        )
    return results


def results_table(
    bake_off_results: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Convert bake_off() output to a summary DataFrame.

    Args:
        bake_off_results: Output of bake_off().

    Returns:
        DataFrame with one row per model and columns for each mean metric
        and its standard deviation.
    """
    rows = []
    for model_name, result in bake_off_results.items():
        row: dict[str, Any] = {"model": model_name}
        for metric, stats in result["summary"].items():
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")
    # Sort by PR-AUC descending
    if "pr_auc_mean" in df.columns:
        df = df.sort_values("pr_auc_mean", ascending=False)
    return df
