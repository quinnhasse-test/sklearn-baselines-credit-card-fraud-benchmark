"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from fraud_benchmark.metrics import (
    pr_auc,
    recall_at_precision,
    roc_auc,
    summarize_cv_scores,
    threshold_metrics,
)


@pytest.fixture
def perfect_scores():
    y_true = np.array([0, 0, 0, 1, 1])
    y_prob = np.array([0.01, 0.02, 0.03, 0.95, 0.97])
    return y_true, y_prob


@pytest.fixture
def random_scores():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    y_prob = rng.uniform(0, 1, size=200)
    return y_true, y_prob


def test_pr_auc_perfect(perfect_scores):
    y_true, y_prob = perfect_scores
    assert pr_auc(y_true, y_prob) == pytest.approx(1.0, abs=1e-6)


def test_pr_auc_range(random_scores):
    y_true, y_prob = random_scores
    score = pr_auc(y_true, y_prob)
    assert 0.0 <= score <= 1.0


def test_roc_auc_perfect(perfect_scores):
    y_true, y_prob = perfect_scores
    assert roc_auc(y_true, y_prob) == pytest.approx(1.0, abs=1e-6)


def test_recall_at_precision_perfect(perfect_scores):
    y_true, y_prob = perfect_scores
    # At 90% precision, a perfect classifier should get recall = 1.0
    r = recall_at_precision(y_true, y_prob, min_precision=0.9)
    assert r == pytest.approx(1.0, abs=1e-6)


def test_recall_at_precision_impossible_target():
    """If no threshold reaches the target precision, return 0.0."""
    # All predictions are the same low probability — precision will be low
    y_true = np.array([0] * 90 + [1] * 10)
    y_prob = np.ones(100) * 0.5  # constant output, precision = base rate
    r = recall_at_precision(y_true, y_prob, min_precision=0.99)
    assert r == 0.0


def test_recall_at_precision_returns_float(random_scores):
    y_true, y_prob = random_scores
    r = recall_at_precision(y_true, y_prob, min_precision=0.5)
    assert isinstance(r, float)
    assert 0.0 <= r <= 1.0


def test_threshold_metrics_all_correct():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    m = threshold_metrics(y_true, y_prob, threshold=0.5)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)


def test_threshold_metrics_all_wrong():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    m = threshold_metrics(y_true, y_prob, threshold=0.5)
    assert m["precision"] == pytest.approx(0.0)
    assert m["recall"] == pytest.approx(0.0)


def test_threshold_metrics_support():
    y_true = np.array([0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8])
    m = threshold_metrics(y_true, y_prob)
    assert m["support_fraud"] == 2


def test_summarize_cv_scores_mean_std():
    scores = [
        {"pr_auc": 0.80, "recall_at_p90": 0.50},
        {"pr_auc": 0.90, "recall_at_p90": 0.60},
        {"pr_auc": 0.85, "recall_at_p90": 0.55},
    ]
    summary = summarize_cv_scores(scores)
    assert summary["pr_auc"]["mean"] == pytest.approx(0.85, abs=1e-4)
    assert summary["pr_auc"]["std"] == pytest.approx(
        np.std([0.80, 0.90, 0.85]), abs=1e-4
    )


def test_summarize_cv_scores_empty():
    assert summarize_cv_scores([]) == {}
