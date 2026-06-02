"""Tests for data loading and synthetic dataset generation."""

from __future__ import annotations

import numpy as np
import pytest

from fraud_benchmark.data import (
    FEATURE_COLS,
    TARGET_COL,
    class_balance_report,
    load_data,
    make_synthetic_dataset,
    split_data,
)


def test_make_synthetic_dataset_shape():
    X, y = make_synthetic_dataset(n_samples=500, fraud_rate=0.04)
    assert len(X) == 500
    assert len(y) == 500
    assert list(X.columns) == FEATURE_COLS


def test_make_synthetic_dataset_fraud_rate():
    X, y = make_synthetic_dataset(n_samples=1000, fraud_rate=0.05)
    actual_rate = y.mean()
    # Allow ±1% tolerance around requested rate
    assert abs(actual_rate - 0.05) < 0.01


def test_make_synthetic_dataset_reproducible():
    X1, y1 = make_synthetic_dataset(n_samples=300, random_state=0)
    X2, y2 = make_synthetic_dataset(n_samples=300, random_state=0)
    np.testing.assert_array_equal(X1.values, X2.values)
    np.testing.assert_array_equal(y1.values, y2.values)


def test_make_synthetic_dataset_different_seeds():
    X1, _ = make_synthetic_dataset(n_samples=300, random_state=1)
    X2, _ = make_synthetic_dataset(n_samples=300, random_state=2)
    # Different seeds should produce different data
    assert not np.array_equal(X1.values, X2.values)


def test_class_balance_report():
    import pandas as pd

    y = pd.Series([0] * 98 + [1] * 2)
    report = class_balance_report(y)
    assert report["n_total"] == 100
    assert report["n_fraud"] == 2
    assert report["n_legit"] == 98
    assert report["fraud_rate_pct"] == pytest.approx(2.0)


def test_split_data_sizes():
    X, y = make_synthetic_dataset(n_samples=1000)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    assert len(X_train) == 800
    assert len(X_test) == 200


def test_split_data_stratified():
    """Train and test splits should maintain similar fraud rates."""
    X, y = make_synthetic_dataset(n_samples=2000, fraud_rate=0.05)
    _, _, y_train, y_test = split_data(X, y, test_size=0.2)
    rate_train = y_train.mean()
    rate_test = y_test.mean()
    assert abs(rate_train - rate_test) < 0.01


def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_data("/nonexistent/path/creditcard.csv")


def test_load_data_missing_columns(tmp_path):
    """load_data should raise KeyError if columns are missing."""
    import pandas as pd

    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]}).to_csv(bad_csv, index=False)
    with pytest.raises(KeyError, match="Missing columns"):
        load_data(bad_csv)
