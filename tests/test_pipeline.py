"""Tests for pipeline construction and determinism."""

from __future__ import annotations

import numpy as np
import pytest

from fraud_benchmark.data import make_synthetic_dataset, split_data
from fraud_benchmark.pipeline import (
    _MODEL_REGISTRY,
    all_pipelines,
    build_pipeline,
    build_preprocessor,
)


@pytest.fixture(scope="module")
def small_dataset():
    """Synthetic dataset for pipeline tests."""
    return make_synthetic_dataset(n_samples=800, fraud_rate=0.05, random_state=0)


def test_build_preprocessor_output_shape(small_dataset):
    """Preprocessor should output 30 features (2 scaled + 28 passthrough)."""
    X, _ = small_dataset
    prep = build_preprocessor()
    X_out = prep.fit_transform(X)
    assert X_out.shape == (len(X), 30)


def test_build_pipeline_known_models():
    """build_pipeline should succeed for every registered model name."""
    for name in _MODEL_REGISTRY:
        pipeline = build_pipeline(name, random_state=99)
        assert pipeline is not None
        assert hasattr(pipeline, "fit")
        assert hasattr(pipeline, "predict_proba")


def test_build_pipeline_unknown_model():
    """build_pipeline should raise ValueError for unknown model names."""
    with pytest.raises(ValueError, match="Unknown model"):
        build_pipeline("svm_not_registered")


def test_pipeline_fit_predict(small_dataset):
    """Pipeline should fit and produce probabilities in [0, 1]."""
    X, y = small_dataset
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    pipeline = build_pipeline("logistic_regression", random_state=42)
    pipeline.fit(X_train, y_train)
    probs = pipeline.predict_proba(X_test)
    assert probs.shape == (len(X_test), 2)
    assert (probs >= 0).all() and (probs <= 1).all()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_pipeline_determinism(small_dataset):
    """Same seed should produce identical predicted probabilities.

    Uses n_jobs=1 to avoid parallel tree aggregation causing floating-point
    reordering at machine-epsilon scale.
    """
    X, y = small_dataset
    X_train, X_test, y_train, _ = split_data(X, y, test_size=0.2)

    p1 = build_pipeline("random_forest", random_state=7, n_jobs=1)
    p2 = build_pipeline("random_forest", random_state=7, n_jobs=1)
    p1.fit(X_train, y_train)
    p2.fit(X_train, y_train)

    probs1 = p1.predict_proba(X_test)[:, 1]
    probs2 = p2.predict_proba(X_test)[:, 1]
    np.testing.assert_array_equal(probs1, probs2)


def test_all_pipelines_returns_all_models():
    """all_pipelines() should return one entry per registered model."""
    pipelines = all_pipelines(random_state=0)
    assert set(pipelines.keys()) == set(_MODEL_REGISTRY.keys())


def test_smote_not_applied_to_test_data(small_dataset):
    """After fit, predict_proba should return exactly len(X_test) rows."""
    X, y = small_dataset
    X_train, X_test, y_train, _ = split_data(X, y, test_size=0.3)
    pipeline = build_pipeline("logistic_regression", random_state=0)
    pipeline.fit(X_train, y_train)
    probs = pipeline.predict_proba(X_test)
    # If SMOTE leaked into predict, shape would be wrong
    assert probs.shape[0] == len(X_test)
