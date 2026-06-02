"""sklearn Pipeline factories for the fraud benchmark.

Each pipeline wraps:
  1. A ColumnTransformer that scales Time/Amount (StandardScaler) and
     passes V1–V28 through unchanged (already PCA-normalized).
  2. SMOTE for oversampling the minority class — applied only within
     training folds to prevent data leakage.
  3. A classifier.

Uses imbalanced-learn's Pipeline so that SMOTE is excluded during
predict/transform on held-out data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from fraud_benchmark.data import RAW_FEATURES, V_FEATURES

# Supported model names and their default constructors
_MODEL_REGISTRY: dict[str, type] = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}

# Default hyperparameters per model (tuned for class imbalance)
_MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    "logistic_regression": {
        "max_iter": 1000,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "C": 0.1,
    },
    "random_forest": {
        "n_estimators": 200,
        "class_weight": "balanced_subsample",
        "max_depth": 12,
        "min_samples_leaf": 4,
        "n_jobs": -1,
    },
    "gradient_boosting": {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "min_samples_leaf": 4,
    },
}


def build_preprocessor() -> ColumnTransformer:
    """Build a ColumnTransformer for the creditcard feature schema.

    Scales Time and Amount with StandardScaler; passes V1–V28 through
    unchanged because they are already PCA-transformed.

    Returns:
        A fitted-ready ColumnTransformer.
    """
    return ColumnTransformer(
        transformers=[
            ("scale_raw", StandardScaler(), RAW_FEATURES),
            ("passthrough_pca", "passthrough", V_FEATURES),
        ],
        remainder="drop",
    )


def build_pipeline(
    model_name: str,
    random_state: int = 42,
    smote_k_neighbors: int = 5,
    calibrate: bool = False,
    **model_kwargs: Any,
) -> ImbPipeline:
    """Build an imbalanced-learn Pipeline for one classifier.

    Pipeline steps:
      preprocessor → smote → classifier

    SMOTE is part of the pipeline so it only sees training data during
    cross-validation — it is skipped on predict calls automatically.

    Args:
        model_name: One of "logistic_regression", "random_forest",
            "gradient_boosting".
        random_state: Seed passed to both SMOTE and the classifier.
        smote_k_neighbors: Number of nearest neighbors for SMOTE.
        calibrate: If True, wrap the classifier in
            CalibratedClassifierCV (sigmoid) for better probability
            estimates. Disabled by default because it adds compute cost
            and all three baseline classifiers already support
            predict_proba.
        **model_kwargs: Override default hyperparameters for the model.

    Returns:
        An unfitted ImbPipeline ready for fit/predict.

    Raises:
        ValueError: If model_name is not in the registry.
    """
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {sorted(_MODEL_REGISTRY)}"
        )

    params = {**_MODEL_DEFAULTS[model_name], **model_kwargs}

    # Inject random_state where the model accepts it
    if "random_state" not in params:
        params["random_state"] = random_state

    clf = _MODEL_REGISTRY[model_name](**params)

    if calibrate:
        clf = CalibratedClassifierCV(clf, method="sigmoid", cv=3)

    preprocessor = build_preprocessor()

    smote = SMOTE(
        k_neighbors=smote_k_neighbors,
        random_state=random_state,
    )

    return ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", smote),
            ("classifier", clf),
        ]
    )


def all_pipelines(
    random_state: int = 42,
) -> dict[str, ImbPipeline]:
    """Return one pipeline per model name, all sharing the same random seed.

    Args:
        random_state: Seed for SMOTE and all classifiers.

    Returns:
        Dict mapping model name → unfitted ImbPipeline.
    """
    return {
        name: build_pipeline(name, random_state=random_state)
        for name in _MODEL_REGISTRY
    }


def feature_names_out(preprocessor: ColumnTransformer) -> list[str]:
    """Extract the output feature names from a fitted ColumnTransformer.

    The order matches the column order produced by preprocessor.transform().

    Args:
        preprocessor: A fitted ColumnTransformer built by build_preprocessor().

    Returns:
        List of feature name strings.
    """
    scaled_names = list(
        preprocessor.named_transformers_["scale_raw"].get_feature_names_out(
            RAW_FEATURES
        )
    )
    return scaled_names + V_FEATURES
