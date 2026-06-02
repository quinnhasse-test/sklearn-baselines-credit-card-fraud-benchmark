"""Data loading and splitting for the credit card fraud dataset.

The dataset is the ULB Credit Card Fraud dataset (284,807 transactions,
492 fraud cases). Download from:
  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Expected CSV columns: Time, V1–V28 (PCA features), Amount, Class
  Class: 0 = legitimate, 1 = fraud
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


# PCA-transformed features from the original dataset
V_FEATURES = [f"V{i}" for i in range(1, 29)]
# Features that need scaling (raw values, not PCA-normalized)
RAW_FEATURES = ["Time", "Amount"]
FEATURE_COLS = RAW_FEATURES + V_FEATURES
TARGET_COL = "Class"


def load_data(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the creditcard CSV and return (X, y).

    Args:
        path: Path to creditcard.csv.

    Returns:
        X: DataFrame with feature columns (Time, Amount, V1–V28).
        y: Series with binary labels (0 = legit, 1 = fraud).

    Raises:
        FileNotFoundError: If the CSV file does not exist at the given path.
        KeyError: If the expected columns are missing from the CSV.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Download creditcard.csv from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
        )

    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in dataset: {missing}")

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split preserving fraud class ratio.

    Args:
        X: Feature DataFrame.
        y: Target series.
        test_size: Fraction of data to hold out for final evaluation.
        random_state: Random seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def class_balance_report(y: pd.Series) -> dict[str, float]:
    """Return fraud rate and counts for a label series.

    Args:
        y: Binary label series (0/1).

    Returns:
        Dict with keys: n_total, n_fraud, n_legit, fraud_rate_pct.
    """
    n_total = len(y)
    n_fraud = int(y.sum())
    n_legit = n_total - n_fraud
    fraud_rate = 100.0 * n_fraud / n_total
    return {
        "n_total": n_total,
        "n_fraud": n_fraud,
        "n_legit": n_legit,
        "fraud_rate_pct": round(fraud_rate, 4),
    }


def make_synthetic_dataset(
    n_samples: int = 5000,
    fraud_rate: float = 0.02,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate a synthetic dataset matching the creditcard CSV schema.

    Used for unit tests and CI — does not require the Kaggle download.

    Args:
        n_samples: Total number of transactions to generate.
        fraud_rate: Fraction of transactions that are fraudulent.
        random_state: Random seed.

    Returns:
        X: DataFrame with Time, Amount, V1–V28 columns.
        y: Binary label Series.
    """
    rng = np.random.default_rng(random_state)
    n_fraud = max(1, int(n_samples * fraud_rate))
    n_legit = n_samples - n_fraud

    # Legit transactions: low amounts, PCA features near zero
    legit_amount = rng.exponential(scale=50.0, size=n_legit)
    fraud_amount = rng.exponential(scale=200.0, size=n_fraud)

    legit_v = rng.normal(0.0, 1.0, size=(n_legit, 28))
    # Fraud transactions have a shifted distribution on some components
    fraud_v = rng.normal(0.0, 1.0, size=(n_fraud, 28))
    fraud_v[:, :5] += rng.normal(2.0, 0.5, size=(n_fraud, 5))

    legit_time = rng.uniform(0, 172800, size=n_legit)
    fraud_time = rng.uniform(0, 172800, size=n_fraud)

    legit_rows = pd.DataFrame(
        np.column_stack([legit_time, legit_amount, legit_v]),
        columns=FEATURE_COLS,
    )
    fraud_rows = pd.DataFrame(
        np.column_stack([fraud_time, fraud_amount, fraud_v]),
        columns=FEATURE_COLS,
    )

    X = pd.concat([legit_rows, fraud_rows], ignore_index=True)
    y = pd.Series(
        [0] * n_legit + [1] * n_fraud, name=TARGET_COL, dtype=int
    )

    # Shuffle
    idx = rng.permutation(len(X))
    X = X.iloc[idx].reset_index(drop=True)
    y = y.iloc[idx].reset_index(drop=True)

    return X, y
