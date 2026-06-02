#!/usr/bin/env python3
"""Train and evaluate all fraud-detection baselines.

Usage:
    python train.py --data creditcard.csv
    python train.py --data creditcard.csv --folds 5 --seed 42 --mlflow

The script:
  1. Loads the creditcard.csv dataset.
  2. Runs stratified k-fold cross-validation for LogisticRegression,
     RandomForest, and GradientBoosting — each in a pipeline with
     SMOTE oversampling inside the fold.
  3. Prints a results table (PR-AUC and recall at 90% precision).
  4. Optionally logs results to an MLflow experiment.
"""

from __future__ import annotations

import argparse
import logging
import sys

import mlflow

from fraud_benchmark.data import class_balance_report, load_data, split_data
from fraud_benchmark.evaluate import bake_off, results_table
from fraud_benchmark.pipeline import all_pipelines

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Credit card fraud baseline bake-off",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        required=True,
        metavar="PATH",
        help="Path to creditcard.csv",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        metavar="N",
        help="Number of stratified CV folds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="SEED",
        help="Random seed for SMOTE, CV splits, and classifiers",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.9,
        metavar="P",
        help="Precision floor for recall@precision metric",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Log results to MLflow (experiment: fraud-baselines)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on success."""
    args = parse_args(argv)

    # --- Data -----------------------------------------------------------------
    logger.info("loading %s", args.data)
    X, y = load_data(args.data)
    balance = class_balance_report(y)
    logger.info(
        "dataset: %d transactions | %d fraud (%.4f%%)",
        balance["n_total"],
        balance["n_fraud"],
        balance["fraud_rate_pct"],
    )

    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=0.2, random_state=args.seed
    )
    logger.info(
        "train=%d  test=%d  fraud_in_test=%d",
        len(y_train),
        len(y_test),
        int(y_test.sum()),
    )

    # --- Bake-off -------------------------------------------------------------
    pipelines = all_pipelines(random_state=args.seed)

    logger.info(
        "running %d-fold CV over %d models", args.folds, len(pipelines)
    )
    results = bake_off(
        pipelines,
        X_train,
        y_train,
        n_splits=args.folds,
        random_state=args.seed,
        min_precision=args.min_precision,
    )

    table = results_table(results)
    print("\n=== Bake-off results (CV on train split) ===")
    print(table.to_string())
    print()

    # --- MLflow ---------------------------------------------------------------
    if args.mlflow:
        mlflow.set_experiment("fraud-baselines")
        for model_name, result in results.items():
            with mlflow.start_run(run_name=model_name):
                mlflow.log_params(
                    {
                        "model": model_name,
                        "n_folds": args.folds,
                        "seed": args.seed,
                        "min_precision": args.min_precision,
                    }
                )
                for metric, stats in result["summary"].items():
                    mlflow.log_metric(f"{metric}_mean", stats["mean"])
                    mlflow.log_metric(f"{metric}_std", stats["std"])
        logger.info("results logged to MLflow experiment 'fraud-baselines'")

    return 0


if __name__ == "__main__":
    sys.exit(main())
