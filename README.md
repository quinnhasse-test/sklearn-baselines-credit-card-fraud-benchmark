# sklearn-baselines: credit card fraud benchmark

Baseline comparison of LogisticRegression, RandomForest, and GradientBoosting on the ULB credit card fraud dataset. Stratified k-fold CV with SMOTE inside each fold, evaluated on PR-AUC and recall at 90% precision.

## Results

| model | PR-AUC (mean ± std) | recall@p90 (mean ± std) |
|---|---|---|
| gradient_boosting | — | — |
| random_forest | — | — |
| logistic_regression | — | — |

_Numbers populated after running `train.py` with `--mlflow`. See the [MLflow run table](#mlflow-tracking) below._

## Dataset

ULB Credit Card Fraud Detection (284,807 transactions, 492 fraud cases — 0.17% positive rate).

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root.

Columns: `Time`, `Amount`, `V1`–`V28` (PCA-transformed), `Class` (0 = legitimate, 1 = fraud).

## Install

```bash
pip install -r requirements.txt
```

## Run the bake-off

```bash
python train.py --data creditcard.csv --folds 5 --seed 42 --mlflow
```

Flags:

| flag | default | description |
|---|---|---|
| `--data` | required | path to creditcard.csv |
| `--folds` | 5 | number of CV folds |
| `--seed` | 42 | random seed for SMOTE, CV, classifiers |
| `--min-precision` | 0.9 | precision floor for recall@p metric |
| `--mlflow` | off | log to MLflow experiment `fraud-baselines` |

## MLflow tracking

```bash
mlflow ui
```

Opens at `http://localhost:5000`. All three models appear under the `fraud-baselines` experiment.

## Streamlit demo

```bash
streamlit run app.py
```

Scores a single transaction as JSON input. Live at: _(URL added after deploy)_

## Tests

```bash
pytest tests/ -v
```

No Kaggle download required — tests use a synthetic dataset generated in `fraud_benchmark/data.py`.

## Structure

```
fraud_benchmark/
  data.py        load_data(), split_data(), synthetic generator for tests
  pipeline.py    build_pipeline() per model, SMOTE inside imblearn Pipeline
  metrics.py     pr_auc(), recall_at_precision(), threshold_metrics()
  evaluate.py    cross_validate_pipeline(), bake_off(), results_table()
train.py         CLI entrypoint
app.py           Streamlit demo (added in later commit)
tests/           pytest suite
```

## Design notes

- SMOTE runs inside the imblearn `Pipeline` so it only touches training folds — held-out folds are never oversampled.
- `StandardScaler` is applied only to `Time` and `Amount`; the PCA features `V1`–`V28` are already normalized.
- All classifiers use `class_weight="balanced"` or `balanced_subsample` in addition to SMOTE — double defense against the imbalance.
- PR-AUC is the primary metric. At 0.17% fraud rate, ROC-AUC inflates due to the large true-negative mass; precision-recall curves show the real operating tradeoff.
- Reproducible via `--seed 42`.
