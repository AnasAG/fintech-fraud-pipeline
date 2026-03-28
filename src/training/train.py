"""
Model training + experiment tracking.

What this module does:
  1. Loads the merged Parquet file
  2. Time-aware train/val/test split (no random shuffle — prevents leakage)
  3. Builds the feature matrix via build_features.py
  4. Applies SMOTE to address class imbalance on training set only
  5. Trains 4 candidate models
  6. Logs every experiment to MLflow (params, metrics, artifacts)
  7. Picks the champion model by PR-AUC and saves to models/champion/

Time-aware split (critical):
  The dataset is ordered by time (TransactionDT). A random split would put
  future transactions in training and past transactions in validation — leaking
  future patterns into the model. This gives artificially optimistic metrics
  that collapse in production. We split by time: train on early data, validate
  on later data.

SMOTE on training only:
  SMOTE synthesises minority-class samples. It must ONLY be applied to training
  data. Applying it to validation/test data would give misleading metrics because
  you'd be evaluating on synthetic fraud samples, not real ones.

Run:
  python -m src.training.train
  or: make train
"""

import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SklearnPipeline
from xgboost import XGBClassifier

from src.features.build_features import build_features, save_pipeline
from src.training.evaluate import evaluate, find_optimal_threshold

load_dotenv()

PROCESSED_DIR = Path(os.getenv("PROCESSED_DATA_PATH", "data/processed"))
MODEL_DIR = Path("models")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "fintech-fraud-detection"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# Test fraction = 1 - TRAIN_FRAC - VAL_FRAC = 0.15


# ── Model definitions ─────────────────────────────────────────────────────────

def get_models() -> dict:
    """
    Four candidate models. Each chosen for a different reason:

    LightGBM:
      Industry standard for tabular fraud detection. Handles missing values
      natively (no imputation needed for tree splits), fast training, good
      with imbalanced data via scale_pos_weight.

    XGBoost:
      Direct comparison to LightGBM. Usually slightly worse on large datasets
      but useful to quantify the difference.

    Logistic Regression:
      Interpretability baseline. If LightGBM wins by a large margin, you can
      articulate *why* a non-linear model adds value over the linear baseline.
      Also useful for explaining individual predictions (SHAP values are less
      intuitive for trees).

    Calibrated LightGBM:
      Raw LightGBM probabilities can be over-confident (spiky near 0 and 1).
      Calibration (Platt scaling or isotonic regression) makes probabilities
      more reliable for threshold-based decisioning. Important when the business
      makes decisions at 0.3 and 0.7 — if those thresholds are poorly calibrated,
      you're blocking/approving incorrectly.
    """
    return {
        "lightgbm": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=30,  # ~1/fraud_rate approximation
            eval_metric="aucpr",
            random_state=42,
            verbosity=0,
        ),
        "logistic_regression": SklearnPipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=0.1,
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            )),
        ]),
        "calibrated_lightgbm": CalibratedClassifierCV(
            LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                class_weight="balanced",
                random_state=42,
                verbose=-1,
            ),
            method="isotonic",
            cv=3,
        ),
    }


# ── Training pipeline ─────────────────────────────────────────────────────────

def time_aware_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by TransactionDT (time order). No shuffling.

    This is the correct split for any time-series or sequential data.
    Random splitting leaks future patterns into training.
    """
    df_sorted = df.sort_values("TransactionDT").reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))

    train = df_sorted.iloc[:train_end]
    val = df_sorted.iloc[train_end:val_end]
    test = df_sorted.iloc[val_end:]

    logger.info(f"Split: train={len(train):,}  val={len(val):,}  test={len(test):,}")
    logger.info(f"  Train fraud rate: {train['isFraud'].mean():.2%}")
    logger.info(f"  Val fraud rate:   {val['isFraud'].mean():.2%}")
    logger.info(f"  Test fraud rate:  {test['isFraud'].mean():.2%}")

    return train, val, test


def train_all_models() -> None:
    """Main training entrypoint."""
    # ── Setup MLflow ──────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"MLflow tracking: {MLFLOW_URI}")

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading merged.parquet ...")
    df = pd.read_parquet(PROCESSED_DIR / "merged.parquet")

    # ── Split (time-aware) ────────────────────────────────────────────────────
    train_df, val_df, test_df = time_aware_split(df)

    # ── Feature engineering (fit on train only) ───────────────────────────────
    logger.info("Building features (training set) ...")
    X_train, y_train, encoders = build_features(train_df, fit=True)
    save_pipeline(encoders)

    logger.info("Building features (val + test sets using fitted encoders) ...")
    X_val, y_val, _ = build_features(val_df, fit=False, encoders=encoders)
    X_test, y_test, _ = build_features(test_df, fit=False, encoders=encoders)

    # ── Handle remaining NaNs (after feature engineering) ────────────────────
    X_train = X_train.fillna(-999)
    X_val = X_val.fillna(-999)
    X_test = X_test.fillna(-999)

    # ── SMOTE on training set only ─────────────────────────────────────────────
    logger.info("Applying SMOTE to training set ...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE: {y_train_resampled.sum():,} fraud / {(y_train_resampled == 0).sum():,} legit")

    # ── Train each model ──────────────────────────────────────────────────────
    models = get_models()
    results = {}

    for name, model in models.items():
        logger.info(f"\nTraining {name} ...")

        with mlflow.start_run(run_name=name):
            # Log model type and key params
            mlflow.log_param("model_type", name)
            mlflow.log_param("train_size", len(X_train_resampled))
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("smote_applied", True)

            # Train
            model.fit(X_train_resampled, y_train_resampled)

            # Evaluate on validation set
            y_val_prob = model.predict_proba(X_val)[:, 1]
            val_metrics = evaluate(y_val, y_val_prob, model_name=f"{name}/val")

            # Evaluate on test set
            y_test_prob = model.predict_proba(X_test)[:, 1]
            test_metrics = evaluate(y_test, y_test_prob, model_name=f"{name}/test")

            # Log all metrics to MLflow
            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

            # Log the model artifact
            mlflow.sklearn.log_model(model, artifact_path="model")

            results[name] = {
                "model": model,
                "val_pr_auc": val_metrics["pr_auc"],
                "test_pr_auc": test_metrics["pr_auc"],
                "run_id": mlflow.active_run().info.run_id,
            }

            logger.info(f"{name}: val PR-AUC={val_metrics['pr_auc']:.4f}  test PR-AUC={test_metrics['pr_auc']:.4f}")

    # ── Select champion model ─────────────────────────────────────────────────
    champion_name = max(results, key=lambda k: results[k]["val_pr_auc"])
    champion = results[champion_name]
    logger.success(f"\nChampion model: {champion_name}  (val PR-AUC={champion['val_pr_auc']:.4f})")

    # ── Save champion ─────────────────────────────────────────────────────────
    champion_dir = MODEL_DIR / "champion"
    champion_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(champion["model"], champion_dir / "model.pkl")
    joblib.dump(encoders, champion_dir / "encoders.pkl")

    # Write a simple manifest so the API knows which model is loaded
    import json
    manifest = {
        "model_name": champion_name,
        "mlflow_run_id": champion["run_id"],
        "val_pr_auc": champion["val_pr_auc"],
        "test_pr_auc": champion["test_pr_auc"],
        "feature_columns": list(X_train.columns),
    }
    with open(champion_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.success(f"Champion saved → {champion_dir}/")
    logger.info(f"MLflow UI: {MLFLOW_URI}")


if __name__ == "__main__":
    train_all_models()
