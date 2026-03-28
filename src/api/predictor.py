"""
Model loading and prediction logic — separated from the FastAPI route handlers.

Separation of concerns:
  main.py   → HTTP routing, request/response handling
  predictor.py → model loading, feature preparation, scoring logic

This separation makes the predictor testable without spinning up a FastAPI server.
You can unit test predict() directly with a dict input.

Key design decisions:
  1. Model + pipeline loaded ONCE at startup (not per request)
     Loading a 50MB LightGBM model takes ~500ms. Doing this per request would
     make every prediction slow. We load once in __init__ and reuse.

  2. Feature pipeline runs at inference time
     The same encoders fitted at training time are applied here. This is what
     prevents training/serving skew — the exact same code path runs in both places.

  3. Missing feature columns are filled with -999 (LightGBM default for 'missing')
     This is consistent with training. Do NOT change this without retraining.
"""

import json
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger


MODEL_DIR = Path(os.getenv("MODEL_PATH", "models/champion"))
THRESHOLD_APPROVE = float(os.getenv("THRESHOLD_APPROVE", "0.3"))
THRESHOLD_DECLINE = float(os.getenv("THRESHOLD_DECLINE", "0.7"))


class FraudPredictor:
    """
    Wraps the champion model + feature pipeline for single-transaction scoring.

    Usage:
        predictor = FraudPredictor()
        result = predictor.predict({"TransactionAmt": 149.5, "card1": 9500, ...})
        # → {"fraud_probability": 0.12, "decision": "APPROVE", ...}
    """

    def __init__(self, model_dir: Path = MODEL_DIR) -> None:
        self.model_dir = model_dir
        self.model = None
        self.encoders = None
        self.manifest: dict = {}
        self.feature_columns: list[str] = []
        self._loaded = False

    def load(self) -> None:
        """
        Load model and encoders from disk.
        Called once at API startup — not per request.
        """
        model_path = self.model_dir / "model.pkl"
        encoders_path = self.model_dir / "encoders.pkl"
        manifest_path = self.model_dir / "manifest.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}\n"
                f"Run 'make train' to generate the champion model."
            )

        logger.info(f"Loading model from {model_path} ...")
        self.model = joblib.load(model_path)
        self.encoders = joblib.load(encoders_path)

        with open(manifest_path) as f:
            self.manifest = json.load(f)

        self.feature_columns = self.manifest.get("feature_columns", [])
        self._loaded = True
        logger.success(
            f"Model loaded: {self.manifest.get('model_name', 'unknown')}  "
            f"(val PR-AUC={self.manifest.get('val_pr_auc', '?')})"
        )

    def predict(self, transaction: dict) -> dict:
        """
        Score a single transaction.

        Parameters
        ----------
        transaction : raw transaction fields (matches TransactionRequest schema)

        Returns
        -------
        dict with fraud_probability, decision, model_version
        """
        if not self._loaded:
            raise RuntimeError("Predictor not loaded. Call load() first.")

        # Build a single-row DataFrame — the feature pipeline expects a DataFrame
        df = pd.DataFrame([transaction])

        # Apply feature engineering using fitted encoders (same as training)
        from src.features.build_features import build_features
        X, _, _ = build_features(df, fit=False, encoders=self.encoders)

        # Align columns to training schema
        # - Add missing columns as NaN (will be -999 after fillna)
        # - Drop extra columns that weren't in training
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = np.nan
        X = X[self.feature_columns]
        X = X.fillna(-999)

        # Score
        fraud_prob = float(self.model.predict_proba(X)[0, 1])

        # Apply threshold-based decision
        if fraud_prob < THRESHOLD_APPROVE:
            decision = "APPROVE"
        elif fraud_prob > THRESHOLD_DECLINE:
            decision = "DECLINE"
        else:
            decision = "REVIEW"

        return {
            "fraud_probability": round(fraud_prob, 4),
            "decision": decision,
            "model_version": self.manifest.get("mlflow_run_id", "unknown")[:8],
            "threshold_approve": THRESHOLD_APPROVE,
            "threshold_decline": THRESHOLD_DECLINE,
        }

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_version(self) -> str:
        return self.manifest.get("mlflow_run_id", "unknown")[:8]

    @property
    def model_type(self) -> Optional[str]:
        return self.manifest.get("model_name")
