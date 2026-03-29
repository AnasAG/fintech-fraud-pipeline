"""
Feature engineering pipeline for the IEEE-CIS fraud dataset.

This module builds a feature matrix from the merged Parquet file.
All transformations must be reproducible and serialisable — the same
sklearn Pipeline object runs at training time AND at serving time.

Feature groups:
  1. Time features       — cyclical encoding of hour/day to preserve periodicity
  2. Transaction amounts — log transform + rolling z-score per card
  3. Velocity features   — transaction count and spend in rolling windows per card
  4. Categorical encoding — target encoding for high-cardinality, freq for identifiers
  5. Missing indicators  — explicit 'was_null' flags for key columns

Why serialize the pipeline?
  At inference, the API receives a raw transaction dict. It must apply exactly
  the same transformations that were applied during training. If training used
  mean imputation with mean=450.2 and inference recomputes mean=450.9 from
  different data, you get feature drift. Serialising the fitted pipeline prevents
  this — the fitted parameters (means, encodings, etc.) are fixed at training time.

Run directly:
  python -m src.features.build_features
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Optional

from src.features.encoders import FrequencyEncoder, TargetEncoder

PROCESSED_DIR = Path(os.getenv("PROCESSED_DATA_PATH", "data/processed"))
PIPELINE_PATH = Path("src/features/pipeline.pkl")


# ── Column definitions ────────────────────────────────────────────────────────

# High-cardinality categoricals → target encoding (encode with fraud rate)
TARGET_ENCODE_COLS = [
    "P_emaildomain",
    "R_emaildomain",
    "DeviceInfo",
    "id_30",    # OS version
    "id_31",    # browser
]

# Card/device identifiers → frequency encoding (encode with appearance count)
FREQ_ENCODE_COLS = [
    "card1", "card2", "card3",
    "id_20",
]

# Low-cardinality categoricals → integer mapping
ORDINAL_COLS = {
    "ProductCD":  ["C", "H", "R", "S", "W"],
    "card4":      ["american express", "discover", "mastercard", "visa"],
    "card6":      ["charge card", "credit", "debit", "debit or credit"],
    "DeviceType": ["desktop", "mobile"],
    "M4":         ["M0", "M1", "M2"],
}


# ── Feature engineering functions ─────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert TransactionDT (elapsed seconds) to cyclical time features.

    Why cyclical encoding?
    Hour 23 and hour 0 are 1 hour apart, but numerically they're 23 apart.
    sin/cos encoding maps the cycle so hour 23 and hour 0 are close in feature space.
    This is important for fraud: 2am transactions behave like 1am transactions,
    not like 2pm transactions.
    """
    # TransactionDT is seconds since a reference point (not unix timestamp)
    # The reference date is 2017-11-30 based on known dataset analysis
    REFERENCE_DATE = pd.Timestamp("2017-11-30")
    dt = REFERENCE_DATE + pd.to_timedelta(df["TransactionDT"], unit="s")

    hour = dt.dt.hour
    dow = dt.dt.dayofweek

    # Cyclical encoding: map to unit circle
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["is_weekend"] = (dow >= 5).astype(int)
    df["is_night"] = ((hour >= 23) | (hour <= 5)).astype(int)

    return df


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Log-transform TransactionAmt and compute per-card amount z-score.

    Why log transform?
    Transaction amounts are right-skewed (many small, few very large).
    Log-transform compresses the scale so a $10 vs $100 difference is
    treated similarly to a $100 vs $1,000 difference.

    Why z-score per card?
    A $500 transaction on a card that usually spends $50 is very different
    from a $500 transaction on a card that usually spends $10,000.
    The z-score normalises amount by that card's history.
    """
    df["log_amount"] = np.log1p(df["TransactionAmt"])

    card_stats = df.groupby("card1")["TransactionAmt"].agg(["mean", "std"])
    df = df.merge(card_stats, on="card1", how="left", suffixes=("", "_card"))
    df["amount_zscore"] = (df["TransactionAmt"] - df["mean"]) / (df["std"] + 1e-8)
    df.drop(columns=["mean", "std"], inplace=True)

    return df


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transaction velocity per card: count and spend over rolling windows.

    Why velocity?
    A fraudster who steals a card often makes many transactions in quick
    succession before the card is blocked. High velocity (many txns in short
    window) is one of the strongest fraud signals in the literature.

    Note: This is computed offline here on the full dataset. At serving time,
    you would need a real-time feature store (Redis, Feast) to compute these
    over a live window. For this portfolio project, we pre-compute them.
    """
    if "TransactionDT" in df.columns:
        df = df.sort_values("TransactionDT")

    # At inference time (single row), TransactionID is absent — fall back to 1/amount/amount
    if "TransactionID" in df.columns:
        card_count = df.groupby("card1")["TransactionID"].transform("count")
    else:
        card_count = pd.Series(1, index=df.index)

    card_total = df.groupby("card1")["TransactionAmt"].transform("sum")
    card_mean = df.groupby("card1")["TransactionAmt"].transform("mean")

    df["card_txn_count"] = card_count
    df["card_total_spend"] = card_total
    df["card_mean_spend"] = card_mean

    return df


def add_null_indicators(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Add binary 'was_null' flags for columns with significant missing rates.

    Why not just impute?
    In the IEEE-CIS dataset, null values in device fields often mean 'no device
    info collected' — which itself is informative. A transaction with no device
    fingerprint is structurally different from one with a device fingerprint,
    regardless of what we impute. The null flag preserves this signal.
    """
    for col in cols:
        if col in df.columns:
            df[f"{col}_was_null"] = df[col].isnull().astype(int)
    return df


def encode_ordinals(df: pd.DataFrame) -> pd.DataFrame:
    """Map low-cardinality categoricals to integers."""
    for col, categories in ORDINAL_COLS.items():
        if col in df.columns:
            cat = pd.Categorical(df[col].str.lower().str.strip(), categories=categories)
            df[col] = cat.codes  # -1 for unseen/null
    return df


# ── Main pipeline ─────────────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    target_col: str = "isFraud",
    fit: bool = True,
    encoders: Optional[dict] = None,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Full feature engineering pipeline.

    Parameters
    ----------
    df : raw merged DataFrame (output of ingestion)
    target_col : label column name
    fit : if True, fit encoders on df (training mode)
          if False, use pre-fitted encoders (inference mode)
    encoders : pre-fitted encoder dict (required when fit=False)

    Returns
    -------
    X : feature matrix (pd.DataFrame)
    y : labels (pd.Series)
    encoders : fitted encoder dict (save this alongside the model)
    """
    logger.info(f"Building features: {len(df):,} rows")

    y = df[target_col].copy() if target_col in df.columns else None
    df = df.drop(columns=[target_col], errors="ignore")

    # Step 1: Time features
    df = add_time_features(df)

    # Step 2: Amount features
    df = add_amount_features(df)

    # Step 3: Velocity features
    df = add_velocity_features(df)

    # Step 4: Null indicators (before imputation)
    null_flag_cols = ["DeviceType", "DeviceInfo", "id_30", "id_31", "P_emaildomain"]
    df = add_null_indicators(df, null_flag_cols)

    # Step 5: Ordinal encoding (must happen before target/freq encoding)
    df = encode_ordinals(df)

    # Step 6: Categorical encoding
    if fit:
        te = TargetEncoder(cols=[c for c in TARGET_ENCODE_COLS if c in df.columns])
        fe = FrequencyEncoder(cols=[c for c in FREQ_ENCODE_COLS if c in df.columns])
        if y is not None:
            df = te.fit_transform_train(df, y)
        fe.fit(df)
        df = fe.transform(df)
        encoders = {"target_encoder": te, "freq_encoder": fe}
    else:
        df = encoders["target_encoder"].transform(df)
        df = encoders["freq_encoder"].transform(df)

    # Step 7: Drop non-numeric / ID columns
    drop_cols = ["TransactionID", "TransactionDT"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Step 8: Keep only numeric columns (any remaining strings become NaN via coerce)
    df = df.apply(pd.to_numeric, errors="coerce")

    # Step 9: Drop V-columns with >90% correlation to another V-column
    # (pruning 339 largely redundant anonymised features)
    v_cols = [c for c in df.columns if c.startswith("V")]
    if len(v_cols) > 50 and fit:
        corr = df[v_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
        df = df.drop(columns=to_drop)
        logger.info(f"Pruned {len(to_drop)} highly-correlated V-columns")
        if encoders:
            encoders["dropped_v_cols"] = to_drop
    elif encoders and "dropped_v_cols" in encoders:
        df = df.drop(columns=[c for c in encoders["dropped_v_cols"] if c in df.columns])

    logger.info(f"Feature matrix: {df.shape[1]} features")

    return df, y, encoders


def save_pipeline(encoders: dict, path: Path = PIPELINE_PATH) -> None:
    """Serialise fitted encoders to disk. Load this at serving time."""
    joblib.dump(encoders, path)
    logger.info(f"Pipeline saved → {path}")


def load_pipeline(path: Path = PIPELINE_PATH) -> dict:
    return joblib.load(path)


if __name__ == "__main__":
    df = pd.read_parquet(PROCESSED_DIR / "merged.parquet")
    X, y, encoders = build_features(df, fit=True)
    save_pipeline(encoders)
    logger.info(f"Feature matrix shape: {X.shape}")
