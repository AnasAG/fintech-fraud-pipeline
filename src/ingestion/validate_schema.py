"""
Schema validation for the IEEE-CIS dataset.

Why validate schema explicitly?
  Silent failures are the worst kind in ML pipelines. A missing column doesn't
  crash immediately — it causes a KeyError three steps later in feature engineering,
  or worse, it silently produces a feature matrix with a wrong shape that trains
  a model on garbage. Schema validation here gives a fast, clear failure message
  at the exact point where something is wrong.

  This is basic data observability — logging not just 'did it load?' but
  'does it look like what we expect?'
"""

from typing import Dict, List

import pandas as pd
from loguru import logger


# ── Expected schema ───────────────────────────────────────────────────────────

TRANSACTION_REQUIRED: List[str] = [
    "TransactionID",
    "isFraud",
    "TransactionDT",     # seconds elapsed — NOT a real timestamp, needs conversion
    "TransactionAmt",
    "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2",
    "dist1", "dist2",
    "P_emaildomain",
    "R_emaildomain",
    # C-columns: counting features (device fingerprint counts, etc.)
    "C1", "C2", "C3", "C4",
    # M-columns: match features (name/address match flags)
    "M1", "M2", "M3",
]

IDENTITY_REQUIRED: List[str] = [
    "TransactionID",
    "DeviceType",
    "DeviceInfo",
]

EXPECTED_DTYPES: Dict[str, str] = {
    "TransactionID": "int",
    "isFraud":       "int",
    "TransactionDT": "int",
    "TransactionAmt": "float",
}

# Realistic bounds for sanity checks
AMOUNT_MIN = 0.0
AMOUNT_MAX = 20_000.0


# ── Validators ────────────────────────────────────────────────────────────────

def _check_required_columns(df: pd.DataFrame, required: List[str], table: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{table}] Schema validation failed — missing columns: {missing}\n"
            f"  Make sure you downloaded train_transaction.csv and train_identity.csv "
            f"from https://www.kaggle.com/c/ieee-fraud-detection/data"
        )
    logger.info(f"[{table}] Required columns: OK ({len(required)} checked)")


def _check_dtypes(df: pd.DataFrame, table: str) -> None:
    for col, expected_prefix in EXPECTED_DTYPES.items():
        if col not in df.columns:
            continue
        actual = str(df[col].dtype)
        if not actual.startswith(expected_prefix):
            logger.warning(f"[{table}] {col}: expected {expected_prefix}*, got {actual}")


def _check_no_duplicate_ids(df: pd.DataFrame, id_col: str, table: str) -> None:
    n_dupes = df[id_col].duplicated().sum()
    if n_dupes > 0:
        raise ValueError(f"[{table}] {n_dupes:,} duplicate {id_col} values — data integrity issue")
    logger.info(f"[{table}] No duplicate {id_col}s: OK")


def _check_fraud_label(df: pd.DataFrame) -> None:
    """
    The isFraud label should be binary (0 or 1) with ~3.5% positive rate.
    If the rate is wildly off, something is wrong with the data slice.
    """
    unique_labels = set(df["isFraud"].dropna().unique())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"isFraud contains unexpected values: {unique_labels - {0, 1}}")

    fraud_rate = df["isFraud"].mean()
    logger.info(f"Fraud label: binary OK  |  fraud rate = {fraud_rate:.2%}")

    if fraud_rate < 0.01 or fraud_rate > 0.20:
        logger.warning(
            f"Fraud rate {fraud_rate:.2%} is outside expected 1–20% range. "
            f"Check that you're not using a pre-filtered sample."
        )


def _check_transaction_amount(df: pd.DataFrame) -> None:
    neg_count = (df["TransactionAmt"] < AMOUNT_MIN).sum()
    if neg_count > 0:
        logger.warning(f"TransactionAmt: {neg_count:,} negative values")

    extreme_count = (df["TransactionAmt"] > AMOUNT_MAX).sum()
    if extreme_count > 0:
        logger.warning(
            f"TransactionAmt: {extreme_count:,} values > {AMOUNT_MAX:,} "
            f"(may be legitimate large transactions — review manually)"
        )


# ── Public API ────────────────────────────────────────────────────────────────

def validate_transactions(df: pd.DataFrame) -> None:
    """Run all transaction table validations. Raises ValueError on hard failures."""
    _check_required_columns(df, TRANSACTION_REQUIRED, "transactions")
    _check_dtypes(df, "transactions")
    _check_no_duplicate_ids(df, "TransactionID", "transactions")
    _check_fraud_label(df)
    _check_transaction_amount(df)
    logger.info("[transactions] Schema validation passed")


def validate_identity(df: pd.DataFrame) -> None:
    """Run all identity table validations."""
    _check_required_columns(df, IDENTITY_REQUIRED, "identity")
    _check_dtypes(df, "identity")
    logger.info("[identity] Schema validation passed")
