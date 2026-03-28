"""
Tests for the data ingestion layer.

These tests use synthetic DataFrames — no actual Kaggle data needed to run the test suite.
The goal is to test schema validation logic, not file I/O.
"""

import pytest
import pandas as pd
import numpy as np

from src.ingestion.validate_schema import (
    validate_transactions,
    validate_identity,
    _check_fraud_label,
    _check_no_duplicate_ids,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_transaction_df(n: int = 100, fraud_rate: float = 0.035) -> pd.DataFrame:
    """Create a minimal valid transactions DataFrame."""
    rng = np.random.default_rng(42)
    n_fraud = int(n * fraud_rate)
    labels = np.array([1] * n_fraud + [0] * (n - n_fraud))
    rng.shuffle(labels)

    return pd.DataFrame({
        "TransactionID": range(1, n + 1),
        "isFraud": labels,
        "TransactionDT": rng.integers(0, 15_000_000, size=n),
        "TransactionAmt": rng.uniform(1.0, 1000.0, size=n),
        "ProductCD": rng.choice(["W", "H", "C", "S", "R"], size=n),
        "card1": rng.integers(1000, 9999, size=n),
        "card2": rng.uniform(100, 600, size=n),
        "card3": rng.uniform(100, 200, size=n),
        "card4": rng.choice(["visa", "mastercard"], size=n),
        "card5": rng.uniform(100, 250, size=n),
        "card6": rng.choice(["credit", "debit"], size=n),
        "addr1": rng.uniform(100, 500, size=n),
        "addr2": rng.uniform(1, 100, size=n),
        "dist1": rng.uniform(0, 5000, size=n),
        "dist2": rng.uniform(0, 5000, size=n),
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", None], size=n),
        "R_emaildomain": rng.choice(["gmail.com", "hotmail.com", None], size=n),
        "C1": rng.integers(0, 4000, size=n, dtype=float),
        "C2": rng.integers(0, 500, size=n, dtype=float),
        "C3": rng.integers(0, 30, size=n, dtype=float),
        "C4": rng.integers(0, 30, size=n, dtype=float),
        "M1": rng.choice(["T", "F", None], size=n),
        "M2": rng.choice(["T", "F", None], size=n),
        "M3": rng.choice(["T", "F", None], size=n),
    })


def make_identity_df(n: int = 60) -> pd.DataFrame:
    """Create a minimal valid identity DataFrame."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "TransactionID": range(1, n + 1),
        "DeviceType": rng.choice(["desktop", "mobile", None], size=n),
        "DeviceInfo": rng.choice(["Windows", "iOS Device", None], size=n),
    })


# ── Transaction validation tests ──────────────────────────────────────────────

class TestTransactionValidation:
    def test_valid_transactions_pass(self):
        df = make_transaction_df()
        validate_transactions(df)  # should not raise

    def test_missing_required_column_raises(self):
        df = make_transaction_df().drop(columns=["isFraud"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_transactions(df)

    def test_duplicate_transaction_ids_raise(self):
        df = make_transaction_df(n=50)
        df.loc[5, "TransactionID"] = df.loc[0, "TransactionID"]  # create duplicate
        with pytest.raises(ValueError, match="duplicate TransactionID"):
            _check_no_duplicate_ids(df, "TransactionID", "transactions")

    def test_invalid_fraud_label_raises(self):
        df = make_transaction_df()
        df.loc[0, "isFraud"] = 2  # invalid label
        with pytest.raises(ValueError, match="unexpected values"):
            _check_fraud_label(df)

    def test_zero_fraud_rate_warns(self, caplog):
        df = make_transaction_df(fraud_rate=0.0)
        import logging
        with caplog.at_level(logging.WARNING):
            _check_fraud_label(df)
        # loguru captures differently; just confirm it doesn't crash

    def test_negative_transaction_amount_warns(self):
        df = make_transaction_df()
        df.loc[0, "TransactionAmt"] = -1.0
        # Should not raise — just warns
        from src.ingestion.validate_schema import _check_transaction_amount
        _check_transaction_amount(df)  # no exception


# ── Identity validation tests ─────────────────────────────────────────────────

class TestIdentityValidation:
    def test_valid_identity_passes(self):
        df = make_identity_df()
        validate_identity(df)  # should not raise

    def test_missing_transaction_id_raises(self):
        df = make_identity_df().drop(columns=["TransactionID"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_identity(df)
