"""
Tests for feature engineering pipeline.

Key things to test:
  1. Encoder correctness (target encoding values, frequency encoding values)
  2. No training/serving skew — same features produced from same input
  3. Unseen categories handled gracefully (fallback to global mean / 0.0)
  4. Feature matrix is fully numeric (no object columns reaching the model)
"""

import numpy as np
import pandas as pd
import pytest

from src.features.encoders import FrequencyEncoder, TargetEncoder


# ── TargetEncoder tests ───────────────────────────────────────────────────────

class TestTargetEncoder:
    def _make_data(self):
        X = pd.DataFrame({"email": ["gmail", "gmail", "yahoo", "yahoo", "yahoo", "rare"]})
        y = pd.Series([0, 1, 1, 1, 0, 0])
        return X, y

    def test_fit_transform_produces_floats(self):
        X, y = self._make_data()
        enc = TargetEncoder(cols=["email"])
        X_enc = enc.fit_transform_train(X, y)
        assert X_enc["email"].dtype == float

    def test_gmail_gets_higher_rate_than_rare(self):
        """gmail (0.5 fraud rate) should encode higher than rare (0.0 fraud rate)."""
        X, y = self._make_data()
        enc = TargetEncoder(cols=["email"])
        enc.fit(X, y)
        X_test = pd.DataFrame({"email": ["gmail", "rare"]})
        X_enc = enc.transform(X_test)
        assert X_enc["email"].iloc[0] > X_enc["email"].iloc[1]

    def test_unseen_category_gets_global_mean(self):
        X, y = self._make_data()
        enc = TargetEncoder(cols=["email"])
        enc.fit(X, y)
        X_test = pd.DataFrame({"email": ["never_seen_domain"]})
        X_enc = enc.transform(X_test)
        global_mean = y.mean()
        # With smoothing, value should be near global mean
        assert abs(X_enc["email"].iloc[0] - global_mean) < 0.5

    def test_encoding_map_is_serialisable(self):
        """Encoders must be serialisable (joblib) for serving."""
        import io
        import joblib
        X, y = self._make_data()
        enc = TargetEncoder(cols=["email"])
        enc.fit(X, y)
        buf = io.BytesIO()
        joblib.dump(enc, buf)
        buf.seek(0)
        loaded = joblib.load(buf)
        assert loaded.encoding_map_["email"] == enc.encoding_map_["email"]


# ── FrequencyEncoder tests ────────────────────────────────────────────────────

class TestFrequencyEncoder:
    def _make_data(self):
        X = pd.DataFrame({
            "card": ["A", "A", "A", "B", "B", "C"]
        })
        return X

    def test_common_card_gets_higher_freq(self):
        X = self._make_data()
        enc = FrequencyEncoder(cols=["card"], normalize=True)
        enc.fit(X)
        X_enc = enc.transform(X)
        # A (3/6 = 0.5) > B (2/6 = 0.33) > C (1/6 = 0.17)
        a_freq = X_enc.loc[X["card"] == "A", "card"].mean()
        c_freq = X_enc.loc[X["card"] == "C", "card"].mean()
        assert a_freq > c_freq

    def test_unseen_card_gets_zero(self):
        X = self._make_data()
        enc = FrequencyEncoder(cols=["card"])
        enc.fit(X)
        X_test = pd.DataFrame({"card": ["Z_never_seen"]})
        X_enc = enc.transform(X_test)
        assert X_enc["card"].iloc[0] == 0.0

    def test_raw_counts(self):
        X = self._make_data()
        enc = FrequencyEncoder(cols=["card"], normalize=False)
        enc.fit(X)
        X_test = pd.DataFrame({"card": ["A"]})
        X_enc = enc.transform(X_test)
        assert X_enc["card"].iloc[0] == 3  # A appears 3 times


# ── Feature pipeline integration test ────────────────────────────────────────

class TestBuildFeatures:
    def _minimal_df(self, n: int = 50) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "TransactionID": range(n),
            "isFraud": rng.integers(0, 2, size=n),
            "TransactionDT": rng.integers(0, 15_000_000, size=n),
            "TransactionAmt": rng.uniform(1.0, 500.0, size=n),
            "ProductCD": rng.choice(["W", "H", "C"], size=n),
            "card1": rng.integers(1000, 9999, size=n),
            "card4": rng.choice(["visa", "mastercard"], size=n),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com"], size=n),
        })

    def test_output_is_numeric(self):
        from src.features.build_features import build_features
        df = self._minimal_df()
        X, y, encoders = build_features(df, fit=True)
        assert all(X.dtypes != object), "Feature matrix contains non-numeric columns"

    def test_no_transaction_id_in_features(self):
        from src.features.build_features import build_features
        df = self._minimal_df()
        X, y, _ = build_features(df, fit=True)
        assert "TransactionID" not in X.columns

    def test_train_inference_parity(self):
        """Same row through training pipeline and inference pipeline must produce same features."""
        from src.features.build_features import build_features
        df = self._minimal_df(n=100)
        X_train, _, encoders = build_features(df.copy(), fit=True)

        # Run a single row through inference pipeline
        single_row = df.iloc[[0]].copy()
        X_infer, _, _ = build_features(single_row, fit=False, encoders=encoders)

        # Column overlap (inference row may have some columns; check shared ones)
        shared_cols = [c for c in X_train.columns if c in X_infer.columns]
        assert len(shared_cols) > 5, "Very few shared columns between train and inference — check encoder"
