"""
Transaction stream simulator for the monitoring dashboard.

What this does:
  Takes the held-out test set (the last 15% of transactions by time) and
  replays it as if transactions were arriving in real time. Each 'tick'
  releases a batch of transactions with their true labels, allowing the
  dashboard to show how model performance evolves.

  Also includes a concept drift injector: you can modify a fraction of
  transactions to simulate a new fraud pattern the model hasn't seen.
  This demonstrates why monitoring matters — a deployed model degrades
  silently without a monitoring layer.

Why simulate drift?
  Real fraud patterns change constantly (new attack vectors, card skimming
  spikes, etc.). A model trained in Q1 may not handle Q3 patterns well.
  The drift simulator lets you visually demonstrate this problem without
  needing live data.
"""

import os
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
from loguru import logger


PROCESSED_DIR = Path(os.getenv("PROCESSED_DATA_PATH", "data/processed"))


def load_test_set(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """
    Load the last 15% of transactions (time-sorted) as the test stream.
    Mirrors the split used in train.py.
    """
    df = pd.read_parquet(processed_dir / "merged.parquet")
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    test_start = int(len(df) * 0.85)
    test_df = df.iloc[test_start:].copy()
    logger.info(f"Test stream: {len(test_df):,} transactions")
    return test_df


def inject_concept_drift(
    df: pd.DataFrame,
    drift_fraction: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate concept drift by modifying a fraction of transactions.

    The drift pattern: multiply TransactionAmt by 5 for a subset of
    legitimate transactions (simulating a new fraud vector where attackers
    make large purchases that look like normal high-value transactions).

    This causes the model's score distribution to shift because it was
    trained on the original distribution — drift detection should flag this.
    """
    df = df.copy()
    rng = np.random.default_rng(seed)

    # Pick drift_fraction of LEGITIMATE transactions to modify
    legit_mask = df["isFraud"] == 0
    legit_idx = df[legit_mask].index
    n_drift = int(len(legit_idx) * drift_fraction)
    drift_idx = rng.choice(legit_idx, size=n_drift, replace=False)

    df.loc[drift_idx, "TransactionAmt"] *= 5
    df.loc[drift_idx, "_drifted"] = True
    df["_drifted"] = df.get("_drifted", pd.Series(False, index=df.index)).fillna(False)

    logger.info(f"Injected drift: {n_drift:,} transactions modified ({drift_fraction:.0%})")
    return df


def stream_batches(
    df: pd.DataFrame,
    batch_size: int = 500,
) -> Generator[pd.DataFrame, None, None]:
    """
    Yield batches of transactions as if they're arriving in real time.

    Each batch represents one 'tick' in the dashboard's sliding window.
    """
    for start in range(0, len(df), batch_size):
        yield df.iloc[start : start + batch_size].copy()


def compute_window_metrics(
    predictions_log: pd.DataFrame,
    window_hours: int = 1,
) -> dict:
    """
    Compute monitoring metrics over a rolling time window.

    Parameters
    ----------
    predictions_log : DataFrame with columns: timestamp, fraud_probability, isFraud, decision
    window_hours : rolling window size in hours

    Returns
    -------
    dict of metric_name → value
    """
    if predictions_log.empty:
        return {}

    recent = predictions_log.copy()

    # Decision distribution
    decision_counts = recent["decision"].value_counts(normalize=True)

    # Score statistics
    score_mean = recent["fraud_probability"].mean()
    score_p95 = recent["fraud_probability"].quantile(0.95)

    # True labels (where available)
    labeled = recent.dropna(subset=["isFraud"])
    if not labeled.empty:
        actual_fraud_rate = labeled["isFraud"].mean()
        # FP rate: DECLINED or REVIEW but actually legitimate
        flagged = labeled[labeled["decision"].isin(["DECLINE", "REVIEW"])]
        fp_rate = (flagged["isFraud"] == 0).mean() if len(flagged) > 0 else 0.0
    else:
        actual_fraud_rate = None
        fp_rate = None

    return {
        "n_transactions": len(recent),
        "pct_approve": decision_counts.get("APPROVE", 0.0),
        "pct_review": decision_counts.get("REVIEW", 0.0),
        "pct_decline": decision_counts.get("DECLINE", 0.0),
        "score_mean": score_mean,
        "score_p95": score_p95,
        "actual_fraud_rate": actual_fraud_rate,
        "false_positive_rate": fp_rate,
    }
