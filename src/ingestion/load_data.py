"""
Data ingestion layer — IEEE-CIS Fraud Detection dataset.

What this module does:
  1. Loads raw CSVs from data/raw/
  2. Validates basic schema (see validate_schema.py for details)
  3. Converts to Parquet for all downstream processing
  4. Optionally writes a sample to PostgreSQL for the monitoring dashboard

Why Parquet instead of CSV?
  - Columnar format: reading one column doesn't scan the whole file
  - Schema enforcement: data types are preserved (no silent int→string coercions)
  - ~70% smaller than CSV with snappy compression on this dataset
  - Supports partition-by-date for time-aware downstream reads

Run directly:
  python -m src.ingestion.load_data
  or: make ingest
"""

import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from loguru import logger

from src.ingestion.validate_schema import validate_identity, validate_transactions

load_dotenv()

RAW_DIR = Path(os.getenv("RAW_DATA_PATH", "data/raw"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DATA_PATH", "data/processed"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_csv(file_path: Path) -> pd.DataFrame:
    """Load a CSV and log basic stats."""
    logger.info(f"Loading {file_path.name} ...")
    df = pd.read_csv(file_path)
    logger.info(f"  {len(df):,} rows  |  {df.shape[1]} columns")
    return df


def log_null_rates(df: pd.DataFrame, name: str) -> None:
    """
    Log null rate per column. Columns with >50% nulls are flagged as warnings.

    For the V-columns in the IEEE-CIS dataset, high null rates are expected and
    meaningful — null often encodes 'feature not applicable for this card type'.
    Document this, don't blindly drop those columns.
    """
    null_rates = df.isnull().mean().sort_values(ascending=False)
    high_null = null_rates[null_rates > 0.5]

    logger.info(f"[{name}] Null rate summary:")
    if high_null.empty:
        logger.info(f"  No columns exceed 50% null threshold")
    else:
        logger.warning(f"  {len(high_null)} columns with >50% nulls (top 10):")
        for col, rate in high_null.head(10).items():
            logger.warning(f"    {col}: {rate:.1%}")


def to_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """Write DataFrame to Parquet with snappy compression."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path, compression="snappy")
    size_mb = output_path.stat().st_size / 1_048_576
    logger.info(f"Saved → {output_path.name}  ({size_mb:.1f} MB)")


# ── Main entrypoint ───────────────────────────────────────────────────────────

def ingest(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
) -> None:
    """
    Full ingestion pipeline.

    Input (place in data/raw/ after downloading from Kaggle):
      train_transaction.csv   — 590,540 rows, 394 columns
      train_identity.csv      — 144,233 rows, 41 columns

    Output (data/processed/):
      transactions.parquet    — raw transaction table
      identity.parquet        — raw identity table
      merged.parquet          — left-joined on TransactionID (used downstream)

    Join strategy:
      LEFT JOIN on TransactionID — we keep all transactions even without identity.
      ~60% of transactions have no identity row. The model must handle this; it's
      not a data quality issue, it's a real-world characteristic of the data.
    """
    # ── Load ──────────────────────────────────────────────────────────────────
    transactions = load_csv(raw_dir / "train_transaction.csv")
    identity = load_csv(raw_dir / "train_identity.csv")

    # ── Validate ──────────────────────────────────────────────────────────────
    validate_transactions(transactions)
    validate_identity(identity)

    # ── Data quality report ───────────────────────────────────────────────────
    log_null_rates(transactions, "transactions")
    log_null_rates(identity, "identity")

    fraud_rate = transactions["isFraud"].mean()
    logger.info(f"Class distribution: {fraud_rate:.2%} fraud  |  {1 - fraud_rate:.2%} legitimate")

    # ── Save individual tables ────────────────────────────────────────────────
    to_parquet(transactions, processed_dir / "transactions.parquet")
    to_parquet(identity, processed_dir / "identity.parquet")

    # ── Merge ─────────────────────────────────────────────────────────────────
    merged = transactions.merge(identity, on="TransactionID", how="left")
    logger.info(f"Merged shape: {merged.shape[0]:,} rows  |  {merged.shape[1]} columns")
    to_parquet(merged, processed_dir / "merged.parquet")

    logger.success("Ingestion complete.")


if __name__ == "__main__":
    ingest()
