# Architecture

## System overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      fintech-fraud-pipeline                         │
│                                                                     │
│  ┌──────────────┐    ┌───────────────┐    ┌─────────────────────┐  │
│  │  Data        │───▶│  Feature      │───▶│  Model Training     │  │
│  │  Ingestion   │    │  Engineering  │    │  + Experiment       │  │
│  │              │    │               │    │  Tracking (MLflow)  │  │
│  │ load_data.py │    │ build_        │    │                     │  │
│  │ validate_    │    │ features.py   │    │  train.py           │  │
│  │ schema.py    │    │ encoders.py   │    │  evaluate.py        │  │
│  └──────────────┘    └───────────────┘    └────────┬────────────┘  │
│        │                                            │               │
│  CSV → Parquet                               champion model         │
│        → PostgreSQL                          + pipeline.pkl         │
│                                                     │               │
│                                          ┌──────────▼──────────┐   │
│                                          │  FastAPI Scoring    │   │
│                                          │  Service            │   │
│                                          │  POST /predict      │   │
│                                          │  GET  /health       │   │
│                                          │  GET  /metrics      │   │
│                                          └──────────┬──────────┘   │
│                                                     │               │
│                                          ┌──────────▼──────────┐   │
│                                          │  Streamlit          │   │
│                                          │  Monitoring         │   │
│                                          │  Dashboard          │   │
│                                          └─────────────────────┘   │
│                                                                     │
│  PostgreSQL ── stores prediction logs, queried by dashboard         │
│  MLflow     ── experiment tracking, model registry                  │
└─────────────────────────────────────────────────────────────────────┘
```

## Key design decisions

### Why Parquet instead of CSV for intermediate storage?

CSV re-parses string types on every read. A column that should be int64 becomes
object if a single cell is missing. Parquet stores the schema alongside the data,
so types are preserved. For a dataset with 434 columns, this matters — silent
type coercions downstream cause hard-to-debug model failures.

Parquet is also ~70% smaller than CSV on this dataset (columnar + snappy compression)
and supports predicate pushdown: reading only the columns you need doesn't scan
the whole file.

### Why time-aware train/val/test split?

The dataset has a natural time ordering via `TransactionDT`. Random splitting
would let the model see future transactions during training — a form of temporal
leakage. A model trained with temporal leakage produces artificially high validation
metrics that don't hold in production.

We split: first 70% → train, next 15% → val, last 15% → test. No shuffling.

### Why serialize the feature pipeline?

At inference, the API receives a raw transaction dict. It applies exactly the
fitted encoders (TargetEncoder, FrequencyEncoder) from training. If the encoding
map is recomputed at serving time from different data, you get training/serving skew:
the model sees different feature values at inference than it was trained on.

The `encoders.pkl` file (saved alongside `model.pkl`) contains the fitted state
of all encoders. The API loads this once at startup and uses it for every request.

### Why threshold-based decisions instead of binary model output?

A model output of `predict([...]) → 0 or 1` bakes the business decision into the
model. Threshold-based decisioning separates concerns:

- The model outputs a probability (a data science artifact)
- The business decides what probability warrants APPROVE / REVIEW / DECLINE
- Thresholds are environment variables — a product manager can adjust them
  without retraining the model

A REVIEW queue (0.3–0.7) is particularly important: it routes ambiguous cases
to human review rather than auto-blocking, which is important for customer experience.

### Why Prometheus metrics on the API?

`GET /metrics` exposes request count, latency histogram, and in-flight request
count in Prometheus format. In production, you'd scrape this with Prometheus and
alert on p95 latency > 200ms or error rate > 1%. Even for a portfolio project,
having the instrumentation in place demonstrates production thinking.

## Data flow

```
make ingest
  data/raw/*.csv
    └── validate_schema.py (schema check, null rates, fraud rate)
    └── load_data.py (CSV → Parquet, left-join transactions + identity)
    └── data/processed/merged.parquet

make train
  data/processed/merged.parquet
    └── build_features.py (time features, amount features, encoders)
    └── train.py (time split → SMOTE → LightGBM/XGBoost/LR → MLflow)
    └── models/champion/model.pkl + encoders.pkl + manifest.json
    └── src/features/pipeline.pkl (serialised encoder state)

make serve
  models/champion/
    └── predictor.py (load model + encoders once)
    └── main.py (POST /predict → apply encoders → model.predict_proba → decision)

make dashboard
  data/processed/merged.parquet
    └── simulate_stream.py (replay test set as batches, optionally inject drift)
    └── dashboard.py (Streamlit: score dist, decisions over time, drift alert)
```

## Component boundaries

| Component | Depends on | Produces |
|---|---|---|
| Ingestion | Raw CSVs | `data/processed/*.parquet` |
| Feature engineering | `merged.parquet` | Feature matrix + `encoders.pkl` |
| Training | Feature matrix + encoders | `models/champion/` + MLflow runs |
| API | `models/champion/` | Fraud decisions via HTTP |
| Dashboard | `merged.parquet` + API | Monitoring panels |

Each component has a clear input/output contract. You can swap the model
(replace LightGBM with a neural net) without touching the ingestion or API layer,
as long as the `model.pkl` interface (predict_proba) stays the same.
