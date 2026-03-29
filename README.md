# fintech-fraud-pipeline

End-to-end fraud detection ML pipeline built on the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection). Designed to demonstrate production ML engineering — not just a notebook with a score.

```
Ingest → Feature Engineering → Train (MLflow) → FastAPI → Streamlit Monitor
```

---

## What this demonstrates

- **Full ML lifecycle** — raw data through to a deployed, monitored scoring API
- **Production thinking** — serialised feature pipelines, time-aware splits, threshold-based decisioning, Prometheus metrics
- **Fraud-specific engineering** — SMOTE for class imbalance, target encoding for high-cardinality fields, null indicators for informative missingness, temporal leakage prevention
- **Clean, deployable code** — Docker Compose, Makefile, pytest suite, environment config

---

## Quick start

**1. Get the data**

Download from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data) and place in `data/raw/`:
```
data/raw/train_transaction.csv
data/raw/train_identity.csv
```

**2. Set up the environment**

```bash
make setup          # creates .venv and installs requirements
source .venv/bin/activate
cp .env.example .env
```

**3. Run the full pipeline**

```bash
make ingest         # CSV → Parquet (data/processed/)
make train          # train 4 models, log to MLflow, save champion
make serve          # start FastAPI at http://localhost:8000
make dashboard      # start Streamlit at http://localhost:8501
```

**Or start everything with Docker:**

```bash
make up             # docker-compose up --build
```

---

## API usage

```bash
# Health check
curl http://localhost:8000/health

# Score a transaction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionAmt": 149.50,
    "ProductCD": "W",
    "card1": 9500,
    "card4": "visa",
    "P_emaildomain": "gmail.com"
  }'

# Response:
# {
#   "fraud_probability": 0.087,
#   "decision": "APPROVE",
#   "model_version": "a3f1c2d4",
#   "threshold_approve": 0.3,
#   "threshold_decline": 0.7
# }
```

Interactive API docs: http://localhost:8000/docs

---

## Architecture

See [docs/architecture.md](docs/architecture.md) for full design rationale.

```
fintech-fraud-pipeline/
├── src/
│   ├── ingestion/      load_data.py, validate_schema.py
│   ├── features/       build_features.py, encoders.py
│   ├── training/       train.py, evaluate.py
│   ├── api/            main.py, schemas.py, predictor.py
│   └── monitoring/     dashboard.py, simulate_stream.py
├── tests/              pytest suite (ingestion, features, API)
├── notebooks/          EDA and model comparison (shareable)
├── docs/               architecture, dataset notes, GCP deploy guide
├── docker/             Dockerfile.api, Dockerfile.dashboard
├── docker-compose.yml  full local stack (postgres, mlflow, api, dashboard)
└── Makefile            ingest / train / serve / dashboard
```

---

## Services

| Service | URL | Description |
|---|---|---|
| FastAPI | http://localhost:8000 | Fraud scoring endpoint |
| API Docs | http://localhost:8000/docs | OpenAPI spec |
| MLflow UI | http://localhost:5001 | Experiment tracking |
| Streamlit | http://localhost:8501 | Monitoring dashboard |

---

## Tech stack

| Layer | Technology |
|---|---|
| Data processing | Python, Pandas, Polars, PyArrow |
| ML | LightGBM, XGBoost, scikit-learn, imbalanced-learn |
| Experiment tracking | MLflow |
| API | FastAPI + Pydantic |
| Monitoring | Streamlit + Plotly |
| Storage | Parquet + PostgreSQL |
| Containers | Docker + Docker Compose |
| Testing | pytest |

---

## Key engineering decisions

**Why Parquet?** Columnar format, schema-enforced, ~70% smaller than CSV. All intermediate data is Parquet — never CSV.

**Why time-aware split?** Random splitting leaks future transaction patterns into training, inflating metrics. We split on `TransactionDT` order: train on early data, validate on later data.

**Why serialize the feature pipeline?** The same `encoders.pkl` used at training time is loaded at serving time. This prevents training/serving skew — the most common source of silent model degradation in production.

**Why threshold-based decisions?** `fraud_probability → APPROVE/REVIEW/DECLINE` separates the model output (probability) from the business decision (threshold). Thresholds are environment variables, adjustable without retraining.

---

## Running tests

```bash
make test
# pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Deploy to GCP

See [docs/deploy_gcp.md](docs/deploy_gcp.md) for full Cloud Run deployment guide.

---

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) — 590,540 transactions, ~3.5% fraud rate.
See [docs/dataset_notes.md](docs/dataset_notes.md) for engineering gotchas specific to this dataset.

---

## Author

Anas Abughazaleh · [LinkedIn](https://www.linkedin.com/in/anas-abughazaleh/) · [GitHub](https://github.com/anasag)
