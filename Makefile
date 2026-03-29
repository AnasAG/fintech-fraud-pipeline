# ──────────────────────────────────────────────────────────────────────────────
# fintech-fraud-pipeline Makefile
#
# Targets:
#   make ingest      — ingest raw CSVs → Parquet
#   make train       — run feature engineering + model training (logs to MLflow)
#   make serve       — start FastAPI scoring service (local, no Docker)
#   make dashboard   — start Streamlit monitoring dashboard (local, no Docker)
#   make test        — run pytest with coverage
#   make up          — docker-compose up (full stack)
#   make down        — docker-compose down
#   make setup       — create virtualenv and install requirements
# ──────────────────────────────────────────────────────────────────────────────

PYTHON   := python3
VENV     := .venv
PIP      := $(VENV)/bin/pip
PYTEST   := $(VENV)/bin/pytest
UVICORN  := $(VENV)/bin/uvicorn

.DEFAULT_GOAL := help

# ── Help ──────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo ""
	@echo "fintech-fraud-pipeline"
	@echo "──────────────────────────────────────────────────────"
	@echo "  make setup      Install Python dependencies in .venv"
	@echo "  make ingest     Load raw CSVs and convert to Parquet"
	@echo "  make train      Run feature pipeline + train models"
	@echo "  make serve      Start FastAPI scoring API (port 8000)"
	@echo "  make dashboard  Start Streamlit dashboard (port 8501)"
	@echo "  make test       Run pytest with coverage report"
	@echo "  make up         docker-compose up --build (full stack)"
	@echo "  make down       docker-compose down"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
.PHONY: setup
setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	cp -n .env.example .env || true
	@echo "Setup complete. Activate with: source $(VENV)/bin/activate"

# ── Ingest ────────────────────────────────────────────────────────────────────
# Expects raw CSVs in data/raw/. Download from:
# https://www.kaggle.com/c/ieee-fraud-detection/data
.PHONY: ingest
ingest:
	@echo "→ Running data ingestion..."
	$(VENV)/bin/python -m src.ingestion.load_data
	@echo "✓ Parquet files written to data/processed/"

# ── Train ─────────────────────────────────────────────────────────────────────
# Requires data/processed/merged.parquet (run make ingest first).
# Logs all experiments to MLflow. Open http://localhost:5000 to compare runs.
.PHONY: train
train:
	@echo "→ Building features and training models..."
	$(VENV)/bin/python -m src.training.train
	@echo "✓ Training complete. Champion model saved to models/champion/"
	@echo "  MLflow UI: http://localhost:5000"

# ── Serve ─────────────────────────────────────────────────────────────────────
# Requires models/champion/ (run make train first).
# API docs auto-generated at http://localhost:8000/docs
.PHONY: serve
serve:
	@echo "→ Starting FastAPI scoring service on port 8000..."
	@echo "  Docs: http://localhost:8000/docs"
	@echo "  Health: http://localhost:8000/health"
	PYTHONPATH=. $(VENV)/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# ── Dashboard ─────────────────────────────────────────────────────────────────
# Requires data/processed/merged.parquet and the API running (make serve).
.PHONY: dashboard
dashboard:
	@echo "→ Starting Streamlit monitoring dashboard on port 8501..."
	PYTHONPATH=. $(VENV)/bin/streamlit run src/monitoring/dashboard.py \
		--server.port 8501 \
		--server.address 0.0.0.0

# ── Test ──────────────────────────────────────────────────────────────────────
.PHONY: test
test:
	@echo "→ Running test suite..."
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing --cov-report=html
	@echo "✓ Coverage report: htmlcov/index.html"

# ── Docker ────────────────────────────────────────────────────────────────────
.PHONY: up
up:
	docker-compose up --build

.PHONY: down
down:
	docker-compose down

.PHONY: up-infra
up-infra:
	@echo "→ Starting postgres + mlflow only (for local dev)"
	docker-compose up postgres mlflow
