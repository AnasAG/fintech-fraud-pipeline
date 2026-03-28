"""
FastAPI fraud scoring service.

Endpoints:
  POST /predict  → score a single transaction
  GET  /health   → liveness check (is model loaded?)
  GET  /metrics  → Prometheus metrics (request count, latency histogram)
  GET  /docs     → auto-generated OpenAPI docs (FastAPI built-in)

Design decisions documented here:
  - lifespan context manager loads model at startup (not per request)
  - Prometheus instrumentation via prometheus-fastapi-instrumentator
  - All validation handled by Pydantic (see schemas.py)
  - Prediction logic is in predictor.py (not here) for testability

Start:
  uvicorn src.api.main:app --reload
  or: make serve
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.predictor import FraudPredictor
from src.api.schemas import FraudDecision, HealthResponse, TransactionRequest

# ── Global predictor instance ─────────────────────────────────────────────────
# One instance shared across all requests. Loaded once at startup.
predictor = FraudPredictor()


# ── Lifespan: startup / shutdown hooks ───────────────────────────────────────
# The lifespan pattern replaced @app.on_event("startup") in FastAPI 0.95+.
# Code before 'yield' runs at startup; code after 'yield' runs at shutdown.

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model
    logger.info("Starting up — loading model ...")
    try:
        predictor.load()
        logger.success("Model loaded — ready to serve")
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        logger.warning("API will start but /predict will return 503 until model is loaded")
    yield
    # Shutdown: nothing to clean up for this service
    logger.info("Shutting down")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fintech Fraud Detection API",
    description=(
        "Real-time fraud scoring for payment transactions. "
        "Trained on the IEEE-CIS Fraud Detection dataset. "
        "See the [GitHub repo](https://github.com/anasabughazaleh/fintech-fraud-pipeline) for full architecture docs."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Expose /metrics endpoint with Prometheus-compatible format
# Instruments: request count, latency histogram, in-flight requests
Instrumentator().instrument(app).expose(app)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post(
    "/predict",
    response_model=FraudDecision,
    summary="Score a transaction for fraud",
    description=(
        "Returns a fraud probability and a decision (APPROVE / REVIEW / DECLINE). "
        "Thresholds are configured via THRESHOLD_APPROVE and THRESHOLD_DECLINE env vars."
    ),
)
async def predict(transaction: TransactionRequest) -> FraudDecision:
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run 'make train' to generate the champion model.",
        )

    result = predictor.predict(transaction.model_dump())

    return FraudDecision(**result)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if predictor.is_loaded else "degraded",
        model_loaded=predictor.is_loaded,
        model_version=predictor.model_version if predictor.is_loaded else "none",
        model_type=predictor.model_type if predictor.is_loaded else None,
    )


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Fraud Detection API — visit /docs for the OpenAPI spec"}
