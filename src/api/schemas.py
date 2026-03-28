"""
Pydantic schemas for the fraud scoring API.

Why Pydantic?
  FastAPI uses Pydantic models to validate all incoming request bodies and
  outgoing responses. This gives you:
  - Automatic type coercion (string '123' → int 123)
  - Clear error messages on invalid input (HTTP 422 with field-level details)
  - Auto-generated OpenAPI docs at /docs
  - Runtime protection against malformed requests crashing your model

TransactionRequest includes only the fields the feature pipeline expects.
Fields not listed here are silently ignored.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class TransactionRequest(BaseModel):
    """
    Incoming transaction payload for fraud scoring.

    All fields mirror the IEEE-CIS dataset schema.
    Most are optional because ~60% of real transactions have no identity data.
    """

    # Core transaction fields
    TransactionAmt: float = Field(..., gt=0, description="Transaction amount in USD")
    ProductCD: Optional[str] = Field(None, description="Product type: W, H, C, S, R")
    TransactionDT: Optional[int] = Field(
        None, description="Seconds elapsed since reference date (used for time features)"
    )

    # Card features
    card1: Optional[int] = None
    card2: Optional[float] = None
    card3: Optional[float] = None
    card4: Optional[str] = Field(None, description="Card brand: visa, mastercard, etc.")
    card5: Optional[float] = None
    card6: Optional[str] = Field(None, description="Card type: credit, debit, etc.")

    # Address features
    addr1: Optional[float] = None
    addr2: Optional[float] = None

    # Distance features
    dist1: Optional[float] = None
    dist2: Optional[float] = None

    # Email domains
    P_emaildomain: Optional[str] = Field(None, description="Purchaser email domain")
    R_emaildomain: Optional[str] = Field(None, description="Recipient email domain")

    # Count/match features (C and M columns)
    C1: Optional[float] = None
    C2: Optional[float] = None
    C3: Optional[float] = None
    C4: Optional[float] = None
    M1: Optional[str] = None
    M2: Optional[str] = None
    M3: Optional[str] = None

    # Device features (from identity table)
    DeviceType: Optional[str] = Field(None, description="desktop or mobile")
    DeviceInfo: Optional[str] = None

    @field_validator("TransactionAmt")
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("TransactionAmt must be positive")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "TransactionAmt": 149.50,
                "ProductCD": "W",
                "card1": 9500,
                "card4": "visa",
                "card6": "debit",
                "P_emaildomain": "gmail.com",
                "DeviceType": "desktop",
            }
        }
    }


class FraudDecision(BaseModel):
    """
    Fraud scoring response.

    Decision thresholds are business parameters (see .env.example):
      APPROVE  → fraud_probability < 0.3
      REVIEW   → 0.3 ≤ fraud_probability ≤ 0.7
      DECLINE  → fraud_probability > 0.7
    """

    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    decision: Literal["APPROVE", "REVIEW", "DECLINE"]
    model_version: str
    threshold_approve: float
    threshold_decline: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    model_loaded: bool
    model_version: str
    model_type: Optional[str] = None
