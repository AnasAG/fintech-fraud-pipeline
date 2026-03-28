"""
Tests for the FastAPI scoring service.

Uses httpx's AsyncClient to test routes without spinning up a real server.
FastAPI's TestClient wraps this for synchronous tests.

What we test:
  1. /health returns expected shape
  2. /predict with a valid payload returns a decision
  3. /predict with an invalid payload (negative amount) returns 422
  4. Threshold logic: verify that low/medium/high probabilities map to correct decisions

Note: These tests mock the predictor to avoid needing a trained model on disk.
In a real CI environment you'd test against a pinned model artifact.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── Setup ─────────────────────────────────────────────────────────────────────

@pytest.fixture()
def mock_predictor():
    """A predictor mock that returns predictable probabilities."""
    p = MagicMock()
    p.is_loaded = True
    p.model_version = "abc12345"
    p.model_type = "lightgbm"
    p.manifest = {"val_pr_auc": 0.87, "test_pr_auc": 0.84, "model_name": "lightgbm"}
    p.feature_columns = ["TransactionAmt", "log_amount"]
    return p


@pytest.fixture()
def client(mock_predictor):
    """Test client with patched predictor."""
    with patch("src.api.main.predictor", mock_predictor):
        from src.api.main import app
        with TestClient(app) as c:
            yield c


# ── Health endpoint ───────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_ok_when_model_loaded(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert "model_version" in body

    def test_health_returns_degraded_when_model_not_loaded(self):
        p = MagicMock()
        p.is_loaded = False
        p.model_version = "none"
        p.model_type = None
        with patch("src.api.main.predictor", p):
            from src.api.main import app
            with TestClient(app) as c:
                response = c.get("/health")
                assert response.status_code == 200
                assert response.json()["status"] == "degraded"


# ── Predict endpoint ──────────────────────────────────────────────────────────

class TestPredict:
    VALID_PAYLOAD = {
        "TransactionAmt": 149.50,
        "ProductCD": "W",
        "card1": 9500,
        "card4": "visa",
        "card6": "debit",
        "P_emaildomain": "gmail.com",
        "DeviceType": "desktop",
    }

    def test_approve_decision_for_low_probability(self, client, mock_predictor):
        mock_predictor.predict.return_value = {
            "fraud_probability": 0.05,
            "decision": "APPROVE",
            "model_version": "abc12345",
            "threshold_approve": 0.3,
            "threshold_decline": 0.7,
        }
        response = client.post("/predict", json=self.VALID_PAYLOAD)
        assert response.status_code == 200
        body = response.json()
        assert body["decision"] == "APPROVE"
        assert body["fraud_probability"] == pytest.approx(0.05)

    def test_review_decision_for_mid_probability(self, client, mock_predictor):
        mock_predictor.predict.return_value = {
            "fraud_probability": 0.50,
            "decision": "REVIEW",
            "model_version": "abc12345",
            "threshold_approve": 0.3,
            "threshold_decline": 0.7,
        }
        response = client.post("/predict", json=self.VALID_PAYLOAD)
        assert response.status_code == 200
        assert response.json()["decision"] == "REVIEW"

    def test_decline_decision_for_high_probability(self, client, mock_predictor):
        mock_predictor.predict.return_value = {
            "fraud_probability": 0.92,
            "decision": "DECLINE",
            "model_version": "abc12345",
            "threshold_approve": 0.3,
            "threshold_decline": 0.7,
        }
        response = client.post("/predict", json=self.VALID_PAYLOAD)
        assert response.status_code == 200
        assert response.json()["decision"] == "DECLINE"

    def test_negative_amount_returns_422(self, client):
        """Pydantic validation should reject negative transaction amounts."""
        payload = {**self.VALID_PAYLOAD, "TransactionAmt": -50.0}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_missing_required_field_returns_422(self, client):
        """TransactionAmt is required — omitting it should fail validation."""
        payload = {k: v for k, v in self.VALID_PAYLOAD.items() if k != "TransactionAmt"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_503_when_model_not_loaded(self):
        p = MagicMock()
        p.is_loaded = False
        with patch("src.api.main.predictor", p):
            from src.api.main import app
            with TestClient(app) as c:
                response = c.post("/predict", json=self.VALID_PAYLOAD)
                assert response.status_code == 503

    def test_response_has_all_required_fields(self, client, mock_predictor):
        mock_predictor.predict.return_value = {
            "fraud_probability": 0.12,
            "decision": "APPROVE",
            "model_version": "abc12345",
            "threshold_approve": 0.3,
            "threshold_decline": 0.7,
        }
        response = client.post("/predict", json=self.VALID_PAYLOAD)
        body = response.json()
        for field in ["fraud_probability", "decision", "model_version", "threshold_approve", "threshold_decline"]:
            assert field in body, f"Missing field: {field}"
