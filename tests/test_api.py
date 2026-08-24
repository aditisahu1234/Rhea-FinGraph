from fastapi.testclient import TestClient

from fingraph_sentinel.main import app

client = TestClient(app)

SCORE_PAYLOAD = {
    "transaction_id": "txn_001",
    "event_time": "2026-08-23T10:00:00Z",
    "customer_id": "customer_a",
    "card_id": "card_a",
    "merchant_id": "merchant_a",
    "amount": "499.00",
}


def _force_no_registry(monkeypatch):
    monkeypatch.setattr("fingraph_sentinel.main.get_registry", lambda: None)


def test_liveness() -> None:
    response = client.get("/api/v1/health/live")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_status_reports_untrained_when_registry_missing(monkeypatch) -> None:
    _force_no_registry(monkeypatch)
    response = client.get("/api/v1/model/status")

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is False


def test_score_contract_returns_safe_review_before_model_exists(monkeypatch) -> None:
    _force_no_registry(monkeypatch)
    response = client.post("/api/v1/transactions/score", json=SCORE_PAYLOAD)

    assert response.status_code == 202
    assert response.json()["action"] == "review"
    assert response.json()["is_model_ready"] is False


def test_score_uses_registered_model(monkeypatch) -> None:
    from fingraph_sentinel.model_registry import ScoreOutcome
    from fingraph_sentinel.schemas import RiskReason

    class StubRegistry:
        def score_event(self, event):
            return ScoreOutcome(
                probability=0.42,
                weighted_probability=0.9,
                action="review",
                reasons=[
                    RiskReason(
                        feature="stub", direction="context", detail="deterministic stub"
                    )
                ],
                model_version="stub_v9",
            )

    monkeypatch.setattr("fingraph_sentinel.main.get_registry", lambda: StubRegistry())
    response = client.post("/api/v1/transactions/score", json=SCORE_PAYLOAD)

    assert response.status_code == 202
    body = response.json()
    assert body["is_model_ready"] is True
    assert body["model_version"] == "stub_v9"
    assert abs(body["fraud_probability"] - 0.42) < 1e-6
    assert body["action"] == "review"
    assert body["reasons"][0]["feature"] == "stub"


def test_score_fails_safe_when_registry_errors(monkeypatch) -> None:
    class BrokenRegistry:
        def score_event(self, event):
            raise RuntimeError("boom")

    monkeypatch.setattr("fingraph_sentinel.main.get_registry", lambda: BrokenRegistry())
    response = client.post("/api/v1/transactions/score", json=SCORE_PAYLOAD)

    assert response.status_code == 202
    body = response.json()
    assert body["action"] == "review"
    assert body["is_model_ready"] is False
