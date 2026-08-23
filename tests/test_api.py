from fastapi.testclient import TestClient

from fingraph_sentinel.main import app

client = TestClient(app)


def test_liveness() -> None:
    response = client.get("/api/v1/health/live")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_score_contract_returns_safe_review_before_model_exists() -> None:
    response = client.post(
        "/api/v1/transactions/score",
        json={
            "transaction_id": "txn_001",
            "event_time": "2026-08-23T10:00:00Z",
            "customer_id": "customer_a",
            "card_id": "card_a",
            "merchant_id": "merchant_a",
            "amount": "499.00",
        },
    )

    assert response.status_code == 202
    assert response.json()["action"] == "review"
    assert response.json()["is_model_ready"] is False
