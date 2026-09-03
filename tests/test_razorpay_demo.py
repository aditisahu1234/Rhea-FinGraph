"""Tests for the Razorpay demo adapter + business impact endpoint.

These are LIMITATION #1 / #2 acceptance tests:
- business impact serves the parity-verified operating point from disk;
- the Razorpay-style flow (create order -> pay -> webhook) runs through the
  SAME scoring/velocity/audit path and always returns an auditable decision,
  failing safe to `review` when no model is registered (never fail-open).
"""

from fastapi.testclient import TestClient

from fingraph_sentinel.main import app
from fingraph_sentinel.razorpay_demo import _ORDERS, build_webhook, create_order
from fingraph_sentinel.schemas import RiskDecision, RiskReason

client = TestClient(app)


def _decision(action: str = "allow", proba: float = 0.01) -> RiskDecision:
    return RiskDecision(
        transaction_id="pay_abc123",
        model_version="xgboost_online_v2",
        fraud_probability=proba,
        action=action,  # type: ignore[arg-type]
        reasons=[
            RiskReason(feature="amount_log1p", direction="increases_risk",
                       detail="amount pushed 0.5 toward fraud", magnitude=0.5)
        ],
        is_model_ready=True,
        processed_at="2026-08-30T10:00:00Z",
    )


# ---- LIMITATION #1: business operating point ----------------------------
def test_business_impact_endpoint_serves_from_disk() -> None:
    resp = client.get("/api/v1/business/impact")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("available") is True
    # the parity-verified protection block is the whole point
    assert "protection" in body
    assert "recall_by_count" in body["protection"]
    assert "recall_by_amount" in body["protection"]
    assert "per_month_protected_inr" in body["protection"]


# ---- LIMITATION #2: Razorpay demo adapter -------------------------------
def test_create_order_returns_event_and_registers_uuid() -> None:
    _ORDERS.clear()
    resp = client.post("/api/v1/razorpay/order",
                       json={"amount_inr": "1999.00", "merchant_id": "TerraMart-5311"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["order_id"].startswith("order_")
    assert body["currency"] == "INR"
    assert body["event"]["amount"]  # canonical USD amount derived from INR
    assert body["order_id"] in _ORDERS


def test_create_order_bad_amount_returns_clean_error() -> None:
    resp = client.post("/api/v1/razorpay/order", json={"amount_inr": "not-a-number"})
    assert resp.status_code == 200
    assert "error" in resp.json()


def test_pay_without_model_fails_safe_to_review(monkeypatch) -> None:
    # score_transaction path is already covered elsewhere; here we force the
    # model-off path by monkeypatching the readiness probe to False.
    monkeypatch.setattr("fingraph_sentinel.main._model_ready", lambda: False)
    _ORDERS.clear()
    o = create_order("499.00", "TerraMart-5311")
    resp = client.post("/api/v1/razorpay/pay", json={"order_id": o["order_id"]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["risk_assessment"]["action"] == "review"  # fail-safe, not allow
    assert body["audit"]["decision_auditable"] is True


def test_pay_unknown_order_404() -> None:
    resp = client.post("/api/v1/razorpay/pay", json={"order_id": "order_does_not_exist"})
    assert resp.status_code == 404


def test_build_webhook_verdict_and_reasons() -> None:
    o = create_order("2500.00", "TerraMart-5311")
    wh = build_webhook(o, _decision(action="hold", proba=0.85))
    assert wh["event"] == "payment.risk_flagged"
    assert wh["risk_assessment"]["fraud_verdict"] == "MANUAL_HOLD"
    assert wh["risk_assessment"]["top_reasons"][0]["feature"] == "amount_log1p"
    assert wh["order"]["order_id"] == o["order_id"]


def test_build_webhook_allow_verdict() -> None:
    o = create_order("100.00", "GoGrocer-5411")
    wh = build_webhook(o, _decision(action="allow", proba=0.001))
    assert wh["event"] == "payment.autocapture.succeeded"
    assert wh["risk_assessment"]["fraud_verdict"] == "APPROVED"


def test_razorpay_flow_endpoint_describes_lifecycle() -> None:
    resp = client.get("/api/v1/razorpay/flow")
    assert resp.status_code == 200
    assert len(resp.json()["flow"]) >= 5
    assert "XGBoost" in " ".join(resp.json()["flow"])
