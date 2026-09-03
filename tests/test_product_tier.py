"""Tests for LIMITATION #3 (security actions + human reasons) and
LIMITATION #4 (cold-start routing) — plus the sprint endpoints
(/api/v1/impact/summary and /api/v1/razorpay/webhook).

All assertions check *behavior*, never fabricated numbers: security_action is a
pure 1:1 map of the model band, reasons only emit when data supports them, and
the cold-start route must be strictly more conservative than the model on an
unknown entity.
"""

from fastapi.testclient import TestClient

from fingraph_sentinel.cold_start import cold_start_risk, is_cold_start
from fingraph_sentinel.explainer_ui import human_reasons, security_action, verdict
from fingraph_sentinel.main import app

client = TestClient(app)


# ---- LIMITATION #3: security action mapping ------------------------------
def test_security_action_maps_bands_deterministically() -> None:
    assert security_action("allow") == "APPROVE"
    assert security_action("review") == "REQUEST_STEP_UP"
    assert security_action("hold") == "DECLINE"
    assert verdict("allow") == "APPROVED"
    assert verdict("hold") == "BLOCKED"


def test_human_reasons_emit_only_when_data_supports() -> None:
    values = {"cust_prev_amount_ratio": 8.4, "cust_v_1h_count": 12.0}
    reasons = human_reasons(values)
    assert any("8.4x" in r or "8.4 times" in r for r in reasons)
    assert any("12 transaction" in r for r in reasons)
    # absent features never produce fabricated clauses
    clean = human_reasons({"amount_log1p": 3.0})
    assert clean and "dominant risk factor" in clean[0]


def test_human_reasons_cold_start_variant() -> None:
    reasons = human_reasons({"amount_log1p": 8.0, "is_night": 1.0}, cold_start=True)
    assert all("unknown entity" in r or "history" in r for r in reasons)


# ---- LIMITATION #4: cold-start detection + conservative route --------------
def test_is_cold_start_true_when_any_entity_lacks_history() -> None:
    assert is_cold_start(None, 50.0, 100.0) is True      # customer unknown
    assert is_cold_start(50.0, None, 100.0) is True      # card unknown
    assert is_cold_start(50.0, 50.0, None) is True       # merchant unknown
    assert is_cold_start(4.0, 50.0, 100.0) is True       # below threshold
    assert is_cold_start(50.0, 50.0, 100.0) is False     # all warm


def test_cold_start_risk_returns_conservative_with_flag() -> None:
    r = cold_start_risk({"amount_log1p": 7.0, "is_night": 1.0, "channel_online": 1.0})
    assert r["is_cold_start"] is True
    assert r["action"] in ("hold", "review")  # high-risk unknowns never "allow"
    assert 0 <= r["risk_score"] <= 1
    assert r["reasons"]


def test_cold_start_risk_low_unknown_allows() -> None:
    r = cold_start_risk({"amount_log1p": 1.0, "is_night": 0.0,
                         "channel_online": 0.0, "channel_swipe": 0.0})
    assert r["action"] == "allow"  # low unknown value still proceeds
    assert r["risk_score"] < 0.6


# ---- score endpoint attaches the new fields -------------------------------
def test_score_model_path_includes_security_and_cold_flag(monkeypatch) -> None:
    from fingraph_sentinel.serving import ScoredReason, ScoreResult

    monkeypatch.setattr("fingraph_sentinel.main._model_ready", lambda: True)
    monkeypatch.setattr("fingraph_sentinel.main.is_cold_start", lambda *a, **k: False)

    def fake_score(values, feature_columns, boilerplate_reasons=None):
        return ScoreResult(
            transaction_id="txn_001", model_version="stub", fraud_probability=0.9,
            action="hold",
            reasons=[ScoredReason(feature="stub", direction="context", detail="x")],
        )

    monkeypatch.setattr("fingraph_sentinel.main.score_event", fake_score)
    resp = client.post("/api/v1/transactions/score", json={
        "transaction_id": "txn_001", "event_time": "2026-08-23T10:00:00Z",
        "customer_id": "c_warm", "card_id": "k_warm", "merchant_id": "m_warm",
        "amount": "499.00",
    })
    body = resp.json()
    assert body["security_action"] == "DECLINE"
    assert body["is_cold_start"] is False
    assert isinstance(body["reasons_human"], list)


def test_score_cold_start_routes_to_rules(monkeypatch) -> None:
    monkeypatch.setattr("fingraph_sentinel.main._model_ready", lambda: True)
    # force cold-start regardless of velocity priors
    monkeypatch.setattr("fingraph_sentinel.main.is_cold_start", lambda *a, **k: True)
    resp = client.post("/api/v1/transactions/score", json={
        "transaction_id": "txn_cold", "event_time": "2026-08-23T10:00:00Z",
        "customer_id": "c_brand_new", "card_id": "k_brand_new",
        "merchant_id": "m_brand_new", "amount": "499.00",
    })
    body = resp.json()
    assert body["is_cold_start"] is True
    assert body["model_version"] == "cold-start-rules"
    assert body["security_action"] in ("APPROVE", "REQUEST_STEP_UP", "DECLINE")


# ---- sprint endpoints ------------------------------------------------------
def test_impact_summary_serves_verified_numbers() -> None:
    resp = client.get("/api/v1/impact/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert body["total_protected_inr"] is not None
    assert body["fraud_amount_blocked_rate"] is not None  # ~0.96
    assert body["fraud_events_blocked_rate"] is not None  # ~0.886
    assert 0 <= body["fraud_amount_blocked_rate"] <= 1
    assert 0 <= body["fraud_events_blocked_rate"] <= 1


def test_razorpay_webhook_endpoint_scores() -> None:
    resp = client.post("/api/v1/razorpay/webhook", json={
        "order_id": "order_wh_1", "payment_id": "pay_wh_1",
        "amount": 199900, "currency": "INR",
        "customer": {"id": "C-WH-1"}, "card": {"id": "K-WH-1"},
        "merchant": {"id": "TerraMart-5311"}, "method": "card",
    })
    body = resp.json()
    assert body["received"] is True
    assert body["risk"]["decision"] in ("allow", "review", "hold")
    assert body["risk"]["security_action"] in (
        "APPROVE", "REQUEST_STEP_UP", "DECLINE")
    assert body["risk"]["is_cold_start"] in (True, False)
    assert body["audit"]["decision_auditable"] is True
