"""Layer 0 serving + runtime + API endpoint tests (real model when present)."""

from datetime import UTC, datetime

import pytest

from fingraph_sentinel.schemas import PaymentEvent


@pytest.fixture
def event() -> PaymentEvent:
    return PaymentEvent(
        transaction_id="tx-test-1",
        event_time=datetime(2020, 1, 15, 3, 30, tzinfo=UTC),
        customer_id="c1",
        card_id="card1",
        merchant_id="1334959",
        merchant_category_code="5411",
        amount="997.00",
        payment_channel="swipe",
    )


def _model_ready() -> bool:
    from pathlib import Path

    return Path("artifacts/models/baseline-online-xgb/model_config.json").exists()


@pytest.mark.skipif(not _model_ready(), reason="serving model not present")
def test_runtime_feature_dict_matches_columns(event):
    import json

    from fingraph_sentinel.runtime import event_feature_dict
    from fingraph_sentinel.serving import MODEL_DIR

    cfg = json.loads((MODEL_DIR / "model_config.json").read_text())
    values = event_feature_dict(event)
    assert set(values) == set(cfg["feature_columns"])
    # swipe channel must map to the swipe flag
    assert values.get("channel_swipe") == 1.0
    assert values.get("channel_online") == 0.0


@pytest.mark.skipif(not _model_ready(), reason="serving model not present")
def test_score_event_returns_decision_and_shap_reasons(event):
    import json

    from fingraph_sentinel.runtime import boilerplate_reasons, event_feature_dict
    from fingraph_sentinel.serving import MODEL_DIR, score_event

    cfg = json.loads((MODEL_DIR / "model_config.json").read_text())
    values = event_feature_dict(event)
    res = score_event(
        values,
        list(cfg["feature_columns"]),
        boilerplate_reasons=boilerplate_reasons(event),
    )
    assert res.action in ("allow", "review", "hold")
    assert 0.0 <= res.fraud_probability <= 1.0
    assert res.reasons, "expected at least the summary reason"
    # at least one reason should carry a SHAP magnitude
    assert any(r.magnitude is not None for r in res.reasons)


@pytest.mark.skipif(not _model_ready(), reason="serving model not present")
def test_helix_drift_endpoint_reports_features():
    from fastapi.testclient import TestClient

    from fingraph_sentinel.main import app

    client = TestClient(app)
    r = client.get("/api/v1/helix/drift")
    assert r.status_code == 200
    body = r.json()
    assert body["trigger"] in ("YES", "NO")
    assert body["features"] is not None


@pytest.mark.skipif(not _model_ready(), reason="serving model not present")
def test_score_endpoint_end_to_end(event):
    from fastapi.testclient import TestClient

    from fingraph_sentinel.main import app

    client = TestClient(app)
    r = client.post("/api/v1/transactions/score", json=event.model_dump(mode="json"))
    assert r.status_code == 200
    body = r.json()
    assert body["transaction_id"] == event.transaction_id
    assert body["action"] in ("allow", "review", "hold")
    assert "reasons" in body
