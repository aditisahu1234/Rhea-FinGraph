"""Layer 0 API behaviour tests (new serving architecture, no real model needed).

Monkeypatches the new seams (_model_ready, score_event, load_helix_drift) so
these tests are deterministic and do not require a trained model on disk.
"""

from pathlib import Path

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


def _set_model_ready(monkeypatch, ready: bool):
    monkeypatch.setattr("fingraph_sentinel.main._model_ready", lambda: ready)


def test_liveness() -> None:
    response = client.get("/api/v1/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_ready_reflects_model_presence(monkeypatch) -> None:
    _set_model_ready(monkeypatch, True)
    assert client.get("/api/v1/health/ready").json()["model_registered"] is True
    _set_model_ready(monkeypatch, False)
    assert client.get("/api/v1/health/ready").json()["model_registered"] is False


def test_model_status_untrained_when_no_model(monkeypatch) -> None:
    _set_model_ready(monkeypatch, False)
    body = client.get("/api/v1/model/status").json()
    assert body["ready"] is False


def test_score_falls_back_to_review_without_model(monkeypatch) -> None:
    _set_model_ready(monkeypatch, False)
    response = client.post("/api/v1/transactions/score", json=SCORE_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["action"] == "review"
    assert body["is_model_ready"] is False


def test_score_passes_through_model_result(monkeypatch) -> None:
    from fingraph_sentinel.serving import ScoredReason, ScoreResult

    _set_model_ready(monkeypatch, True)
    # warm entity so the model path (not cold-start rules) is exercised
    monkeypatch.setattr("fingraph_sentinel.main.is_cold_start", lambda *a, **k: False)

    def fake_score(values, feature_columns, boilerplate_reasons=None):
        return ScoreResult(
            transaction_id="txn_001",
            model_version="stub_v9",
            fraud_probability=0.42,
            action="review",
            reasons=[
                ScoredReason(
                    feature="stub",
                    direction="context",
                    detail="deterministic stub",
                )
            ],
        )

    monkeypatch.setattr("fingraph_sentinel.main.score_event", fake_score)
    response = client.post("/api/v1/transactions/score", json=SCORE_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["is_model_ready"] is True
    assert body["model_version"] == "stub_v9"
    assert abs(body["fraud_probability"] - 0.42) < 1e-6
    assert body["action"] == "review"
    assert body["reasons"][0]["feature"] == "stub"


def test_score_fails_safe_when_scoring_errors(monkeypatch) -> None:
    _set_model_ready(monkeypatch, True)
    monkeypatch.setattr("fingraph_sentinel.main.is_cold_start", lambda *a, **k: False)

    def broken_score(values, feature_columns, boilerplate_reasons=None):
        raise RuntimeError("boom")

    monkeypatch.setattr("fingraph_sentinel.main.score_event", broken_score)
    response = client.post("/api/v1/transactions/score", json=SCORE_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    # scoring must fail safe, never fail open
    assert body["action"] == "review"
    assert body["is_model_ready"] is False


def test_helix_drift_empty_default(monkeypatch) -> None:
    monkeypatch.setattr("fingraph_sentinel.main.load_helix_drift", lambda: None)
    body = client.get("/api/v1/helix/drift").json()
    assert body["trigger"] == "NO"


# ---- Layer 2 graph status -------------------------------------------------


def test_graph_status_shape_and_neo4j_flag() -> None:
    body = client.get("/api/v1/graph/status").json()
    assert "neo4j" in body
    assert body["neo4j"]["url"].startswith("bolt://")
    assert body["neo4j"]["reachable"] in (True, False)  # honest either way
    assert "pipeline" in body
    assert "snapshots" in body["pipeline"]
    assert "top_merchants" in body["pipeline"]
    for row in body["pipeline"]["top_merchants"]:
        assert "merchant_id" in row and "failures" in row
    # pipeline totals must be numbers when a meta.json exists locally
    total = body["pipeline"].get("total_edges")
    assert total is None or total >= 0
    # gnn summary is optional but shape-consistent when present
    gnn = body.get("gnn")
    assert gnn is None or "architecture" in gnn


def test_graph_status_no_pipeline_when_no_meta(monkeypatch) -> None:
    monkeypatch.setattr(
        "fingraph_sentinel.main._best_graph_meta", lambda: (None, Path(""))
    )
    body = client.get("/api/v1/graph/status").json()
    assert body["pipeline"]["source"] == "none"
    assert body["pipeline"].get("n_merchants") is None
    assert body["gnn"] is None


def test_model_race_shape_and_serving_row() -> None:
    body = client.get("/api/v1/model/race").json()
    assert "models" in body and isinstance(body["models"], list)
    assert "gate_report" in body
    serving = [m for m in body["models"] if m["role"] == "serving"]
    assert any(m["name"] == "baseline-online-xgb" for m in serving)
    for m in body["models"]:
        assert set(("name", "val_roc", "test_roc", "role")) <= set(m)


def test_model_switcher_status_shape() -> None:
    body = client.get("/api/v1/model/switcher/status").json()
    assert body["serving_model"] == "baseline-online-xgb"
    assert "last_decision" in body
    assert "drift_report" in body
