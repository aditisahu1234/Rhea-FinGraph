"""Layer 5 v2 — healing API endpoints (feedback / memory / status / heal)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import fingraph_sentinel.main as main
from fingraph_sentinel.audit import InMemoryLedger, Ledger
from fingraph_sentinel.healing import HealingEngine
from fingraph_sentinel.streaming import InMemoryBackend, VelocityFeatureService, VelocityStore

EVENT = {
    "transaction_id": "tx-heal-1",
    "event_time": "2026-08-23T10:00:00Z",
    "customer_id": "C-1",
    "card_id": "K-1",
    "merchant_id": "m-heal",
    "amount": "99.00",
}


def _fresh_api(tmp_path: Path) -> TestClient:
    """Isolated engine dirs + in-memory ledger/velocity so tests never touch
    repo artifacts."""
    main._ledger = Ledger(InMemoryLedger())
    main._velocity = VelocityFeatureService(VelocityStore(InMemoryBackend()))
    main._healing = HealingEngine(
        model_dir=tmp_path / "model",
        healing_dir=tmp_path / "healing",
    )
    (tmp_path / "model").mkdir(parents=True, exist_ok=True)
    return TestClient(main.app)


def test_feedback_against_audited_decision(tmp_path: Path) -> None:
    c = _fresh_api(tmp_path)
    r = c.post("/api/v1/transactions/score", json=EVENT)
    assert r.status_code == 200
    action = r.json()["action"]
    assert action in ("allow", "review", "hold")

    fb = c.post(
        "/api/v1/healing/feedback",
        json={"transaction_id": "tx-heal-1", "outcome": "fraud"},
    )
    assert fb.status_code == 200
    ep = fb.json()["episode"]
    assert ep["transaction_id"] == "tx-heal-1"
    assert ep["outcome"] == "fraud"
    # only an "allow" decision that turns out fraud is a remembered miss
    assert ep["fail_type"] == ("missed_fraud" if action == "allow" else None)

    mem = c.get("/api/v1/healing/memory").json()
    assert mem["stats"]["episodes"] == 1
    assert mem["stats"]["failures"] == (1 if action == "allow" else 0)


def test_feedback_unknown_transaction_returns_error(tmp_path: Path) -> None:
    c = _fresh_api(tmp_path)
    r = c.post("/api/v1/healing/feedback", json={"transaction_id": "nope", "outcome": "fraud"})
    assert r.json()["ok"] is False
    assert "no audited decision" in r.json()["error"]


def test_status_and_heal_cycle(tmp_path: Path) -> None:
    c = _fresh_api(tmp_path)
    # score two txns at the same merchant and confirm their real actions
    acts: list[str] = []
    for i in (2, 3):
        ev = dict(EVENT, transaction_id=f"tx-heal-{i}", merchant_id="m-hot")
        acts.append(c.post("/api/v1/transactions/score", json=ev).json()["action"])
    for i in (2, 3):
        r = c.post(
            "/api/v1/healing/feedback",
            json={"transaction_id": f"tx-heal-{i}", "outcome": "fraud"},
        )
        assert r.json()["ok"] is True

    failures = sum(1 for a in acts if a == "allow")
    heal = c.post("/api/v1/healing/heal").json()
    assert heal["memory"]["failures"] == failures
    assert len(heal["hot_merchants"]) == (1 if failures >= 2 else 0)
    assert heal["retrain_queued"] == (failures >= 2)

    st = c.get("/api/v1/healing/status").json()
    assert st["memory"]["episodes"] == 2
    assert st["retrain_queue_len"] == (1 if failures >= 2 else 0)
    assert st["heal_report_exists"] is True


def test_heal_report_persisted(tmp_path: Path) -> None:
    c = _fresh_api(tmp_path)
    c.post("/api/v1/transactions/score", json=EVENT)
    c.post("/api/v1/healing/feedback", json={"transaction_id": "tx-heal-1", "outcome": "fraud"})
    c.post("/api/v1/healing/heal")
    assert (tmp_path / "model" / "heal_report.json").exists()