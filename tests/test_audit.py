"""Layer 6 audit ledger: hash-chain integrity, fail-safe, API endpoints."""

from __future__ import annotations

from fingraph_sentinel.audit import GENESIS_HASH, InMemoryLedger, Ledger, sha256
from fingraph_sentinel.schemas import AuditRecord


def _ledger() -> Ledger:
    return Ledger(InMemoryLedger())


def test_append_chains_hashes() -> None:
    led = _ledger()
    r1 = led.append("decision.scored", {"transaction_id": "t1", "action": "allow"})
    r2 = led.append("decision.scored", {"transaction_id": "t2", "action": "hold"})

    assert r1["prev_hash"] == GENESIS_HASH
    assert r2["prev_hash"] == r1["hash"]
    assert r1["hash"] != r2["hash"]
    assert r1["payload"]["transaction_id"] == "t1"
    assert led.count() == 2


def test_verify_valid_on_untampered_chain() -> None:
    led = _ledger()
    for i in range(5):
        led.append("decision.scored", {"transaction_id": f"t{i}", "action": "review"})
    rep = led.verify()
    assert rep["valid"] is True
    assert rep["records"] == 5


def test_verify_detects_payload_tamper() -> None:
    led = _ledger()
    for i in range(3):
        led.append("decision.scored", {"transaction_id": f"t{i}", "action": "allow"})
    # Tamper with a stored payload in place (simulating someone editing the log).
    store = led.store
    assert isinstance(store, InMemoryLedger)
    store._records[1]["payload"]["action"] = "review"  # noqa: SLF001 - test hook
    rep = led.verify()
    assert rep["valid"] is False
    assert rep["first_broken_index"] == 1


def test_verify_detects_deleted_record() -> None:
    led = _ledger()
    for i in range(4):
        led.append("decision.scored", {"transaction_id": f"t{i}", "action": "allow"})
    store = led.store
    assert isinstance(store, InMemoryLedger)
    store._records.pop(2)  # noqa: SLF001 - test hook - remove the middle record
    rep = led.verify()
    assert rep["valid"] is False


def test_fail_safe_append_buffers_when_store_breaks() -> None:
    led = _ledger()
    # Append once fine, then break the store so the second append must not raise.
    led.append("decision.scored", {"transaction_id": "t0"})

    class BrokenStore(InMemoryLedger):
        def append(self, record) -> None:  # noqa: ANN001
            raise ConnectionError("db gone")

    led.store = BrokenStore()
    sentinels = []
    try:
        r = led.append("decision.scored", {"transaction_id": "t1"})
        sentinels.append(r)
    except ConnectionError:  # pragma: no cover - must not escape
        sentinels.append(None)

    assert sentinels and sentinels[0] is not None
    assert led.store.is_healthy() is False
    h = led.health()
    assert h["healthy"] is False
    assert h["buffered"] == 1
    # The buffered record is still surfaced on reads + counted.
    assert led.count() == 1
    assert led.recent(1)[0]["payload"]["transaction_id"] == "t1"


def test_recent_orders_newest_first() -> None:
    led = _ledger()
    for i in range(3):
        led.append("decision.scored", {"transaction_id": f"t{i}"})
    recent = led.recent(2)
    assert [r["payload"]["transaction_id"] for r in recent] == ["t2", "t1"]


def test_records_map_to_schema() -> None:
    led = _ledger()
    led.append("decision.scored", {"transaction_id": "t1", "action": "hold"})
    rec = led.recent(1)[0]
    parsed = AuditRecord(**rec)
    assert parsed.event_type == "decision.scored"
    assert parsed.payload["action"] == "hold"
    assert isinstance(parsed.hash, str) and len(parsed.hash) == 64


def test_sha256_stable() -> None:
    assert sha256("a") == sha256("a")
    assert sha256("a") != sha256("b")


def test_default_falls_back_to_memory_without_dsn() -> None:
    led = Ledger.default()
    assert isinstance(led.store, InMemoryLedger)
    led.append("decision.scored", {"transaction_id": "t1"})
    assert led.count() == 1


# ---- API-level: decision scoring audits + observability endpoints -------


def _fresh_api_ledger(monkeypatch) -> Ledger:
    """Point the app singleton at a fresh in-memory ledger for the test."""
    led = Ledger(InMemoryLedger())
    monkeypatch.setattr("fingraph_sentinel.main.get_ledger", lambda: led)
    return led


def test_score_route_audits_failsafe_decision(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from fingraph_sentinel.schemas import PaymentEvent

    led = _fresh_api_ledger(monkeypatch)
    monkeypatch.setattr("fingraph_sentinel.main._model_ready", lambda: False)

    from fingraph_sentinel.main import app

    client = TestClient(app)
    ev = PaymentEvent(
        transaction_id="tx-audit-1",
        event_time="2020-01-15T03:30:00Z",
        customer_id="c1",
        card_id="card1",
        merchant_id="1334959",
        amount="997.00",
    )
    resp = client.post("/api/v1/transactions/score", json=ev.model_dump(mode="json"))
    assert resp.status_code == 200
    assert resp.json()["action"] == "review"  # failsafe (no model)

    assert led.count() == 1
    rec = led.recent(1)[0]
    assert rec["event_type"] == "decision.review_failsafe"
    assert rec["payload"]["transaction_id"] == "tx-audit-1"
    assert rec["payload"]["action"] == "review"
    assert rec["prev_hash"] == GENESIS_HASH
    assert led.verify()["valid"] is True


def test_observability_endpoints_report_audit(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from fingraph_sentinel.main import app

    led = _fresh_api_ledger(monkeypatch)
    led.append("decision.scored", {"transaction_id": "t1", "action": "allow"})
    led.append("decision.scored", {"transaction_id": "t2", "action": "hold"})

    client = TestClient(app)

    r = client.get("/api/v1/audit/health")
    assert r.status_code == 200
    assert r.json()["healthy"] is True
    assert r.json()["total"] == 2

    r = client.get("/api/v1/audit/recent?limit=10")
    assert r.status_code == 200
    entries = r.json()
    assert len(entries) == 2
    assert entries[0]["payload"]["transaction_id"] == "t2"

    r = client.get("/api/v1/audit/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    assert body["valid"] is True

    r = client.get("/api/v1/audit/verify")
    assert r.status_code == 200
    assert r.json()["valid"] is True
    assert r.json()["records"] == 2


def test_ledger_daily_rollup() -> None:
    led = _ledger()
    led.append("decision.scored", {"transaction_id": "t1", "action": "allow"})
    led.append("decision.scored", {"transaction_id": "t2", "action": "hold"})
    led.append("decision.review_failsafe", {"transaction_id": "t3"})
    roll = led.daily(days=7)
    assert roll, "expected at least one day bucket"
    top = roll[0]
    assert top["total"] == 3
    assert top["by_action"].get("allow") == 1
    assert top["by_action"].get("hold") == 1
    assert top["by_event"].get("decision.scored") == 2


def test_audit_daily_endpoint(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from fingraph_sentinel.main import app

    led = _fresh_api_ledger(monkeypatch)
    for i in range(3):
        led.append("decision.scored", {"transaction_id": f"t{i}", "action": "review"})
    client = TestClient(app)
    r = client.get("/api/v1/audit/daily?days=7")
    assert r.status_code == 200
    roll = r.json()
    assert len(roll) == 1
    assert roll[0]["total"] == 3
