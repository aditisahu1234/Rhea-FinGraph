"""Layer 5 v2 — failure memory (durable episodic store of feedback)."""

from __future__ import annotations

from pathlib import Path

from fingraph_sentinel.failure_memory import Episode, FailureMemory


def _ep(txid: str, outcome: str, action: str = "allow", mid: str = "m1") -> Episode:
    return Episode(
        transaction_id=txid,
        model_version="xgb_v1",
        action=action,
        fraud_probability=0.001,
        outcome=outcome,
        event={"merchant_id": mid, "amount": "100.00", "payment_channel": "swipe"},
        reasons=["amount_log1p"],
    )


def test_record_and_reload_is_durable(tmp_path: Path) -> None:
    mem = FailureMemory(tmp_path / "mem.jsonl")
    mem.record(_ep("t1", "fraud"))
    mem.record(_ep("t2", "legit"))

    # A fresh instance reading the same file sees the same episodes.
    reloaded = FailureMemory(tmp_path / "mem.jsonl")
    assert [e.transaction_id for e in reloaded.episodes()] == ["t1", "t2"]
    assert (tmp_path / "mem.jsonl").exists()


def test_failure_taxonomy() -> None:
    missed = _ep("t1", "fraud", action="allow")
    false_hold = _ep("t2", "legit", action="hold")
    ok_fraud = _ep("t3", "fraud", action="hold")  # caught
    ok_legit = _ep("t4", "legit", action="allow")
    assert missed.fail_type == "missed_fraud" and missed.is_failure
    assert false_hold.fail_type == "false_hold" and false_hold.is_failure
    assert ok_fraud.is_failure is False
    assert ok_legit.is_failure is False


def test_stats_and_hot_merchants(tmp_path: Path) -> None:
    mem = FailureMemory(tmp_path / "mem.jsonl")
    mem.record(_ep("t1", "fraud", mid="m1"))
    mem.record(_ep("t2", "fraud", mid="m1"))
    mem.record(_ep("t3", "legit", mid="m2"))
    mem.record(_ep("t4", "legit", action="allow", mid="m1"))  # not a failure

    s = mem.stats()
    assert s["episodes"] == 4
    assert s["failures"] == 2
    assert s["missed_fraud"] == 2
    assert s["false_hold"] == 0
    assert s["hot_merchants"] == 1

    hot = mem.hot_merchants(min_failures=2)
    assert len(hot) == 1 and hot[0]["merchant_id"] == "m1"
    assert hot[0]["failures"] == 2

    roll = mem.merchant_rollup()
    assert roll["m1"]["txns"] == 3
    assert roll["m1"]["failures"] == 2


def test_clear(tmp_path: Path) -> None:
    mem = FailureMemory(tmp_path / "mem.jsonl")
    mem.record(_ep("t1", "fraud"))
    assert len(mem.episodes()) == 1
    mem.clear()
    assert mem.episodes() == []
    assert not mem._path.exists()  # noqa: SLF001


def test_corrupt_lines_are_skipped(tmp_path: Path) -> None:
    p = tmp_path / "mem.jsonl"
    p.write_text(
        '{"transaction_id": "ok", "action": "allow", "outcome": "legit", '
        '"event": {}, "reasons": []}\nnot-json\n',
        encoding="utf-8",
    )
    mem = FailureMemory(p)
    assert len(mem.episodes()) == 1