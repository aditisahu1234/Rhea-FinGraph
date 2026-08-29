"""Layer 5 v2 — healing engine: hot-list, threshold overrides, retrain queue."""

from __future__ import annotations

import json
from pathlib import Path

from fingraph_sentinel.failure_memory import Episode, FailureMemory
from fingraph_sentinel.healing import (
    HOTLIST_NAME,
    RETRAIN_QUEUE_NAME,
    THRESHOLD_OVERRIDE_NAME,
    HealingEngine,
    load_retrain_queue,
)


def make_model_dir(tmp_path: Path, hold: float = 0.002, review: float = 0.001) -> Path:
    d = tmp_path / "model"
    d.mkdir(parents=True, exist_ok=True)
    (d / "model_config.json").write_text(
        json.dumps(
            {
                "model_name": "test_online_v1",
                "model_file": "model.json",
                "thresholds": {"hold": hold, "review": review},
                "feature_columns": ["amount_log1p"],
            }
        ),
        encoding="utf-8",
    )
    return d


def _feed(memory: FailureMemory, txid: str, outcome: str, mid: str) -> None:
    memory.record(
        Episode(
            transaction_id=txid,
            model_version="test_online_v1",
            action="allow" if outcome == "legit" else "allow",
            fraud_probability=0.0005,
            outcome=outcome,
            event={"merchant_id": mid, "amount": "200.00", "payment_channel": "swipe"},
        )
    )


def test_heal_writes_hotlist_override_and_retrain_queue(tmp_path: Path) -> None:
    model_dir = make_model_dir(tmp_path)
    healing_dir = tmp_path / "healing"
    mem = FailureMemory(healing_dir / "failure_memory.jsonl")
    # 3 episodes, 2 missed frauds at m1 -> miss_rate 0.667 -> all three actions
    _feed(mem, "t1", "fraud", "m1")
    _feed(mem, "t2", "fraud", "m1")
    _feed(mem, "t3", "legit", "m2")
    engine = HealingEngine(memory=mem, model_dir=model_dir, healing_dir=healing_dir)

    report = engine.heal()

    assert report["memory"]["failures"] == 2
    assert len(report["hot_merchants"]) == 1
    assert report["hot_merchants"][0]["merchant_id"] == "m1"
    assert report["retrain_queued"] is True

    # hot-list file next to the model
    hot = json.loads((model_dir / HOTLIST_NAME).read_text(encoding="utf-8"))
    assert hot["merchants"][0]["merchant_id"] == "m1"

    # threshold override tightened (miss rate high)
    over = json.loads((model_dir / THRESHOLD_OVERRIDE_NAME).read_text(encoding="utf-8"))
    assert over["hold"] > 0.002  # tightened above the base hold

    # retrain queue durable + deduped
    queue = load_retrain_queue(healing_dir / RETRAIN_QUEUE_NAME)
    assert len(queue) >= 1
    engine.heal()  # second same-day cycle must not duplicate requests
    assert len(load_retrain_queue(healing_dir / RETRAIN_QUEUE_NAME)) == len(queue)


def test_heal_clears_override_when_rates_recover(tmp_path: Path) -> None:
    model_dir = make_model_dir(tmp_path)
    healing_dir = tmp_path / "healing"
    mem = FailureMemory(healing_dir / "failure_memory.jsonl")
    _feed(mem, "t1", "fraud", "m1")
    engine = HealingEngine(memory=mem, model_dir=model_dir, healing_dir=healing_dir)
    engine.heal()
    assert (model_dir / THRESHOLD_OVERRIDE_NAME).exists()

    # 40 legit allows -> rates drop well below the hysteresis floor
    for i in range(40):
        mem.record(
            Episode(
                transaction_id=f"ok-{i}",
                model_version="test_online_v1",
                action="allow",
                fraud_probability=0.0001,
                outcome="legit",
                event={"merchant_id": "m1", "amount": "10.00"},
            )
        )
    engine.heal()
    assert not (model_dir / THRESHOLD_OVERRIDE_NAME).exists()


def test_repair_dataset_and_skip_when_too_small(tmp_path: Path) -> None:
    model_dir = make_model_dir(tmp_path)
    healing_dir = tmp_path / "healing"
    mem = FailureMemory(healing_dir / "failure_memory.jsonl")
    _feed(mem, "t1", "fraud", "m1")
    engine = HealingEngine(memory=mem, model_dir=model_dir, healing_dir=healing_dir)

    # too few rows -> honest skip, no model emitted
    res = engine.train_repair(out_dir=tmp_path / "repair")
    assert res["trained"] is False
    assert "only" in res["reason"]
    assert not (tmp_path / "repair" / "model.json").exists()

    # enough rows with positives -> capped train succeeds
    for i in range(9):
        _feed(mem, f"t{i}", "fraud" if i % 2 else "legit", f"m{i}")
    res = engine.train_repair(out_dir=tmp_path / "repair", max_rows=100)
    assert res["trained"] is True
    assert res["positives"] >= 1
    assert (tmp_path / "repair" / "model.json").exists()
    assert (tmp_path / "repair" / "model_config.json").exists()


def test_stats_surface(tmp_path: Path) -> None:
    model_dir = make_model_dir(tmp_path)
    healing_dir = tmp_path / "healing"
    mem = FailureMemory(healing_dir / "failure_memory.jsonl")
    _feed(mem, "t1", "fraud", "m1")
    _feed(mem, "t2", "fraud", "m2")
    _feed(mem, "t3", "legit", "m2")
    engine = HealingEngine(memory=mem, model_dir=model_dir, healing_dir=healing_dir)
    engine.heal()

    s = engine.stats()
    assert s["memory"]["episodes"] == 3
    assert s["memory"]["failures"] == 2
    assert s["retrain_queue_len"] == 1  # failures crossed the floor once
    assert s["drift"] is None  # no helix report in the test model dir
    assert s["heal_report_exists"] is True
    assert (model_dir / "heal_report.json").exists()