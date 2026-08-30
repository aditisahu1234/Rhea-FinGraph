"""Layer 1 — historical velocity replay + velocity-trained model plumbing."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from fingraph_sentinel.velocity_replay import OUT_COLUMNS, replay_split


def _mini_timeline(path: Path) -> None:
    """Three customers with 3 same-hour transactions each (chronological)."""
    base = datetime(2020, 1, 1, 10, 0, 0)
    rows = []
    for c in ("c1", "c2", "c3"):
        for i in range(3):
            rows.append(
                {
                    "transaction_id": f"{c}-{i}",
                    "event_time": base + timedelta(minutes=5 * i),
                    "customer_id": c,
                    "card_id": f"{c}::card",
                    "merchant_id": f"m{i}",
                    "merchant_category_code": "5411",
                    "amount": float(100.0 + i),
                    "currency": "USD",
                    "payment_channel": "Swipe Transaction",
                    "is_fraud": 1 if i == 2 else 0,
                }
            )
    pl.DataFrame(rows).write_parquet(path)


def test_replay_strictly_past_and_window_accumulation(tmp_path: Path) -> None:
    src = tmp_path / "mini.parquet"
    _mini_timeline(src)
    res = replay_split(src, tmp_path / "out")
    assert res["rows"] == 9

    parts = sorted((tmp_path / "out").glob("part-*.parquet"))
    frame = pl.concat([pl.read_parquet(p) for p in parts]).sort("transaction_id")
    assert frame.columns == OUT_COLUMNS

    by_id = {r["transaction_id"]: r for r in frame.iter_rows(named=True)}
    # strictly-past: an event never counts itself
    assert by_id["c1-0"]["cust_v_1h_count"] == 0.0
    prior = by_id["c1-0"]["cust_txn_count_prior"]
    assert prior is None or prior == 0.0
    # third same-customer event within the hour sees the two previous ones
    assert by_id["c1-2"]["cust_v_1h_count"] == 2.0
    assert by_id["c1-2"]["cust_v_1h_amount"] == 201.0  # 100 + 101
    # distinct merchants: c1 visits m0, m1, m2
    assert by_id["c1-2"]["cust_v_7d_distinct_merchants"] == 2.0


def test_replay_backfills_missing_device_columns(tmp_path: Path) -> None:
    src = tmp_path / "mini.parquet"
    _mini_timeline(src)
    replay_split(src, tmp_path / "out")
    frame = pl.read_parquet(tmp_path / "out" / "part-0000.parquet")
    dev_cols = [c for c in frame.columns if c.startswith("device_")]
    assert dev_cols, "device velocity columns must exist even without device ids"
    for col in dev_cols:
        assert frame[col].null_count() == frame.height


def test_train_baseline_velocity_feature_set(tmp_path: Path) -> None:
    """Velocity feature set trains end-to-end on a capped slice and records
    the full ONLINE_VELOCITY column set in its config."""
    import json

    replay_split(
        Path("data/processed/ibm_full/train.parquet"),
        tmp_path / "vel" / "train",
        max_rows=3000,
    )
    replay_split(
        Path("data/processed/ibm_full/validation.parquet"),
        tmp_path / "vel" / "validation",
        max_rows=1200,
    )
    replay_split(
        Path("data/processed/ibm_full/test.parquet"),
        tmp_path / "vel" / "test",
        max_rows=1200,
    )

    from fingraph_sentinel.features import ONLINE_VELOCITY_FEATURE_COLUMNS
    from fingraph_sentinel.train_baseline import main as train_main

    args = [
        "--feature-set", "velocity",
        "--velocity-dir", str(tmp_path / "vel"),
        "--out", str(tmp_path / "model"),
        "--max-train-rows", "2000",
        "--max-val-rows", "500",
        "--max-test-rows", "500",
        "--backend", "xgboost",
        "--device", "cpu",
    ]
    # main() parses sys.argv; run it in-process with swapped argv.
    import sys  # noqa: PLC0415

    old = sys.argv
    sys.argv = ["train_baseline", *args]
    try:
        train_main()
    finally:
        sys.argv = old

    cfg = json.loads((tmp_path / "model" / "model_config.json").read_text())
    assert cfg["feature_columns"] == ONLINE_VELOCITY_FEATURE_COLUMNS
    assert cfg["model_name"] == "xgboost_velocity_v3"
    assert "roc_auc" in cfg["metrics_validation"]
    assert "roc_auc" in cfg["metrics_test_locked"]
    assert (tmp_path / "model" / "model.json").exists()