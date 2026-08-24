"""Causality guarantees of the feature engine, verified on tiny fixtures."""

from datetime import UTC, datetime

import polars as pl

from fingraph_sentinel.features import FEATURE_COLUMNS, build_feature_frame


def _sample_frame() -> pl.DataFrame:
    ts = lambda day, hour: datetime(2020, 1, day, hour, tzinfo=UTC)  # noqa: E731
    return pl.DataFrame(
        {
            "transaction_id": ["t1", "t2", "t3", "t4"],
            "event_time": [ts(1, 10), ts(1, 11), ts(2, 9), ts(3, 8)],
            "customer_id": ["c1", "c1", "c1", "c2"],
            "card_id": ["k1", "k1", "k1", "k2"],
            "merchant_id": ["m1", "m1", "m2", "m1"],
            "merchant_category_code": ["5411", "5411", "5945", "5411"],
            "amount": [100.0, 50.0, 200.0, 75.0],
            "currency": ["USD", "USD", "USD", "USD"],
            "payment_channel": [
                "Swipe Transaction",
                "Online Transaction",
                "Chip Transaction",
                None,
            ],
            "payment_error": [None, "Insufficient Balance", None, None],
            "is_fraud": [0, 0, 1, 0],
        }
    )


def _row(frame: pl.DataFrame, txn_id: str) -> dict[str, object]:
    return frame.filter(pl.col("transaction_id") == txn_id).row(0, named=True)


def test_feature_columns_all_present_and_ordered() -> None:
    out = build_feature_frame(_sample_frame().lazy()).collect()
    assert all(name in out.columns for name in FEATURE_COLUMNS)


def test_first_customer_transaction_has_no_history() -> None:
    out = build_feature_frame(_sample_frame().lazy()).collect()
    t1 = _row(out, "t1")
    assert t1["cust_txn_count_prior"] == 0
    assert t1["cust_time_since_prev_log"] is None


def test_expanding_stats_exclude_current_row() -> None:
    out = build_feature_frame(_sample_frame().lazy()).collect()
    t2 = _row(out, "t2")
    t3 = _row(out, "t3")
    # c1 history before t2 is exactly the t1 amount.
    assert t2["cust_txn_count_prior"] == 1
    assert abs(t2["cust_amount_mean_prior"] - 100.0) < 1e-3
    # Before t3 the customer made two transactions averaging 75.
    assert t3["cust_txn_count_prior"] == 2
    assert abs(t3["cust_amount_mean_prior"] - 75.0) < 1e-3


def test_card_and_merchant_windows_are_independent() -> None:
    out = build_feature_frame(_sample_frame().lazy()).collect()
    t3 = _row(out, "t3")
    assert t3["card_txn_count_prior"] == 2  # same card as t1,t2
    t2 = _row(out, "t2")
    assert t2["merch_txn_count_prior"] == 1  # m1 seen once before t2


def test_channel_and_error_flags() -> None:
    out = build_feature_frame(_sample_frame().lazy()).collect()
    assert _row(out, "t2")["channel_online"] == 1
    assert _row(out, "t2")["had_payment_error"] == 1
    assert _row(out, "t3")["had_payment_error"] == 0
