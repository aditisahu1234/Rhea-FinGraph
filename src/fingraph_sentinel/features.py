"""Causal feature engineering for Rhea FinGraph.

Design rule: every feature may only use information available at (or
strictly before) the transaction's own timestamp. Expanding statistics are
shifted by one row so the current event never contributes to its own
features, and label-derived priors (merchant fraud rate) are fitted on the
training period only, then applied unchanged to later periods.
"""

from __future__ import annotations

import math

import polars as pl

# Final numeric feature vector consumed by the baseline model.
FEATURE_COLUMNS: list[str] = [
    # static / calendar
    "amount_log1p",
    "hour_sin",
    "hour_cos",
    "is_weekend",
    "is_night",
    "channel_swipe",
    "channel_chip",
    "channel_online",
    "had_payment_error",
    # customer behaviour (causal)
    "cust_txn_count_prior",
    "cust_amount_mean_prior",
    "cust_time_since_prev_log",
    "cust_prev_amount_ratio",
    # card behaviour (causal)
    "card_txn_count_prior",
    "card_amount_mean_prior",
    "card_time_since_prev_log",
    # merchant context
    "merch_txn_count_prior",
    "merch_freq_share",
    "merch_fraud_rate_prior",
    "mcc_freq_share",
]

ID_TIME_LABEL = ["transaction_id", "event_time", "is_fraud"]
JOIN_KEYS = ["merchant_id", "merchant_category_code"]
PRIOR_PLACEHOLDERS = ["merch_freq_share", "merch_fraud_rate_prior", "mcc_freq_share"]

# Features computable for a single inbound event without any behavioural
# history store. The online-serving model uses exactly this subset so its
# thresholds transfer faithfully to production traffic.
ONLINE_FEATURE_COLUMNS: list[str] = [
    "amount_log1p",
    "hour_sin",
    "hour_cos",
    "is_weekend",
    "is_night",
    "channel_swipe",
    "channel_chip",
    "channel_online",
    "had_payment_error",
    "merch_freq_share",
    "merch_fraud_rate_prior",
    "mcc_freq_share",
]


def _static_and_calendar(lf: pl.LazyFrame) -> pl.LazyFrame:
    hour = pl.col("event_time").dt.hour()
    channel = pl.col("payment_channel").str.to_lowercase()
    return lf.with_columns(
        pl.col("amount").clip(lower_bound=0.0).log1p().alias("amount_log1p"),
        (hour * math.pi / 12).sin().alias("hour_sin"),
        (hour * math.pi / 12).cos().alias("hour_cos"),
        (pl.col("event_time").dt.weekday() >= 6).cast(pl.Int8).alias("is_weekend"),
        ((hour <= 5) | (hour >= 23)).cast(pl.Int8).alias("is_night"),
        channel.str.contains("swipe").cast(pl.Int8).alias("channel_swipe"),
        channel.str.contains("chip").cast(pl.Int8).alias("channel_chip"),
        channel.str.contains("online").cast(pl.Int8).alias("channel_online"),
        pl.col("payment_error").fill_null("").str.strip_chars().ne("")
        .cast(pl.Int8)
        .alias("had_payment_error"),
    )


def _entity_history(lf: pl.LazyFrame, entity: str, prefix: str) -> pl.LazyFrame:
    """Backward-looking per-entity behaviour features.

    Sorting by (entity, event_time) makes shift(1) inside the window refer to
    that entity's previous transaction; cumulative aggregates are shifted
    before use so every value excludes the current row -- causal by design.
    """
    ordered = lf.sort([entity, "event_time", "transaction_id"])
    prior_count = pl.col("amount").cum_count().over(entity).shift(1)
    prior_sum = pl.col("amount").cum_sum().over(entity).shift(1)
    gap_seconds = (
        pl.col("event_time") - pl.col("event_time").shift(1).over(entity)
    ).dt.total_seconds()

    extras: list[pl.Expr] = []
    if prefix == "cust":
        extras.append(
            (pl.col("amount") / pl.col("amount").shift(1).over(entity))
            .clip(0.0, 50.0)
            .cast(pl.Float32)
            .alias("cust_prev_amount_ratio")
        )

    return ordered.with_columns(
        *extras,
        prior_count.fill_null(0).cast(pl.Float32).alias(f"{prefix}_txn_count_prior"),
        (prior_sum / prior_count).clip(0.0, 1e7).cast(pl.Float32).alias(
            f"{prefix}_amount_mean_prior"
        ),
        gap_seconds.clip(lower_bound=0).add(1).log1p().cast(pl.Float32).alias(
            f"{prefix}_time_since_prev_log"
        ),
    )


def build_feature_frame(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Return a lazy frame holding FEATURE_COLUMNS plus id/time/label."""
    frame = _static_and_calendar(lf)
    frame = _entity_history(frame, "customer_id", "cust")
    frame = _entity_history(frame, "card_id", "card")

    frame = frame.sort(["merchant_id", "event_time", "transaction_id"]).with_columns(
        pl.col("amount")
        .cum_count()
        .over("merchant_id")
        .shift(1)
        .fill_null(0)
        .cast(pl.Float32)
        .alias("merch_txn_count_prior")
    )
    frame = frame.with_columns(
        pl.lit(None, dtype=pl.Float32).alias(name) for name in PRIOR_PLACEHOLDERS
    )
    return frame.select(FEATURE_COLUMNS + ID_TIME_LABEL + JOIN_KEYS)


def fit_merchant_priors(train_frame: pl.DataFrame) -> dict[str, float]:
    """Merchant fraud-rate priors fitted on the TRAINING period only."""
    stats = train_frame.group_by("merchant_id").agg(
        pl.len().alias("n"), pl.col("is_fraud").mean().alias("rate")
    )
    mapping = {
        str(row["merchant_id"]): round(float(row["rate"]), 6)
        for row in stats.iter_rows(named=True)
    }
    mapping["__default__"] = round(float(train_frame["is_fraud"].mean()), 8)
    return mapping


def fit_frequency_shares(train_frame: pl.DataFrame, column: str) -> dict[str, float]:
    """Share of training transactions per category (frequency encoding)."""
    counts = train_frame.group_by(column).agg(pl.len().alias("n"))
    total = float(train_frame.height) or 1.0
    mapping = {
        str(row[column]): round(float(row["n"]) / total, 8)
        for row in counts.iter_rows(named=True)
    }
    mapping["__default__"] = 0.0
    return mapping
