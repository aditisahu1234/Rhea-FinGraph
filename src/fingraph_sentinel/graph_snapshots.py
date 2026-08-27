"""Build leakage-safe temporal graph snapshots from the parquet splits.

Each snapshot = one calendar month of transactions. Node features are
computed from STRICTLY PAST months only (cumulative history up to and
excluding the snapshot's own month), so no future information leaks into
any node or edge.

Graph schema per snapshot (node sets are GLOBAL; snapshots differ in edges):
  ("customer", "purchased", "merchant")    -- labeled edges (is_fraud)
  ("merchant", "rev_purchased", "customer")-- message passing only
  ("customer", "has_card", "card")         -- message passing only
  ("card", "rev_has_card", "customer")     -- message passing only

Run:  python -m fingraph_sentinel.graph_snapshots [--max-rows N] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import polars as pl
import torch
from torch_geometric.data import HeteroData

REQUIRED_COLS = [
    "transaction_id", "event_time", "customer_id", "card_id", "merchant_id",
    "merchant_category_code", "amount", "payment_channel", "is_fraud",
]

EDGE_FEATURES = [
    "amount_log1p", "hour_sin", "hour_cos", "is_night", "is_weekend",
    "channel_swipe", "channel_chip", "channel_online", "month_frac",
]

NODE_FEATURE_DIM = 4  # log1p(n), log1p(amount), log1p(extra), fraud_rate


def _month_idx(ts: pl.Series) -> pl.Series:
    """Integer month index: (year - 1970) * 12 + (month - 1)."""
    return (ts.dt.year() - 1970) * 12 + (ts.dt.month() - 1)


def _cumulative_history(df: pl.DataFrame, id_col: str, extra_count_col: str) -> pl.DataFrame:
    """Per (entity, month) strictly-past cumulative stats.

    Output columns: [id_col, month_idx, hist_n, hist_amt, hist_extra, hist_fraud]
    hist_* EXCLUDE the current month (strictly past history only).
    """
    return (
        df.group_by([id_col, "month_idx"])
        .agg(
            pl.len().alias("n"),
            pl.col("amount").sum().alias("amt"),
            pl.col(extra_count_col).n_unique().alias("extra"),
            pl.col("is_fraud").sum().alias("fraud"),
        )
        .sort([id_col, "month_idx"])
        .with_columns(
            pl.col("n").cum_sum().over(id_col).alias("n_cum"),
            pl.col("amt").cum_sum().over(id_col).alias("amt_cum"),
            pl.col("extra").cum_sum().over(id_col).alias("extra_cum"),
            pl.col("fraud").cum_sum().over(id_col).alias("fraud_cum"),
        )
        .select(
            id_col,
            "month_idx",
            (pl.col("n_cum") - pl.col("n")).alias("hist_n"),
            (pl.col("amt_cum") - pl.col("amt")).alias("hist_amt"),
            (pl.col("extra_cum") - pl.col("extra")).alias("hist_extra"),
            (pl.col("fraud_cum") - pl.col("fraud")).alias("hist_fraud"),
        )
    )


def _feat_matrix_for_month(
    hist: pl.DataFrame,
    id_col: str,
    ids: list[str],
    month: int,
    dim: int,
) -> torch.Tensor:
    """Vectorized per-entity feature matrix for one month (history-only).

    For each entity takes the LATEST strictly-past month's cumulative stats;
    entities with no history before `month` get zero features (cold start).
    """
    past = hist.filter(pl.col("month_idx") < month)
    missing = pl.DataFrame({id_col: ids})
    if past.height:
        latest = (
            past.sort([id_col, "month_idx"])
            .group_by(id_col, maintain_order=True)
            .last()
        )
        frame = missing.join(latest, on=id_col, how="left")
    else:
        frame = missing.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("hist_n"),
            pl.lit(None, dtype=pl.Float64).alias("hist_amt"),
            pl.lit(None, dtype=pl.Float64).alias("hist_extra"),
            pl.lit(None, dtype=pl.Float64).alias("hist_fraud"),
        )

    frame = frame.with_columns(
        pl.col("hist_n").fill_null(0.0).alias("n"),
        pl.col("hist_amt").fill_null(0.0).clip(lower_bound=0.0).alias("amt"),
        pl.col("hist_extra").fill_null(0.0).alias("extra"),
        pl.col("hist_fraud").fill_null(0.0).alias("fraud"),
    ).with_columns(
        # 0/0 yields NaN; guard explicitly (fill_null does NOT catch NaN)
        pl.when(pl.col("n") > 0)
        .then(pl.col("fraud") / pl.col("n"))
        .otherwise(0.0)
        .fill_nan(0.0)
        .clip(lower_bound=0.0, upper_bound=1.0)
        .alias("rate"),
    )

    mat = torch.stack(
        [
            torch.tensor((frame["n"].log1p()).to_numpy(), dtype=torch.float32),
            torch.tensor((frame["amt"].log1p()).to_numpy(), dtype=torch.float32),
            torch.tensor((frame["extra"].log1p()).to_numpy(), dtype=torch.float32),
            torch.tensor(frame["rate"].to_numpy(), dtype=torch.float32),
        ],
        dim=1,
    )
    assert mat.shape == (len(ids), dim), (mat.shape, len(ids), dim)
    return mat


def _edge_tensors(
    sub: pl.DataFrame, month_frac: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized edge_index (customer->merchant), edge_attr, edge_label."""
    amt = pl.col("amount").clip(lower_bound=0.0).log1p()
    hour = pl.col("_hour")
    channel = pl.col("payment_channel").fill_null("").str.to_lowercase()
    attr = sub.select(
        [
            amt.alias("amount_log1p"),
            (2 * math.pi * hour / 24).sin().alias("hour_sin"),
            (2 * math.pi * hour / 24).cos().alias("hour_cos"),
            ((hour >= 23) | (hour < 5)).cast(pl.Int8).alias("is_night"),
            pl.col("_weekend").alias("is_weekend"),
            (channel == "swipe").cast(pl.Int8).alias("channel_swipe"),
            (channel == "chip").cast(pl.Int8).alias("channel_chip"),
            (channel == "online").cast(pl.Int8).alias("channel_online"),
            pl.lit(month_frac, dtype=pl.Float64).alias("month_frac"),
        ]
    )
    attr_t = torch.tensor(attr.to_numpy(), dtype=torch.float32)
    src = torch.tensor(sub["_cust_idx"].to_numpy(), dtype=torch.long)
    dst = torch.tensor(sub["_merch_idx"].to_numpy(), dtype=torch.long)
    label = torch.tensor(sub["is_fraud"].to_numpy(), dtype=torch.float32)
    return torch.stack([src, dst], dim=0), attr_t, label


def build_snapshots(
    parquet_files: list[Path],
    out_dir: Path,
    max_rows: int | None = None,
    bucket_months: int = 12,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load + bucket assignment ──────────────────────────────────────────
    frames = []
    for f in parquet_files:
        df = pl.read_parquet(f, columns=REQUIRED_COLS)
        if max_rows and df.height > max_rows:
            # random sample keeps fraud spread across all buckets (smoke tests)
            df = df.sample(n=max_rows, shuffle=True, seed=42)
        frames.append(df)
    df = pl.concat(frames)

    df = df.with_columns(
        _month_idx(pl.col("event_time")).alias("month_idx"),
        pl.col("event_time").dt.hour().alias("_hour"),
        pl.col("event_time").dt.weekday().alias("_dow"),
    ).with_columns(
        (pl.col("_dow") >= 6).cast(pl.Int8).alias("_weekend"),
        # bucket key: floor(month_idx / bucket_months); 12 = yearly snapshot
        (pl.col("month_idx") // bucket_months).alias("bucket_idx"),
    )

    months = sorted(df["bucket_idx"].unique().to_list())
    month_min = months[0]
    n_months = len(months)
    frac_for = {m: i / max(n_months - 1, 1) for i, m in enumerate(months)}

    # ── global node id maps ───────────────────────────────────────────────
    customer_ids = df["customer_id"].unique().sort().to_list()
    merchant_ids = df["merchant_id"].unique().sort().to_list()
    card_ids = df["card_id"].unique().sort().to_list()
    cust_map = {c: i for i, c in enumerate(customer_ids)}
    merch_map = {m: i for i, m in enumerate(merchant_ids)}
    card_map = {c: i for i, c in enumerate(card_ids)}
    n_cust, n_merch, n_card = len(customer_ids), len(merchant_ids), len(card_ids)
    print(
        f"[snapshots] {n_cust} customers, {n_merch} merchants, {n_card} cards;"
        f" {n_months} months ({month_min}..{months[-1]})",
        flush=True,
    )

    # ── cumulative history (strictly past) ────────────────────────────────
    hist_cust = _cumulative_history(df, "customer_id", extra_count_col="merchant_id")
    hist_merch = _cumulative_history(df, "merchant_id", extra_count_col="customer_id")
    hist_card = _cumulative_history(df, "card_id", extra_count_col="merchant_id")

    # index rows: _cust_idx / _merch_idx for edge building
    df = df.with_columns(
        pl.col("customer_id").replace_strict(cust_map).alias("_cust_idx"),
        pl.col("merchant_id").replace_strict(merch_map).alias("_merch_idx"),
    )

    # ── static card ownership pairs (customer -> card) ────────────────────
    pairs = (
        df.select(["customer_id", "card_id"])
        .unique()
        .to_numpy()
    )
    hc_src = torch.tensor([cust_map[c] for c in pairs[:, 0]], dtype=torch.long)
    hc_dst = torch.tensor([card_map[c] for c in pairs[:, 1]], dtype=torch.long)
    hc_pairs = torch.stack([hc_src, hc_dst], dim=0)

    # ── build + save one snapshot per bucket ──────────────────────────────
    snap_meta = []
    for s, month in enumerate(months):
        sub = df.filter(pl.col("bucket_idx") == month)
        frac = frac_for[month]

        ei, attr, label = _edge_tensors(sub, frac)

        data = HeteroData()
        data["customer"].x = _feat_matrix_for_month(
            hist_cust, "customer_id", customer_ids, month, NODE_FEATURE_DIM
        )
        data["merchant"].x = _feat_matrix_for_month(
            hist_merch, "merchant_id", merchant_ids, month, NODE_FEATURE_DIM
        )
        data["card"].x = _feat_matrix_for_month(
            hist_card, "card_id", card_ids, month, NODE_FEATURE_DIM
        )

        data["customer", "purchased", "merchant"].edge_index = ei
        data["customer", "purchased", "merchant"].edge_attr = attr
        data["customer", "purchased", "merchant"].edge_label = label
        data["merchant", "rev_purchased", "customer"].edge_index = ei.flip(0)
        data["merchant", "rev_purchased", "customer"].edge_attr = attr

        data["customer", "has_card", "card"].edge_index = hc_pairs
        data["card", "rev_has_card", "customer"].edge_index = hc_pairs.flip(0)

        torch.save(data, out_dir / f"snapshot_{s:03d}.pt")

        snap_meta.append({
            "month_idx": int(month),
            "month_frac": frac,
            "n_edges": int(sub.height),
            "n_fraud": int(sub["is_fraud"].sum()),
        })
        print(
            f"  month {month}: {sub.height:,} edges,"
            f" {int(sub['is_fraud'].sum()):,} fraud  -> snapshot_{s:03d}.pt",
            flush=True,
        )

    meta = {
        "n_customers": n_cust,
        "n_merchants": n_merch,
        "n_cards": n_card,
        "n_months": n_months,
        "month_min": month_min,
        "month_max": months[-1],
        "bucket_months": bucket_months,
        "edge_feature_cols": EDGE_FEATURES,
        "node_feature_dim": NODE_FEATURE_DIM,
        "months": [int(m) for m in months],
        "snapshots": snap_meta,
        "customer_ids": customer_ids,
        "merchant_ids": merchant_ids,
        "card_ids": card_ids,
    }
    with (out_dir / "meta.json").open("w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[snapshots] wrote {n_months} snapshots to {out_dir}")
    return meta


def main():
    parser = argparse.ArgumentParser(description="Build temporal graph snapshots.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Cap rows per split (smoke testing)")
    parser.add_argument("--bucket-months", type=int, default=12,
                        help="Months per snapshot bucket (12 = yearly; 1 = monthly)")
    parser.add_argument("--out", type=Path, default=Path("artifacts/graph/snapshots"))
    args = parser.parse_args()

    base = Path("data/processed/ibm_full")
    pmap = {"train": "train.parquet", "val": "validation.parquet", "test": "test.parquet"}
    files = [base / pmap[s] for s in args.splits if s in pmap]
    if not files:
        print("No valid splits.")
        return
    build_snapshots(files, args.out, max_rows=args.max_rows,
                    bucket_months=args.bucket_months)


if __name__ == "__main__":
    main()