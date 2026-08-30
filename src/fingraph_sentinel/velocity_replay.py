"""Historical velocity replay (Layer 1) - vectorized twin of the live store.

Two engines:

* ``replay_split`` (serial)      - drives the *live* ``VelocityStore`` (dict
  backend) event-by-event. This is the production code path; used as the
  ground-truth oracle in ``--verify``.
* ``replay_split_vectorized``    - polars-native recomputation of the *exact*
  same features (window counts/amounts/distinct merchants + cumulative priors)
  in one pass. Every formula mirrors ``VelocityStore.compute/observe``,
  including inclusive window bounds and same-timestamp (tie) semantics:

    - temporal rolling windows in polars include ALL rows with
      ``t in [t-dt, t]`` regardless of position, so the strictly-past count is
      ``rolled - (tie_size - tie_rank)`` (the observed-before same-timestamp
      surplus, identical to the store's ``lo <= t <= hi`` lookups).
    - cumulative priors are plain ``cum_count/cum_sum/shift`` per entity.

  Both engines are verified byte-for-byte (1e-6 tolerance) by
  ``--verify N`` on the head(N) rows of a split.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import polars as pl

from fingraph_sentinel.streaming import (
    WINDOWS,
    InMemoryBackend,
    VelocityStore,
    prior_feature_names,
    velocity_feature_names,
)

SPLITS = ("train", "validation", "test")
OUT_COLUMNS = ["transaction_id", *velocity_feature_names(), *prior_feature_names()]

#: entity id column per streaming entity (device_id is absent from the IBM
#: schema, so device features are backfilled as null for every split).
ENTITY_COLUMNS = {"cust": "customer_id", "card": "card_id", "merch": "merchant_id"}
WINDOW_SECONDS = {win: int(secs) for win, secs in WINDOWS.items()}


def _emit(frame: pl.DataFrame, out_dir: Path, part: int) -> int:
    missing = [c for c in OUT_COLUMNS if c not in frame.columns]
    if missing:
        frame = frame.with_columns(
            [pl.lit(None, dtype=pl.Float32).alias(c) for c in missing]
        )
    frame.select(OUT_COLUMNS).write_parquet(out_dir / f"part-{part:04d}.parquet")
    return part + 1


def replay_split(
    parquet_path: Path,
    out_dir: Path,
    chunk_rows: int = 500_000,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Replay (a prefix of) one split chronologically through the LIVE store.

    Serial, exact; used as the ground-truth oracle for ``--verify`` and for
    small slices. Full-scale production runs use the vectorized engine.
    """
    store = VelocityStore(InMemoryBackend())
    start = time.time()
    part = 0
    rows: list[dict[str, Any]] = []
    seen = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    total = pl.scan_parquet(parquet_path).select(pl.len()).collect().item()
    if max_rows is not None:
        total = min(total, max_rows)

    def flush() -> None:
        nonlocal part, rows
        if not rows:
            return
        frame = _emit(pl.DataFrame(rows), out_dir, part)
        part = frame
        rows = []

    lf = pl.scan_parquet(parquet_path)  # parquet row groups are time-ordered
    while seen < total:
        take = min(chunk_rows, total - seen)
        batch = lf.slice(seen, take).collect()
        for ev in batch.iter_rows(named=True):
            feats = store.compute(ev)
            store.observe(ev)
            rows.append({"transaction_id": str(ev.get("transaction_id", "")), **feats})
            seen += 1
        flush()
        elapsed = time.time() - start
        print(
            f"[replay:{out_dir.name}] {seen:,}/{total:,} in {elapsed:.1f}s "
            f"({elapsed / max(seen, 1) * 1e6:.1f} us/ev)",
            flush=True,
        )
    elapsed = time.time() - start
    print(
        f"[replay] {out_dir.name}: {seen:,} events, {part} batches, {elapsed:.1f}s",
        flush=True,
    )
    return {"rows": seen, "batches": part, "seconds": round(elapsed, 1)}


def _window_count(rolled: pl.Expr, tie_size: pl.Expr, tie_rank: pl.Expr) -> pl.Expr:
    """Strictly-past count: closed-both rolling minus the same-timestamp
    surplus (rows at t == t_r observed at-or-after position r, incl. self)."""
    return rolled - (tie_size - tie_rank)


def replay_split_vectorized(
    parquet_path: Path,
    out_dir: Path,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """One-pass polars recomputation of the exact velocity feature set.

    Execution order within each entity group is the split's chronological row
    order (the parquet is already non-decreasing in ``event_time``); equal
    timestamps keep their original (observation) order, matching the store.
    """
    start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    lf = pl.scan_parquet(parquet_path)
    if max_rows is not None:
        lf = lf.head(max_rows)

    df = (
        lf.select(
            "transaction_id",
            pl.col("event_time").alias("dt"),
            # μs-exact float seconds — matches VelocityStore._ts exactly
            (pl.col("event_time").dt.epoch("us").cast(pl.Float64) / 1e6).alias("t"),
            pl.col("amount").cast(pl.Float64),
            *ENTITY_COLUMNS.values(),
        )
        .with_row_index("pos")
        .with_columns(pl.lit(1.0, dtype=pl.Float64).alias("one"))
        .collect()
    )
    pos = pl.col("pos")
    amount = pl.col("amount")
    t = pl.col("t")

    frame = df

    # --- window + prior expressions, all evaluated in ONE with_columns pass ---
    exprs: list[pl.Expr] = []
    for ent, win, with_amt, with_distinct in [
        ("cust", "1h", True, False),
        ("cust", "24h", True, True),
        ("cust", "7d", True, True),
        ("card", "1h", True, False),
        ("card", "24h", True, False),
        ("card", "7d", True, False),
        ("merch", "24h", False, False),
        ("merch", "7d", False, False),
    ]:
        idcol = ENTITY_COLUMNS[ent]
        delta = float(WINDOW_SECONDS[win])
        win_s = win  # '1h' | '24h' | '7d' — polars fixed-second windows == store Δ
        tie_rank = (
            pl.col("pos").cum_count().over([idcol, "dt"]) - 1
        )  # 0-based within (entity, instant)
        tie_size = pl.col("dt").count().over([idcol, "dt"])

        # NOTE: surplus = tie_size - tie_rank computed per entity id (the SAME tie group as the
        # store's window key: id + window; the tie-size/rank correction only
        # depends on (id, instant) which is window-independent).
        cnt = (
            pl.col("one")
            .rolling_sum_by(by="dt", window_size=win_s, closed="both")
            .over(idcol)
        )
        exprs.append(_window_count(cnt, tie_size, tie_rank).alias(f"{ent}_v_{win}_count"))
        if with_amt:
            roll_amt = amount.rolling_sum_by(by="dt", window_size=win_s, closed="both").over(idcol)
            tot = amount.sum().over([idcol, "dt"])
            prev_cum = amount.cum_sum().over([idcol, "dt"]) - amount
            surplus_amt = tot - prev_cum
            exprs.append((roll_amt - surplus_amt).alias(f"{ent}_v_{win}_amount"))
        if with_distinct:
            # Exact distinct-merchant count over the strictly-past window.
            # Each event o "covers" query rows r with  pos_o < pos_r <= cover_o
            # where  cover_o = min(next_o, ub_o)  with
            #   next_o = next position of the same (id, merchant) > pos_o
            #   ub_o   = max position with t <= t_o + delta   (window upper
            #            bound; store looks ups are inclusive on BOTH ends).
            # Since cover >= pos_o, every cover < x belongs to a strictly-prior
            # event, so
            #   distinct(x) = #{pos < x} - #{cover < x}   (covers from events
            #   that also fall strictly before x)        == A(x) - B_lt(x)
            d = df.select("pos", "t", idcol, "merchant_id")
            nxt = (
                pl.col("pos")
                .shift(-1)
                .over([idcol, "merchant_id"])
                .fill_null(pl.col("pos").max().over(idcol) + 1)  # sentinel > any pos
            )
            # ub must be the LARGEST pos per (id, t) so that same-timestamp
            # ties map to the correct window upper bound
            right = (
                d.group_by(idcol, "t")
                .agg(pl.col("pos").max().alias("pos"))
                .sort(idcol, "t")
            )
            left = d.with_columns((pl.col("t") + delta).alias("t_ub")).select(
                "pos", idcol, "t_ub"
            ).sort(idcol, "t_ub")
            ub = (
                left.join_asof(
                    right,
                    left_on="t_ub",
                    right_on="t",
                    by=idcol,
                    strategy="backward",
                )
                .select("pos", pl.col("pos_right").alias("ub_pos"))
            )
            d = (
                d.with_columns(nxt.alias("next_m_pos"))
                .join(ub, on="pos")
                .with_columns(
                    pl.min_horizontal("next_m_pos", "ub_pos").alias("cover")
                )
            )
            # B_lt(x) = #{cover < x} == #{cover <= x-1}; computed as the
            # per-customer cover rank (cover_x / cum_count) matched by a
            # backward asof on (pos - 1) so strict inequality holds.
            cov = (
                d.select(idcol, "cover")
                .with_columns(pl.col("cover").cast(pl.Int64).alias("cover"))
                .sort(idcol, "cover")
                .with_columns(
                    pl.col("cover").cum_count().over(idcol).cast(pl.Float64).alias("cover_rank")
                )
            )
            q = (
                df.select("pos", idcol)
                .sort(idcol, "pos")
                .with_columns((pl.col("pos").cast(pl.Int64) - 1).alias(f"posm1_{win}"))
            )
            b_lt_df = (
                q.join_asof(
                    cov,
                    left_on=f"posm1_{win}",
                    right_on="cover",
                    by=idcol,
                    strategy="backward",
                ).select(
                    "pos",
                    pl.col("cover_rank").fill_null(0.0).alias(f"b_lt_{win}"),
                )
            )
            frame = frame.join(b_lt_df, on="pos", how="left")
            a_rank = (pl.col("pos").rank("ordinal").over(idcol) - 1).cast(pl.Float64)
            distinct = (
                a_rank.alias("A")
                - pl.col(f"b_lt_{win}").fill_null(0.0).alias("b_lt")
            )
            exprs.append(distinct.alias(f"{ent}_v_{win}_distinct_merchants"))

        # NOTE: window expressions are accumulated into ``exprs`` across the
        # loop iterations above (they were appended with .alias directly).

    # --- cumulative causal priors (before each event is recorded) -------------
    for ent, idcol in ENTITY_COLUMNS.items():
        count_prior = (pos.cum_count().over(idcol) - 1).cast(pl.Float64)
        amount_mean = ((amount.cum_sum().over(idcol) - amount) / pl.max_horizontal(
            count_prior, pl.lit(1.0)
        )).fill_null(0.0)
        prev_t = t.shift(1).over(idcol)
        time_log = (
            pl.when(prev_t.is_not_null())
            .then((t - prev_t).clip(lower_bound=0.0).log1p())
            .otherwise(0.0)
        )
        exprs.append(count_prior.alias(f"{ent}_txn_count_prior"))
        if ent == "cust":
            exprs += [
                amount_mean.alias(f"{ent}_amount_mean_prior"),
                time_log.alias(f"{ent}_time_since_prev_log"),
            ]
            prev_amt = amount.shift(1).over(idcol)
            ratio = (
                pl.when(prev_amt > 0.0)
                .then((amount / prev_amt).clip(lower_bound=0.0, upper_bound=50.0))
                .otherwise(1.0)
            )
            exprs.append(ratio.alias("cust_prev_amount_ratio"))
        elif ent == "card":
            exprs += [
                amount_mean.alias(f"{ent}_amount_mean_prior"),
                time_log.alias(f"{ent}_time_since_prev_log"),
            ]
    frame = frame.with_columns(exprs)

    part = _emit(frame, out_dir, 0)
    elapsed = time.time() - start
    print(
        f"[replay-vec:{out_dir.name}] {len(df):,} events, "
        f"{len(OUT_COLUMNS)} cols, {elapsed:.1f}s",
        flush=True,
    )
    return {"rows": len(df), "batches": part, "seconds": round(elapsed, 1)}


def verify_parity(
    parquet_path: Path,
    num_rows: int = 100_000,
    chunk_rows: int = 100_000,
) -> dict[str, Any]:
    """Vectorized vs live-store parity check on the head(num_rows) rows."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        serial_dir = Path(td) / "serial"
        vec_dir = Path(td) / "vec"
        serial_dir.mkdir(parents=True)
        vec_dir.mkdir(parents=True)
        replay_split(parquet_path, serial_dir, chunk_rows=chunk_rows, max_rows=num_rows)
        replay_split_vectorized(parquet_path, vec_dir, max_rows=num_rows)
        s = pl.read_parquet(serial_dir / "part-0000.parquet").sort("transaction_id")
        v = pl.read_parquet(vec_dir / "part-0000.parquet").sort("transaction_id")
        joined = s.join(v, on="transaction_id", how="inner")
        diffs: dict[str, float] = {}
        worst = 0.0
        for c in OUT_COLUMNS[1:]:
            left = joined.select(pl.col(c) + 0.0).to_series()
            right = joined.select(pl.col(f"{c}_right") + 0.0).to_series()
            d = (left.fill_null(0.0) - right.fill_null(0.0)).abs().max()
            diffs[c] = float(d)
            worst = max(worst, float(d))
        return {
            "rows_compared": int(joined.height),
            "worst_abs_diff": worst,
            "per_column_max_abs_diff": diffs,
            "exact_match": worst < 1e-6,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Historical velocity replay (Layer 1)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/ibm_full"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/data/velocity"))
    parser.add_argument(
        "--split",
        choices=[*SPLITS, "all"],
        default="all",
        help="Which split to replay (all = train+validation+test).",
    )
    parser.add_argument("--chunk-rows", type=int, default=500_000)
    parser.add_argument(
        "--engine",
        choices=["serial", "vectorized"],
        default="vectorized",
        help="serial = live VelocityStore (oracle); vectorized = polars twin.",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--verify",
        type=int,
        default=0,
        metavar="N",
        help="Parity-check vectorized vs serial on head(N) rows of train, then exit.",
    )
    args = parser.parse_args()

    if args.verify:
        res = verify_parity(args.data_dir / "train.parquet", num_rows=args.verify)
        print(f"rows compared: {res['rows_compared']:,}")
        print(f"worst abs diff: {res['worst_abs_diff']:.3e}")
        print(f"exact parity (1e-6): {res['exact_match']}")
        for col, d in res["per_column_max_abs_diff"].items():
            print(f"  {col:36s} {d:.3e}")
        raise SystemExit(0 if res["exact_match"] else 1)

    splits = SPLITS if args.split == "all" else (args.split,)
    for name in splits:
        src = args.data_dir / f"{name}.parquet"
        if not src.exists():
            raise SystemExit(f"missing {src}")
        if args.engine == "serial":
            replay_split(src, args.out_dir / name, chunk_rows=args.chunk_rows,
                         max_rows=args.max_rows)
        else:
            replay_split_vectorized(src, args.out_dir / name, max_rows=args.max_rows)


if __name__ == "__main__":
    main()