"""Profile and validate the raw IBM card-transaction file before model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

IBM_CANONICAL_COLUMNS = {
    "customer_id": "User",
    "card_id": "Card",
    "year": "Year",
    "month": "Month",
    "day": "Day",
    "time": "Time",
    "amount": "Amount",
    "payment_channel": "Use Chip",
    "merchant_id": "Merchant Name",
    "merchant_city": "Merchant City",
    "merchant_state": "Merchant State",
    "merchant_zip": "Zip",
    "merchant_category_code": "MCC",
    "errors": "Errors?",
    "is_fraud": "Is Fraud?",
}


def _parse_fraud_label() -> pl.Expr:
    """Convert IBM's Yes/No label to a compact binary target without guessing unknown values."""
    return (
        pl.col(IBM_CANONICAL_COLUMNS["is_fraud"])
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.to_lowercase()
        .replace_strict({"yes": 1, "no": 0}, default=None)
        .cast(pl.Int8)
        .alias("is_fraud")
    )


def normalize_ibm_transactions(source: Path, row_limit: int | None = None) -> pl.LazyFrame:
    """Return the canonical, pseudonymous model frame from the raw IBM CSV.

    The transform is deliberately source-only: it never computes aggregate features or
    uses labels beyond retaining the supervised target. This keeps split-specific
    feature engineering leakage-safe in the next pipeline stage.
    """
    frame = pl.scan_csv(source, infer_schema_length=10_000, ignore_errors=False)
    if row_limit is not None:
        frame = frame.head(row_limit)

    expected = set(IBM_CANONICAL_COLUMNS.values())
    actual = set(frame.collect_schema().names())
    missing = sorted(expected - actual)
    if missing:
        raise ValueError(f"IBM CSV is missing expected columns: {', '.join(missing)}")

    timestamp = pl.concat_str(
        [
            pl.col(IBM_CANONICAL_COLUMNS["year"]).cast(pl.Utf8),
            pl.col(IBM_CANONICAL_COLUMNS["month"]).cast(pl.Utf8).str.zfill(2),
            pl.col(IBM_CANONICAL_COLUMNS["day"]).cast(pl.Utf8).str.zfill(2),
            pl.col(IBM_CANONICAL_COLUMNS["time"]).cast(pl.Utf8),
        ],
        separator="-",
    ).str.strptime(pl.Datetime, format="%Y-%m-%d-%H:%M", strict=True)

    return (
        frame.with_row_index("raw_row_id")
        .select(
            pl.col("raw_row_id").cast(pl.Utf8).alias("transaction_id"),
            timestamp.alias("event_time"),
            pl.col(IBM_CANONICAL_COLUMNS["customer_id"]).cast(pl.Utf8).alias("customer_id"),
            pl.concat_str(
                [
                    pl.col(IBM_CANONICAL_COLUMNS["customer_id"]).cast(pl.Utf8),
                    pl.col(IBM_CANONICAL_COLUMNS["card_id"]).cast(pl.Utf8),
                ],
                separator="::",
            ).alias("card_id"),
            pl.col(IBM_CANONICAL_COLUMNS["merchant_id"]).cast(pl.Utf8).alias("merchant_id"),
            pl.col(IBM_CANONICAL_COLUMNS["merchant_category_code"])
            .cast(pl.Utf8)
            .alias("merchant_category_code"),
            pl.col(IBM_CANONICAL_COLUMNS["amount"])
            .cast(pl.Utf8)
            .str.replace_all(r"[^0-9.\\-]", "")
            .cast(pl.Float64, strict=True)
            .alias("amount"),
            pl.lit("USD").alias("currency"),
            pl.col(IBM_CANONICAL_COLUMNS["merchant_city"]).cast(pl.Utf8).alias("merchant_city"),
            pl.col(IBM_CANONICAL_COLUMNS["merchant_state"]).cast(pl.Utf8).alias("merchant_state"),
            pl.col(IBM_CANONICAL_COLUMNS["merchant_zip"]).cast(pl.Utf8).alias("merchant_zip"),
            pl.col(IBM_CANONICAL_COLUMNS["payment_channel"])
            .cast(pl.Utf8)
            .alias("payment_channel"),
            pl.col(IBM_CANONICAL_COLUMNS["errors"]).cast(pl.Utf8).alias("payment_error"),
            _parse_fraud_label(),
        )
        .filter(pl.col("event_time").is_not_null() & pl.col("amount").is_not_null())
    )


def write_temporal_splits(
    source: Path,
    destination: Path,
    row_limit: int | None = None,
) -> dict[str, object]:
    """Write 60/20/20 chronological parquet splits and a leak-checkable manifest."""
    frame = normalize_ibm_transactions(source, row_limit).sort("event_time")
    boundaries = frame.select(
        pl.col("event_time").quantile(0.60, interpolation="nearest").alias("train_end"),
        pl.col("event_time").quantile(0.80, interpolation="nearest").alias("validation_end"),
    ).collect().row(0, named=True)
    train_end = boundaries["train_end"]
    validation_end = boundaries["validation_end"]

    if train_end is None or validation_end is None:
        raise ValueError("Unable to calculate chronological split boundaries.")
    if train_end >= validation_end:
        raise ValueError("Temporal split boundaries must be strictly ordered.")

    destination.mkdir(parents=True, exist_ok=True)
    split_frames = {
        "train": frame.filter(pl.col("event_time") <= pl.lit(train_end)),
        "validation": frame.filter(
            (pl.col("event_time") > pl.lit(train_end))
            & (pl.col("event_time") <= pl.lit(validation_end))
        ),
        "test": frame.filter(pl.col("event_time") > pl.lit(validation_end)),
    }

    manifest: dict[str, object] = {
        "source": str(source),
        "row_limit": row_limit,
        "boundaries": {
            "train_end": train_end.isoformat(),
            "validation_end": validation_end.isoformat(),
        },
        "splits": {},
    }
    for name, lazy_frame in split_frames.items():
        output = destination / f"{name}.parquet"
        lazy_frame.sink_parquet(output, compression="zstd")
        summary = pl.scan_parquet(output).select(
            pl.len().alias("rows"),
            pl.col("event_time").min().alias("min_event_time"),
            pl.col("event_time").max().alias("max_event_time"),
            pl.col("is_fraud").mean().alias("fraud_rate"),
        ).collect().row(0, named=True)
        manifest["splits"][name] = {
            "path": str(output),
            "rows": summary["rows"],
            "min_event_time": summary["min_event_time"].isoformat(),
            "max_event_time": summary["max_event_time"].isoformat(),
            "fraud_rate": summary["fraud_rate"],
        }

    manifest_path = destination / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest


def profile_csv(source: Path, row_limit: int | None = None) -> dict[str, object]:
    frame = pl.scan_csv(source, infer_schema_length=10_000, ignore_errors=False)
    if row_limit is not None:
        frame = frame.head(row_limit)

    schema = frame.collect_schema()
    columns = list(schema.names())
    missing = {
        canonical: expected
        for canonical, expected in IBM_CANONICAL_COLUMNS.items()
        if expected not in columns
    }
    sample = frame.head(3).collect().to_dicts()

    return {
        "source": str(source),
        "column_count": len(columns),
        "columns": columns,
        "schema": {name: str(dtype) for name, dtype in schema.items()},
        "missing_expected_columns": missing,
        "sample_rows": sample,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the IBM fraud-data CSV safely.")
    parser.add_argument("source", type=Path, help="Path to the downloaded IBM transaction CSV")
    parser.add_argument("--output", type=Path, default=Path("artifacts/data_profile.json"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--write-splits",
        type=Path,
        default=None,
        help="Write leakage-safe 60/20/20 chronological parquet splits to this folder.",
    )
    args = parser.parse_args()

    report = profile_csv(args.source, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"Wrote schema profile to {args.output}")
    if args.write_splits is not None:
        manifest = write_temporal_splits(args.source, args.write_splits, args.limit)
        print(
            "Wrote chronological splits: "
            + ", ".join(f"{name}={details['rows']}" for name, details in manifest["splits"].items())
        )


if __name__ == "__main__":
    main()
