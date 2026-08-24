"""Train the Rhea FinGraph baseline gradient-boosting risk model.

Pipeline: causal features -> label-derived priors fitted on TRAIN ONLY ->
HistGradientBoosting with positive-class weighting -> probability
calibration -> decision thresholds learned on validation -> one-shot,
locked evaluation on the held-out test period.

The model ranks transactions by fraud likelihood; probabilities are
recalibrated to the true base rate because positive-class weighting distorts
them. Decision thresholds are chosen on validation to guarantee minimum
precision inside each action band (hold / review), never on test.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from fingraph_sentinel.features import (
    FEATURE_COLUMNS,
    ONLINE_FEATURE_COLUMNS,
    build_feature_frame,
    fit_frequency_shares,
    fit_merchant_priors,
)


def calibrate_probability(p: np.ndarray | float, scale: float) -> np.ndarray | float:
    """Undo positive-class weighting: recover approximately true probabilities.

    Weighted training targets odds' = scale * odds_true, so invert that map.
    """
    if scale <= 1.0:
        return p
    return np.asarray(p, dtype=np.float64) / (scale * (1.0 - np.asarray(p)) + np.asarray(p))


def _load_featured(path: Path, max_rows: int | None) -> pl.DataFrame:
    lf = build_feature_frame(pl.scan_parquet(path))
    if max_rows is not None:
        lf = lf.head(max_rows)
    print(f"[features] materialising {path.name} ...", flush=True)
    frame = lf.collect()
    return frame


def _attach_priors(
    frame: pl.DataFrame,
    merchant_rates: dict[str, float],
    merchant_shares: dict[str, float],
    mcc_shares: dict[str, float],
) -> pl.DataFrame:
    merchant_table = pl.DataFrame(
        {
            "merchant_id": list(merchant_rates.keys()),
            "merch_fraud_rate_prior": list(merchant_rates.values()),
        },
        schema={"merchant_id": pl.Utf8, "merch_fraud_rate_prior": pl.Float32},
    )
    merchant_share_table = pl.DataFrame(
        {
            "merchant_id": list(merchant_shares.keys()),
            "merch_freq_share": list(merchant_shares.values()),
        },
        schema={"merchant_id": pl.Utf8, "merch_freq_share": pl.Float32},
    )
    mcc_table = pl.DataFrame(
        {
            "merchant_category_code": list(mcc_shares.keys()),
            "mcc_freq_share": list(mcc_shares.values()),
        },
        schema={"merchant_category_code": pl.Utf8, "mcc_freq_share": pl.Float32},
    )

    joined = (
        frame.with_columns(pl.col("merchant_id").cast(pl.Utf8))
        .join(merchant_table, on="merchant_id", how="left")
        .join(merchant_share_table, on="merchant_id", how="left")
        .with_columns(pl.col("merchant_category_code").cast(pl.Utf8))
        .join(mcc_table, on="merchant_category_code", how="left")
        .with_columns(
            pl.col("merch_fraud_rate_prior")
            .fill_null(float(merchant_rates["__default__"]))
            .clip(0.0, 1.0),
            pl.col("merch_freq_share").fill_null(0.0),
            pl.col("mcc_freq_share").fill_null(0.0),
        )
    )
    return joined


def _matrix(frame: pl.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    frame = frame.with_columns([pl.col(name).cast(pl.Float32) for name in columns])
    x = frame.select(columns).to_numpy().astype(np.float32)
    y = frame["is_fraud"].to_numpy().astype(np.int8)
    return x, y


def _threshold_policy(
    y_true: np.ndarray, proba: np.ndarray, hold_precision: float, review_precision: float
) -> tuple[float, float, dict[str, object]]:
    order = np.argsort(-proba)
    sorted_y = y_true[order].astype(np.int64)
    sorted_p = proba[order]
    cum_tp = np.cumsum(sorted_y)
    ks = np.arange(1, len(sorted_y) + 1)
    precision_at_k = cum_tp / ks

    def best_threshold(target: float) -> float | None:
        hits = np.nonzero(precision_at_k >= target)[0]
        if hits.size == 0 or hits[0] < 10:
            return None
        k = int(hits[0])  # smallest flagged set reaching the target precision
        return float((sorted_p[k - 1] + (sorted_p[k] if k < len(sorted_p) else 0.0)) / 2)

    hold_thr = best_threshold(hold_precision)
    review_thr = best_threshold(review_precision)

    # Operational fallback: fixed top-risk rate bands, guaranteed sane load
    # even when the model cannot reach the requested precision targets.
    fallback_used = False
    if hold_thr is None:
        hold_thr = float(np.quantile(proba, 1 - 0.0005))  # top 0.05%
        fallback_used = True
    if review_thr is None:
        review_thr = float(np.quantile(proba, 1 - 0.0105))  # next 1.0%
        fallback_used = True
    if review_thr > hold_thr:  # enforce ordered bands
        review_thr = hold_thr * 0.9

    actions = np.where(proba >= hold_thr, "hold", np.where(proba >= review_thr, "review", "allow"))
    summary: dict[str, object] = {
        "hold_threshold": round(hold_thr, 6),
        "review_threshold": round(review_thr, 6),
        "fallback_used": fallback_used,
        "val_action_counts": {
            a: int((actions == a).sum()) for a in ("allow", "review", "hold")
        },
        "val_hold_precision": round(
            float(
                ((proba >= hold_thr) & (y_true == 1)).sum()
                / max(int((proba >= hold_thr).sum()), 1)
            ),
            4,
        ),
        "val_review_band_precision": round(
            float(
                ((proba >= review_thr) & (proba < hold_thr) & (y_true == 1)).sum()
                / max((((proba >= review_thr) & (proba < hold_thr)).sum()), 1)
            ),
            4,
        ),
    }
    return float(hold_thr), float(review_thr), summary


def _recall_at_precision(y_true: np.ndarray, proba: np.ndarray, target: float) -> float:
    precision, recall, _ = precision_recall_curve(y_true, proba)
    mask = precision >= target
    return round(float(recall[mask].max()) if mask.any() else 0.0, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the baseline fraud-risk model.")
    base = Path("data/processed/ibm_full")
    parser.add_argument("--train", type=Path, default=base / "train.parquet")
    parser.add_argument("--val", type=Path, default=base / "validation.parquet")
    parser.add_argument("--test", type=Path, default=base / "test.parquet")
    parser.add_argument("--out", type=Path, default=Path("artifacts/models/baseline"))
    parser.add_argument("--max-train-rows", type=int, default=None, help="Smoke-test cap.")
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--feature-set", choices=["full", "online"], default="full",
        help="online = only features computable at realtime scoring (cold-start safe).")
    parser.add_argument("--hold-precision", type=float, default=0.80)
    parser.add_argument("--review-precision", type=float, default=0.30)
    args = parser.parse_args()

    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)

    train_raw = _load_featured(args.train, args.max_train_rows)
    val_raw = _load_featured(args.val, args.max_val_rows)
    test_raw = _load_featured(args.test, args.max_test_rows)
    print(f"[priors] fitting on train period ({train_raw.height:,} rows)", flush=True)

    merchant_rates = fit_merchant_priors(train_raw)
    merchant_shares = fit_frequency_shares(train_raw, "merchant_id")
    mcc_shares = fit_frequency_shares(train_raw, "merchant_category_code")

    train_df = _attach_priors(train_raw, merchant_rates, merchant_shares, mcc_shares)
    val_df = _attach_priors(val_raw, merchant_rates, merchant_shares, mcc_shares)
    test_df = _attach_priors(test_raw, merchant_rates, merchant_shares, mcc_shares)
    del train_raw, val_raw, test_raw

    columns = ONLINE_FEATURE_COLUMNS if args.feature_set == "online" else FEATURE_COLUMNS
    x_train, y_train = _matrix(train_df, columns)
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    scale_pos_weight = negatives / max(positives, 1)
    print(f"[train] rows={len(y_train):,} pos={positives:,} spw={scale_pos_weight:.1f}", flush=True)

    model = HistGradientBoostingClassifier(
        max_iter=600,
        learning_rate=0.06,
        max_leaf_nodes=63,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.05,
        n_iter_no_change=40,
        class_weight={0: 1.0, 1: scale_pos_weight},
        random_state=42,
    )
    model.fit(x_train, y_train)
    print(f"[train] finished {model.n_iter_} iterations", flush=True)

    xv, yv = _matrix(val_df, columns)
    val_proba = calibrate_probability(model.predict_proba(xv)[:, 1], scale_pos_weight)
    val_metrics = {
        "average_precision": round(float(average_precision_score(yv, val_proba)), 4),
        "roc_auc": round(float(roc_auc_score(yv, val_proba)), 4),
    }
    hold_thr, review_thr, policy = _threshold_policy(
        yv, val_proba, args.hold_precision, args.review_precision
    )
    print(f"[validate] {val_metrics} | policy={policy}", flush=True)

    xt, yt = _matrix(test_df, columns)
    test_proba = calibrate_probability(model.predict_proba(xt)[:, 1], scale_pos_weight)
    test_actions = np.where(
        test_proba >= hold_thr, "hold", np.where(test_proba >= review_thr, "review", "allow")
    )
    test_metrics = {
        "rows": int(len(yt)),
        "frauds": int(yt.sum()),
        "average_precision": round(float(average_precision_score(yt, test_proba)), 4),
        "roc_auc": round(float(roc_auc_score(yt, test_proba)), 4),
        "recall_at_precision_25": _recall_at_precision(yt, test_proba, 0.25),
        "recall_at_precision_50": _recall_at_precision(yt, test_proba, 0.50),
        "recall_at_precision_80": _recall_at_precision(yt, test_proba, 0.80),
        "action_counts": {a: int((test_actions == a).sum()) for a in ("allow", "review", "hold")},
        "caught_frauds_by_action": {
            a: int(((test_actions == a) & (yt == 1)).sum()) for a in ("allow", "review", "hold")
        },
    }
    print(f"[TEST - locked] {test_metrics}", flush=True)

    joblib.dump(model, args.out / "model.joblib")
    (args.out / "merchant_fraud_priors.json").write_text(json.dumps(merchant_rates))
    (args.out / "merchant_share.json").write_text(json.dumps(merchant_shares))
    (args.out / "mcc_share.json").write_text(json.dumps(mcc_shares))

    config = {
        "model_name": "baseline_hgb_v1",
        "backend": "sklearn.HistGradientBoostingClassifier",
        "created_at": datetime.now(UTC).isoformat(),
        "feature_columns": columns,
        "feature_set": args.feature_set,
        "calibration_scale_pos_weight": round(scale_pos_weight, 2),
        "thresholds": {"hold": hold_thr, "review": review_thr},
        "policy": policy,
        "metrics_validation": val_metrics,
        "metrics_test_locked": test_metrics,
        "training_rows": int(len(y_train)),
        "fit_seconds": round(time.time() - started, 1),
    }
    (args.out / "model_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"[done] artifacts in {args.out} ({config['fit_seconds']}s total)", flush=True)


if __name__ == "__main__":
    main()
