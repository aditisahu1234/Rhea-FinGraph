"""Train Rhea FinGraph risk models with production gradient boosting.

Backends (--backend):
  xgboost    default; GPU-capable (`--device cuda` on a Kaggle T4)
  lightgbm   cross-check booster
  sklearn    HistGradientBoosting baseline, retained deliberately so the
             SOTA models can be compared against it like in published work

Pipeline: causal features -> label-derived priors fitted on TRAIN ONLY ->
class-weighted boosting -> probability calibration -> decision thresholds
learned on validation -> one-shot locked evaluation on the test period.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from fingraph_sentinel.features import (
    FEATURE_COLUMNS,
    ONLINE_FEATURE_COLUMNS,
    ONLINE_VELOCITY_FEATURE_COLUMNS,
    build_feature_frame,
    fit_frequency_shares,
    fit_merchant_priors,
)


def calibrate_probability(p: np.ndarray | float, scale: float) -> np.ndarray | float:
    """Undo positive-class weighting: recover approximately true probabilities."""
    if scale <= 1.0:
        return p
    p_arr = np.asarray(p, dtype=np.float64)
    return p_arr / (scale * (1.0 - p_arr) + p_arr)


def _load_featured(
    path: Path,
    max_rows: int | None,
    velocity_dir: Path | None = None,
) -> pl.DataFrame:
    lf = build_feature_frame(pl.scan_parquet(path))
    if max_rows is not None:
        lf = lf.head(max_rows)
    if velocity_dir is not None:
        # Layer 1: join the historical strictly-past velocity replay (one row
        # per transaction, same id). Misses stay null (cold-start entities),
        # which every supported backend treats natively.
        vel = pl.scan_parquet(f"{velocity_dir}/*.parquet")
        lf = lf.join(vel, on="transaction_id", how="left")
    print(f"[features] materialising {path.name} "
          f"(velocity={velocity_dir is not None}) ...", flush=True)
    return lf.collect()


def _attach_priors(
    frame: pl.DataFrame,
    merchant_rates: dict[str, float],
    merchant_shares: dict[str, float],
    mcc_shares: dict[str, float],
) -> pl.DataFrame:
    def table(keys: dict[str, float], key_col: str, value_col: str) -> pl.DataFrame:
        return pl.DataFrame(
            {key_col: list(keys.keys()), value_col: list(keys.values())},
            schema={key_col: pl.Utf8, value_col: pl.Float32},
        )

    return (
        frame.with_columns(pl.col("merchant_id").cast(pl.Utf8))
        .join(
            table(merchant_rates, "merchant_id", "merch_fraud_rate_prior"),
            on="merchant_id",
            how="left",
        )
        .join(
            table(merchant_shares, "merchant_id", "merch_freq_share"),
            on="merchant_id",
            how="left",
        )
        .with_columns(pl.col("merchant_category_code").cast(pl.Utf8))
        .join(
            table(mcc_shares, "merchant_category_code", "mcc_freq_share"),
            on="merchant_category_code",
            how="left",
        )
        .with_columns(
            pl.col("merch_fraud_rate_prior")
            .fill_null(float(merchant_rates["__default__"]))
            .clip(0.0, 1.0),
            pl.col("merch_freq_share").fill_null(0.0),
            pl.col("mcc_freq_share").fill_null(0.0),
        )
    )


def _matrix(frame: pl.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    frame = frame.with_columns([pl.col(name).cast(pl.Float32) for name in columns])
    x = frame.select(columns).to_numpy().astype(np.float32)
    np.nan_to_num(x, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    y = frame["is_fraud"].to_numpy().astype(np.int8)
    return x, y


def _fit_backend(backend: str, device: str, spw: float, x, y, xv, yv):
    """Fit the chosen booster; early-stops on the validation split where supported."""
    if backend == "xgboost":
        import xgboost as xgb

        model = xgb.XGBClassifier(
            n_estimators=1500,
            learning_rate=0.05,
            max_depth=8,
            min_child_weight=5,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=spw,
            tree_method="hist",
            device=device,
            eval_metric="aucpr",
            early_stopping_rounds=100,
            random_state=42,
        )
        model.fit(x, y, eval_set=[(xv, yv)], verbose=50)
        print(f"[train] best_iteration={model.best_iteration}", flush=True)
        return model

    if backend == "lightgbm":
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=100,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=spw,
            device_type=device if device == "cuda" else "cpu",
            random_state=42,
        )
        model.fit(
            x,
            y,
            eval_set=[(xv, yv)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)],
        )
        return model

    if backend == "sklearn":
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(
            max_iter=600,
            learning_rate=0.06,
            max_leaf_nodes=63,
            min_samples_leaf=200,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.05,
            n_iter_no_change=40,
            class_weight={0: 1.0, 1: spw},
            random_state=42,
        )
        model.fit(x, y)
        print(f"[train] finished {model.n_iter_} iterations", flush=True)
        return model

    raise ValueError(f"Unknown backend: {backend}")


def _predict_raw(model, backend: str, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, 1]


def _save_model(model, backend: str, out_dir: Path) -> str:
    if backend == "xgboost":
        path = out_dir / "model.json"
        model.save_model(path)
    elif backend == "lightgbm":
        path = out_dir / "model.txt"
        model.booster_.save_model(str(path))
    else:
        import joblib

        path = out_dir / "model.joblib"
        joblib.dump(model, path)
    return path.name


# ----------------------------------------------------------------- thresholding


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

    # Operational fallback: fixed top-risk rate bands, guaranteed sane load.
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
            float(((proba >= hold_thr) & (y_true == 1)).sum())
            / max(int((proba >= hold_thr).sum()), 1),
            4,
        ),
        "val_review_band_precision": round(
            float(((proba >= review_thr) & (proba < hold_thr) & (y_true == 1)).sum())
            / max(int(((proba >= review_thr) & (proba < hold_thr)).sum()), 1),
            4,
        ),
    }
    return float(hold_thr), float(review_thr), summary


def _recall_at_precision(y_true: np.ndarray, proba: np.ndarray, target: float) -> float:
    precision, recall, _ = precision_recall_curve(y_true, proba)
    mask = precision >= target
    return round(float(recall[mask].max()) if mask.any() else 0.0, 4)


# ------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fraud-risk booster.")
    base = Path("data/processed/ibm_full")
    parser.add_argument("--train", type=Path, default=base / "train.parquet")
    parser.add_argument("--val", type=Path, default=base / "validation.parquet")
    parser.add_argument("--test", type=Path, default=base / "test.parquet")
    parser.add_argument("--out", type=Path, default=Path("artifacts/models/baseline"))
    parser.add_argument(
        "--backend", choices=["xgboost", "lightgbm", "sklearn"], default="xgboost"
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--feature-set", choices=["full", "online", "velocity"], default="full",
        help="online = features computable at realtime scoring; velocity = "
        "online + strictly-past streaming features from the Layer 1 replay.",
    )
    parser.add_argument(
        "--velocity-dir", type=Path, default=None,
        help="Velocity replay dir (artifacts/data/velocity/<split>) joined by "
        "transaction_id; required when --feature-set velocity.",
    )
    parser.add_argument("--max-train-rows", type=int, default=None, help="Smoke-test cap.")
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--hold-precision", type=float, default=0.80)
    parser.add_argument("--review-precision", type=float, default=0.30)
    args = parser.parse_args()

    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.feature_set == "velocity" and args.velocity_dir is None:
        raise SystemExit("--feature-set velocity requires --velocity-dir <split replay dir>")
    if args.feature_set == "velocity":
        columns = ONLINE_VELOCITY_FEATURE_COLUMNS
    elif args.feature_set == "online":
        columns = ONLINE_FEATURE_COLUMNS
    else:
        columns = FEATURE_COLUMNS

    split_vel = (
        {"train": args.velocity_dir / "train",
         "validation": args.velocity_dir / "validation",
         "test": args.velocity_dir / "test"}
        if args.feature_set == "velocity"
        else {}
    )
    train_raw = _load_featured(args.train, args.max_train_rows, split_vel.get("train"))
    val_raw = _load_featured(args.val, args.max_val_rows, split_vel.get("validation"))
    test_raw = _load_featured(args.test, args.max_test_rows, split_vel.get("test"))
    print(f"[priors] fitting on train period ({train_raw.height:,} rows)", flush=True)

    merchant_rates = fit_merchant_priors(train_raw)
    merchant_shares = fit_frequency_shares(train_raw, "merchant_id")
    mcc_shares = fit_frequency_shares(train_raw, "merchant_category_code")

    train_df = _attach_priors(train_raw, merchant_rates, merchant_shares, mcc_shares)
    val_df = _attach_priors(val_raw, merchant_rates, merchant_shares, mcc_shares)
    test_df = _attach_priors(test_raw, merchant_rates, merchant_shares, mcc_shares)
    del train_raw, val_raw, test_raw

    x_train, y_train = _matrix(train_df, columns)
    x_val, y_val = _matrix(val_df, columns)
    x_test, y_test = _matrix(test_df, columns)
    del train_df, val_df, test_df

    positives = int(y_train.sum())
    spw = float(len(y_train) - positives) / max(positives, 1)
    print(
        f"[train] backend={args.backend} device={args.device} set={args.feature_set} "
        f"rows={len(y_train):,} pos={positives:,} spw={spw:.1f}",
        flush=True,
    )

    model = _fit_backend(args.backend, args.device, spw, x_train, y_train, x_val, y_val)

    val_proba = calibrate_probability(_predict_raw(model, args.backend, x_val), spw)
    val_metrics = {
        "average_precision": round(float(average_precision_score(y_val, val_proba)), 4),
        "roc_auc": round(float(roc_auc_score(y_val, val_proba)), 4),
    }
    hold_thr, review_thr, policy = _threshold_policy(
        y_val, val_proba, args.hold_precision, args.review_precision
    )
    print(f"[validate] {val_metrics} | policy={policy}", flush=True)

    test_proba = calibrate_probability(_predict_raw(model, args.backend, x_test), spw)
    test_actions = np.where(
        test_proba >= hold_thr, "hold", np.where(test_proba >= review_thr, "review", "allow")
    )
    test_metrics = {
        "rows": int(len(y_test)),
        "frauds": int(y_test.sum()),
        "average_precision": round(float(average_precision_score(y_test, test_proba)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, test_proba)), 4),
        "recall_at_precision_25": _recall_at_precision(y_test, test_proba, 0.25),
        "recall_at_precision_50": _recall_at_precision(y_test, test_proba, 0.50),
        "recall_at_precision_80": _recall_at_precision(y_test, test_proba, 0.80),
        "action_counts": {a: int((test_actions == a).sum()) for a in ("allow", "review", "hold")},
        "caught_frauds_by_action": {
            a: int(((test_actions == a) & (y_test == 1)).sum())
            for a in ("allow", "review", "hold")
        },
    }
    print(f"[TEST - locked] {test_metrics}", flush=True)

    model_file = _save_model(model, args.backend, args.out)
    (args.out / "merchant_fraud_priors.json").write_text(json.dumps(merchant_rates))
    (args.out / "merchant_share.json").write_text(json.dumps(merchant_shares))
    (args.out / "mcc_share.json").write_text(json.dumps(mcc_shares))

    config = {
        "model_name": (
            f"{args.backend}_{args.feature_set}_v3"
            if args.feature_set == "velocity"
            else f"{args.backend}_{args.feature_set}_v2"
        ),
        "backend": args.backend,
        "model_file": model_file,
        "device_used": args.device,
        "created_at": datetime.now(UTC).isoformat(),
        "feature_columns": columns,
        "feature_set": args.feature_set,
        "calibration_scale_pos_weight": round(spw, 2),
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
