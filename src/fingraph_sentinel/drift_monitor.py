"""Concept-drift monitoring for the risk ensemble (Layer 4).

Watches the score stream for distribution drift so retraining / Helix
memory refresh can be triggered -- this is the honest counterpart to the
observed val->test AUC collapse of the XGBoost baseline (0.89 -> 0.60).

Two complementary detectors on per-window statistics (default: monthly):

  * EWMA  -- exponentially weighted moving average of the mean score (and
             fraud rate when labels are available); alarming z-score comes
             from the training-period baseline distribution.
  * CUSUM -- Page's cumulative-sum control chart (two-sided); signals when
             the cumulative deviation from the training mean exceeds h.

Plus the Population Stability Index (PSI) of the score histogram vs the
train reference -- a standard model-monitoring signal.

Reference statistics are ALWAYS fitted on train-period scores only; each
later window (val, then each test month) is compared against them.

Run:
    python -m fingraph_sentinel.drift_monitor score-streams   # produce scores.parquet
    python -m fingraph_sentinel.drift_monitor monitor         # monthly drift report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from fingraph_sentinel.train_baseline import (
    _attach_priors,
    _load_featured,
    _matrix,
    calibrate_probability,
)

DEFAULT_MODEL_DIR = Path("artifacts/models/baseline-online-xgb")
DEFAULT_SCORES = DEFAULT_MODEL_DIR / "scores.parquet"
PSI_WARN = 0.25  # standard PSI warning threshold


# ── detector math (pure numpy) ────────────────────────────────────────────
def ewma_series(values: np.ndarray, span: float = 20.0) -> np.ndarray:
    """Exponentially weighted moving average (span-based smoothing)."""
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def cusum_series(
    values: np.ndarray,
    baseline_mean: float,
    baseline_std: float,
    k: float = 0.5,
    h: float = 5.0,
    two_sided: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Page's CUSUM of standardized deviations.

    Returns (chart, fired), where the chart resets to 0 after a signal and
    ``fired[i]`` marks the indices where the statistic crossed ``h`` (so the
    caller can report the exact alert month).
    """
    z = (values - baseline_mean) / max(baseline_std, 1e-9)
    n = len(values)
    pos = np.zeros(n)
    neg = np.zeros(n)
    fired = np.zeros(n, dtype=bool)
    for i in range(n):
        if i == 0:
            pos[i] = max(0.0, z[i] - k)
            neg[i] = max(0.0, -z[i] - k)
        else:
            pos[i] = max(0.0, pos[i - 1] + z[i] - k)
            neg[i] = max(0.0, neg[i - 1] - z[i] - k)
        if pos[i] > h or neg[i] > h:
            fired[i] = True
            pos[i] = 0.0
            neg[i] = 0.0
    chart = np.maximum(pos, neg) if two_sided else pos
    return chart, fired


def _logit(p: np.ndarray) -> np.ndarray:
    """Logit transform (score -> log-odds) before PSI binning."""
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def psi_score(reference: np.ndarray, observed: np.ndarray, bins: int = 10) -> float:
    """Population stability index of two score distributions."""
    if reference.size == 0 or observed.size == 0:
        return float("nan")
    edges = np.quantile(reference, np.linspace(0.0, 1.0, bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_hist, _ = np.histogram(reference, bins=edges)
    obs_hist, _ = np.histogram(observed, bins=edges)
    ref_p = ref_hist / ref_hist.sum()
    obs_p = obs_hist / obs_hist.sum()
    ref_p = np.clip(ref_p, 1e-6, None)
    obs_p = np.clip(obs_p, 1e-6, None)
    return float(np.sum((obs_p - ref_p) * np.log(obs_p / ref_p)))


# ── windowing + reference fitting ──────────────────────────────────────────
def monthly_windows(df: pl.DataFrame) -> list[tuple[str, pl.DataFrame]]:
    """Split a score stream into (year-month, window) pairs, oldest first."""
    out: list[tuple[str, pl.DataFrame]] = []
    month_col = pl.col("event_time").dt.strftime("%Y-%m")
    for m in df.select(month_col.alias("mo")).to_series().unique().sort().to_list():
        out.append((m, df.filter(month_col == m)))
    return out


def window_stats(window: pl.DataFrame) -> dict:
    score = window["score"].to_numpy()
    stats = {
        "rows": int(window.height),
        "mean_score": float(score.mean()),
        "std_score": float(score.std()),
    }
    if "is_fraud" in window.columns:
        y = window["is_fraud"].to_numpy()
        stats["fraud_rate"] = float(y.mean())
        stats["frauds"] = int(y.sum())
    return stats


def monitor_report(
    scores: pl.DataFrame,
    reference_months: int = 3,
    ewma_span: float = 3.0,
    cusum_k: float = 0.5,
    cusum_h: float = 5.0,
) -> dict:
    """Reference = LAST `reference_months` train windows (the model as
    shipped); walk = every window AFTER the train block (val, then test),
    compared against that reference.

    Level detectors (z / CUSUM) use the month-to-month volatility of the
    reference windows' mean scores, not the per-row std -- a shift must
    exceed the historical month-to-month variation to alarm. PSI is
    computed on logit(score) bins (raw probs here are concentrated near
    0.005, where tiny level moves would look like massive shift).
    """
    if "split" not in scores.columns:
        raise ValueError("scores need a 'split' column (train/val/test)")
    train_f = scores.filter(pl.col("split") == "train")
    walk_f = scores.filter(pl.col("split") != "train")
    train_windows = monthly_windows(train_f)
    tail = train_windows[-reference_months:]
    if len(tail) < reference_months:
        raise ValueError("not enough train windows for a reference tail")
    ref_scores = np.concatenate([w["score"].to_numpy() for _, w in tail])
    ref_means = np.array([w["score"].to_numpy().mean() for _, w in tail])
    ref_mean = float(ref_scores.mean())
    ref_std = float(max(ref_means.std(), 1e-9))  # month-to-month volatility

    walk = monthly_windows(walk_f)
    if not walk:
        raise ValueError("no val/test windows to walk")

    rows = []
    for m, window in walk:
        stats = window_stats(window)
        stats["month"] = m
        stats["z_mean_score"] = float(
            (stats["mean_score"] - ref_mean) / ref_std
        )
        logit = _logit(window["score"].to_numpy())
        stats["psi"] = psi_score(_logit(ref_scores), logit)
        rows.append(stats)

    series = [r["mean_score"] for r in rows]
    ewma = ewma_series(np.array(series), span=ewma_span)
    chart, fired = cusum_series(
        np.array(series), baseline_mean=ref_mean, baseline_std=ref_std,
        k=cusum_k, h=cusum_h,
    )
    alerts = {
        "cusum_first_alert_month": None,
        "psi_first_alert_month": None,
    }
    for i, row in enumerate(rows):
        row["ewma_mean_score"] = float(ewma[i])
        row["cusum_stat"] = float(chart[i])
        if fired[i] and alerts["cusum_first_alert_month"] is None:
            alerts["cusum_first_alert_month"] = row["month"]
        if row["psi"] > PSI_WARN and alerts["psi_first_alert_month"] is None:
            alerts["psi_first_alert_month"] = row["month"]

    return {
        "reference": {
            "months": [m for m, _ in tail],
            "n_rows": int(ref_scores.size),
            "mean_score": ref_mean,
            "std_score_monthly_means": ref_std,
        },
        "detectors": {
            "ewma_span": ewma_span,
            "cusum_k": cusum_k,
            "cusum_h": cusum_h,
            "psi_warn": PSI_WARN,
        },
        "windows": rows,
        "alerts": {k: v for k, v in alerts.items() if v is not None},
    }


def print_report(report: dict) -> None:
    print("month      rows     mean_score   z     psi     ewma    cusum  fraud_rate")
    for r in report["windows"]:
        print(
            f"{r['month']}  {r['rows']:>8,}  {r['mean_score']:>10.5f} "
            f"{r['z_mean_score']:>5.1f} {r['psi']:>7.3f} "
            f"{r['ewma_mean_score']:>7.5f} {r['cusum_stat']:>7.2f} "
            f"{r.get('fraud_rate', float('nan')):>10.5f}"
        )
    print(f"\nalerts: {report['alerts'] or 'none'}")


# ── CLI ────────────────────────────────────────────────────────────────────
def _score_splits(model_dir: Path, out: Path, max_train: int | None,
                  max_eval: int | None) -> None:
    import xgboost as xgb

    cfg = json.loads((model_dir / "model_config.json").read_text())
    booster = xgb.Booster()
    booster.load_model(model_dir / cfg["model_file"])
    scale = float(cfg.get("calibration_scale_pos_weight", 1.0))

    def score(path: Path, max_rows: int | None, split: str) -> pl.DataFrame:
        df = _attach_priors(
            _load_featured(path, max_rows),
            *(
                json.loads((model_dir / name).read_text())
                for name in ("merchant_fraud_priors.json", "merchant_share.json",
                             "mcc_share.json")
            ),
        )
        x, y = _matrix(df, cfg["feature_columns"])
        raw = booster.predict(xgb.DMatrix(x))
        p = calibrate_probability(raw, scale).astype(np.float32)
        return pl.DataFrame(
            {
                "event_time": df["event_time"],
                "score": p,
                "is_fraud": y.astype(np.int8),
                "split": [split] * df.height,
            }
        )

    base = Path("data/processed/ibm_full")
    frames = [
        score(base / "train.parquet", max_train, "train"),
        score(base / "validation.parquet", max_eval, "val"),
        score(base / "test.parquet", max_eval, "test"),
    ]
    full = pl.concat(frames)
    out.parent.mkdir(parents=True, exist_ok=True)
    full.write_parquet(out)
    print(f"[drift] wrote {out} ({full.height:,} rows)")


def main():
    parser = argparse.ArgumentParser(description="Concept-drift monitoring.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("score-streams", help="Score train/val/test with a saved model")
    ps.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ps.add_argument("--out", type=Path, default=DEFAULT_SCORES)
    ps.add_argument("--max-train-rows", type=int, default=None)
    ps.add_argument("--max-eval-rows", type=int, default=None)

    pm = sub.add_parser("monitor", help="Monthly EWMA/CUSUM/PSI drift report")
    pm.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    pm.add_argument("--reference-months", type=int, default=3)
    pm.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    if args.cmd == "score-streams":
        _score_splits(args.model_dir, args.out, args.max_train_rows,
                      args.max_eval_rows)
    else:
        if not args.scores.exists():
            raise SystemExit(f"no {args.scores} -- run 'drift_monitor score-streams' first")
        scores = pl.read_parquet(args.scores)
        report = monitor_report(scores, reference_months=args.reference_months)
        print_report(report)
        if args.out_json:
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            args.out_json.write_text(json.dumps(report, indent=2))
            print(f"[drift] report written to {args.out_json}")


if __name__ == "__main__":
    main()