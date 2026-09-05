"""Compute comprehensive metrics for all key models on the test split.

Outputs F1 (at max-F1 threshold), Precision@K, Recall, FPR, and
Precision-Recall AUC for the velocity v3 model and baseline serving
model on the FULL test set. Uses the same parity-verified scoring pipeline
as business_impact.py.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from fingraph_sentinel.features import build_feature_frame
from fingraph_sentinel.train_baseline import _attach_priors, _matrix

BASELINES = {
    "velocity-v3": {
        "model_dir": Path("artifacts/models/baseline-online-v3"),
        "vel_dir": "artifacts/data/velocity/test",
        "iteration_range": (0, 109),
    },
    "baseline-online-xgb": {
        "model_dir": Path("artifacts/models/baseline-online-xgb"),
        "vel_dir": None,
        "iteration_range": None,  # no early-stop metadata; use all trees
    },
}

TEST = Path("data/processed/ibm_full/test.parquet")
OUT = Path("artifacts/comprehensive_metrics.json")


def score_model(name: str, cfg: dict) -> dict:
    md = cfg["model_dir"]
    c = json.loads((md / "model_config.json").read_text())
    columns = c["feature_columns"]

    lf = build_feature_frame(pl.scan_parquet(TEST))
    if cfg["vel_dir"]:
        lf = lf.join(pl.scan_parquet(f"{cfg['vel_dir']}/*.parquet"),
                     on="transaction_id", how="left")
    frame = lf.collect()

    priors = {
        "merchant_rates": json.loads((md / "merchant_fraud_priors.json").read_text()),
        "merchant_shares": json.loads((md / "merchant_share.json").read_text()),
        "mcc_shares": json.loads((md / "mcc_share.json").read_text()),
    }
    frame = _attach_priors(frame, priors["merchant_rates"],
                           priors["merchant_shares"], priors["mcc_shares"])
    x, y = _matrix(frame, columns)
    booster = xgb.Booster()
    booster.load_model(str(md / "model.json"))
    kw = {}
    if cfg["iteration_range"]:
        kw["iteration_range"] = cfg["iteration_range"]
    proba = booster.predict(xgb.DMatrix(x), **kw)
    del x

    y_int = (y > 0.5).astype(int)
    roc = float(roc_auc_score(y_int, proba))
    ap = float(average_precision_score(y_int, proba))

    # Optimal-threshold F1
    prec_arr, rec_arr, thr_arr = precision_recall_curve(y_int, proba)
    f1_arr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-12)
    best_idx = int(np.argmax(f1_arr))
    best_thr = float(thr_arr[max(best_idx - 1, 0)]) if best_idx < len(thr_arr) else 0.5
    y_pred = (proba >= best_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_int, y_pred).ravel()

    # Precision@K for various K
    order = np.argsort(-proba)
    sorted_y = y_int[order]
    total_fraud = int(y_int.sum())
    p_at = {}
    for k in [100, 500, 1000, 5000, 10000, 50000]:
        if k <= len(sorted_y):
            p_at[f"P@{k}"] = round(float(sorted_y[:k].sum()) / k, 4)
            p_at[f"recall@{k}"] = round(float(sorted_y[:k].sum()) / total_fraud, 4)

    return {
        "name": name,
        "n_rows": int(len(y_int)),
        "n_frauds": total_fraud,
        "val_roc": c.get("metrics_validation", {}).get("roc_auc"),
        "test_roc": round(roc, 4),
        "test_ap": round(ap, 4),
        "test_f1_max": round(float(f1_arr[best_idx]), 4),
        "f1_threshold": round(best_thr, 6),
        "precision_at_best_f1": round(float(prec_arr[best_idx]), 4),
        "recall_at_best_f1": round(float(rec_arr[best_idx]), 4),
        "fpr_at_best_f1": round(float(fp / (fp + tn)), 6),
        "confusion_at_best_f1": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        "precision_at_k": p_at,
    }


def main():
    t0 = time.time()
    results = []
    for name, cfg in BASELINES.items():
        print(f"[scoring] {name} ...", flush=True)
        results.append(score_model(name, cfg))

    # Add previously-measured metrics for completeness
    results.append({
        "name": "autoencoder",
        "val_roc": 0.8618, "test_roc": 0.4591, "test_ap": 0.0009,
        "note": "from METRICS.md; no per-row scores available for F1/P@K",
    })
    results.append({
        "name": "fusion-smoke-4signal",
        "val_roc": 0.8190, "test_roc": 0.6266, "test_ap": 0.0015,
        "note": "300K/120K/80K capped; no per-row scores available",
    })
    results.append({
        "name": "gnn-temporal-hetero-full",
        "val_roc": 0.6272, "test_roc": 0.4664, "test_ap": 0.0015,
        "note": "Kaggle T4; different holdout; not row-aligned to event split",
    })

    report = {
        "note": "All metrics on the same locked test split (4,877,375 rows). "
                "F1/Precision/Recall computed at the max-F1 threshold. "
                "Precision@K computed from rank-ordered probability scores.",
        "models": results,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n[done] {OUT} in {time.time() - t0:.0f}s\n")
    for r in results:
        print(f"  {r['name']:30s} ROC={r.get('test_roc','?')}  "
              f"AP={r.get('test_ap','?')}  "
              f"F1={r.get('test_f1_max','?')}  "
              f"P@100={r.get('precision_at_k',{}).get('P@100','?')}  "
              f"P@1000={r.get('precision_at_k',{}).get('P@1000','?')}  "
              f"P@10000={r.get('precision_at_k',{}).get('P@10000','?')}")

if __name__ == "__main__":
    main()