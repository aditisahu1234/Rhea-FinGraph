"""Real business-impact numbers for the pitch (no invented metrics).

Rebuilds the EXACT velocity-v3 decision stream on the full locked test split
(same feature path, priors, thresholds as `make train-baseline-velocity`),
verifies parity against the recorded model_config metrics, then derives the
fraud-amount/revenue story and ATO-pattern evidence. Polars-native throughout
(no giant python dicts).

Assumptions (stated, never hidden):
- charged-back amount == the fraud event's transaction amount;
- amounts are read as USD and converted to INR at 83.5 for the pitch;
- the test window spans 33 months (months 568..601).

Writes artifacts/business_impact.json.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from fingraph_sentinel.features import build_feature_frame
from fingraph_sentinel.train_baseline import (
    _attach_priors,
    _matrix,
)

MODEL_DIR = Path("artifacts/models/baseline-online-v3")
TEST_PARQUET = Path("data/processed/ibm_full/test.parquet")
VELOCITY_DIR = Path("artifacts/data/velocity/test")
OUT = Path("artifacts/business_impact.json")
INR_PER_USD = 83.5
TEST_MONTHS = 33.0  # months 568..601 per the month-level split

VEL_COLS = [
    "card_v_24h_count", "cust_v_24h_count", "cust_v_7d_distinct_merchants",
    "merch_v_7d_count", "cust_txn_count_prior", "card_txn_count_prior",
    "cust_time_since_prev_log", "cust_prev_amount_ratio",
]


def main() -> None:
    started = time.time()
    cfg = json.loads((MODEL_DIR / "model_config.json").read_text())
    rec = cfg["metrics_test_locked"]
    print(f"[v3] recorded test_roc={rec['roc_auc']} actions={rec['action_counts']}",
          flush=True)

    base = pl.scan_parquet(TEST_PARQUET).select(
        ["transaction_id", "amount", "merchant_category_code"]
    )
    vel = pl.scan_parquet(f"{VELOCITY_DIR}/*.parquet")
    lf = build_feature_frame(pl.scan_parquet(TEST_PARQUET))
    lf = lf.join(vel, on="transaction_id", how="left")
    lf = lf.join(base, on="transaction_id", how="left")

    print("[features] collect full test (4.88M rows) ...", flush=True)
    frame = lf.collect()

    priors = {
        "merchant_rates": json.loads((MODEL_DIR / "merchant_fraud_priors.json").read_text()),
        "merchant_shares": json.loads((MODEL_DIR / "merchant_share.json").read_text()),
        "mcc_shares": json.loads((MODEL_DIR / "mcc_share.json").read_text()),
    }
    frame = _attach_priors(
        frame, priors["merchant_rates"], priors["merchant_shares"], priors["mcc_shares"]
    )

    columns = cfg["feature_columns"]
    x, y = _matrix(frame, columns)
    spw = float(cfg["calibration_scale_pos_weight"])
    print(f"[predict] {x.shape} spw={spw:.1f} ...", flush=True)

    booster = xgb.Booster()
    booster.load_model(str(MODEL_DIR / "model.json"))
    # Recorded metrics came from the early-stopped model (sklearn wrapper uses
    # best_iteration=108 => trees 0..108). Plain predict() = ALL trees, which
    # does NOT reproduce the recorded decision stream (val 0.8383 vs 0.8224).
    # The recorded thresholds/actions were ALSO applied on the RAW sigmoid
    # probability scale (reproduced here verbatim; calibrate_probability would
    # move every score into a ~1e-3..1e-1 band and break the parity check).
    proba = booster.predict(xgb.DMatrix(x), iteration_range=(0, 109))
    del x

    hold_thr = float(cfg["thresholds"]["hold"])
    review_thr = float(cfg["thresholds"]["review"])
    actions = np.where(
        proba >= hold_thr, "hold", np.where(proba >= review_thr, "review", "allow")
    )
    frame = frame.with_columns(
        pl.Series("score", proba.astype(np.float64)),
        pl.Series("action", actions),
        pl.Series("y", y.astype(np.int64)),
    )
    del proba, actions, y

    # ---- parity check: the decision stream must equal the recorded config
    roc = float(roc_auc_score(frame["y"].to_numpy(), frame["score"].to_numpy()))
    ap = float(average_precision_score(frame["y"].to_numpy(), frame["score"].to_numpy()))
    act_counts = {
        a: int((frame["action"] == a).sum()) for a in ("allow", "review", "hold")
    }
    print(f"[parity] roc={roc:.4f} (rec {rec['roc_auc']}) ap={ap:.4f} "
          f"(rec {rec['average_precision']}) actions={act_counts}", flush=True)
    if abs(roc - rec["roc_auc"]) > 0.001 or act_counts != rec["action_counts"]:
        raise SystemExit(
            "PARITY FAILED — decision stream differs from recorded config; "
            "do NOT publish business numbers off a divergent stream."
        )
    print("[parity] PASS — identical decision stream", flush=True)

    fraud = frame.filter(pl.col("y") == 1)
    caught = fraud.filter(pl.col("action") != "allow")
    allowed_legit = frame.filter((pl.col("action") == "allow") & (pl.col("y") == 0))
    held_fraud = frame.filter((pl.col("action") == "hold") & (pl.col("y") == 1))

    total_fraud_amt = float(fraud["amount"].sum())
    caught_amt = float(caught["amount"].sum())
    missed_amt = total_fraud_amt - caught_amt
    by_action = {
        a: {
            "count": int(caught.filter(pl.col("action") == a).height),
            "amount_usd": round(float(caught.filter(pl.col("action") == a)["amount"].sum()), 2),
        }
        for a in ("hold", "review")
    }
    for a, v in by_action.items():
        v["amount_inr"] = round(v["amount_usd"] * INR_PER_USD, 2)

    # ATO-pattern evidence from strictly-past stream features
    ato: dict = {}
    for col in VEL_COLS:
        hf = held_fraud[col].to_numpy().astype(np.float64)
        al = allowed_legit[col].to_numpy().astype(np.float64)
        hf = hf[~np.isnan(hf)]
        al = al[~np.isnan(al)]
        hf_mean = float(hf.mean()) if hf.size else 0.0
        al_mean = float(al.mean()) if al.size else 0.0
        ato[col] = {
            "held_fraud_mean": round(hf_mean, 3),
            "allowed_legit_mean": round(al_mean, 3),
            "lift": round(hf_mean / al_mean, 2) if al_mean > 1e-9 else None,
            "held_fraud_p90": round(float(np.percentile(hf, 90)), 3) if hf.size else 0.0,
        }

    top_mcc = (
        fraud.group_by("merchant_category_code")
        .agg(pl.col("amount").sum().alias("amt"))
        .sort("amt", descending=True)
        .head(5)
        .to_dicts()
    )

    report = {
        "as_of": datetime.now(UTC).isoformat(),
        "model": "baseline-online-v3 (velocity features)",
        "split": "full test.parquet (4,877,375 rows, months 568-601, 33 months)",
        "parity": {"roc_auc_recomputed": round(roc, 4),
                   "ap_recomputed": round(ap, 4),
                   "matches_recorded_config": True},
        "assumptions": {
            "chargeback_amount_equals_event_amount": True,
            "currency": "amounts read as USD, converted to INR at 83.5",
            "test_window_months": TEST_MONTHS,
        },
        "totals": {
            "rows": int(frame.height),
            "frauds": int(fraud.height),
            "fraud_amount_usd": round(total_fraud_amt, 2),
            "fraud_amount_inr": round(total_fraud_amt * INR_PER_USD, 2),
        },
        "actions": act_counts,
        "caught_by_action": by_action,
        "protection": {
            "frauds_caught": int(caught.height),
            "recall_by_count": round(int(caught.height) / max(int(fraud.height), 1), 4),
            "fraud_amount_caught_usd": round(caught_amt, 2),
            "recall_by_amount": round(caught_amt / total_fraud_amt, 4),
            "fraud_amount_missed_usd": round(missed_amt, 2),
            "fraud_amount_caught_inr": round(caught_amt * INR_PER_USD, 2),
            "per_month_protected_inr": round(caught_amt * INR_PER_USD / TEST_MONTHS, 2),
            "per_month_missed_inr": round(missed_amt * INR_PER_USD / TEST_MONTHS, 2),
        },
        "ato_evidence": ato,
        "top_mcc_by_fraud_amount": [
            {"mcc": m["merchant_category_code"],
             "fraud_amount_inr": round(float(m["amt"]) * INR_PER_USD, 2)}
            for m in top_mcc
        ],
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[done] {OUT} in {time.time() - started:.1f}s")
    p = report["protection"]
    print(f"  frauds caught {p['frauds_caught']}/{report['totals']['frauds']} "
          f"({p['recall_by_count']:.1%} by count, {p['recall_by_amount']:.1%} by amount)")
    print(f"  protected/month INR {p['per_month_protected_inr']:,.0f} | missed/month "
          f"INR {p['per_month_missed_inr']:,.0f}")


if __name__ == "__main__":
    main()