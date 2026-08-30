"""Local repair-promotion gate sim (Item 3 closing proof).

Pipeline (all local, leakage-safe):
  1. Memory episodes: last 500K rows of validation.parquet (chronologically
     'served' feedback with true labels).
  2. Candidate repair model: healing.train_repair on those episodes (capped).
  3. Locked gate slice L: last 800K rows of test.parquet — never touched by
     memory or repair training.
  4. Score L with the SERVING baseline (online features, via build_feature_frame)
     and with the REPAIR model (native compact features).
  5. Emit gate_report.json with the verdict (pass / fail / insufficient_evidence).
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from fingraph_sentinel.failure_memory import Episode, FailureMemory
from fingraph_sentinel.healing import HealingEngine
from fingraph_sentinel.train_baseline import build_feature_frame

ROOT = Path(".")
SERVING_DIR = ROOT / "artifacts/models/baseline-online-xgb"
GATE_DIR = ROOT / "artifacts/healing"
# Fixed chrono position ranges (never overlap: memory from validation,
# gate slice from test). Each holds ~1-1.7K frauds in 800K rows.
MEMORY_LO, MEMORY_HI = 3_000_000, 3_800_000
GATE_LO, GATE_HI = 3_000_000, 3_800_000
FRAUD_LABEL = "is_fraud"


def _slice(split: str, lo: int, hi: int) -> pl.DataFrame:
    df = pl.scan_parquet(f"data/processed/ibm_full/{split}.parquet").with_row_index()
    return df.filter((pl.col("index") >= lo) & (pl.col("index") < hi)).collect()


def _episode_events(df: pl.DataFrame) -> list[tuple[str, dict]]:
    """(transaction_id, event-dict) pairs in row order (true labels = is_fraud)."""
    out = []
    for r in df.iter_rows(named=True):
        ev = {
            "transaction_id": r["transaction_id"],
            "event_time": r["event_time"].isoformat(),
            "customer_id": str(r["customer_id"]),
            "card_id": str(r["card_id"]),
            "merchant_id": str(r["merchant_id"]),
            "amount": str(r["amount"]),
            "payment_channel": str(r["payment_channel"] or "").lower(),
        }
        out.append((r["transaction_id"], ev))
    return out


def main() -> None:
    t0 = time.time()
    GATE_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1) memory = validation rows [MEMORY_LO, MEMORY_HI) (chronological feedback)
    mem_df = _slice("validation", MEMORY_LO, MEMORY_HI)
    mem_path = GATE_DIR / "failure_memory.jsonl"
    if mem_path.exists():
        mem_path.unlink()
    mem = FailureMemory(mem_path)
    label_by_id = dict(zip(mem_df["transaction_id"].to_list(),
                           mem_df[FRAUD_LABEL].to_list()))
    for tid, ev in _episode_events(mem_df):
        outcome = "fraud" if label_by_id[tid] == 1 else "legit"
        mem.record(Episode(
            transaction_id=tid, model_version="baseline-online-xgb",
            action="allow", fraud_probability=0.0, outcome=outcome,
            event=ev, reasons=[], source="sim",
        ))
    n_mem = len(mem.episodes())
    print(f"[gate] memory episodes: {n_mem:,} "
          f"({mem.stats()['failures']:,} frauds)", flush=True)

    # ---- 2) candidate repair model
    engine = HealingEngine(model_dir=SERVING_DIR, healing_dir=GATE_DIR)
    res = engine.train_repair(out_dir=GATE_DIR / "repair-candidate",
                              max_rows=MEMORY_HI - MEMORY_LO)
    print(f"[gate] train-repair: {json.dumps(res, indent=2, default=str)}", flush=True)
    if not res["trained"]:
        print(json.dumps({"verdict": "not_trained", **res}, indent=2))
        return

    # ---- 3) locked gate slice L = test rows [GATE_LO, GATE_HI)
    L = _slice("test", GATE_LO, GATE_HI)
    y = L[FRAUD_LABEL].to_numpy()
    print(f"[gate] locked slice L: {len(L):,} rows, {int(y.sum()):,} frauds", flush=True)

    # ---- 4a) serving baseline scores on L (real online features)
    svc_cfg = json.loads((SERVING_DIR / "model_config.json").read_text())
    on_cols = svc_cfg["feature_columns"]
    import xgboost as xgb
    serving = xgb.XGBClassifier()
    serving.load_model(str(SERVING_DIR / "model.json"))
    X_on = build_feature_frame(
        pl.scan_parquet("data/processed/ibm_full/test.parquet")
        .with_row_index()
        .filter((pl.col("index") >= GATE_LO) & (pl.col("index") < GATE_HI))
    ).collect().select(on_cols).to_numpy()
    p_serving = serving.predict_proba(X_on)[:, 1]
    auc_serving = float(roc_auc_score(y, p_serving))
    print(f"[gate] serving baseline on L: roc_auc={auc_serving:.4f}", flush=True)

    # ---- 4b) repair model scores on L (native compact features)
    import xgboost as rxg
    repair = rxg.Booster()
    repair.load_model(str(GATE_DIR / "repair-candidate" / "model.json"))
    roll = mem.merchant_rollup()
    feats = []
    for _, ev in _episode_events(L):
        amount = float(ev["amount"] or 0.0)
        ch = ev["payment_channel"]
        mid = ev["merchant_id"]
        r = roll.get(mid, {})
        feats.append([
            float(math.log1p(max(amount, 0.0))),
            1.0 if ch == "swipe" else 0.0,
            1.0 if ch == "chip" else 0.0,
            1.0 if ch == "online" else 0.0,
            1.0 if r.get("failures", 0) >= engine.min_failures_hot else 0.0,
            r.get("failures", 0) / r.get("txns", 1) if r.get("txns") else 0.0,
        ])
    X_rep = np.array(feats, dtype=np.float32)
    drep = rxg.DMatrix(X_rep, feature_names=[
        "amount_log1p", "channel_swipe", "channel_chip", "channel_online",
        "merchant_is_hot", "merchant_failure_rate"])
    p_repair = repair.predict(drep)
    auc_repair = float(roc_auc_score(y, p_repair))
    print(f"[gate] repair model on L: roc_auc={auc_repair:.4f}", flush=True)

    # top-5k fraud capture (5k ~= decision volume on this slice)
    k = 5000
    topk_serving = int(y[np.argsort(p_serving)[-k:]].sum())
    topk_repair = int(y[np.argsort(p_repair)[-k:]].sum())

    # ---- 5) verdict (doc: REPAIR_PROMOTION_GATE.md §2.4)
    n_fraud = int(y.sum())
    margin = auc_repair - auc_serving
    if n_fraud < 500:
        verdict = "insufficient_evidence"
    elif margin >= 0.02 and topk_repair >= topk_serving + 5:
        verdict = "pass_with_caveat"
    elif margin <= -0.02 and topk_repair <= topk_serving - 5:
        verdict = "fail"
    else:
        verdict = "insufficient_evidence"

    report = {
        "verdict": verdict,
        "slice": {"rows": int(len(L)), "frauds": n_fraud,
                  "note": "test.parquet rows [3000000, 3800000) — fixed chrono "
                          "range, never in memory (see gate_L_ids.json)"},
        "memory": {"episodes": n_mem, "frauds": mem.stats()["failures"]},
        "serving": {"roc_auc": round(auc_serving, 4), "top5k_caught": topk_serving},
        "repair": {"roc_auc": round(auc_repair, 4), "top5k_caught": topk_repair,
                   "note": "native compact features; not apples-to-apples with "
                           "serving (different feature spaces, see doc §1.2). "
                           "pass_with_caveat => confirm on shared representation "
                           "on T4 before actual promotion"},
        "gate_criteria": {"min_frauds": 500, "margin": round(margin, 4),
                          "margin_required": 0.02},
        "gate_seconds": round(time.time() - t0, 1),
    }
    (GATE_DIR / "gate_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # lock the slice ids so re-runs reuse the same L (doc §3)
    (GATE_DIR / "gate_L_ids.json").write_text(
        json.dumps({"split": "test", "lo": GATE_LO, "hi": GATE_HI,
                    "ids": L["transaction_id"].to_list()}, indent=2),
        encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\n[done] {GATE_DIR / 'gate_report.json'}")


if __name__ == "__main__":
    sys.exit(main())