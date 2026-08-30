# Rhea FinGraph — honest model scoreboard

All numbers below come from real runs on the IBM synthetic dataset
(`data/processed/ibm_full/*.parquet`, chronological 60/20/20, leakage-safe).
No fabricated metrics. Where two models use different splits, that caveat is
noted; the numbers are honest *for each model's own split*.

## Scoreboard

| Model | Val ROC | Test ROC | Test AP | Source |
|---|---|---|---|---|
| XGBoost serving baseline (`baseline-online-xgb`) | **0.8937** | **0.5967** | 0.0015 | `make train-baseline-online` |
| Velocity model (`baseline-online-v3`, capped 2.5M-train) | 0.8224 | **0.7646** | 0.0038 | `make train-baseline-velocity` (capped) + full-set rescore |
| Autoencoder anomaly detector | 0.8618 | 0.4591 | 0.0009 | `make train-ae` (smoke) |
| 4-signal fusion stacker (xgb+lgbm+catboost+AE) | 0.8190 | 0.6266 | 0.0015 | `make fusion-smoke` (300K/120K/80K) |
| **Temporal Heterogeneous GNN (full graph)** | 0.6272 | 0.4664 | 0.0015 | Kaggle T4, `artifacts/graph/gnn_kaggle/` |

## Velocity model (2026-08-29)

Layer-1 velocity features (strictly-past 1h/24h/7d counts, amounts, distinct
merchants; cumulative priors) replayed chronologically through the production
`VelocityStore` semantics.

- **Replay**: vectorized polars twin of the live store, verified byte-parity
  against the serial oracle on 100K rows (worst abs diff 2.3e-12 ≤ 1e-6).
  Full replay (14.63M train / 4.88M val / 4.88M test): 114s + 25s + 26s locally.
- **Training (honest cap)**: 2.5M-row train slice (full-correct velocity
  features), best_iteration=108, 68.8s. Full-data 14.6M velocity training is a
  Kaggle step (~2h CPU; see `docs/FUSION_KAGGLE_RUNBOOK.md` §4b).
- **Full-set rescore** (4,877,380 val / 4,877,375 test rows, same sets as the
  serving baseline): val ROC **0.8224** (AP 0.0642), test ROC **0.7646**
  (AP 0.0038); test action counts allow 2.25M / review 282K / hold 2.34M,
  caught frauds 4,130 hold + 153 review of 4,833 test frauds.

### ⚠️ Promotion gate verdict: NOT promoted
`make promote-velocity` gate requires val ROC ≥ serving's 0.8937. Velocity v3
val ROC is **0.8224 < 0.8937 → gate rejects promotion**. Serve-online stays
`baseline-online-xgb`. The gate is doing its job — the capped model is not a
clean all-around win. Two honest observations for the story:

1. **Test ROC is dramatically higher (0.7646 vs 0.5967)** and val→test decay is
   far milder (0.822→0.765 vs 0.894→0.597): velocity features make ranking
   *robust to the channel shift* documented in the Layer 5 finding.
2. The gap is partly training-size (2.5M vs 14.6M rows). A full-data velocity
   run on the T4 is the next evidence step, not a promise it will beat 0.8937.

### Backend comparison on the same capped velocity slice (2026-08-30)

| Backend | Val ROC | Test ROC | Verdict |
|---|---|---|---|
| XGBoost (`baseline-online-v3`) | **0.8224** | **0.7646** | serving candidate (gate: not promoted yet) |
| LightGBM (same 2.5M slice) | 0.3175 | 0.7373 | **rejected — val worse than random**; early-stopped at iter 2 with logloss ~8.0. The trainer's pos-weight/calibration path for lightgbm on this feature set does not hold; XGBoost stays the velocity backend. |

Recorded from `make train-baseline-velocity` (xgb) and the lightgbm variant run.
LightGBM's test ROC looks high only because its broken val calibration
collapses to near-random ordering by design threshold fallback — the honest
read is the val ROC.

## Serving latency (Layer 0, 2026-08-30)

Per-event scoring path was re-loading the 2.6 MB booster + rebuilding a SHAP
TreeExplainer on EVERY event (~140 ms/event warm, >1 s cold). Now cached
in-process (mtime-keyed; see `docs/LATENCY.md` for the full table):

| Stage (full core path: velocity + features + predict + SHAP + observe) | mean | throughput |
|---|---|---|
| before fix (warm) | ~140 ms/event | ~7/s |
| **after fix** | **0.466 ms/event** | **2,148/s** |

Full HTTP round-trip adds ~1 ms (FastAPI + audit write) → realistic service
ceiling ≈ 1.5 ms/event on the MacBook.

## Repair-model promotion gate sim (2026-08-29)

Local end-to-end pass of the Helix v2 promotion loop
(`scripts/repair_gate_sim.py`), run exactly as documented in
`docs/REPAIR_PROMOTION_GATE.md`:

- **Memory**: validation rows `[3,000,000, 3,800,000)` — 800,000 episodes,
  1,691 frauds (simulated post-hoc feedback with true labels).
- **Candidate repair model**: `artifacts/healing/repair-candidate/`
  (in-sample recall 0.0993 / precision 0.875 — in-sample only, never the
  promotion argument).
- **Locked gate slice L**: test rows `[3,000,000, 3,800,000)` — 800,000 rows,
  1,160 frauds; ids locked in `artifacts/healing/gate_L_ids.json`.
- **Serving baseline on L**: ROC **0.5107**, top-5k caught **7** frauds.
- **Repair model on L**: ROC **0.5989**, top-5k caught **52** frauds.
  (margin +0.088 ≥ 0.02 gate, top-5k 52 ≥ 12 gate) → verdict
  **`pass_with_caveat`**.

### ⚠️ Caveat
The repair model's native feature space (amount + channel + merchant
hot/failure-rate) is not the serving model's 40-feature space, so this
head-to-head is **illustrative, not apples-to-apples**. Before any real
promotion, confirm on a **shared feature representation on the T4**
(see REPAIR_PROMOTION_GATE.md §2.4). The gate is a decision *record*, not a
rubber stamp; actual serving remains `baseline-online-xgb`.

## GNN (first full-data run, 2026-08-29)

- 30 yearly snapshots (months 21..50), 24.39M edges, 29,757 fraud edges
- T4 CUDA: hidden 32, 1 layer, 2 heads, 8 epochs, 46,113 params, 67.7s fit,
  no pre-train init
- GNN split is over the 30 buckets (60/20/20): train=buckets 0-18,
  val=18-24, test=24-30
  - val: 9,351,299 rows / 9,433 frauds -> ROC 0.6272
  - test: 8,915,708 rows / 11,693 frauds -> ROC 0.4664

### ⚠️ Caveats (honesty)
1. **Not a fair head-to-head with XGBoost.** The baseline's test is months
   2017-2020 (by the month-level split); the GNN's test is the last 6 *yearly*
   buckets. Different holdouts; read the GNN's val->test gap (0.627 -> 0.466)
   as the *same ranking-drift problem*, not as a baseline comparison.
2. **GNN score stream is not row-aligned to fusion.** `gnn_scores.parquet`
   (24.39M rows) is ordered by GNN split (train 6.12M / val 9.35M /
   test 8.92M), but the fusion feature matrix is `ibm_full` (train 14.63M /
   val 4.88M / test 4.88M). Positionally gluing them would be misleading.
   Fusion-with-GNN requires the GNN score stream regenerated on the
   event-aligned split.

## Layer 5 finding (why the baseline decays)

Level monitors (EWMA/CUSUM/PSI on score) stay flat (~0.0058 mean) while test
AUC falls, because the *input distribution* migrates:

| Feature | Train mean | Test mean | PSI |
|---|---|---|---|
| `channel_swipe` | 0.998 | 0.20 | ~5.9 |
| `channel_chip` | 0.000 | 0.79 | large |

`make helix` emits a per-feature drift table + retrain trigger from this.
