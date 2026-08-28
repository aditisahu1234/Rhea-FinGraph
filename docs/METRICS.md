# Rhea FinGraph — honest model scoreboard

All numbers below come from real runs on the IBM synthetic dataset
(`data/processed/ibm_full/*.parquet`, chronological 60/20/20, leakage-safe).
No fabricated metrics. Where two models use different splits, that caveat is
noted; the numbers are honest *for each model's own split*.

## Scoreboard

| Model | Val ROC | Test ROC | Test AP | Source |
|---|---|---|---|---|
| XGBoost serving baseline (`baseline-online-xgb`) | **0.8937** | **0.5967** | 0.0015 | `make train-baseline-online` |
| Autoencoder anomaly detector | 0.8618 | 0.4591 | 0.0009 | `make train-ae` (smoke) |
| 4-signal fusion stacker (xgb+lgbm+catboost+AE) | 0.8190 | 0.6266 | 0.0015 | `make fusion-smoke` (300K/120K/80K) |
| **Temporal Heterogeneous GNN (full graph)** | 0.6272 | 0.4664 | 0.0015 | Kaggle T4, `artifacts/graph/gnn_kaggle/` |

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
