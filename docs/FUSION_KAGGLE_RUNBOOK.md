# Full-Data Fusion with GNN — Kaggle Runbook

Objective: fuse all four signals — XGBoost serving baseline (online),
velocity model, Autoencoder, and the temporal heterogeneous GNN — into one
calibrated stacker, evaluated on the **locked test split** (permanent
scoreboard: `docs/METRICS.md`).

> **Honesty rule (read first).** Every number in METRICS.md is real. The
> fusion numbers below are *to be recorded after you run them* — this doc
> never invents metrics. The single most important caveat is the
> **row-alignment contract** in §3; violating it silently corrupts every
> downstream number.

---

## 1. What is fused, and why

| Signal | Where it comes from | Role |
|---|---|---|
| XGBoost serving baseline (`online_v2`) | `make train-baseline-online` | strong realtime features |
| Velocity model (`velocity_v3`) | velocity replay parquet + `make train-baseline-velocity` | strictly-past velocity (1h/24h/7d counts, amounts, distinct merchants; cumulative priors) |
| Autoencoder (AE) | `make train-ae` | unsupervised anomaly signal on the same 12 online features |
| Temporal Heterogeneous GNN | Kaggle T4 notebook `kaggle/train_gnn_t4.ipynb` | graph-structure signal (merchant-customer purchase graph over time) |

`src/fingraph_sentinel/ensemble_fusion.py` trains three GBDTs on the base
features, scores train/val/test with them (plus AE), then a **stacker** learns
to weight the four signals, calibrates, and writes `model_config.json` with
`metrics_validation` / `metrics_test_locked`.

---

## 2. Local capped proof (already run)

Before the full Kaggle pass, the fusion path is proven locally in capped mode:

```
make fusion-smoke        # 300K train / 120K val / 80K test rows
```

Real recorded numbers (METRICS.md, 2026-08-29):

| Model | Val ROC | Test ROC | Test AP |
|---|---|---|---|
| 4-signal fusion stacker (xgb+lgbm+catboost+AE) | 0.8190 | 0.6266 | 0.0015 |
| XGBoost serving baseline (for comparison) | 0.8937 | 0.5967 | 0.0015 |

The smoke run is **not** a claim that fusion beats the baseline — note fusion's
val ROC (0.8190) is *below* the baseline's (0.8937); the stacker's test ROC
(0.6266) edges ahead. A capped row subset trained differently per split makes
this an approximation, not a final verdict.

---

## 3. The row-alignment contract (⚠️ read this twice)

`ensemble_fusion fit` reads the four parquets in **row order** and expects the
GNN score stream to line up positionally:

```
need = len(train.parquet) + len(validation.parquet) + len(test.parquet)
# ibm_full: 14,632,145 + 4,877,380 + 4,877,375 = 24,386,900

g = pl.read_parquet(gnn_scores.parquet)["score"]
assert len(g) >= need          # g[0:train] -> train, [train:train+val] -> val, rest -> test
```

**Current GNN output does NOT satisfy this silently.** `gnn_scores.parquet`
emitted by `train_gnn.py` carries only `split` + `score` columns (no
`transaction_id`) and its train/val/test sizes depend on how snapshot *months*
partition against the calendar cutoffs `--event-cutoffs 534 568`
(`event_split_months` in `train_gnn.py` assigns each monthly snapshot by its
dominant calendar month). The baseline's 60/20/20 split cuts *events*, so the
two partitions can disagree at month boundaries.

**Before gluing GNN scores into fusion, VERIFY:**

```python
import polars as pl
g = pl.read_parquet("gnn_scores.parquet")
print(g.group_by("split").len())          # must print 14,632,145 / 4,877,380 / 4,877,375
```

If the counts differ from the table above by even one row, **do NOT glue**.
Options, in order of preference:

1. **Align on `transaction_id`.** Modify the GNN scorer (or a post-step in the
   notebook) to also emit `transaction_id` per edge, then left-join the score
   onto the baseline parquet by id so the stream becomes exactly
   train→val→test ordered. This is the honest fix and the recommended path.
2. If the month partition is exactly equal at these cutoffs, positional gluing
   is safe *for this dataset version* — but still print the counts into the
   run record (notebook cell output) so the evidence is committed.

The existing stacked fusion *without* GNN (`fusion-smoke`) is already valid;
GNN adds the 4th signal only when the alignment above is proven.

---

## 4. Kaggle steps (T4, ~2h quota)

The notebook `kaggle/train_gnn_t4.ipynb` already contains:

- **cell 4** — build temporal snapshots from the parquets (yearly buckets);
- **cell 7** — self-supervised pre-training (optional, label-free);
- **cell 8** — event-aligned retrain: `--event-cutoffs 534 568`, real
  architecture (hidden 192, 3 layers, 8 heads), 40 epochs, patience 8,
  `--with-sage` baseline, initialized from pre-trained embeddings when present;
- **cell 10** — archive `rhea_gnn_artifacts.zip` (gnn_config.json +
  gnn_scores.parquet) to the Output page.

After the notebook run, download `rhea_gnn_artifacts.zip`, unzip, and record
the GNN's own numbers (printed by cell 10) — then run the alignment verify
(§3) on the extracted `gnn_scores.parquet`.

### 4b. Full-data velocity training (optional but recommended)

The velocity replay is cheap and local:

```
make velocity-replay        # ~3 min locally: train 114s + val 25s + test 26s
```

Local capped training (already done, 2.5M-row cap to stay Mac-cool):

```
make train-baseline-velocity          # full 14.6M; ~2h on CPU
# or capped local proof (real numbers, see METRICS.md):
python -m fingraph_sentinel.train_baseline --feature-set velocity \
  --velocity-dir artifacts/data/velocity \
  --max-train-rows 2500000 --max-val-rows 1000000 --max-test-rows 1000000 \
  --out artifacts/models/baseline-online-v3
```

Full-data velocity training on the T4 is a one-command swap inside the
notebook (`device=cuda`, no caps) and the same `model_config.json` contract.

---

## 5. Fusion on the T4 (full data, with GNN)

With the alignment verified (§3) and the artifact paths in place:

```
python -m fingraph_sentinel.ensemble_fusion fit \
  --gnn-score-file <path>/gnn_scores.parquet \
  --out artifacts/models/ensemble-fusion-full
```

Record the printed `val_metrics` / `test_metrics` (and the per-model val AUC
block) into `docs/METRICS.md` under a new `## Full-data fusion (T4)` section,
**including the alignment evidence**: the printed split counts from §3 and the
`gnn_scores.parquet` row counts.

If you prefer to run fusion locally with the downloaded GNN scores, the same
command works on the Mac with `--smoke` caps for a quick sanity pass first.

---

## 6. What gets recorded (honest framing)

- Fusion **train** numbers are meaningless (in-sample by construction) and are
  never recorded.
- **Val** ROC/AP is the tuning signal; **test locked** is the scoreboard row.
- Every entry stays tied to its exact command (reproducibility) — a number
  without its command is not a number.
- If the GNN does not align (§3), fusion-with-GNN stays **not published**,
  exactly as METRICS.md currently states, until a run with proven alignment
  exists.