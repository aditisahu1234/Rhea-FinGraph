# Strengthening the Temporal GNN — honest playbook

## Why the first full run looks weak (and is NOT a fair loss)

First full-data run: **val ROC 0.6272 / test ROC 0.4664**. Three honest reasons
this understates the GNN, and each maps to a concrete fix:

1. **The holdout is the hardest slice.** The GNN's test = the *last 6 yearly
   buckets* (the most fraud-dense era of the graph). XGBoost's test = the
   2017–2020 *month* window. Different splits → the two are **not** a fair
   head-to-head. Fix: re-run the GNN on the **same event-aligned chronological
   split** as the baseline (train→2014-07, val→2017-05, test→2020-02) so the
   comparison is apples-to-apples AND the score stream lines up for honest
   fusion.

2. **It was a smoke-tier architecture.** hidden 32 · 1 layer · 2 heads ·
   8 epochs · 46,113 params · `init_from: null`. That is intentionally
   undersized. SOTA temporal GNNs need depth + pre-training.

3. **It was blindfolded.** Node features are only **4-dim**. The GNN leans on
   pure topology while XGBoost gets 12 rich tabular features. Give the graph
   real node signals (amount velocity, fraud priors, degree) and it stops
   guessing.

## The honest pitch: it's the ENSEMBLE, not "GNN alone"

A temporal GNN is not meant to *replace* a strong GBDT on tabular features — it
adds the **structural/relational** signal no tabular model can see (fraud rings,
merchant-collusion motifs, time-evolving neighborhoods). The defensible headline
for Razorpay is:

> Tabular GBDT for feature risk + Temporal GNN for **structural** risk + Autoencoder
> for anomaly + Helix for **drift-awareness** → fused by a calibrated stacker.

So the goal is a fusion where the GNN contributes signal that lifts the ensemble
above the best single model — not to out-ROC XGBoost in isolation.

## Concrete upgrade path (ranked by impact)

> **Implementation status (2026-08-29):** code for items 1–3 is now **built,
> tested and in the repo** (commit after `be72496`). The Kaggle notebook
> `kaggle/train_gnn_t4.ipynb` is patched to run them one-click. What remains is
> the **T4 re-run itself** (a manual Kaggle step) and then honest fusion. See
> "Run it" at the end.

### 1. Fair, event-aligned split (do FIRST — enables everything else) ✅ built
- `train_gnn.py --event-cutoffs C0 C1` partitions snapshots by **calendar
  month-idx** instead of the raw bucket-count 60/20/20.
- use `--event-cutoffs 534 568` to match the XGBoost baseline exactly
  (train<2014-07=534, val 534..568=2017-05, test>=568).
- Then `gnn_scores.parquet` row-counts match `ibm_full` and it can be fed to
  `ensemble_fusion.py --gnn-score-file` **honestly** (the old count mismatch is
  gone for this split).

### 2. Richer node features ✅ built (4-dim -> 8-dim)
- Per-customer / merchant / card, strictly-past: **log1p count, log1p amount,
  log1p distinct counterparties, fraud rate, log1p fraud volume, log1p avg
  amount, partner density, log1p spend velocity** (spend per active month).
- **Latent bug fixed:** the old builder passed the yearly *bucket index* (21–50)
  as the history-cutoff but compared it against *calendar month-idx* (252–601),
  so the filter was always false and **every node feature was all-zero**. The
  GNN was effectively learning with no node signal. `calendar_cut = month *
  bucket_months` restores real node features (verified: later-snapshot customer
  features go from 0.00 to abs-mean ~2.1).

### 3. Real architecture + pre-train init ✅ built
- hidden 32 → **192**, layers 1 → **3**, heads 2 → **8**, `--dropout 0.2`,
  `--lr 1e-3`, **early stopping** `--patience 8`, train **40 epochs**.
- `--init-from <gnn_pretrained.pt>` wires the self-supervised pre-trained
  encoder (this run left it off) — the notebook auto-enables it.

### 4. Tune + calibrate, then fuse
- Sweep the usual suspects on the fair split (hidden, layers, lr, dim).
- Calibrate the GNN score stream and confirm per-split row alignment, then run
  the stacker with GNN as one more signal.
- Report the *delta*: ensemble-with-GNN vs ensemble-without-GNN on the same
  holdout. That delta is the honest, pitchable number.

## What we will NOT do (honesty guardrail)
- We will not cherry-pick a split or metric to "win". Every comparison will be
  on the same held-out window, and we will report what the data actually says —
  including a negative result if the GNN adds nothing, with the analysis of why.

## Effort split
- **Kaggle/T4 (you):** re-train with new architecture on the fair split once
  the code + split mode are ready (code pushed here for a one-click notebook).
- **Local (me):** feature pipelines, split-mode code, fusion wiring, honest
  metric tables, and the ensemble-delta report.

## Run it (one-click Kaggle re-train)

The notebook `kaggle/train_gnn_t4.ipynb` is patched to run the strong,
event-aligned config. To kick off the re-train:

1. Upload the **updated repo** to Kaggle (or re-copy `src/` + `kaggle/`); the
   notebook now calls `graph_snapshots` (8-dim node features) and
   `train_gnn --event-cutoffs 534 568 --hidden 192 --layers 3 --heads 8
   --epochs 40 --patience 8`.
2. In the notebook: **Session options → Accelerator → GPU T4 x2**; the Input
   dataset `rhea-fingraph-ibm-splits` (train/val/test parquets) must be
   attached.
3. **File → Save Version → Save & Run All (Commit)**, wait for *Save
   complete*, then download `rhea_gnn_artifacts.zip` from the Output page and
   send it back.
4. I ingest the new `gnn_scores.parquet` + `gnn_config.json` and run the honest
   fusion + ensemble-delta comparison against the (now event-aligned, fair)
   baseline.

