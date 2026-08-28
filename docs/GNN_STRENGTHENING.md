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

### 1. Fair, event-aligned split (do FIRST — enables everything else)
- Add an event-level temporal split mode to `train_gnn.py` so the GNN trains on
  *transactions* cut at the same train/val/test months as the baseline, not the
  raw bucket-count split.
- Then `gnn_scores.parquet` row-counts match `ibm_full` and it can be fed to
  `ensemble_fusion.py --gnn-score-file` **honestly** (currently blocked).

### 2. Richer node features (biggest raw accuracy lever per effort)
- Per-card / per-customer / per-merchant **rolling amount velocities** and
  **fraud-rate priors** (reuse the priors JSONs already computed).
- Node **degree + 2-hop neighbor fraud density** as structural priors.
- Add as additional node feature dims (4-dim → ~12–16-dim).

### 3. Real architecture + pre-train init
- hidden 32 → **128–192**, layers 1 → **2–3**, heads 2 → **4–8**, add dropout
  and a learning-rate schedule.
- Enable `--init-from pretrain_gnn` — the GraphMAE pre-trained encoder already
  built in `pretrain_gnn.py` exists exactly for this (this run left it off).
- Train **20–40 epochs** with early stopping on val AUC instead of 8 flat.

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
