# Repair-Model Promotion Gate (Helix v2)

## Purpose

Helix v2 remembers post-hoc feedback (`failure_memory.jsonl`), derives merchant
hot-lists and threshold overrides, and can train a **capped repair model** on
the remembered episodes (`train_repair`). This doc defines the honest gate that
decides whether a repair model may actually **replace the serving model**.

> The current boundary (in code and docs) is deliberate: `train_repair` emits
> `metrics_in_sample` only and states *"promotion to serving is
> manual/Kaggle"*. This gate makes that manual decision **explicit,
> evidence-driven, and leakage-safe** — it does not rubber-stamp promotions.

---

## 1. Why a gate at all (honest framing)

The repair model's training data is the **failure memory itself**: episodes the
deployed model mis-scored. Three traps make naive promotion dangerous:

1. **In-sample optimism.** Evaluating the repair model on the same episodes it
   trained on inflates every metric. `metrics_in_sample` is exactly that and is
   never a promotion argument.
2. **Feature-space mismatch.** The repair model sees a *compact hand-built
   vector* (`amount_log1p`, channel one-hots, `merchant_is_hot`,
   `merchant_failure_rate`) while the serving model sees 40 features
   (online + velocity). A head-to-head must therefore be scored **on a shared
   feature representation** or be caveated as not apples-to-apples.
3. **Feedback bias.** Memory only holds transactions the model saw — misses and
   holds — so a repair model trained on it is conditioned on a skewed slice of
   reality. Positive labels dominate by construction once feedback lands.

The gate therefore requires **held-out evidence**, not memory.

---

## 2. The gate procedure

Let `S` = serving model, `R` = repair model, and `L` = a **locked held-out
slice** of events with true labels that was **never used** to build the repair
model (see §3 for constructing one).

1. **Record feedback** for a sample of recent decisions (API:
   `POST /api/v1/healing/feedback`, `outcome: fraud|legit`) so memory has
   real episodes.
2. **Train the repair model on memory only**:
   ```
   python -m fingraph_sentinel.healing train-repair \
     --model-dir artifacts/models/baseline-online-xgb \
     --healing-dir artifacts/healing \
     --max-rows 2000 --out-dir artifacts/healing/repair-candidate
   ```
   Output must say `trained: true` and report `metrics_in_sample` (recorded for
   transparency, never used as the promotion argument).
3. **Score `L` with both models** on the **same feature vector**. Two options:
   - (a) *Shared-feature head-to-head*: feed both models the repair feature
     vector (repair model natively; serving model's scores on the same vector
     are NOT its real scores — this measures "repair's features vs serving's
     features", not the models). Caveat explicitly.
   - (b) *Native head-to-head (recommended)*: score `L` with `S` on its real
     40-feature vector (velocity overlays via the runtime), and with `R` on
     its native vector, and compare **decision quality on `L`** (fraud recall
     among `hold`+`review`, hold precision) rather than raw AUC across
     different feature spaces. Report both AUCs with the caveat in §1.2.
4. **Apply the gate criteria**, all must hold:
   - `R` beats `S` on the held-out slice on the **primary decision metric**
     (default: fraud caught among actions ≠ allow, or hold-set precision) by a
     margin ≥ 2× the slice's sampling noise (e.g., at least 5 additional caught
     frauds on ≥ 200-fraud slice), AND
   - `R` does not regress the other metric (no precision drop > 0.05), AND
   - the slice `L` is large enough to trust (≥ 500 held-out frauds for a
     promotion claim; otherwise the gate returns `insufficient_evidence`).
5. **Promote** = copy repair `model.json` + `model_config.json` into the
   serving dir **after** the gate passes, with a `promoted_by` + `gate_id`
   field in HELIX_MEMORY.md / METRICS.md. **No gate pass = no promotion.**
   A helper script `scripts/`-style one-liner or `Makefile` target
   (`promote-repair`) may automate the copy, but the *decision* stays manual
   and is recorded in the docs.

---

## 3. Constructing the locked held-out slice (no leakage)

`L` must be excluded from the repair model's training data (the failure
memory). Recommended construction, reusing the existing products:

1. Take the **test split** events (`data/processed/ibm_full/test.parquet`) —
   none of them can ever enter `failure_memory.jsonl` (feedback is collected
   only for *served* decisions; simply never record feedback for test rows in
   experiments).
2. Slice `L` = the **last 20% of the test split by time** and hold it out from
   any experiment entirely until the gate runs.
3. Record the slice's `transaction_id` set into the run record
   (`artifacts/healing/gate_L_ids.json`) so re-runs reuse the same locked slice
   (a changed `L` invalidates comparisons).

The velocity replay guarantees strictly-past features on any chronological
slice (see `docs/FUSION_KAGGLE_RUNBOOK.md` §4b), so scoring `L` with velocity
features is leakage-safe by construction.

---

## 4. Honest failure modes (write these into the run record)

- `R` wins on `metrics_in_sample` but loses on `L` → **expected** (trap 1);
  do not promote, record the gap as evidence the gate works.
- `R` wins on `L` only via option (b) with different feature spaces → promote
  **with the caveat** that the win may be a feature-representation artifact,
  and plan a real shared-representation evaluation on the T4.
- Slice too small / too few frauds → return `insufficient_evidence` (the gate
  prefers "no decision" over a wrong one).

---

## 5. Where the evidence lives

- `artifacts/healing/repair-candidate/model_config.json` — the candidate's
  self-reported in-sample metrics (labeled as such).
- `artifacts/healing/gate_report.json` — gate inputs (slice size, frauds,
  per-model metric table) and verdict.
- `docs/HELIX_MEMORY.md` / `docs/METRICS.md` — the human-readable record of
  what was promoted, when, and on what evidence.

No invented numbers here: the gate report is produced by a script you run, and
its verdict is only as good as the honesty of the slice construction.