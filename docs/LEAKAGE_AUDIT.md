# Leakage Audit — zero-future-leakage checklist

Every claim here is **code-verified**, not assumed. The audit applies to the
IBM pipeline (all four models) and to the FraudTransformer.

---

## 1. Temporal splits (dataset.py)

`write_temporal_splits` produces strict chronological 60/20/20 splits by
`event_time` and writes `data/processed/ibm_full/split_manifest.json`
(rows / min−max `event_time` / fraud rate per split).

**Verified (manifest, 2026-09-02):**

| Split | Rows | Min event_time | Max event_time | Gap to next |
|---|---|---|---|---|
| train | 14,632,145 | 1991-01-02 | 2014-07-02T12:06 | val starts +1 min |
| validation | 4,877,380 | 2014-07-02T12:07 | 2017-05-14T10:36 | test starts +1 min |
| test | 4,877,375 | 2017-05-14T10:37 | 2020-02-28T23:58 | — |

No overlap: each split's data ends before the next begins (`<=` vs `>`).
Same rule used by BankShield-2M / IEEE-CIS runbook.

## 2. Feature engineering (features.py) — past-only by construction

- `_entity_history(lf, entity, prefix)`: sorts by `(entity, event_time,
  transaction_id)`, then every cumulative aggregate is `.shift(1)` —
  **the current row is never included** in its own features (`cum_count`
  and `cum_sum` shifted, `prev_amount_ratio` = `amount / amount.shift(1)`,
  `time_since_prev` = gap to the *previous* event).
- Merchant prior `merch_txn_count_prior`: `cum_count().over(merchant_id)`
  `.shift(1)` — past purchases only.
- **Priors fitted on train only**: `fit_merchant_priors(train_frame)` and
  `fit_frequency_shares(train_frame, col)` take the *training* frame as their
  only input; the fitted maps are attached to validation/test frames, never
  refit on them.
- Static/calendar columns (hour, weekday, card-present, channel dummies) are
  either per-event or calendar-derived → no lookahead.

## 3. FraudTransformer (fraud_transformer.py) — unit-tested causality

The GPT-style temporal transformer uses a **hand-rolled causal attention
mask** (explicit additive `-inf` per head). PyTorch 2.13's
`TransformerEncoder` fast paths were empirically found to leak future tokens
(dup-tail test showed logit delta 0.39); the encoder was replaced with a
`_CausalBlock` whose masking is entirely under our control.

Unit tests (`tests/test_fraud_transformer.py`):
- `test_causal_future_not_seen`: appending 4 future tokens to a 4-token
  sequence changes token-3's logit by `< 1e-4` → **asserted**.
- `test_forward_shape_and_dtype`, `test_focal_loss_weights_positives_and_masks_padding`
  (padding positions contribute zero loss).

## 4. Training discipline (train_baseline.py + train_fraud_transformer.py)

- Chronological validation: the model is validated on the most recent
  (validation) fold — equivalent to time-series holdout CV.
- Anti-overfitting toolkit, all in place: L1/L2 (XGBoost reg params;
  FraudTransformer AdamW weight decay 1e-2), dropout 0.25 (+0.1 pos dropout),
  early stopping (XGBoost wrapper; FT patience 3 on val AUC), focal loss to
  fight label imbalance.
- **Locked test gate**: the test split is touched exactly once; FT computes
  `metrics_test_locked` only on uncapped runs (`--limit None`), smoke runs
  deliberately leave it `null` (see `smoke_note` in
  `artifacts/models/fraud-transformer/model_config.json`).
- Metrics: ROC-AUC / Average Precision / Precision@K (never F1 as the
  headline at 0.099% fraud rate) — `scripts/comprehensive_metrics.py`.

## 5. Score-stream honesty (drift switcher)

Drift detection runs on the serving model's score stream (inference-time
scores, no future labels). The auto-switch recommendation
(`artifacts/healing/switch_decision_latest.json`) uses each candidate's
**recorded** test ROC — read from disk, not recomputed on the fly.

---

### Result

No future leakage found in the audited paths; the one PyTorch fast-path leak
was found *because* of the audit and fixed in the same session
(commit 42403e8). Re-run `pytest tests/test_fraud_transformer.py` after any
change to the attention path.