# Cross-Dataset Generalization Prep — Kaggle Runbook

Objective: prove our pipeline is not overfit to the IBM dataset by running the
**exact same FraudTransformer + XGBoost** code on two totally different
public fraud datasets, with the same leakage-safe discipline:

- **BankShield-2M** — HuggingFace `Abdulmajeedyahya/BankShield-Enterprise-Grade-Multi-Pattern-Dataset`
  (2M rows, 5 tables, multi-country, ~0.84% fraud rate).
- **IEEE-CIS** — Kaggle `ieee-fraud-detection` (academic benchmark, 590K
  transactions, device/identity tables).

> **Honesty rule (read first).** This runbook builds the *data-prep + train
> + evaluate* path. It does **not** promise a number — the numbers come from
> running this on Kaggle's free T4. Our realistic, defensible target: the
> FraudTransformer holds ROC-AUC ≥ 0.82 on both unseen datasets. If it does
> not, that is a real finding, we record it, and we say so in the pitch.

---

## 1. Why two more datasets

| Dataset | Fraud rate | Tables | Why it matters |
|---|---|---|---|
| IBM (current) | 0.099% | 1 | primary; chronological 60/20/20 splits |
| BankShield-2M | 0.84% | 5 (accounts, merchants, terminal, transaction, fraud) | 20× denser fraud; industrial multi-pattern; tests whether our *feature engineering* generalizes |
| IEEE-CIS | ~0.019% | 3 (train, transaction, identity) | the academic gold standard; judges recognize it; tests temporal robustness |

The pitch line: *"trained on IBM, validated on BankShield-2M and IEEE-CIS —
same architecture, same leakage-safe splits, no retuning."*

---

## 2. Prereqs (done locally already)

- `src/fingraph_sentinel/fraud_transformer.py` — pure-PyTorch causal temporal
  transformer (no `transformers` dependency).
- `src/fingraph_sentinel/train_fraud_transformer.py` — CLI trainer with
  strict chronological splits, focal loss, early stopping, locked-test gate.
- `src/fingraph_sentinel/features.py` / `train_baseline.py` — the same GBDT
  baseline for fair comparison (needs features to exist on the new tables).

---

## 3. BankShield-2M prep (HF dataset)

```bash
# On Kaggle / Colab (needs hf hub; run as the USER, not the agent)
pip install -q "datasets>=2.19" pyarrow polars

python - <<'EOF'
from datasets import load_dataset
import polars as pl

ds = load_dataset("Abdulmajeedyahya/BankShield-Enterprise-Grade-Multi-Pattern-Dataset")
# 5 tables: accounts, merchants, terminal, transaction, fraud
tables = {k: pl.DataFrame(ds[k].to_polars() if hasattr(ds[k], "to_polars") else pl.from_arrow(ds[k].data.table)) for k in ds}
for k, t in tables.items():
    print(k, t.shape)
EOF
```

Mapping to OUR canonical columns (see `dataset.py` → `IBM_CANONICAL_COLUMNS`):

| BankShield field | Our canonical | Notes |
|---|---|---|
| `transaction_id` | `transaction_id` | key |
| `customer_id` / `account_id` | `customer_id` | use account-level id |
| `transaction_time` / `timestamp` | `event_time` | parse to UTC `datetime` |
| `amount` | `amount` | float |
| `merchant_id` | `merchant_id` | text |
| `terminal_id` | `terminal_id` | text |
| `mcc` / `category` | `merchant_category_code` | text |
| `channel` / `txn_type` | `payment_channel` | text |
| `is_fraud` / `fraud_label` | `is_fraud` | 0/1 |

Prepared frame → `/kaggle/working/bankshield.parquet` with columns:
`transaction_id, customer_id, event_time, amount, merchant_id, merchant_category_code, payment_channel, payment_error, is_fraud`

Then run our own trainer unchanged:

```bash
cd /kaggle/working/build-x20
python -m fingraph_sentinel.train_fraud_transformer \
  --train /kaggle/working/bankshield.parquet \
  --val  /kaggle/working/bankshield_val.parquet \
  --test /kaggle/working/bankshield_test.parquet \
  --device cuda --epochs 12 --batch-size 512 --max-len 64 \
  --out artifacts/models/fraud-transformer-bankshield
```

> Splits are chronological 60/20/20 by `event_time` (same rule as IBM). Use
> the same `--window-start` trick if a prefix slice is fraud-free:
> `--window-start 0 --limit 400000` smoke first.

---

## 4. IEEE-CIS prep (Kaggle dataset)

```bash
# attach "IEEE-CIS Fraud Detection" dataset to the notebook
python - <<'EOF'
import polars as pl

tr = pl.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv",
                 infer_schema_length=10000)
idf = pl.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv",
                  infer_schema_length=10000)
# join identity onto transaction (left join on TransactionID)
df = tr.join(idf, on="TransactionID", how="left")
df = df.with_columns(
    pl.col("TransactionDT").cast(pl.Int64).alias("event_unix"),
    pl.col("TransactionAmt").cast(pl.Float64).alias("amount"),
    pl.col("isFraud").cast(pl.Int8).alias("is_fraud"),
)
# event_time from TransactionDT (seconds since a reference; monotonic => ok)
from datetime import datetime, UTC
ref = datetime(2017, 1, 1, tzinfo=UTC)  # arbitrary monotonic anchor
df = df.with_columns(
    (pl.lit(ref) + pl.duration(seconds="event_unix")).alias("event_time"),
    pl.lit("UNK").alias("merchant_category_code"),
    pl.lit(pl.col("ProductCD").cast(pl.Utf8)).alias("payment_channel"),
    pl.lit("None").alias("payment_error"),
    pl.col("TransactionID").cast(pl.Utf8).alias("transaction_id"),
    pl.lit("UNK").alias("customer_id"),   # IEEE-CIS has no customer id
)
out = df.select(["transaction_id", "customer_id", "event_time", "amount",
                 "merchant_category_code", "payment_channel", "payment_error",
                 "is_fraud"]).drop_nulls("amount")
out.write_parquet("/kaggle/working/ieee_cis.parquet")
print(out.shape)
EOF
```

Then identical trainer call:

```bash
python -m fingraph_sentinel.train_fraud_transformer \
  --train /kaggle/working/ieee_cis.parquet \
  --val  /kaggle/working/ieee_cis_val.parquet \
  --test /kaggle/working/ieee_cis_test.parquet \
  --device cuda --epochs 12 --batch-size 512 --max-len 64 \
  --out artifacts/models/fraud-transformer-ieee
```

> IEEE-CIS has no customer id, so sequences collapse to single events with
> `prev_amount_ratio=0.0` and `interval_log1p=0.0` — the model degrades to a
> plain cross-sectional transformer on amount + channel + error. That is fine;
> the point is measuring generalization of the *training loop + loss + eval
> discipline*, and it sets an honest lower bound.

---

## 5. What to record back (report these numbers, no fabrication)

After each run, copy from `model_config.json`:

```
metrics_validation.roc_auc   (val AUC)
metrics_test_locked.roc_auc  (locked test AUC, only exists when --limit was None)
metrics_test_locked.average_precision
```

Update `docs/METRICS.md` → new section "Cross-dataset generalization" with
real rows. **Never** quote the 0.9784 / 97.3% SOTA paper numbers as ours —
those are on the authors' own splits, not ours.

---

## 6. Failure is a finding

If BankShield or IEEE-CIS ROCC-AUC **< 0.80**, that is an honest, documented
generalization gap → the pitch says: *"we know where the edge is; the drift
switcher and cross-dataset validation tell us when to stop trusting the
model."* Judges value that over a fabricated 0.97.