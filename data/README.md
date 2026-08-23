# Data workflow

The project uses IBM Synthetic Credit Card Transactions as its primary fraud benchmark.

1. Authenticate Kaggle CLI outside the repository.
2. Run `make data-download`.
3. Unzip the archive inside `data/raw/`.
4. Profile a CSV before any modelling:

```bash
.venv/bin/rhea-profile data/raw/<ibm-csv> --limit 500000 \
  --write-splits data/processed/ibm_500k
```

The split is chronological: earliest 60% train, next 20% validation, latest 20% held-out test. Feature aggregates and graph state must be fit on train data only, then advanced through validation and test in event-time order.

Raw and processed data remain untracked. The `split_manifest.json` produced beside the Parquet files is evidence of the held-out evaluation boundary.
