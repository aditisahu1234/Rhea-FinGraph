"""Ensemble-fusion orchestrator (Layer 4): heterogeneous stack.

Combines the Layer-4 signals into one calibrated risk score:

    base models  XGBoost + LightGBM + CatBoost (trained fresh on the
                 online feature set, positive-class weighted)
    + anomaly    autoencoder reconstruction error (committed artifact)
    + optional   GNN score stream (a parquet with event_id -> score,
                 produced by the Kaggle GNN scorer; wired behind
                 --gnn-score-file so nothing breaks until it arrives)

Meta-model: logistic regression on logit-transformed base probabilities,
fit on TRAIN stacked scores only -> evaluated on VAL, TEST locked.
Leakage-safe: every base model's train scores come from the same fit
that produced its val/test scores; the stacker never sees val/test.

macOS note: torch's libiomp conflicts with xgboost/lightgbm/catboost
OpenMP runtimes (hard crash or deadlock) unless every GBDT runs with
--n-jobs 1 and OMP_NUM_THREADS=1 -- the Makefile targets encode this.
On Kaggle/Linux the GBDTs are not imported into a torch process until
after they finish, so full parallelism is safe there.

CLI:
    python -m fingraph_sentinel.ensemble_fusion fit --smoke
    python -m fingraph_sentinel.ensemble_fusion fit --gnn-score-file <f.parquet>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import lightgbm  # noqa: F401  (imported first: macOS OpenMP runtime order)
import numpy as np
import polars as pl
import xgboost  # noqa: F401

from fingraph_sentinel.train_baseline import (
    _attach_priors,
    _load_featured,
    _matrix,
    calibrate_probability,
)

# NOTE torch is intentionally NOT imported at module level: torch ships its
# own libiomp on macOS, and xgboost/lightgbm/catboost hard-crash when their
# OpenMP runtimes initialize after torch's. Keeping the GBDT imports above
# and importing torch lazily inside load_ae/ae_scores avoids the segfault.
try:  # catboost optional (auto-skip when missing)
    import catboost  # noqa: F401
except ImportError:  # pragma: no cover - exercised only on minimal envs
    catboost = None  # type: ignore[assignment]

BASE = Path("data/processed/ibm_full")
PRIORS_DIR = Path("artifacts/models/baseline-online-xgb")
AE_DIR = Path("artifacts/models/anomaly-ae")
OUT = Path("artifacts/models/ensemble-fusion")
PRIOR_FILES = ("merchant_fraud_priors.json", "merchant_share.json",
               "mcc_share.json")


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _pos_weight(y: np.ndarray) -> float:
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    return neg / pos if pos > 0 else 1.0


def load_frame(
    path: Path, max_rows: int | None
) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    priors = tuple(
        json.loads((PRIORS_DIR / name).read_text()) for name in PRIOR_FILES
    )
    df = _attach_priors(_load_featured(path, max_rows), *priors)
    x, y = _matrix(df, ON_FEATURES)
    return x, y, df


def train_gbdt(backend: str, x: np.ndarray, y: np.ndarray,
               n_jobs: int = 4) -> tuple[object, float]:
    """Train one GBDT on the (capped) train matrix; return (model, pos_weight)."""
    w = _pos_weight(y)
    if backend == "xgboost":
        import xgboost as xgb

        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=w,
            objective="binary:logistic", n_jobs=n_jobs, tree_method="hist",
            random_state=42,
        )
    elif backend == "lightgbm":
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=w,
            n_jobs=n_jobs, random_state=42, verbosity=-1,
        )
    else:  # catboost
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            iterations=200, depth=6, learning_rate=0.05,
            scale_pos_weight=w, thread_count=n_jobs, verbose=False,
            random_seed=42,
        )
    model.fit(x, y)
    return model, w


def predict_calibrated(model: object, backend: str, x: np.ndarray,
                       w: float) -> np.ndarray:
    p = model.predict_proba(x)[:, 1]  # same API for all three backends
    return np.asarray(calibrate_probability(p, w), dtype=np.float64)


def load_ae(ae_dir: Path = AE_DIR) -> object:
    from fingraph_sentinel.anomaly_autoencoder import Autoencoder

    cfg = json.loads((ae_dir / "ae_config.json").read_text())
    model = Autoencoder(
        in_dim=len(cfg["feature_columns"]),
        hidden=tuple(cfg["hidden_dims"]),
        dropout=0.1,
    )
    import torch

    state = torch.load(ae_dir / "ae.pt", map_location="cpu")
    model.load_state_dict(state if not hasattr(state, "state_dict") else state.state_dict())
    model.eval()
    model.mean = np.load(ae_dir / "scaler_mean.npy")  # type: ignore[attr-defined]
    model.std = np.load(ae_dir / "scaler_std.npy")  # type: ignore[attr-defined]
    return model


def ae_scores(model: object, x: np.ndarray) -> np.ndarray:
    import torch

    from fingraph_sentinel.anomaly_autoencoder import (
        reconstruct_error_scores,
        standardize,
    )

    xs = standardize(x, model.mean, model.std)  # type: ignore[attr-defined]
    return reconstruct_error_scores(model, xs, torch.device("cpu"))


def fit_stack(
    train_scores: np.ndarray, y_train: np.ndarray,
    val_scores: np.ndarray, y_val: np.ndarray,
    test_scores: np.ndarray, y_test: np.ndarray,
    names: list[str],
) -> dict:
    """LogisticRegression on logit(prob) meta-features; returns metrics."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score

    def feats(s: np.ndarray) -> np.ndarray:
        out = []
        for j, name in enumerate(names):
            col = s[:, j]
            if name.endswith("_ae") or name.endswith("_gnn"):
                out.append(col)  # unbounded anomaly / gnn score, keep raw
            else:
                out.append(_logit(col))
        return np.column_stack(out)

    stacker = LogisticRegression(C=1.0, max_iter=2000)
    stacker.fit(feats(train_scores), y_train)
    p_val = stacker.predict_proba(feats(val_scores))[:, 1]
    p_test = stacker.predict_proba(feats(test_scores))[:, 1]

    def metrics(p: np.ndarray, y: np.ndarray) -> dict:
        if y.sum() == 0 or y.sum() == len(y):
            return {"rows": int(len(y)), "frauds": int(y.sum()),
                    "roc_auc": None, "average_precision": None}
        return {
            "rows": int(len(y)), "frauds": int(y.sum()),
            "roc_auc": float(roc_auc_score(y, p)),
            "average_precision": float(average_precision_score(y, p)),
        }

    return {
        "stacker": stacker,
        "names": names,
        "p_val": p_val,
        "p_test": p_test,
        "val_metrics": metrics(p_val, y_val),
        "test_metrics": metrics(p_test, y_test),
    }


def main():
    ap = argparse.ArgumentParser(description="Ensemble-fusion orchestrator")
    ap.add_argument("--train", type=Path, default=BASE / "train.parquet")
    ap.add_argument("--val", type=Path, default=BASE / "validation.parquet")
    ap.add_argument("--test", type=Path, default=BASE / "test.parquet")
    ap.add_argument("--max-train-rows", type=int, default=None)
    ap.add_argument("--max-val-rows", type=int, default=None)
    ap.add_argument("--max-test-rows", type=int, default=None)
    ap.add_argument("--ae-dir", type=Path, default=AE_DIR)
    ap.add_argument("--gnn-score-file", type=Path, default=None,
                    help="Optional GNN (event_id, score) stream parquet")
    ap.add_argument("--skip-catboost", action="store_true",
                    help="Force-disable CatBoost (import may be missing)")
    ap.add_argument("--n-jobs", type=int, default=4,
                    help="Threads per GBDT (use 1 on macOS alongside torch)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--smoke", action="store_true",
                    help="Capped-row CPU smoke run")
    args = ap.parse_args()

    if args.smoke:
        args.max_train_rows = args.max_train_rows or 300_000
        args.max_val_rows = args.max_val_rows or 120_000
        args.max_test_rows = args.max_test_rows or 80_000

    global ON_FEATURES
    ON_FEATURES = json.loads(
        (PRIORS_DIR / "model_config.json").read_text()
    )["feature_columns"]

    x_train, y_train, _ = load_frame(args.train, args.max_train_rows)
    x_val, y_val, _ = load_frame(args.val, args.max_val_rows)
    x_test, y_test, _ = load_frame(args.test, args.max_test_rows)
    print(f"[fusion] matrices: train {x_train.shape} val {x_val.shape} "
          f"test {x_test.shape}")

    backends = ["xgboost", "lightgbm"]
    if not args.skip_catboost and catboost is not None:
        backends.append("catboost")

    models = {}
    for backend in backends:
        t0 = time.time()
        model, w = train_gbdt(backend, x_train, y_train, n_jobs=args.n_jobs)
        models[backend] = (model, w)
        print(f"[fusion] {backend:10s} fit {time.time()-t0:.1f}s "
              f"(pos_weight {w:.1f})")

    names: list[str] = list(backends) + ["ae"]
    ae = load_ae(args.ae_dir)
    cols: dict[str, dict[str, np.ndarray]] = {}
    for split_name, x, y in (("train", x_train, y_train),
                             ("val", x_val, y_val),
                             ("test", x_test, y_test)):
        cols[split_name] = {
            backend: predict_calibrated(model, backend, x, w)
            for backend, (model, w) in models.items()
        }
        cols[split_name]["ae"] = ae_scores(ae, x)

    gnn_included = args.gnn_score_file is not None
    if gnn_included:
        gnn_df = pl.read_parquet(args.gnn_score_file)
        if "score" not in gnn_df.columns:
            raise ValueError(f"{args.gnn_score_file} needs a 'score' column")
        g = gnn_df["score"].to_numpy().astype(np.float64)
        need = len(y_train) + len(y_val) + len(y_test)
        if len(g) < need:
            raise ValueError(
                f"gnn stream has {len(g)} rows, need >= {need} "
                "(train+val+test, split order)"
            )
        n_tr, n_va = len(y_train), len(y_val)
        cols["train"]["gnn"] = g[:n_tr]
        cols["val"]["gnn"] = g[n_tr : n_tr + n_va]
        cols["test"]["gnn"] = g[n_tr + n_va : need]
        names.append("gnn")
        print(f"[fusion] gnn score stream included ({len(g):,} rows)")

    def stack(name: str) -> np.ndarray:
        return np.column_stack([cols[name][b] for b in names])

    t0 = time.time()
    res = fit_stack(stack("train"), y_train, stack("val"), y_val,
                    stack("test"), y_test, names)
    print(f"[fusion] stacker fit {time.time()-t0:.1f}s "
          f"({len(names)} signals)")

    print("\n  split    rows   frauds   roc_auc   avg_precision")
    for split_name, y in (("val", y_val), ("test", y_test)):
        m = res["val_metrics"] if split_name == "val" else res["test_metrics"]
        auc = "n/a" if m["roc_auc"] is None else f"{m['roc_auc']:.4f}"
        ap = "n/a" if m["average_precision"] is None else f"{m['average_precision']:.5f}"
        print(f"  {split_name:7s} {m['rows']:>7,} {m['frauds']:>7,} "
              f"{auc:>9s} {ap:>14s}")

    # per-model val AUC for the config
    from sklearn.metrics import roc_auc_score

    per_model = {}
    for name in names:
        p = cols["val"][name]
        try:
            per_model[name] = round(float(roc_auc_score(y_val, p)), 4)
        except ValueError:
            per_model[name] = None

    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(res["stacker"], args.out / "stacker.joblib")
    np.save(args.out / "p_val.npy", res["p_val"])
    np.save(args.out / "p_test.npy", res["p_test"])
    config = {
        "signals": names,
        "gnn_included": gnn_included,
        "gnn_score_file": str(args.gnn_score_file) if gnn_included else None,
        "n_train": int(len(y_train)),
        "per_model_val_roc_auc": per_model,
        "metrics_validation": res["val_metrics"],
        "metrics_test_locked": res["test_metrics"],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (args.out / "fusion_config.json").write_text(
        json.dumps(config, indent=2)
    )
    print(f"\n[fusion] artifacts -> {args.out}")


if __name__ == "__main__":
    main()