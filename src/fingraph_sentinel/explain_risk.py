"""SHAP / LIME explainability harness for the serving XGBoost model (Layer 4).

Explains the model behind `POST /api/v1/transactions/score`: for any row of
the engineered feature matrix we produce feature attributions -- SHAP
(TreeExplainer margin-space values, monotone with the calibrated
probability) or LIME (surrogate on calibrated probabilities) -- and turn
them into the `reasons` list the API attaches to alerts.

CLI:
    python -m fingraph_sentinel.explain_risk batch --parquet path --n 2000
    python -m fingraph_sentinel.explain_risk one --row-idx 42
    python -m fingraph_sentinel.explain_risk lime --row-idx 7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from fingraph_sentinel.train_baseline import (
    _attach_priors,
    _load_featured,
    _matrix,
    calibrate_probability,
)

MODEL_DIR = Path("artifacts/models/baseline-online-xgb")
PRIOR_FILES = ("merchant_fraud_priors.json", "merchant_share.json",
               "mcc_share.json")


def load_serving_model(model_dir: Path = MODEL_DIR) -> dict:
    """Return booster + config + priors for the serving model."""
    import xgboost as xgb

    cfg = json.loads((model_dir / "model_config.json").read_text())
    booster = xgb.Booster()
    booster.load_model(model_dir / cfg["model_file"])
    priors = tuple(
        json.loads((model_dir / name).read_text()) for name in PRIOR_FILES
    )
    return {"booster": booster, "config": cfg, "priors": priors}


def _sigmoid(m: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-m))


def build_batch(
    parquet: Path, model: dict, max_rows: int | None
) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    """Feature matrix + labels + source frame for a split parquet."""
    cfg = model["config"]
    df = _attach_priors(
        _load_featured(parquet, max_rows), *model["priors"]
    )
    x, y = _matrix(df, cfg["feature_columns"])
    return x, y, df


def calibrated_probability(model: dict, raw_margin: np.ndarray) -> np.ndarray:
    """Map booster margin -> calibrated fraud probability."""
    scale = float(model["config"]["calibration_scale_pos_weight"])
    return np.asarray(
        calibrate_probability(_sigmoid(raw_margin), scale), dtype=np.float64
    )


def shap_batch(
    x: np.ndarray, model: dict, n_background: int = 2000
) -> dict:
    """TreeExplainer attributions for rows of `x`.

    shap 0.51 parses an xgboost Booster natively; values are margin-space
    (linear with the calibrated probability), the base margin is the
    expected value, and per-feature mean |SHAP| gives the global view.
    """
    import shap

    explainer = shap.TreeExplainer(model["booster"], model_output="raw")
    values = explainer.shap_values(x)  # shape (n, n_features)
    names = model["config"]["feature_columns"]
    return {
        "values": np.asarray(values, dtype=np.float64),
        "expected": float(np.asarray(explainer.expected_value)),
        "mean_abs": {
            name: float(np.abs(values[:, i]).mean())
            for i, name in enumerate(names)
        },
    }


def top_reasons(
    values_row: np.ndarray, names: list[str], expected: float, k: int = 3
) -> list[dict]:
    """Top-k |SHAP| reasons for one row, in API `reasons` shape."""
    order = np.argsort(-np.abs(values_row))
    out = []
    for i in order[:k]:
        val = float(values_row[i])
        out.append(
            {
                "feature": names[i],
                "attribution": val,
                "direction": "increases" if val > 0 else "decreases",
                "detail": (
                    f"SHAP {val:+.5f}; margin base {expected:.5f}"
                ),
            }
        )
    return out


def lime_explain(
    x: np.ndarray, row_idx: int, model: dict, feature_names: list[str],
    num_features: int = 8,
) -> dict:
    """LIME surrogate explanation for one row (calibrated probabilities)."""
    import lime.lime_tabular
    import xgboost as xgb

    scale = float(model["config"]["calibration_scale_pos_weight"])

    def predict_proba(x2: np.ndarray) -> np.ndarray:
        margin = model["booster"].predict(xgb.DMatrix(x2))
        p = calibrate_probability(_sigmoid(margin), scale)
        return np.column_stack([1.0 - p, p])

    explainer = lime.lime_tabular.LimeTabularExplainer(
        x, feature_names=feature_names, mode="classification",
        class_names=["allow", "hold"], discretize_continuous=True,
        random_state=42,
    )
    pred_class = int(np.argmax(predict_proba(x[row_idx : row_idx + 1])[0]))
    exp = explainer.explain_instance(
        x[row_idx], predict_proba, num_features=num_features,
        top_labels=1,
    )
    labels = {name: float(w) for name, w in exp.as_list(label=pred_class)}
    return {
        "row_idx": int(row_idx),
        "predicted_class": pred_class,
        "calibrated_probability": float(predict_proba(x[row_idx : row_idx + 1])[0, 1]),
        "weights": labels,
        "intercept": float(exp.intercept[pred_class]),
    }


def main():
    ap = argparse.ArgumentParser(description="Explainability harness (Layer 4)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("batch", help="SHAP attributions for a slice of a split")
    pb.add_argument("--parquet", type=Path,
                    default=Path("data/processed/ibm_full/validation.parquet"))
    pb.add_argument("--n", type=int, default=2000)
    pb.add_argument("--out", type=Path,
                    default=MODEL_DIR / "explain.parquet")

    po = sub.add_parser("one", help="Top reasons for a single row of the batch")
    po.add_argument("--parquet", type=Path,
                    default=Path("data/processed/ibm_full/validation.parquet"))
    po.add_argument("--n", type=int, default=2000)
    po.add_argument("--row-idx", type=int, default=0)

    plm = sub.add_parser("lime", help="LIME explanation for one row")
    plm.add_argument("--parquet", type=Path,
                     default=Path("data/processed/ibm_full/validation.parquet"))
    plm.add_argument("--n", type=int, default=2000)
    plm.add_argument("--row-idx", type=int, default=0)
    plm.add_argument("--num-features", type=int, default=8)
    args = ap.parse_args()

    model = load_serving_model()
    features = model["config"]["feature_columns"]

    if args.cmd == "batch":
        x, y, _ = build_batch(args.parquet, model, args.n)
        res = shap_batch(x, model, n_background=min(args.n, 2000))
        margin = model["booster"].predict(__import__("xgboost").DMatrix(x))
        prob = calibrated_probability(model, margin)
        rows = []
        for i in range(len(x)):
            rows.append(
                {
                    "row_idx": i,
                    "is_fraud": int(y[i]),
                    "score": float(prob[i]),
                    "reasons": top_reasons(
                        res["values"][i], features, res["expected"]
                    ),
                }
            )
        out = pl.DataFrame(
            {"row_idx": [r["row_idx"] for r in rows],
             "is_fraud": [r["is_fraud"] for r in rows],
             "score": [r["score"] for r in rows],
             "reasons_json": [json.dumps(r["reasons"]) for r in rows]}
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out.write_parquet(args.out)
        mean_abs = sorted(res["mean_abs"].items(), key=lambda kv: -kv[1])[:6]
        print(f"[shap] wrote {args.out} ({len(x):,} rows)")
        print("[shap] top features by mean|SHAP| (margin units):")
        for name, m in mean_abs:
            print(f"        {name:28s} {m:.5f}")

    elif args.cmd == "one":
        x, y, _ = build_batch(args.parquet, model, max(args.n, args.row_idx + 1))
        single = shap_batch(x[args.row_idx : args.row_idx + 1], model, 0)
        margin = model["booster"].predict(
            __import__("xgboost").DMatrix(x[args.row_idx : args.row_idx + 1])
        )
        prob = calibrated_probability(model, margin)[0]
        print(f"row {args.row_idx}: predicted score {prob:.6f} "
              f"(label fraud={int(y[args.row_idx])})")
        for r in top_reasons(single["values"][0], features, single["expected"]):
            print(f"  {r['feature']:28s} {r['attribution']:+.5f} "
                  f"({r['direction']} risk)")

    else:  # lime
        x, y, _ = build_batch(args.parquet, model, args.n)
        exp = lime_explain(x, args.row_idx, model, features,
                           args.num_features)
        cls_name = "hold" if exp["predicted_class"] == 1 else "allow"
        print(f"row {exp['row_idx']}: calibrated probability "
              f"{exp['calibrated_probability']:.6f} -> class "
              f"'{cls_name}' (label fraud={int(y[exp['row_idx']])})")
        for name, w in sorted(exp["weights"].items(), key=lambda kv: -abs(kv[1])):
            print(f"  {name:28s} {w:+.5f}")


if __name__ == "__main__":
    main()