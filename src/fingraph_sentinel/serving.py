"""Layer 0 serving service: model scoring + SHAP reasons + Helix drift context.

Thin, lazy wrapper the FastAPI layer calls so the API stays declarative and
the same functions are reusable by offline scripts / the dashboard data
source. Everything loads on first use (no import-time heavy work).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MODEL_DIR = Path("artifacts/models/baseline-online-xgb")


@dataclass(slots=True)
class ScoredReason:
    feature: str
    direction: str
    detail: str
    magnitude: float | None = None


@dataclass(slots=True)
class ScoreResult:
    transaction_id: str
    model_version: str
    fraud_probability: float
    action: str
    reasons: list[ScoredReason] = field(default_factory=list)
    is_model_ready: bool = True


def score_event(
    values: dict[str, float | None],
    feature_columns: list[str],
    model_dir: Path = MODEL_DIR,
    boilerplate_reasons: list[dict] | None = None,
) -> ScoreResult:
    """Score a single fully-materialised feature dict against the serving model.

    ``values`` maps feature name -> value (None for realtime velocity that is
    not wired yet). Returns a ScoreResult with a calibrated probability,
    decision band, SHAP-style top reasons and contextual reasons.
    """
    import xgboost as xgb

    cfg = json.loads((model_dir / "model_config.json").read_text())
    model_file = cfg["model_file"]
    booster = xgb.Booster()
    booster.load_model(model_dir / model_file)

    x = np.array(
        [
            np.nan if values[name] is None else float(values[name])
            for name in feature_columns
        ],
        dtype=np.float32,
    ).reshape(1, -1)

    raw_margin = float(booster.predict(xgb.DMatrix(x))[0])
    raw = 1.0 / (1.0 + np.exp(-raw_margin))
    scale = float(cfg.get("calibration_scale_pos_weight", 1.0))
    calibrated = raw / (scale * (1.0 - raw) + raw)
    calibrated = float(min(max(calibrated, 0.0), 1.0))

    thresholds = cfg["thresholds"]
    action = (
        "hold"
        if calibrated >= thresholds["hold"]
        else "review" if calibrated >= thresholds["review"] else "allow"
    )

    reasons = [ScoredReason(**r) for r in (boilerplate_reasons or [])]

    # SHAP top contributers (margin space), cheap single-row TreeExplainer
    reasons += _shap_reasons(booster, x, feature_columns, model_dir)

    # contextual summary reason last
    reasons.append(
        ScoredReason(
            feature="model_score",
            direction="context",
            detail=(
                f"Calibrated fraud probability {calibrated:.4f} from "
                f"{cfg.get('model_name', 'baseline')}."
            ),
            magnitude=calibrated,
        )
    )

    return ScoreResult(
        transaction_id=str(values.get("transaction_id", "")),
        model_version=str(cfg.get("model_name", "baseline")),
        fraud_probability=calibrated,
        action=action,
        reasons=reasons,
    )


def _shap_reasons(
    booster,
    x: np.ndarray,
    feature_columns: list[str],
    model_dir: Path,
    top_n: int = 5,
) -> list[ScoredReason]:
    """Top-N SHAP contributions for a single row, ordered by |value| desc."""
    try:
        import shap
    except Exception:  # noqa: BLE001 - optional dependency
        return []
    try:
        explainer = shap.TreeExplainer(booster, model_output="raw")
        values = explainer.shap_values(x)[0]  # margin-space contributions
    except Exception:  # noqa: BLE001 - never fail scoring because of SHAP
        return []
    order = np.argsort(-np.abs(values))
    out = []
    for i in order[:top_n]:
        v = float(values[i])
        out.append(
            ScoredReason(
                feature=feature_columns[i],
                direction="increases_risk" if v > 0 else "reduces_risk",
                detail=(
                    f"{feature_columns[i]} pushed {abs(v):.4f} "
                    "toward fraud" if v > 0 else
                    f"{feature_columns[i]} pushed {abs(v):.4f} away from fraud"
                ),
                magnitude=round(v, 4),
            )
        )
    return out
