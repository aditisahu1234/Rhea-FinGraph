"""Layer 0 serving service: model scoring + SHAP reasons + Helix drift context.

Thin, lazy wrapper the FastAPI layer calls so the API stays declarative and
the same functions are reusable by offline scripts / the dashboard data
source. Everything loads on first use (no import-time heavy work).

Latency contract (Layer 0): the XGBoost booster, its config, and the SHAP
TreeExplainer are loaded ONCE per model snapshot and cached in-process, keyed
on the model files' mtimes, so a *promoted* model (new files copied into the
serving dir) transparently rebuilds the cache while steady-state scoring never
touches disk (measured: ~140 ms/event before caching -> ~3 ms/event after).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xgboost as xgb

MODEL_DIR = Path("artifacts/models/baseline-online-xgb")

# In-process asset cache: key = str(model_dir), value = {"config", "booster",
# "explainer", "config_mtime"}. Rebuilt whenever model_config.json mtime moves.
_ASSET_CACHE: dict[str, dict] = {}


def _assets(model_dir: Path) -> dict:
    """Return the cached {config, booster, explainer} for a model snapshot.

    ``model_config.json`` mtime is the invalidation key: promotion copies a
    new config + model together, so an mtime move forces a rebuild while an
    unchanged config reuses the loaded booster/explainer (no disk reads in the
    steady state).
    """
    cfg_path = model_dir / "model_config.json"
    cfg_mtime = cfg_path.stat().st_mtime_ns if cfg_path.exists() else 0
    cached = _ASSET_CACHE.get(str(model_dir))
    if cached is not None and cached["config_mtime"] == cfg_mtime:
        return cached

    cfg = json.loads(cfg_path.read_text())
    booster = xgb.Booster()
    booster.load_model(model_dir / cfg["model_file"])
    explainer = None
    try:
        import shap

        explainer = shap.TreeExplainer(booster, model_output="raw")
    except Exception:  # noqa: BLE001 - SHAP is optional; never fail scoring
        explainer = None
    assets = {
        "config": cfg,
        "booster": booster,
        "explainer": explainer,
        "config_mtime": cfg_mtime,
    }
    _ASSET_CACHE[str(model_dir)] = assets
    return assets


def clear_model_cache() -> None:
    """Drop cached assets (tests / model promotion tooling)."""
    _ASSET_CACHE.clear()


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
    assets = _assets(model_dir)
    cfg = assets["config"]
    booster = assets["booster"]

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
    reasons += _shap_reasons(assets, x, feature_columns)

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
    assets: dict,
    x: np.ndarray,
    feature_columns: list[str],
    top_n: int = 5,
) -> list[ScoredReason]:
    """Top-N SHAP contributions for a single row, ordered by |value| desc.

    Uses the cached TreeExplainer from ``assets`` (built once per model
    snapshot) so per-event explainability is microseconds, not a rebuild.
    """
    explainer = assets.get("explainer")
    if explainer is None:
        return []
    try:
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
