from datetime import UTC, datetime

from fastapi import FastAPI, status

from fingraph_sentinel.config import get_settings
from fingraph_sentinel.runtime import (
    boilerplate_reasons,
    event_feature_dict,
    load_helix_drift,
)
from fingraph_sentinel.schemas import (
    FeatureDrift,
    HelixDriftReport,
    ModelStatus,
    PaymentEvent,
    RiskDecision,
    RiskReason,
)
from fingraph_sentinel.serving import MODEL_DIR, score_event

settings = get_settings()

app = FastAPI(
    title=settings.project_name,
    version="0.3.0",
    description=(
        "Defense-only merchant fraud risk intelligence. Recommendations are auditable "
        "and never execute payment actions."
    ),
)


def _model_ready() -> bool:
    return (MODEL_DIR / "model_config.json").exists()


@app.get("/api/v1/health/live", tags=["health"])
def liveness() -> dict[str, str]:
    return {"status": "ok", "service": "risk-api"}


@app.get("/api/v1/health/ready", tags=["health"])
def readiness() -> dict[str, str | bool]:
    return {"status": "ok", "model_registered": _model_ready(),
            "message": "Service ready to score transactions."}


@app.get("/api/v1/model/status", response_model=ModelStatus, tags=["risk"])
def model_status() -> ModelStatus:
    if not _model_ready():
        return ModelStatus(ready=False, model_version=settings.model_version)
    cfg = _config()
    return ModelStatus(
        ready=True,
        model_version=str(cfg.get("model_name", "baseline")),
        backend=str(cfg.get("backend")),
        trained_at=str(cfg.get("created_at", "")),
        training_rows=cfg.get("training_rows"),
        thresholds=cfg.get("thresholds"),
        metrics_validation=cfg.get("metrics_validation"),
        metrics_test_locked=cfg.get("metrics_test_locked"),
    )


@app.post(
    "/api/v1/transactions/score",
    response_model=RiskDecision,
    status_code=status.HTTP_200_OK,
    tags=["risk"],
)
def score_transaction(event: PaymentEvent) -> RiskDecision:
    """Score a single payment event and explain the decision."""
    if not _model_ready():
        return _safe_review_decision(event)
    try:
        values = event_feature_dict(event)
        result = score_event(
            values,
            feature_columns=list(_config()["feature_columns"]),
            boilerplate_reasons=boilerplate_reasons(event),
        )
    except Exception:  # noqa: BLE001 - scoring must fail safe, never fail open
        return _safe_review_decision(event)
    return RiskDecision(
        transaction_id=event.transaction_id,
        model_version=result.model_version,
        fraud_probability=round(result.fraud_probability, 6),
        action=result.action,  # type: ignore[arg-type]
        reasons=[
            RiskReason(
                feature=r.feature,
                direction=r.direction,  # type: ignore[arg-type]
                detail=r.detail,
                magnitude=r.magnitude,
            )
            for r in result.reasons
        ],
        is_model_ready=True,
        processed_at=datetime.now(UTC).isoformat(),
    )


@app.get("/api/v1/helix/drift", response_model=HelixDriftReport, tags=["helix"])
def helix_drift() -> HelixDriftReport:
    """Layer 5 per-feature drift + retrain trigger for the serving model."""
    report = load_helix_drift()
    if not report:
        return HelixDriftReport(trigger="NO", reasons=["no drift data yet"])
    triggers: list[dict] = []
    features: dict[str, FeatureDrift] = {}
    for window, part in report.items():
        trig = part.get("trigger", {}) if isinstance(part, dict) else {}
        if isinstance(trig, dict):
            triggers.append(trig)
        for f in part.get("features", []) if isinstance(part, dict) else []:
            fd = FeatureDrift(**f)
            features[fd.feature] = fd
    culprits = [c for t in triggers for c in t.get("culprits", [])]
    reasons = [r for t in triggers for r in t.get("reasons", [])]
    scores = [float(t["score"]) for t in triggers if t.get("score") is not None]
    return HelixDriftReport(
        trigger="YES" if "YES" in [t.get("trigger") for t in triggers] else "NO",
        score=max(scores, default=None),
        n_culprits=len(culprits),
        culprits=list(dict.fromkeys(culprits)),
        reasons=list(dict.fromkeys(reasons)),
        features=list(features.values()),
    )


@app.get("/api/v1/meta", tags=["health"])
def metadata() -> dict[str, str]:
    return {
        "project": settings.project_name,
        "environment": settings.environment,
        "generated_at": datetime.now(UTC).isoformat(),
        "safety_mode": "defense-only",
    }


def _config() -> dict:
    import json

    return json.loads((MODEL_DIR / "model_config.json").read_text())


def _safe_review_decision(event: PaymentEvent) -> RiskDecision:
    """Fail-safe default: no model, no guessing -- always human review."""
    _ = event
    return RiskDecision(
        transaction_id=event.transaction_id,
        model_version=settings.model_version,
        fraud_probability=0.0,
        action="review",
        reasons=[
            RiskReason(
                feature="model_registry",
                direction="context",
                detail="No trained model is registered yet; manual review is required.",
            )
        ],
        is_model_ready=False,
        processed_at=datetime.now(UTC).isoformat(),
    )
