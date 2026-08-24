from datetime import UTC, datetime

from fastapi import FastAPI, status

from fingraph_sentinel.config import get_settings
from fingraph_sentinel.model_registry import get_registry
from fingraph_sentinel.schemas import PaymentEvent, RiskDecision, RiskReason

settings = get_settings()

app = FastAPI(
    title=settings.project_name,
    version="0.2.0",
    description=(
        "Defense-only merchant fraud risk intelligence. Recommendations are auditable "
        "and never execute payment actions."
    ),
)


@app.get("/api/v1/health/live", tags=["health"])
def liveness() -> dict[str, str]:
    return {"status": "ok", "service": "risk-api"}


@app.get("/api/v1/health/ready", tags=["health"])
def readiness() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "model_registered": get_registry() is not None,
        "message": "Service ready to score transactions.",
    }


@app.get("/api/v1/model/status", tags=["risk"])
def model_status() -> dict[str, object]:
    registry = get_registry()
    if registry is None:
        return {
            "ready": False,
            "model_version": settings.model_version,
            "message": (
                "No trained model found on disk; every transaction is routed "
                "to manual review."
            ),
        }
    registry._ensure_loaded()
    cfg = registry.config
    return {
        "ready": True,
        "model_version": cfg.get("model_name"),
        "backend": cfg.get("backend"),
        "trained_at": cfg.get("created_at"),
        "training_rows": cfg.get("training_rows"),
        "thresholds": cfg.get("thresholds"),
        "metrics_validation": cfg.get("metrics_validation"),
        "metrics_test_locked": cfg.get("metrics_test_locked"),
    }


@app.post(
    "/api/v1/transactions/score",
    response_model=RiskDecision,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["risk"],
)
def score_transaction(event: PaymentEvent) -> RiskDecision:
    """Score a payment event with the registered model when available."""
    registry = get_registry()
    if registry is None:
        return _safe_review_decision(event)

    try:
        outcome = registry.score_event(event)
    except Exception:  # noqa: BLE001 - scoring must fail safe, never fail open
        return _safe_review_decision(event)

    return RiskDecision(
        transaction_id=event.transaction_id,
        model_version=outcome.model_version,
        fraud_probability=round(outcome.probability, 6),
        action=outcome.action,  # type: ignore[arg-type]
        reasons=outcome.reasons,
        is_model_ready=True,
    )


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
    )


@app.get("/api/v1/meta", tags=["health"])
def metadata() -> dict[str, str]:
    return {
        "project": settings.project_name,
        "environment": settings.environment,
        "generated_at": datetime.now(UTC).isoformat(),
        "safety_mode": "defense-only",
    }
