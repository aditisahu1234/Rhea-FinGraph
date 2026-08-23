from datetime import UTC, datetime

from fastapi import FastAPI, status

from fingraph_sentinel.config import get_settings
from fingraph_sentinel.schemas import PaymentEvent, RiskDecision, RiskReason

settings = get_settings()

app = FastAPI(
    title=settings.project_name,
    version="0.1.0",
    description=(
        "Defense-only merchant fraud risk intelligence. Recommendations are auditable "
        "and never execute payment actions."
    ),
)


@app.get("/api/v1/health/live", tags=["health"])
def liveness() -> dict[str, str]:
    return {"status": "ok", "service": "risk-api"}


@app.get("/api/v1/health/ready", tags=["health"])
def readiness() -> dict[str, str]:
    return {
        "status": "starting",
        "model_version": settings.model_version,
        "message": "Infrastructure and model readiness checks will be added with their clients.",
    }


@app.post(
    "/api/v1/transactions/score",
    response_model=RiskDecision,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["risk"],
)
def score_transaction(event: PaymentEvent) -> RiskDecision:
    """Validate the canonical event contract until the trained model registry is connected."""
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
