from datetime import UTC, datetime

from fastapi import FastAPI, status

from fingraph_sentinel.audit import Ledger
from fingraph_sentinel.config import get_settings
from fingraph_sentinel.runtime import (
    boilerplate_reasons,
    event_feature_dict,
    load_helix_drift,
)
from fingraph_sentinel.schemas import (
    AuditHealth,
    AuditRecord,
    AuditSummary,
    FeatureDrift,
    HelixDriftReport,
    ModelStatus,
    PaymentEvent,
    RiskDecision,
    RiskReason,
)
from fingraph_sentinel.serving import MODEL_DIR, score_event
from fingraph_sentinel.streaming import VelocityFeatureService

settings = get_settings()

# Layer 6 audit ledger: Postgres when configured, in-memory fail-safe otherwise.
# Purposely constructed at import with .default() so an unreachable DB never
# raises and the API stays up (the ledger buffers and reports unhealthy).
_ledger: Ledger | None = None


def get_ledger() -> Ledger:
    global _ledger  # noqa: PLW0603 - lazy singleton so tests can reset it
    if _ledger is None:
        _ledger = Ledger.default(settings.postgres_url)
    return _ledger


# Layer 1 streaming velocity store: Redis when configured, in-memory fail-safe
# otherwise. Built the same way as the ledger so an unreachable store never
# raises and never breaks scoring.
_velocity: VelocityFeatureService | None = None


def get_velocity() -> VelocityFeatureService:
    global _velocity  # noqa: PLW0603 - lazy singleton so tests can reset it
    if _velocity is None:
        _velocity = VelocityFeatureService.default(settings.redis_url)
    return _velocity


app = FastAPI(
    title=settings.project_name,
    version="0.4.0",
    description=(
        "Defense-only merchant fraud risk intelligence. Recommendations are auditable "
        "and never execute payment actions."
    ),
)

# --- Startup seeding: populate audit + streaming stores so the dashboard ---
# has data on first load instead of showing "no scored decisions yet".      ---
_SEED_EVENTS = [
    {"transaction_id": "seed-001", "event_time": "2026-08-23T10:15:00Z",
     "customer_id": "C-1001", "card_id": "K-2001", "merchant_id": "1334959",
     "amount": "49.99", "device_id": "D-3001", "merchant_country": "IN"},
    {"transaction_id": "seed-002", "event_time": "2026-08-23T10:16:30Z",
     "customer_id": "C-1001", "card_id": "K-2001", "merchant_id": "5411",
     "amount": "299.00", "device_id": "D-3001", "merchant_country": "US"},
    {"transaction_id": "seed-003", "event_time": "2026-08-23T10:17:45Z",
     "customer_id": "C-1002", "card_id": "K-2002", "merchant_id": "5999",
     "amount": "1250.00", "device_id": "D-3002", "merchant_country": "GB"},
    {"transaction_id": "seed-004", "event_time": "2026-08-23T10:18:20Z",
     "customer_id": "C-1003", "card_id": "K-2003", "merchant_id": "7299",
     "amount": "12.50", "device_id": "D-3003", "merchant_country": "IN"},
    {"transaction_id": "seed-005", "event_time": "2026-08-23T10:19:00Z",
     "customer_id": "C-1001", "card_id": "K-2001", "merchant_id": "5411",
     "amount": "875.50", "device_id": "D-3001", "merchant_country": "US"},
]


@app.on_event("startup")
def _seed_on_startup() -> None:
    """Score 5 sample events so the dashboard has audit + velocity data."""
    from fastapi.testclient import TestClient  # noqa: PLC0415

    try:
        client = TestClient(app, raise_server_exceptions=False)
        for evt in _SEED_EVENTS:
            client.post("/api/v1/transactions/score", json=evt)
    except Exception:  # noqa: BLE001 — seeding is best-effort
        pass


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
    """Score a single payment event, explain the decision, and audit it.

    Layer 1 ordering guarantee: the streaming velocity features are computed
    (strictly-past, read-only) before the event is committed to the store, so a
    transaction never counts towards its own risk. The event is always committed
    afterwards — even on a scoring failure — so live velocity state accumulates
    as real traffic flows.
    """
    velocity = get_velocity().compute(event)
    try:
        if not _model_ready():
            decision = _safe_review_decision(event)
            _audit("decision.review_failsafe", event, decision)
            return decision
        try:
            values = event_feature_dict(event, velocity=velocity)
            result = score_event(
                values,
                feature_columns=list(_config()["feature_columns"]),
                boilerplate_reasons=boilerplate_reasons(event),
            )
        except Exception:  # noqa: BLE001 - scoring must fail safe, never fail open
            decision = _safe_review_decision(event)
            _audit("decision.fail_open_blocked", event, decision)
            return decision
        decision = RiskDecision(
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
        _audit("decision.scored", event, decision)
        return decision
    finally:
        # Commit the event into the streaming store after read + scoring, always.
        get_velocity().observe(event)


def _audit(event_type: str, event: PaymentEvent, decision: RiskDecision) -> None:
    """Append one decision to the Layer 6 ledger. Never raises (fail-safe)."""
    try:
        get_ledger().append(
            event_type,
            {
                "transaction_id": event.transaction_id,
                "model_version": decision.model_version,
                "action": decision.action,
                "fraud_probability": decision.fraud_probability,
                "is_model_ready": decision.is_model_ready,
                "n_reasons": len(decision.reasons),
                "reasons": [
                    {
                        "feature": r.feature,
                        "direction": r.direction,
                        "detail": r.detail,
                        "magnitude": r.magnitude,
                    }
                    for r in decision.reasons
                ],
                "amount": str(event.amount),
                "currency": event.currency,
                "customer_id": event.customer_id,
                "merchant_id": event.merchant_id,
                "payment_channel": event.payment_channel,
                "processed_at": decision.processed_at,
            },
        )
    except Exception:  # noqa: BLE001 - audit must never break scoring
        return


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


@app.get("/api/v1/audit/health", response_model=AuditHealth, tags=["audit"])
def audit_health() -> AuditHealth:
    """Layer 6 audit store health: backend, healthy, buffered fallback count."""
    led = get_ledger()
    h = led.health()
    return AuditHealth(
        healthy=h["healthy"], backend=h["backend"], buffered=h["buffered"],
        total=h["total"],
    )


@app.get("/api/v1/audit/recent", response_model=list[AuditRecord], tags=["audit"])
def audit_recent(limit: int = 20) -> list[AuditRecord]:
    """Most recent tamper-evident audit entries (newest first)."""
    limit = max(1, min(int(limit), 200))
    return [AuditRecord(**r) for r in get_ledger().recent(limit)]


@app.get("/api/v1/audit/summary", response_model=AuditSummary, tags=["audit"])
def audit_summary() -> AuditSummary:
    """Count + integrity verification of the whole audit hash chain."""
    led = get_ledger()
    return AuditSummary(
        total=led.count(),
        backend=type(led.store).__name__,
        buffered=led.health()["buffered"],
        valid=bool(led.verify()["valid"]),
        verified_records=led.verify()["records"],
        store_healthy=led.store.is_healthy(),
    )


@app.get("/api/v1/audit/verify", tags=["audit"])
def audit_verify() -> dict:
    """Full chain-integrity verification: detects any tamper / reorder."""
    return get_ledger().verify()


@app.get("/api/v1/audit/daily", tags=["audit"])
def audit_daily(days: int = 14) -> list[dict]:
    """Per-day decision/exception volume rollup (UTC), newest first."""
    return get_ledger().daily(days=max(1, min(int(days), 90)))


@app.get("/api/v1/streaming/health", tags=["streaming"])
def streaming_health() -> dict:
    """Layer 1 streaming store health: backend, observations, window state."""
    h = get_velocity().health()
    return {"layer": "streaming-velocity", "read_contract": "strictly-past", **h}


@app.get("/api/v1/streaming/snapshot", tags=["streaming"])
def streaming_snapshot(entity: str = "cust", entity_id: str = "") -> dict:
    """Per-key rolling-window + cumulative-prior view for an entity."""
    if not entity_id:
        return {"error": "entity_id is required"}
    try:
        return get_velocity().snapshot(entity, entity_id)
    except ValueError as exc:  # unknown entity -> 422-style message
        return {"error": str(exc)}


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
