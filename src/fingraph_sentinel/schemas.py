from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PaymentEvent(BaseModel):
    """Canonical, pseudonymous payment event used by every downstream model."""

    model_config = ConfigDict(str_strip_whitespace=True)

    transaction_id: str = Field(min_length=1, max_length=128)
    event_time: datetime
    customer_id: str = Field(min_length=1, max_length=128)
    card_id: str = Field(min_length=1, max_length=128)
    merchant_id: str = Field(min_length=1, max_length=256)
    merchant_category_code: str | None = Field(default=None, max_length=16)
    amount: Decimal = Field(gt=0, max_digits=12, decimal_places=2)
    currency: str = Field(default="USD", min_length=3, max_length=3)
    merchant_city: str | None = Field(default=None, max_length=128)
    merchant_state: str | None = Field(default=None, max_length=128)
    merchant_country: str | None = Field(default=None, min_length=2, max_length=2)
    device_id: str | None = Field(default=None, max_length=256)
    ip_hash: str | None = Field(default=None, max_length=128)
    payment_channel: str | None = Field(default=None, max_length=64)

    @field_validator("currency")
    @classmethod
    def uppercase_currency(cls, value: str) -> str:
        return value.upper()


class RiskReason(BaseModel):
    feature: str
    direction: Literal["increases_risk", "reduces_risk", "context"]
    detail: str
    magnitude: float | None = Field(
        default=None, description="Signed contribution, e.g. SHAP margin value"
    )
    human: str | None = Field(
        default=None, description="Product-grade human-readable sentence (LIMITATION #3)"
    )


class RiskDecision(BaseModel):
    transaction_id: str
    model_version: str
    fraud_probability: float = Field(ge=0, le=1)
    action: Literal["allow", "review", "hold"]
    reasons: list[RiskReason]
    is_model_ready: bool
    processed_at: str | None = Field(
        default=None, description="ISO-8601 processing timestamp"
    )
    # LIMITATION #3 — concrete payment-security action on top of the model band.
    security_action: Literal["APPROVE", "REQUEST_STEP_UP", "DECLINE"] = "REVIEW"
    # LIMITATION #4 — cold-start routing flag (conservative rule engine).
    is_cold_start: bool = False
    # Product-facing summary reasons (human-readable clauses).
    reasons_human: list[str] = Field(default_factory=list)


class FeatureDrift(BaseModel):
    feature: str
    ref_mean: float | None = None
    obs_mean: float | None = None
    psi: float | None = None
    z: float | None = None


class HelixDriftReport(BaseModel):
    trigger: str
    score: float | None = None
    n_features: int = 0
    n_culprits: int = 0
    culprits: list[str] = []
    reasons: list[str] = []
    features: list[FeatureDrift] = []


class ModelStatus(BaseModel):
    ready: bool
    model_version: str
    backend: str | None = None
    trained_at: str | None = None
    training_rows: int | None = None
    thresholds: dict | None = None
    metrics_validation: dict | None = None
    metrics_test_locked: dict | None = None


# ---- Layer 2: graph store (local snapshots + Neo4j) ---------------------


class Neo4jStatus(BaseModel):
    reachable: bool
    detail: str
    url: str


class GraphStatus(BaseModel):
    """Honest snapshot of the Layer-2 graph pipeline + Neo4j connectivity.

    ``pipeline`` is always derived from the LOCAL graph-snapshot artifacts
    (torch snapshots + meta.json from graph_snapshots), so the dashboard
    renders real nodes/edges even when Neo4j is not running. ``neo4j`` reports
    live reachability of the bolt endpoint — offline locally is the expected
    state until the user runs ``make ingest-graph``.
    """

    neo4j: Neo4jStatus
    pipeline: dict
    gnn: dict | None = None


# ---- Layer 6: compliance audit + observability --------------------------


class AuditRecord(BaseModel):
    """One tamper-evident entry in the Layer 6 audit hash chain."""

    id: str
    event_type: str
    payload: dict = {}
    prev_hash: str = ""
    hash: str = ""
    audited_at: float | None = None
    seq: int | None = None


class AuditHealth(BaseModel):
    healthy: bool
    backend: str
    buffered: int = 0
    total: int = 0


class AuditSummary(BaseModel):
    total: int
    backend: str
    buffered: int = 0
    valid: bool = False
    verified_records: int = 0
    store_healthy: bool = False
    metrics_test_locked: dict | None = None


class FeedbackIn(BaseModel):
    """Helix v2: one outcome against an audited decision."""

    transaction_id: str = Field(min_length=1, max_length=128)
    outcome: Literal["fraud", "legit"]
