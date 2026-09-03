from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, status

from fingraph_sentinel.audit import Ledger
from fingraph_sentinel.config import get_settings
from fingraph_sentinel.healing import HealingEngine
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
    FeedbackIn,
    GraphStatus,
    HelixDriftReport,
    ModelStatus,
    Neo4jStatus,
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


# Layer 5 v2 healing engine: failure memory + heal actions. Durable JSONL
# under settings.healing_dir, so feedback survives restarts with no Docker.
_healing: HealingEngine | None = None


def get_healing() -> HealingEngine:
    global _healing  # noqa: PLW0603 - lazy singleton so tests can reset it
    if _healing is None:
        _healing = HealingEngine(model_dir=MODEL_DIR, healing_dir=settings.healing_dir)
    return _healing


app = FastAPI(
    title=settings.project_name,
    version="0.4.0",
    description=(
        "Defense-only merchant fraud risk intelligence. Recommendations are auditable "
        "and never execute payment actions."
    ),
)

# --- Startup seeding: populate audit + streaming + healing stores so the --
# --- dashboard has data on first load instead of empty panels.          ----
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

# Demo outcomes (clearly synthetic) so the healing panel starts with real
# memory: 2 confirmed frauds at merchant 5411 that the model allowed
# (missed fraud -> hot-list + tightened hold + retrain queue) and 1 legit.
_SEED_FEEDBACK = [
    {"transaction_id": "seed-002", "outcome": "fraud"},
    {"transaction_id": "seed-005", "outcome": "fraud"},
    {"transaction_id": "seed-001", "outcome": "legit"},
]


@app.on_event("startup")
def _seed_on_startup() -> None:
    """Score demo events + record demo outcomes so panels have data on boot."""
    if not settings.demo_seed:
        return
    from fastapi.testclient import TestClient  # noqa: PLC0415

    try:
        client = TestClient(app, raise_server_exceptions=False)
        for evt in _SEED_EVENTS:
            client.post("/api/v1/transactions/score", json=evt)
        for fb in _SEED_FEEDBACK:
            client.post("/api/v1/healing/feedback", json=fb)
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


# ---- Layer 2: graph pipeline status + Neo4j connectivity -----------------


# Layer 2 top-fraud-merchant rollup: one vectorized pass over the Helix failure
# memory per process (mtime-keyed), so the dashboard poll never rescans the
# 338 MB / 800K-episode JSONL. Real confirmed-fraud merchants, not a promise.
_TOP_MERCHANTS_CACHE: dict[str, tuple[int, list[dict]]] = {}


def _top_fraud_merchants(limit: int = 10) -> list[dict]:
    """Top merchants by confirmed-fraud failures from failure_memory.jsonl."""
    mem_path = Path(settings.healing_dir) / "failure_memory.jsonl"
    if not mem_path.exists():
        return []
    try:
        key = str(mem_path)
        mtime = mem_path.stat().st_mtime_ns
        cached = _TOP_MERCHANTS_CACHE.get(key)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        import polars as pl  # noqa: PLC0415

        rows = (
            pl.scan_ndjson(key)
            .select(
                pl.col("event").struct.field("merchant_id").alias("mid"),
                pl.col("fail_type"),
                pl.col("outcome"),
            )
            .group_by("mid")
            .agg(
                [
                    pl.len().alias("txns"),
                    pl.col("fail_type").is_not_null().sum().alias("failures"),
                    (pl.col("fail_type") == "missed_fraud")
                    .sum()
                    .alias("missed_fraud"),
                    (pl.col("outcome") == "fraud").sum().alias("confirmed_fraud"),
                ]
            )
            .filter(pl.col("mid").is_not_null())
            .sort("failures", descending=True)
            .head(limit)
            .collect()
            .to_dicts()
        )
        result: list[dict] = [
            {"merchant_id": r.pop("mid", "") or "", **r} for r in rows
        ]
        _TOP_MERCHANTS_CACHE[key] = (mtime, result)
        return result
    except Exception:  # noqa: BLE001 - any rollup failure degrades to no table
        return []


def _best_graph_meta() -> tuple[dict | None, Path]:
    """Best local graph-snapshot meta.json (prefer the full-data Kaggle run)."""

    candidates = [
        Path("artifacts/graph/gnn_kaggle/graph/meta.json"),
        Path("artifacts/graph/snapshots-smoke/meta.json"),
    ]
    for p in candidates:
        if p.exists():
            import json  # noqa: PLC0415

            try:
                return json.loads(p.read_text()), p
            except Exception:  # noqa: BLE001 - corrupt meta should not 500
                return None, p
    return None, Path("")


def _neo4j_reachable() -> Neo4jStatus:
    """Live bolt handshake against settings.neo4j_url (short timeout).

    Offline is the expected local state until the user runs
    ``make ingest-graph``; the check uses verify_connectivity() which fails
    fast (connection refused) when no Neo4j server is listening.
    """
    try:  # noqa: PLW1101
        from neo4j import GraphDatabase  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - driver not installed -> not reachable
        return Neo4jStatus(reachable=False, detail="neo4j driver not installed",
                           url=settings.neo4j_url)
    driver = None
    try:
        driver = GraphDatabase.driver(
            settings.neo4j_url,
            auth=(settings.neo4j_username, settings.neo4j_password),
            connection_timeout=2.0,
        )
        driver.verify_connectivity()
        return Neo4jStatus(reachable=True,
                           detail="bolt endpoint reachable",
                           url=settings.neo4j_url)
    except Exception as exc:  # noqa: BLE001 - any failure => offline, honest
        return Neo4jStatus(reachable=False, detail=f"{type(exc).__name__}: {exc}",
                           url=settings.neo4j_url)
    finally:
        if driver is not None:
            try:
                driver.close()
            except Exception:  # noqa: BLE001 - best-effort close
                pass


@app.get("/api/v1/graph/status",
         response_model=GraphStatus,
         tags=["graph"])
def graph_status() -> GraphStatus:
    """Layer-2 graph store status: local snapshot pipeline + Neo4j reachability."""
    meta, meta_path = _best_graph_meta()
    pipeline: dict = {"source": "none" if meta is None else str(meta_path)}
    gnn: dict | None = None
    if meta is not None:
        snapshots = meta.get("snapshots") or []
        pipeline.update({
            "n_customers": meta.get("n_customers"),
            "n_merchants": meta.get("n_merchants"),
            "n_cards": meta.get("n_cards"),
            "n_snapshots": meta.get("n_months") or len(snapshots),
            "month_range": [meta.get("month_min"), meta.get("month_max")],
            "bucket_months": meta.get("bucket_months"),
            "total_edges": sum(int(s.get("n_edges", 0)) for s in snapshots),
            "total_fraud_edges": sum(int(s.get("n_fraud", 0)) for s in snapshots),
            "snapshots": [
                {"month_idx": s.get("month_idx"), "n_edges": s.get("n_edges"),
                 "n_fraud": s.get("n_fraud")}
                for s in snapshots
            ],
        })
        # Real confirmed-fraud merchants from the Helix failure memory
        # (800K val-slice episodes). Provenance: failure-memory rollup, NOT
        # graph-node fraud counts (per-edge snapshots live on Kaggle).
        top = _top_fraud_merchants()
        pipeline["top_merchants"] = top
        pipeline["top_merchants_source"] = (
            "helix failure memory rollup (failure_memory.jsonl)"
        )
        import json  # noqa: PLC0415

        gnn_cfg = meta_path.parent / "gnn_config.json"
        if not gnn_cfg.exists():
            gnn_cfg = (Path("artifacts/graph/gnn_kaggle/gnn/gnn_config.json")
                       if (Path("artifacts/graph/gnn_kaggle/gnn/gnn_config.json")
                           .exists()) else None)
        if gnn_cfg is not None:
            try:
                c = json.loads(gnn_cfg.read_text())
                gnn = {
                    "architecture": c.get("architecture"),
                    "params": c.get("params"),
                    "epochs": c.get("epochs"),
                    "fit_seconds": c.get("fit_seconds"),
                    "device_used": c.get("device_used"),
                    "best_val_auc": c.get("best_val_auc"),
                    "metrics_validation": c.get("metrics_validation"),
                    "metrics_test_locked": c.get("metrics_test_locked"),
                }
            except Exception:  # noqa: BLE001 - corrupt config => no GNN row
                gnn = None
    return GraphStatus(
        neo4j=_neo4j_reachable(),
        pipeline=pipeline,
        gnn=gnn,
    )


# ---- Layer 4: model fight card (honest model race) ------------------------


@app.get("/api/v1/model/race", tags=["models"])
def model_race() -> dict:
    """Real model registry from disk: serving, candidates, repair verdict.

    Every row comes from the recorded model_config.json metrics (locked test
    split where present) + the repair gate report. No fabricated rows; a
    model without a config simply does not appear.
    """
    import json  # noqa: PLC0415

    MODELS_DIR = Path("artifacts/models")
    SERVING_NAME = "baseline-online-xgb"
    rows: list[dict] = []
    if MODELS_DIR.exists():
        for d in sorted(MODELS_DIR.iterdir()):
            cfg_path = d / "model_config.json"
            if not cfg_path.exists():
                continue
            try:
                c = json.loads(cfg_path.read_text())
            except Exception:  # noqa: BLE001 - corrupt config skipped
                continue
            mv = c.get("metrics_validation") or {}
            mt = c.get("metrics_test_locked") or {}
            if d.name == SERVING_NAME:
                role = "serving"
            elif mt.get("action_counts"):
                role = "promotion-candidate"
            else:
                role = "candidate"
            rows.append({
                "name": d.name,
                "label": c.get("model_name", d.name),
                "backend": c.get("backend"),
                "feature_set": c.get("feature_set"),
                "training_rows": c.get("training_rows"),
                "created_at": c.get("created_at"),
                "val_roc": mv.get("roc_auc"),
                "test_roc": mt.get("roc_auc"),
                "test_ap": mt.get("average_precision"),
                "test_action_counts": mt.get("action_counts"),
                "caught_frauds_by_action": mt.get("caught_frauds_by_action"),
                "thresholds": c.get("thresholds"),
                "role": role,
            })
    gate: dict | None = None
    gate_path = Path(settings.healing_dir) / "gate_report.json"
    if gate_path.exists():
        try:
            gate = json.loads(gate_path.read_text())
        except Exception:  # noqa: BLE001 - corrupt report => no verdict
            gate = {"verdict": "unreadable"}
    return {"models": rows, "serving_name": SERVING_NAME, "gate_report": gate}


@app.get("/api/v1/model/switcher/status", tags=["models"])
def model_switcher_status() -> dict:
    """Concept-drift auto-switch status: last decision + detector state.

    Reads the persisted switch decision written by ``drift_switcher`` and the
    monthly drift report from ``drift_monitor``. When drift was detected and a
    better candidate exists on disk, the dashboard shows a "MODEL AUTO-SWITCHED"
    alert with the honest from->to chain. No decision -> no alert.
    """
    import json  # noqa: PLC0415

    import polars as pl  # noqa: PLC0415

    from fingraph_sentinel.drift_monitor import (
        DEFAULT_SCORES,  # noqa: PLC0415
        monitor_report,  # noqa: PLC0415
    )

    decision: dict | None = None
    path = Path(settings.healing_dir) / "switch_decision_latest.json"
    if path.exists():
        try:
            decision = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            decision = None

    drift: dict | None = None
    if DEFAULT_SCORES.exists():
        try:
            scores = pl.read_parquet(DEFAULT_SCORES)
            drift = monitor_report(scores)
        except Exception:  # noqa: BLE001 - stale/corrupt score stream
            drift = None

    return {
        "serving_model": "baseline-online-xgb",
        "last_decision": decision,
        "drift_report": drift,
    }


@app.get("/api/v1/business/impact", tags=["risk"])
def business_impact() -> dict:
    """Razorpay-relevant operating-point recap (LIMITATION #1).

    Serves the parity-verified business numbers computed offline by
    scripts/business_impact.py (locked test split, velocity-v3 decision
    stream reproduced byte-for-byte against the recorded model_config):
    allow/review/hold volumes, frauds caught, recall by count & amount,
    protected/missed ₹, top MCCs by fraud amount. All fields are read from
    the artifact, never synthesised here.
    """
    import json  # noqa: PLC0415

    path = Path("artifacts/business_impact.json")
    if not path.exists():
        return {
            "available": False,
            "note": "Run scripts/business_impact.py to generate the operating-point recap.",
        }
    try:
        report = json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return {"available": False, "note": "business_impact.json unreadable."}
    report["available"] = True
    return report


@app.post("/api/v1/razorpay/order", tags=["razorpay"])
def razorpay_create_order(body: dict) -> dict:
    """Razorpay demo: create a test order (LIMITATION #2).

    Accepts amount_inr (string) + optional merchant/customer/card selectors and
    returns a created order with the canonical PaymentEvent that represents
    its payment. This is the FINGRAPH-facing step of the merchant flow:
    Razorpay-like event -> RISK MANAGER.
    """
    amount_inr = str(body.get("amount_inr", "1999.00"))
    from fingraph_sentinel.razorpay_demo import create_order  # noqa: PLC0415

    try:
        return create_order(
            amount_inr=amount_inr,
            merchant_key=str(body.get("merchant_id", "TerraMart-5311")),
            customer_id=str(body.get("customer_id", "C-DEMO-1001")),
            card_id=str(body.get("card_id", "K-DEMO-2001")),
            payment_error=body.get("payment_error"),
        )
    except Exception as exc:  # noqa: BLE001 - bad input => clean 400-style message
        return {"error": "invalid_amount", "detail": str(exc)}


@app.post("/api/v1/razorpay/pay", tags=["razorpay"])
def razorpay_pay(body: dict) -> dict:
    """Razorpay demo: run one order through FINGRAPH and return the decision.

    Takes the order_id from /razorpay/order (or a bare PaymentEvent), scores it
    through the exact same velocity -> XGBoost -> SHAP -> calibrated action ->
    audit path as live traffic, and returns the Razorpay-style webhook payload.
    The decision is never hidden behind a fake threshold — it is the audited
    ScoreResult from serving.score_event.
    """
    from fastapi import HTTPException  # noqa: PLC0415

    from fingraph_sentinel.razorpay_demo import (  # noqa: PLC0415
        _ORDERS,
        build_webhook,
        create_order,
    )

    order_id = body.get("order_id")
    event = None
    if order_id:
        order = _ORDERS.get(order_id)
        if order is None:
            raise HTTPException(status_code=404, detail="unknown order_id")
        event = order.event
    else:
        # allow a bare PaymentEvent payload directly (no pre-created order)
        try:
            evt_body = {k: v for k, v in body.items()
                        if k in {"transaction_id", "event_time", "customer_id",
                                 "card_id", "merchant_id", "merchant_category_code",
                                 "amount", "payment_channel", "merchant_city",
                                 "merchant_country", "payment_error"}}
            if not evt_body.get("event_time"):
                import time  # noqa: PLC0415
                evt_body["event_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            event = PaymentEvent(**evt_body)
            # register a matching order for the webhook
            order = create_order(
                amount_inr=str(float(event.amount) * 83.5),
                merchant_key=event.merchant_id if event.merchant_id else "TerraMart-5311",
                customer_id=event.customer_id,
                card_id=event.card_id,
                payment_error=event.payment_error,
            )
        except Exception as exc:  # noqa: BLE001 - malformed event
            raise HTTPException(status_code=422, detail=f"invalid payment event: {exc}")

    velocity = get_velocity().compute(event)
    try:
        if not _model_ready():
            decision = _safe_review_decision(event)
        else:
            values = event_feature_dict(event, velocity=velocity)
            import json as _json  # noqa: PLC0415

            from fingraph_sentinel.serving import MODEL_DIR as _SERVING_DIR  # noqa: PLC0415
            _cfg = _json.loads((_SERVING_DIR / "model_config.json").read_text())
            result = score_event(
                values,
                feature_columns=list(_cfg["feature_columns"]),
                boilerplate_reasons=boilerplate_reasons(event),
            )
            decision = RiskDecision(
                transaction_id=event.transaction_id,
                model_version=result.model_version,
                fraud_probability=round(result.fraud_probability, 6),
                action=result.action,  # type: ignore[arg-type]
                reasons=[
                    RiskReason(feature=r.feature, direction=r.direction,  # type: ignore[arg-type]
                               detail=r.detail, magnitude=r.magnitude)
                    for r in result.reasons
                ],
                is_model_ready=True,
                processed_at=datetime.now(UTC).isoformat(),
            )
            _apply_threshold_override(decision)
    except Exception:  # noqa: BLE001 - fail safe to review, never fail open
        decision = _safe_review_decision(event)
    finally:
        get_velocity().observe(event)

    _audit("decision.razorpay_demo", event, decision)
    webhook = build_webhook(order, decision)
    webhook["order"]["order_id"] = (body.get("order_id") or webhook["order"]["order_id"])
    return webhook


@app.get("/api/v1/razorpay/flow", tags=["razorpay"])
def razorpay_flow() -> dict:
    """Describe the demo payment lifecycle for the dashboard/docs."""
    return {
        "flow": [
            "create order (Razorpay test-mode shape)",
            "payment event reaches FINGRAPH",
            "velocity (strictly-past) computed",
            "XGBoost serve + SHAP -> calibrated action",
            "decision ALLOW / REVIEW / HOLD",
            "webhook -> audit ledger",
        ],
        "endpoints": {
            "create_order": "POST /api/v1/razorpay/order",
            "pay": "POST /api/v1/razorpay/pay",
            "flow": "GET /api/v1/razorpay/flow",
        },
        "note": "Demo adapter; no live Razorpay keys or network calls.",
    }


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
        _apply_threshold_override(decision)  # Layer 5 healing: miss/false-hold
        _audit("decision.scored", event, decision)
        return decision
    finally:
        # Commit the event into the streaming store after read + scoring, always.
        get_velocity().observe(event)


def _apply_threshold_override(decision: RiskDecision) -> None:
    """Layer 5 healing: re-derive the decision band from a live override."""
    over = get_healing().threshold_overrides()
    hold = over.get("hold")
    if hold is None:
        return
    try:
        base = _config()["thresholds"]
        review = float(over.get("review", base["review"]))
        p = float(decision.fraud_probability)
        decision.action = (  # type: ignore[assignment]
            "hold" if p >= float(hold) else "review" if p >= review else "allow"
        )
    except Exception:  # noqa: BLE001 - override must never break scoring
        return


def _find_audited_decision(transaction_id: str) -> dict | None:
    """Look up one audited decision payload by transaction id (Layer 6)."""
    try:
        records = get_ledger().store.scan()
    except Exception:  # noqa: BLE001 - fail-safe lookup
        return None
    for rec in records:
        payload = rec.get("payload") or {}
        if payload.get("transaction_id") == transaction_id:
            return payload
    return None


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


@app.post("/api/v1/healing/feedback", tags=["healing"])
def healing_feedback(body: FeedbackIn) -> dict:
    """Helix v2: record an outcome (fraud/legit) against an audited decision.

    The decision must exist in the Layer 6 ledger — feedback always references
    a real, audited decision; the ledger chain itself is never modified.
    """
    decision = _find_audited_decision(body.transaction_id)
    if decision is None:
        return {
            "ok": False,
            "error": f"no audited decision for transaction '{body.transaction_id}'",
        }
    ep = get_healing().record_feedback(body.transaction_id, body.outcome, decision)
    return {"ok": True, "episode": {
        "transaction_id": ep.transaction_id,
        "outcome": ep.outcome,
        "action": ep.action,
        "fail_type": ep.fail_type,
        "model_version": ep.model_version,
    }}


@app.get("/api/v1/healing/memory", tags=["healing"])
def healing_memory() -> dict:
    """Helix v2: what the system remembers — episodes, failures, hot-lists."""
    eng = get_healing()
    return {
        "stats": eng.memory.stats(),
        "merchant_rollup": eng.memory.merchant_rollup(),
        "hot_merchants": eng.memory.hot_merchants(),
    }


@app.get("/api/v1/healing/status", tags=["healing"])
def healing_status() -> dict:
    """Helix v2: full healing state — memory, drift, overrides, retrain queue."""
    return get_healing().stats()


@app.post("/api/v1/healing/heal", tags=["healing"])
def healing_heal() -> dict:
    """Helix v2: run one healing cycle now (hot-list + overrides + queue)."""
    return get_healing().heal()


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


_config_cache: dict[str, object] = {}


def _config() -> dict:
    """model_config.json, read once per file mtime (not per request)."""
    import json

    p = MODEL_DIR / "model_config.json"
    mtime = p.stat().st_mtime_ns if p.exists() else 0
    if _config_cache.get("_mtime") == mtime:
        return _config_cache["cfg"]  # type: ignore[return-value]
    cfg = json.loads(p.read_text()) if p.exists() else {}
    _config_cache["_mtime"] = mtime
    _config_cache["cfg"] = cfg
    return cfg


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
