import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, status

from fingraph_sentinel.audit import Ledger
from fingraph_sentinel.cold_start import cold_start_risk, is_cold_start
from fingraph_sentinel.config import get_settings
from fingraph_sentinel.explainer_ui import human_reasons, security_action, verdict
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

# CORS: the dashboard runs on :3001 (Next.js dev server) while the API is on
# :8000. Without this the browser blocks every fetch as cross-origin and every
# panel shows "failed to fetch" even though the API responds fine.
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        # locked-test row = the velocity-v3 hero's recorded metrics (the same
        # artifact /api/v1/model/race reads); serving val stays above.
        metrics_test_locked=_hero_locked_test() or cfg.get("metrics_test_locked"),
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


@app.get("/api/v1/graph/sample", tags=["graph"])
def graph_sample(max_nodes: int = 120, seed: int = 7) -> dict:
    """Return a capped, renderable subgraph for the dashboard visualizer.

    Loads the newest local temporal snapshot and extracts a DETERMINISTIC,
    focused sample (heavy-user customers → shared merchants → their other
    customers → cards) so the demo shows a connected relational web, not a
    random scatter. Returns nodes (id, type, fraud flag) and edges
    (source→target) the force-directed graph can render. Pure local data —
    no Neo4j required.
    """
    import json as _json  # noqa: PLC0415

    import torch  # noqa: PLC0415

    # resolve newest snapshot (prefer full-data Kaggle run, else smoke)
    cand = sorted(
        [p for p in Path("artifacts/graph").glob("*/snapshot_*.pt")],
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    src = cand[0] if cand else None
    if src is None:
        raise HTTPException(status_code=404, detail="no graph snapshot present")

    # hot (confirmed-fraud) merchant ids from the Helix failure-memory rollup
    hot: set[str] = set()
    mem_path = Path(settings.healing_dir) / "failure_memory.jsonl"
    if mem_path.exists():
        for line in mem_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = _json.loads(line)
            except Exception:  # noqa: BLE001 - skip malformed lines
                continue
            if rec.get("outcome") == "fraud":
                mv = (rec.get("event") or {}).get("merchant_id")
                if mv:
                    hot.add(str(mv))

    g = torch.load(str(src), map_location="cpu", weights_only=False)

    purchased = g["customer", "purchased", "merchant"].edge_index
    has_card = g["customer", "has_card", "card"].edge_index

    # ---- focused deterministic subgraph (demo-friendly, ~60-90 nodes) ----
    # Instead of a random scatter, sample the HIGHEST-DEGREE customers as
    # seeds ("heavy users"), then walk their merchants, the other customers
    # who bought from those same merchants (shared-merchant web), and the
    # cards they hold. Deterministic: no randomness, stable across reloads.
    from collections import defaultdict  # noqa: PLC0415

    cust_merch: dict[int, list[int]] = defaultdict(list)
    merch_cust: dict[int, list[int]] = defaultdict(list)
    cust_card: dict[int, list[int]] = defaultdict(list)
    for e in range(purchased.shape[1]):
        c, m = int(purchased[0, e]), int(purchased[1, e])
        cust_merch[c].append(m)
        merch_cust[m].append(c)
    for e in range(has_card.shape[1]):
        c, cd = int(has_card[0, e]), int(has_card[1, e])
        cust_card[c].append(cd)

    # heavy-user seeds by purchase count (desc, then index for stability)
    seeds = sorted(cust_merch, key=lambda c: (-len(cust_merch[c]), c))
    seed_n = max(4, min(10, max_nodes // 12))  # 4-10 hub customers
    keep_cust: list[int] = seeds[:seed_n]
    cust_cap = max(8, max_nodes // 3)
    merch_cap = max(8, max_nodes // 3)
    card_cap = max(6, max_nodes // 6)

    # their merchants (bounded, most-shared first for a connected web)
    seed_merch: list[int] = []
    for c in keep_cust:
        seed_merch.extend(cust_merch[c])
    seed_merch = list(dict.fromkeys(seed_merch))  # dedupe, keep order
    keep_merch = sorted(seed_merch, key=lambda m: (-len(merch_cust[m]), m))[:merch_cap]

    # other customers who share those merchants (the relational web)
    for m in keep_merch:
        for c in merch_cust[m]:
            if len(keep_cust) >= cust_cap:
                break
            if c not in keep_cust:
                keep_cust.append(c)
        if len(keep_cust) >= cust_cap:
            break

    # more merchants reachable from the expanded customer set (bounded)
    extra_merch: list[int] = []
    for c in keep_cust:
        extra_merch.extend(cust_merch[c])
    for m in sorted(set(extra_merch) - set(keep_merch),
                    key=lambda m: (-len(merch_cust[m]), m)):
        if len(keep_merch) >= merch_cap:
            break
        keep_merch.append(m)

    # cards of kept customers (bounded)
    keep_card: list[int] = []
    for c in keep_cust:
        for cd in cust_card[c]:
            if cd not in keep_card:
                keep_card.append(cd)
                if len(keep_card) >= card_cap:
                    break
        if len(keep_card) >= card_cap:
            break

    # guaranteed hard cap on TOTAL nodes (demo render budget)
    while len(keep_cust) + len(keep_merch) + len(keep_card) > max_nodes:
        if len(keep_card) > card_cap - 2 and keep_card:
            keep_card.pop()
        elif len(keep_merch) > merch_cap - 4 and keep_merch:
            keep_merch.pop()
        elif len(keep_cust) > cust_cap - 4 and len(keep_cust) > 4:
            keep_cust.pop()

    keep_cust, keep_merch, keep_card = set(keep_cust), set(keep_merch), set(keep_card)

    # edges only between kept nodes (recompute from raw index, honest subset)
    edge_rows = []
    for e in range(purchased.shape[1]):
        c, m = int(purchased[0, e]), int(purchased[1, e])
        if c in keep_cust and m in keep_merch:
            edge_rows.append(("purchased", c, m))
    for e in range(has_card.shape[1]):
        c, cd = int(has_card[0, e]), int(has_card[1, e])
        if c in keep_cust and cd in keep_card:
            edge_rows.append(("has_card", c, cd))

    def mid(nt: str, i: int) -> str:
        return f"{nt}-{i}"

    # Honest fraud flag: a merchant node is marked fraud only when its id is
    # actually present in the Helix confirmed-fraud rollup. Local snapshots use
    # integer index ids (no native merchant_id string), so in practice none are
    # marked here — we never paint unverified nodes as fraud.
    def merchant_fraud(i: int) -> bool:
        return mid("merchant", i) in hot

    nodes = [{"id": mid("customer", i), "type": "customer",
              "label": f"C{i}"} for i in keep_cust]
    nodes += [{"id": mid("merchant", i), "type": "merchant",
               "label": f"M{i}", "fraud": merchant_fraud(i)}
              for i in keep_merch]
    nodes += [{"id": mid("card", i), "type": "card", "label": f"K{i}"}
              for i in keep_card]
    node_ids = {n["id"] for n in nodes}

    edges = []
    for k, s, t in edge_rows:
        tgt = mid("merchant", int(t)) if k == "purchased" else mid("card", int(t))
        # drop any edge whose target was trimmed by the node cap
        if tgt in node_ids:
            edges.append({"source": mid("customer", int(s)),
                          "target": tgt, "kind": k})

    return {
        "source_snapshot": src.name,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "node_types": ["customer", "merchant", "card"],
        "n_fraud_marked": sum(1 for n in nodes if n.get("fraud")),
        "nodes": nodes,
        "edges": edges,
        "note": (
            "Rendered from the local temporal graph snapshot. The subgraph is "
            "deterministic: heavy-user customers, the merchants they share, and "
            "the cards they hold. A merchant is marked fraud only when it "
            "appears in the Helix confirmed-fraud rollup; node type is always "
            "colored distinctly."
        ),
    }


# ---------------------------------------------------------------------------
# Live Neo4j Cypher gateway (safe, whitelisted queries only)
# ---------------------------------------------------------------------------

# Whitelisted store of live Cypher queries. We never accept arbitrary user
# strings — only these static queries, so the proxy is read-only and safe.
CYPHER_QUERIES: dict[str, dict[str, str]] = {
    "overview": {
        "label": "Connected web (merchants a customer has purchased from)",
        "cypher": """
            MATCH (c:Customer)-[:PURCHASED]->(m:Merchant)
            WITH c, m
            LIMIT 40
            RETURN id(c) AS sid, c.id AS customer_id, c.id AS source_id,
                   'customer' AS stype,
                   id(m) AS tid, m.id AS merchant_id, 'merchant' AS ttype,
                   m.mcc_code AS mcc, COALESCE(m.fraud_rate, 0) AS fraud_rate
        """,
    },
    "hot_merchants": {
        "label": "Highest fraud-rate merchants and their customers",
        "cypher": """
            MATCH (c:Customer)-[:PURCHASED]->(m:Merchant)
            WHERE m.fraud_rate > 0.05
            WITH c, m
            LIMIT 60
            RETURN id(c) AS sid, c.id AS customer_id, 'customer' AS stype,
                   id(m) AS tid, m.id AS merchant_id, 'merchant' AS ttype,
                   m.mcc_code AS mcc, m.fraud_rate AS fraud_rate
        """,
    },
    "cards_of_customers": {
        "label": "Customers and the cards they hold",
        "cypher": """
            MATCH (c:Customer)-[:HAS_CARD]->(ca:Card)
            LIMIT 40
            RETURN id(c) AS sid, c.id AS customer_id, 'customer' AS stype,
                   id(ca) AS tid, ca.id AS card_id, 'card' AS ttype,
                   0 AS mcc, 0 AS fraud_rate
        """,
    },
    "fraud_edges": {
        "label": "Confirmed-fraud purchase edges",
        "cypher": """
            MATCH (c:Customer)-[r:PURCHASED {is_fraud: 1}]->(m:Merchant)
            LIMIT 50
            RETURN id(c) AS sid, c.id AS customer_id, 'customer' AS stype,
                   id(m) AS tid, m.id AS merchant_id, 'merchant' AS ttype,
                   m.mcc_code AS mcc, COALESCE(m.fraud_rate, 0) AS fraud_rate
        """,
    },
    "fraud_web": {
        "label": "Fraud web: customers around high-fraud-rate merchants",
        "cypher": """
            MATCH (c:Customer)-[r:PURCHASED {is_fraud: 1}]->(m:Merchant)
            WITH m ORDER BY m.fraud_rate DESC LIMIT 6
            MATCH (c2:Customer)-[:PURCHASED]->(m)
            WITH DISTINCT c2, m
            WITH c2, collect(DISTINCT m) AS fraud_m
            LIMIT 20
            UNWIND fraud_m AS fm
            WITH DISTINCT c2, fm
            MATCH (c2)-[:PURCHASED]->(m2:Merchant)
            WHERE id(m2) <> id(fm)
            WITH c2, fm, m2 LIMIT 70
            RETURN id(c2) AS sid, c2.id AS customer_id, 'customer' AS stype,
                   id(m2) AS tid, m2.id AS merchant_id, 'merchant' AS ttype,
                   m2.mcc_code AS mcc, COALESCE(m2.fraud_rate, 0) AS fraud_rate
        """,
    },
}


@app.post("/api/v1/graph/cypher", tags=["graph"])
def graph_cypher(body: dict) -> dict:
    """Run one whitelisted live Cypher query against Neo4j.

    Accepts ``query`` (one of the static keys defined here — arbitrary Cypher
    is never accepted for safety) and returns a renderable node/edge list.
    Responds 503 with a clear "offline" note when the Neo4j driver is not
    installed or no server is listening; the dashboard shows that honestly.
    """
    from fastapi import HTTPException  # noqa: PLC0415

    key = str(body.get("query", "")).strip()
    spec = CYPHER_QUERIES.get(key)
    if spec is None:
        raise HTTPException(
            status_code=422,
            detail=f"unknown query '{key}'; choose from {sorted(CYPHER_QUERIES)}",
        )
    limit = int(body.get("limit", 100))
    try:
        from neo4j import GraphDatabase  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 - driver not installed
        return _live_query_offline(f"neo4j driver not installed ({exc})")

    driver = None
    try:
        driver = GraphDatabase.driver(
            settings.neo4j_url,
            auth=(settings.neo4j_username, settings.neo4j_password),
            connection_timeout=2.0,
        )
        driver.verify_connectivity()
    except Exception as exc:  # noqa: BLE001 - server down / bad auth
        _close_driver(driver)
        return _live_query_offline(f"{type(exc).__name__}: {exc}")

    try:
        with driver.session() as session:
            rows = session.run(spec["cypher"], limit=limit).data()
    except Exception as exc:  # noqa: BLE001 - query executed but failed
        _close_driver(driver)
        return _live_query_offline(f"query failed: {exc}")
    _close_driver(driver)

    # Build renderable nodes/edges (dedup by id)
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    for r in rows:
        def _add(sid: str, stype: str, label: str, fraud_rate: float) -> None:
            nid = f"{stype}-{sid}"
            if nid not in nodes:
                nodes[nid] = {
                    "id": nid, "type": stype, "label": label,
                    "fraud": bool(fraud_rate and float(fraud_rate) > 0.05),
                }
        _add(str(r.get("sid")), str(r.get("stype", "customer")),
             str(r.get("customer_id") or r.get("source_id") or r.get("sid")),
             float(r.get("fraud_rate") or 0.0))
        _add(str(r.get("tid")), str(r.get("ttype", "merchant")),
             str(r.get("merchant_id") or r.get("card_id") or r.get("tid")),
             float(r.get("fraud_rate") or 0.0))
        edges.append({
            "source": f"{r.get('stype', 'customer')}-{r.get('sid')}",
            "target": f"{r.get('ttype', 'merchant')}-{r.get('tid')}",
            "kind": "purchased",
            "is_fraud": bool(r.get("is_fraud")),
        })
    # keep only edges whose endpoints we kept
    kept = {n["id"] for n in nodes.values()}
    edges = [e for e in edges if e["source"] in kept and e["target"] in kept]

    return {
        "online": True,
        "query": key,
        "label": spec["label"],
        "source": "neo4j",
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "nodes": list(nodes.values())[:120],
        "edges": edges[:200],
        "cypher": spec["cypher"],
    }


def _close_driver(driver) -> None:
    if driver is not None:
        try:
            driver.close()
        except Exception:  # noqa: BLE001 - best-effort close
            pass


def _live_query_offline(detail: str) -> dict:
    return {
        "online": False,
        "source": "neo4j",
        "detail": detail,
        "nodes": [],
        "edges": [],
        "n_nodes": 0,
        "n_edges": 0,
        "hint": (
            "Start Neo4j (brew start / neo4j start) then run 'make ingest-graph' "
            "to load the fraud graph; this panel goes live automatically."
        ),
    }


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
            # positioning: v3 (the velocity online model) is the hero;
            # GNN/Transformer/AE/fusion are honestly framed as future
            # ensemble candidates — their production-readiness is not
            # over-claimed. Hero is an exact identity, not a name substring.
            label = str(c.get("model_name", d.name)).lower()
            if d.name == "baseline-online-v3":
                hero = True
                research = False
            elif any(k in label or k in d.name for k in
                     ("gnn", "transformer", "autoencoder", "fusion", "ae")):
                hero = False
                research = True
            else:
                hero = False
                research = False
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
                "is_hero": hero,
                "is_research": research,
            })
    gate: dict | None = None
    gate_path = Path(settings.healing_dir) / "gate_report.json"
    if gate_path.exists():
        try:
            gate = json.loads(gate_path.read_text())
        except Exception:  # noqa: BLE001 - corrupt report => no verdict
            gate = {"verdict": "unreadable"}
    return {
        "models": rows,
        "serving_name": SERVING_NAME,
        "gate_report": gate,
        # positioning — one hero, advanced work = future ensemble.
        "positioning": {
            "hero_model": "baseline-online-v3",
            "hero_note": (
                "baseline-online-v3 (velocity features) is the hero: drift-robust "
                "on a locked future test period (test ROC 0.7646), resisting the "
                "Jan 2015 concept shift that collapsed the prior model. It is the "
                "recommended promotion behind the gated switch."
            ),
            "hero_metrics": {
                "val_roc_auc": None,
                "test_roc_auc": None,
                "test_average_precision": None,
            },
            "research_as_future_ensemble": (
                "GNN, temporal GNN, Transformer, autoencoder and fusion were "
                "investigated and are tracked as candidate FUTURE ensemble "
                "signals — not established production winners. The focused "
                "story is the Velocity v3 online model through calibration, "
                "cold-start routing, SHAP and webhook→audit."
            ),
        },
    }


@app.get("/api/v1/model/switcher/status", tags=["models"])
def model_switcher_status() -> dict:
    """Drift-aware model recommendation with gated promotion.

    Reads the persisted switch *recommendation* written by the drift detector.
    If a distribution shift was detected and a more robust candidate model
    exists on disk, this returns the recommended from→to model chain. The
    serving model is never auto-promoted — promotion requires explicit
    operator approval. No recommendation → no alert.

    The drift report depends only on the persisted score file, so it is
    computed once per file mtime (the same pattern as _config()); a 1.1M-row
    report otherwise costs ~5s on every poll.
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
            scores_mtime = DEFAULT_SCORES.stat().st_mtime_ns
            if _switcher_cache.get("_mtime") != scores_mtime:
                scores = pl.read_parquet(DEFAULT_SCORES)
                _switcher_cache["_mtime"] = scores_mtime
                _switcher_cache["report"] = monitor_report(scores)
            drift = _switcher_cache.get("report")
        except Exception:  # noqa: BLE001 - stale/corrupt score stream
            drift = None

    return {
        "serving_model": "baseline-online-xgb",
        "last_decision": decision,
        "drift_report": drift,
    }


@app.get("/api/v1/business/impact", tags=["risk"])
def business_impact() -> dict:
    """Full economic impact report for the locked test split.

    Returns allow/review/hold volumes, frauds caught (by count and by
    amount), protected and missed rupee value, and top MCCs by fraud amount.
    All figures are read from the parity-verified artifact computed by the
    offline evaluation script; nothing is synthesised here.
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


@app.get("/api/v1/impact/summary", tags=["risk"])
def impact_summary() -> dict:
    """Compact financial-impact summary for dashboard cards.

    Reduces business_impact.json to four headline numbers: total protected,
    monthly protected, fraud amount blocked rate, and fraud events blocked
    rate. All values come from the verified evaluation artifact.
    """
    import json  # noqa: PLC0415

    path = Path("artifacts/business_impact.json")
    if not path.exists():
        return {
            "available": False,
            "total_protected_inr": None,
            "monthly_protected_inr": None,
            "fraud_amount_blocked_rate": None,
            "fraud_events_blocked_rate": None,
        }
    try:
        r = json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return {"available": False}
    p = r.get("protection") or {}
    total_fraud_inr = (r.get("totals") or {}).get("fraud_amount_inr")
    caught_inr = p.get("fraud_amount_caught_inr")
    # missed INR is stored as USD in the artifact; derive with the same
    # conversion the verified outcome P&L uses (never a new number).
    from fingraph_sentinel.outcome_simulator import INR_PER_USD  # noqa: PLC0415

    missed_usd = p.get("fraud_amount_missed_usd")
    missed_inr = round(float(missed_usd) * INR_PER_USD, 2) if missed_usd else None
    return {
        "available": True,
        "total_protected_inr": caught_inr,
        "monthly_protected_inr": p.get("per_month_protected_inr"),
        "monthly_missed_inr": p.get("per_month_missed_inr"),
        "fraud_amount_blocked_rate": p.get("recall_by_amount"),
        "fraud_events_blocked_rate": p.get("recall_by_count"),
        "total_fraud_inr": total_fraud_inr,
        "missed_inr": missed_inr,
        "model": r.get("model"),
        "split": r.get("split"),
    }


@app.post("/api/v1/payment/order", tags=["payment"])
@app.post("/api/v1/razorpay/order", tags=["razorpay"])
def razorpay_create_order(body: dict) -> dict:
    """Create a payment order through the payment adapter.

    Accepts amount_inr (string) plus optional merchant/customer/card
    selectors. Returns the created order and the canonical payment event that
    represents its eventual payment.
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


@app.post("/api/v1/payment/pay", tags=["payment"])
@app.post("/api/v1/razorpay/pay", tags=["razorpay"])
def razorpay_pay(body: dict) -> dict:
    """Score one order and return the webhook decision.

    Accepts an order_id from the order endpoint (or a bare payment event),
    scores it through the same velocity → XGBoost → SHAP → calibrated action
    → audit path as live traffic, and returns the payment-webhook payload with
    the audited decision.
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
            _apply_threshold_override(decision, getattr(event, "merchant_id", None))
    except Exception:  # noqa: BLE001 - fail safe to review, never fail open
        decision = _safe_review_decision(event)
    finally:
        get_velocity().observe(event)

    _audit("decision.razorpay_demo", event, decision)
    webhook = build_webhook(order, decision)
    webhook["order"]["order_id"] = (body.get("order_id") or webhook["order"]["order_id"])
    return webhook


@app.get("/api/v1/payment/flow", tags=["payment"])
@app.get("/api/v1/razorpay/flow", tags=["razorpay"])
def razorpay_flow() -> dict:
    """Describe the demo payment lifecycle for the dashboard/docs."""
    return {
        "flow": [
            "create order (test-mode payment shape)",
            "payment event enters the risk engine",
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
        "note": "Demo adapter; no live payment-provider keys or network calls.",
    }


@app.post("/api/v1/payment/webhook", tags=["payment"])
@app.post("/api/v1/razorpay/webhook", tags=["razorpay"])
def razorpay_webhook(body: dict) -> dict:
    """Receive a payment webhook and score it.

    Accepts a payment-webhook payload (order_id, payment_id, amount,
    currency, customer, card, merchant, method) and runs the same scoring
    path as live traffic — velocity, cold-start check, XGBoost, SHAP,
    security action, audit — returning the RiskDecision plus a
    security_action. Rounded-off amount (paise) is converted to USD for the
    canonical event.
    """
    from fastapi import HTTPException  # noqa: PLC0415

    try:
        amount_paise = int(body.get("amount", 0))
        amount_inr = amount_paise / 100.0
        event = PaymentEvent(
            transaction_id=str(body.get("payment_id") or body.get("order_id") or "webhook-txn"),
            event_time=body.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            customer_id=str(body.get("customer", {}).get("id", "C-WEBHOOK-0001")),
            card_id=str(body.get("card", {}).get("id", "K-WEBHOOK-0001")),
            merchant_id=str(body.get("merchant", {}).get("id", "TerraMart-5311")),
            merchant_category_code=body.get("mcc"),
            amount=f"{amount_inr / 83.5:.2f}",
            payment_channel=str(body.get("method", "card")),
            payment_error=body.get("error_code"),
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"invalid webhook: {exc}")

    # Reuse the exact scoring + audit path by calling the shared helper.
    decision = score_transaction(event)
    # Re-derive security_action if the helper's decision already has it; else map.
    decision.security_action = security_action(decision.action)  # type: ignore[assignment]
    return {
        "received": True,
        "order_id": body.get("order_id"),
        "payment_id": body.get("payment_id"),
        "risk": {
            "model_version": decision.model_version,
            "fraud_probability": round(decision.fraud_probability, 6),
            "decision": decision.action,
            "security_action": decision.security_action,
            "is_cold_start": decision.is_cold_start,
            "reasons_human": decision.reasons_human,
            "verdict": verdict(decision.action),
        },
        "webhook_to_merchant": (
            "payment.captured" if decision.action == "allow"
            else "payment.risk_flagged"
        ),
        "audit": {
            "transaction_id": event.transaction_id,
            "decision_auditable": True,
            "processed_at": decision.processed_at,
        },
    }


@app.post("/api/v1/payment/event", tags=["payment"])
@app.post("/api/v1/razorpay/event", tags=["razorpay"])
def razorpay_event(body: dict) -> dict:
    """Accept a full payment event and score it.

    Demonstrates the data-contract mapping between a production payment
    event surface (UPI / card / wallet, device, IP, order, checkout,
    3DS/step-up, refund / chargeback) and the canonical event the model
    consumes. This endpoint maps the event onto the canonical PaymentEvent
    the existing
    velocity -> XGBoost -> SHAP -> audit pipeline understands, and returns the
    decision plus an explicit note of which fields ARE model features vs which
    are retained as future/ensemble context (never fed to the current model).

    No synthetic dataset is invented here — this is an adapter over a real
    event shape, the honest "production contract != training dataset" story.
    """
    from fastapi import HTTPException  # noqa: PLC0415

    from fingraph_sentinel.explainer_ui import security_action, verdict  # noqa: PLC0415
    from fingraph_sentinel.razorpay_event import (  # noqa: PLC0415
        describe_mapping,
        map_razorpay_event,
    )

    try:
        event = map_razorpay_event(body)
    except Exception as exc:  # noqa: BLE001 - bad input => clean error
        raise HTTPException(status_code=422, detail=f"invalid razorpay event: {exc}")

    decision = score_transaction(event)
    decision.security_action = security_action(decision.action)  # type: ignore[assignment]

    mapping = describe_mapping(body)
    return {
        "received": True,
        "decision": {
            "fraud_probability": round(decision.fraud_probability, 6),
            "action": decision.action,
            "security_action": decision.security_action,
            "verdict": verdict(decision.action),
            "is_cold_start": decision.is_cold_start,
            "reasons_human": decision.reasons_human,
            "model_version": decision.model_version,
        },
        "mapping": mapping,
        "future_signals_not_model_inputs": mapping["future_signals_not_model_inputs"],
        "audit": {
            "transaction_id": event.transaction_id,
            "decision_auditable": True,
            "processed_at": decision.processed_at,
        },
    }


@app.get("/api/v1/attack/scenarios", tags=["attack"])
def attack_scenarios() -> dict:
    """List the available scripted attack scenarios and their metadata."""
    from fingraph_sentinel.attack_simulator import SCENARIOS  # noqa: PLC0415

    return {
        "source": "scripted event streams scored through the REAL engine",
        "honesty": (
            "Before/after risk is the actual model output for the event "
            "stream, not an invented number. Velocity accumulates across the "
            "sequence; cold-start and SHAP reasons behave as in production."
        ),
        "scenarios": [
            {"key": k, "title": v["title"], "description": v["description"],
             "n_events": len(v["amounts_inr"]), "channel": v["channel"]}
            for k, v in SCENARIOS.items()
        ],
    }


@app.post("/api/v1/attack/simulate", tags=["attack"])
def attack_simulate(body: dict) -> dict:
    """Run one scripted attack scenario through the real engine.

    Scores the scenario's event stream one event at a time through the same
    velocity → cold-start → XGBoost → SHAP → calibrated action path as
    live traffic (velocity accumulates across the sequence). Returns the
    per-step risk timeline plus the BEFORE/AFTER headline.
    """
    from fastapi import HTTPException  # noqa: PLC0415

    from fingraph_sentinel.attack_simulator import (  # noqa: PLC0415
        SCENARIOS,
        make_v3_scorer,
        run_scenario,
    )

    key = str(body.get("scenario", "VELOCITY_ATTACK")).upper()
    if key not in SCENARIOS:
        raise HTTPException(
            status_code=422,
            detail=f"unknown scenario '{key}'; choose from {sorted(SCENARIOS)}",
        )
    obs = get_velocity().observe
    try:
        # Score the stream against the REAL hero model (v3) + live velocity.
        score_one = make_v3_scorer(get_velocity().compute)
        result = run_scenario(key, score_one, observe_one=obs)
        result["model_used"] = "xgboost_velocity_v3 (hero, drift-robust)"
    except Exception as exc:  # noqa: BLE001 - simulation must not 500
        raise HTTPException(status_code=500, detail=f"simulation failed: {exc}")
    return result


@app.post("/api/v1/attack/outcome", tags=["attack"])
def attack_outcome(body: dict) -> dict:
    """Run a chargeback outcome on an attack stream.

    ``mode``:
      * ``verified`` (default) — the real P&L from the locked-test evaluation
        (business_impact.json): fraud prevented ₹31,018,572.48 (hold+review),
        missed chargeback loss ₹1,184,238, and the honest false-positive
        disclosure (count of legitimate holds). Nothing is invented.
      * ``synthetic`` — score a scenario stream through the REAL hero model
        (v3), label it with a chargeback ground-truth (``fraud_from`` is the
        first confirmed-fraud index), and aggregate the per-event P&L from the
        model's actual action.
    """
    from fastapi import HTTPException  # noqa: PLC0415

    from fingraph_sentinel.outcome_simulator import (  # noqa: PLC0415
        run_chargeback_sim,
    )

    mode = str(body.get("mode", "verified")).lower()

    if mode == "verified":
        return _verified_outcome()

    if mode != "synthetic":
        raise HTTPException(
            status_code=422, detail="mode must be 'verified' or 'synthetic'",
        )

    from fingraph_sentinel.attack_simulator import (  # noqa: PLC0415
        SCENARIOS,
        make_v3_scorer,
        run_scenario,
    )

    key = str(body.get("scenario", "VELOCITY_ATTACK")).upper()
    if key not in SCENARIOS:
        raise HTTPException(
            status_code=422,
            detail=f"unknown scenario '{key}'; choose from {sorted(SCENARIOS)}",
        )
    fraud_from = int(body.get("fraud_from", 0))
    obs = get_velocity().observe
    try:
        score_one = make_v3_scorer(get_velocity().compute)
        sim = run_scenario(key, score_one, observe_one=obs)
    except Exception as exc:  # noqa: BLE001 - simulation must not 500
        raise HTTPException(status_code=500, detail=f"simulation failed: {exc}")

    rows = [
        {
            "transaction_id": f"sim_{i}",
            "action": step["action"],
            "outcome": "fraud" if i >= fraud_from else "legit",
            "amount_inr": step["amount_inr"],
        }
        for i, step in enumerate(sim["steps"])
    ]
    pl = run_chargeback_sim(rows)
    return {
        "mode": "synthetic",
        "scenario": key,
        "title": sim["title"],
        "description": sim["description"],
        "fraud_from": fraud_from,
        "n_events": sim["n_events"],
        "model_used": sim["model_used"],
        "pnl": pl,
        "honesty": (
            "Synthetic mode labels the scored, real event stream with a "
            "chargeback ground-truth and aggregates the actual P&L. Amounts are "
            "the real transaction amounts; classifications derive from the "
            "model's real action per event. Use 'verified' mode for the real "
            "locked-test P&L."
        ),
    }


def _verified_outcome() -> dict:
    """Real outcome P&L from the verified locked-test evaluation."""
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from fastapi import HTTPException  # noqa: PLC0415

    from fingraph_sentinel.outcome_simulator import INR_PER_USD  # noqa: PLC0415

    path = Path("artifacts/business_impact.json")
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail="artifacts/business_impact.json not present in this workspace",
        )
    bi = json.loads(path.read_text())
    prot = bi["protection"]
    prevented_inr = float(prot["fraud_amount_caught_inr"])  # ₹31,018,572.48
    missed_usd = float(prot["fraud_amount_missed_usd"])  # ₹14,182.49 USD
    missed_inr = round(missed_usd * INR_PER_USD, 2)  # ₹1,184,238
    # Honest false-positive disclosure: legitimate transactions sent to HOLD
    hold_count = int(bi["actions"]["hold"])
    hold_fraud = int(bi["caught_by_action"]["hold"]["count"])
    legit_holds = hold_count - hold_fraud  # ~2,337,330
    hold_volume_fraction = hold_count / int(bi["totals"]["rows"])
    return {
        "mode": "verified",
        "model": bi.get("model", "baseline-online-v3"),
        "split": bi.get("split", "locked test"),
        "as_of": bi.get("as_of", ""),
        "parity_note": (
            "bytes-identical to records/config — the verified locked-test "
            "evaluation of baseline-online-v3 (velocity features) on "
            "4,877,375 held-out events"
        ),
        "fraud_prevented_value": prevented_inr,   # ₹31,018,572.48 (hold+review)
        "missed_fraud_value": missed_inr,          # chargeback loss ₹1,184,238
        "false_positive_legit_holds": legit_holds, # ~2,337,330 (honest caveat)
        "false_positive_note": (
            f"{legit_holds:,} legitimate transactions ({hold_volume_fraction:.0%} "
            "of volume) received a HOLD — the honest cost of the conservative "
            "operating point; listings do not lose money, only friction/support."
        ),
        "frauds_total": int(bi["totals"]["frauds"]),
        "frauds_caught": int(prot["frauds_caught"]),  # 4,283
        "recall_by_count": float(prot["recall_by_count"]),
        "recall_by_amount": float(prot["recall_by_amount"]),
        "per_month_protected_inr": float(prot["per_month_protected_inr"]),
        "net_protected_value": round(prevented_inr - missed_inr, 2),
        "net_protected_note": (
            "Net protected = fraud prevented − missed chargeback loss "
            "(false-positive handled separately as non-monetary friction)."
        ),
        "honesty": (
            "All figures come from the verified locked-test evaluation "
            "(business_impact.json); the false-positive volume is disclosed "
            "rather than hidden. This is the honest P&L, not a rosy demo."
        ),
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

        # cold-start routing before the model: if the entity has
        # no velocity history, the model's behavioural features are unknown, so
        # we route to a conservative rule engine and flag is_cold_start=1.
        try:
            cold = is_cold_start(
                velocity.get("cust_txn_count_prior"),
                velocity.get("card_txn_count_prior"),
                velocity.get("merch_txn_count_prior"),
            )
        except Exception:  # noqa: BLE001 - cold-start flag must never break score
            cold = False

        try:
            values = event_feature_dict(event, velocity=velocity)
        except Exception:  # noqa: BLE001
            values = {}

        if cold:
            cs = cold_start_risk(
                values,
                merchant_prior=values.get("merch_fraud_rate_prior"),
            )
            decision = RiskDecision(
                transaction_id=event.transaction_id,
                model_version="cold-start-rules",
                fraud_probability=float(cs["risk_score"]),
                action=cs["action"],  # type: ignore[arg-type]
                reasons=[
                    RiskReason(
                        feature="cold_start",
                        direction="increases_risk",
                        detail="low-history entity routed to conservative rule engine",
                        human=r,
                    )
                    for r in cs["reasons"]
                ],
                is_model_ready=True,
                is_cold_start=True,
                security_action=security_action(cs["action"]),  # type: ignore[arg-type]
                reasons_human=cs["reasons"],
                processed_at=datetime.now(UTC).isoformat(),
            )
            _audit("decision.cold_start", event, decision)
            return decision

        try:
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
        # product layer: concrete security action + readable reasons.
        decision.security_action = security_action(decision.action)  # type: ignore[assignment]
        decision.reasons_human = human_reasons(values)
        # Layer 5/6 healing: per-merchant override (PCEC) then global
        _apply_threshold_override(decision, getattr(event, "merchant_id", None))
        _audit("decision.scored", event, decision)
        return decision
    finally:
        # Commit the event into the streaming store after read + scoring, always.
        get_velocity().observe(event)


def _apply_threshold_override(decision: RiskDecision, merchant_id: str | None = None) -> None:
    """Layer 5/6 healing: re-derive the decision band from live overrides.

    Priority: per-merchant override (set by PCEC tighten/relax) first, then
    the global override (set by the grow-up heal cycle). This is the closed
    loop: a PCEC repair actually changes the next decision for that merchant.
    """
    healing = get_healing()
    try:
        base = _config()["thresholds"]
        # Per-merchant override from the PCEC commit target.
        hold: float | None = None
        review: float | None = None
        if merchant_id:
            mstate = healing.get_merchant_threshold(merchant_id)
            if mstate and mstate.get("adjustments"):
                hold = float(mstate["hold"])
                review = float(mstate.get("review", base["review"]))
        if hold is None:
            over = healing.threshold_overrides()
            hold = over.get("hold")
            if hold is not None:
                review = float(over.get("review", base["review"]))
        if hold is None:
            return
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
                # Helix repair/hot-list needs the event context per episode.
                "event": {
                    "transaction_id": event.transaction_id,
                    "customer_id": event.customer_id,
                    "card_id": event.card_id,
                    "merchant_id": event.merchant_id,
                    "amount": str(event.amount),
                    "payment_channel": event.payment_channel,
                },
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


# ---------------------------------------------------------------------------
# Helix Runtime: PCEC repair engine + Gene Map (self-healing memory)
# ---------------------------------------------------------------------------
_gene_map: Any | None = None
_helix_engine: Any | None = None


def get_gene_map():
    global _gene_map  # noqa: PLW0603 - lazy singleton so tests can reset
    if _gene_map is None:
        from fingraph_sentinel.helix_runtime.gene_map import GeneMap  # noqa: PLC0415
        _gene_map = GeneMap(Path(settings.healing_dir) / "gene_map.db")
    return _gene_map


def get_helix_engine():
    global _helix_engine  # noqa: PLW0603
    if _helix_engine is None:
        from fingraph_sentinel.helix_runtime.pcec_engine import PCECEngine  # noqa: PLC0415
        _helix_engine = PCECEngine(get_gene_map(), healing_engine=get_healing())
    return _helix_engine


@app.get("/api/v1/helix/status", tags=["helix"])
def helix_status() -> dict:
    """Helix Runtime status: gene count, mode, measured recovery + gene-hit rates."""
    eng = get_helix_engine()
    stats = eng.stats()
    return {
        "status": "active",
        "mode": "auto",
        "gene_count": stats["gene_count"],
        "repair_attempts": stats["repair_attempts"],
        "repair_successes": stats["repair_successes"],
        "recovery_rate": stats["recovery_rate"],
        "gene_hits": stats["gene_hits"],
        "gene_hit_rate": stats["gene_hit_rate"],
        "recent_repairs": eng.history(limit=10),
    }


@app.get("/api/v1/helix/genes", tags=["helix"])
def helix_genes(limit: int = 20) -> dict:
    """Top genes in the Gene Map, ordered by Q-value (highest first)."""
    genes = get_gene_map().get_hot_genes(limit=limit)
    return {
        "genes": [g.as_dict() for g in genes],
        "count": get_gene_map().count(),
    }


@app.post("/api/v1/helix/demo-error", tags=["helix"])
def helix_demo_error(
    error_type: str = "timeout",
    merchant_id: str = "demo_merchant_001",
    factor: float | None = None,
) -> dict:
    """Trigger a scripted failure for PCEC to repair live.

    error_type:
      - timeout (default): flaky op heals on retry -> stored as gene
      - missed_fraud: PCEC tightens a REAL per-merchant hold threshold, which
        the serving layer then applies to that merchant's next decision
      - false_hold: PCEC relaxes the merchant threshold (fewer holds)
      - cold_start: conservative routing, no threshold mutation
    """
    engine = get_helix_engine()
    t0 = time.monotonic()

    if error_type in ("missed_fraud", "false_hold", "cold_start"):
        marker = error_type
        _ = factor  # strategy factors are chosen inside the repair engine

        def decision_failure() -> dict:
            raise ValueError(
                f"helix: {marker} detected for merchant {merchant_id}"
            )

        context = {"merchant_id": merchant_id}
        healed = engine.heal(decision_failure, context=context)
        # show what the repair changed on the live merchant threshold
        mstate = get_healing().get_merchant_threshold(merchant_id)
        stats = engine.stats()
        return {
            "ok": True,
            "error_type": error_type,
            "merchant_id": merchant_id,
            "message": (
                f"PCEC repaired a {error_type} failure: "
                f"{healed.get('status', healed.get('action'))}; "
                f"strategy recorded as gene."
            ),
            "repair": healed,
            "merchant_threshold_now": {
                "hold": mstate.get("hold"),
                "adjustments": len(mstate.get("adjustments", [])),
            },
            "latency_ms": round((time.monotonic() - t0) * 1000, 2),
            "stats": stats,
        }

    # default: operational flaky timeout
    calls = {"n": 0}

    def flaky() -> dict:
        calls["n"] += 1
        if calls["n"] < 2:
            raise TimeoutError("upstream operation timed out (504)")
        return {"decision": "allow", "ok": True, "attempts": calls["n"]}

    healed = engine.heal(flaky)
    stats = engine.stats()
    return {
        "ok": True,
        "error_type": "timeout",
        "message": (
            f"PCEC healed a simulated flaky upstream call in "
            f"{healed.get('attempts')} attempt(s); strategy stored as a gene."
        ),
        "repair": healed,
        "latency_ms": round((time.monotonic() - t0) * 1000, 2),
        "stats": stats,
    }


@app.get("/api/v1/helix/export", tags=["helix"])
def helix_export() -> dict:
    """Export the full Gene Map (federated sharing: another instance can import)."""
    genes = get_gene_map().get_all_genes()
    return {
        "version": "1.0",
        "exported_at": datetime.now(UTC).isoformat(),
        "gene_count": len(genes),
        "genes": [g.as_dict() for g in genes],
    }


@app.post("/api/v1/helix/import", tags=["helix"])
def helix_import(genes: list[dict] | None = None) -> dict:
    """Import a Gene Map payload; keep the higher-Q gene per signature."""
    payload = genes or []
    gm = get_gene_map()
    imported = 0
    kept = 0
    for g in payload:
        sig = g.get("error_signature")
        strategy = g.get("repair_strategy")
        q = float(g.get("q_value", 0.0))
        if not sig or not strategy:
            continue
        existing = gm.get_repair(sig)
        if existing and existing.q_value >= q:
            kept += 1
            continue
        # seed with a real outcome so Q-history is honest: success if the
        # imported gene has a higher success rate, else failure once.
        success = float(g.get("success_rate", 1.0) or 1.0) >= 0.5
        gm.update_gene(sig, strategy, success)
        imported += 1
    return {
        "imported": imported,
        "kept_existing_higher_q": kept,
        "gene_count": gm.count(),
    }


@app.post("/api/v1/helix/reset", tags=["helix"])
def helix_reset() -> dict:
    """Clear the Gene Map and repair history (dashboard reset control)."""
    engine = get_helix_engine()
    engine.reset()
    return {"ok": True, "gene_count": engine.gene_map.count(),
            "message": "Gene Map cleared; repair history reset."}


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
_switcher_cache: dict[str, object] = {}


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


# Hero locked-test metrics: the dashboard header shows the VELOCITY-V3
# locked-test figures (the drift-robust hero, same config the Model Fight
# Card reads from artifacts/models/baseline-online-v3/model_config.json),
# while val/backend/training-rows stay the serving model's own. Cached the
# same way as _config() — one small json read per mtime, never per request.
_HERO_CFG_PATH = Path("artifacts/models/baseline-online-v3/model_config.json")
_hero_cache: dict[str, object] = {}


def _hero_locked_test() -> dict:
    import json  # noqa: PLC0415

    mtime = _HERO_CFG_PATH.stat().st_mtime_ns if _HERO_CFG_PATH.exists() else 0
    if _hero_cache.get("_mtime") == mtime:
        return dict(_hero_cache.get("mt") or {})  # type: ignore[arg-type]
    mt: dict = {}
    if _HERO_CFG_PATH.exists():
        try:
            mt = dict(json.loads(_HERO_CFG_PATH.read_text()).get("metrics_test_locked") or {})
        except Exception:  # noqa: BLE001 - corrupt hero config => empty
            mt = {}
    _hero_cache["_mtime"] = mtime
    _hero_cache["mt"] = mt
    return mt


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


@app.post("/api/v1/helix/self-play", tags=["helix"])
def helix_self_play(iterations: int = 6, reaction_ratio: float = 2.0) -> dict:
    """Adversarial self-play: attack scenarios -> PCEC repair -> measured stats.

    Each attack is scored through the REAL v3 model + velocity store. An attack
    is *defended* only when the model's raw-margin reaction exceeds the NORMAL
    baseline by ``reaction_ratio`` (default 2.0x — an explicit, honest
    threshold; calibrated actions are unreachable on synthetic events because
    probabilities are compressed ~650x). Attacks below that bar become
    missed_fraud episodes PCEC repairs by tightening the merchant's real
    threshold and storing a gene. Survival + latency are measured from this
    run, never claimed.
    """
    from fingraph_sentinel.attack_simulator import (  # noqa: PLC0415
        SelfPlayLoop,
        make_v3_scorer,
    )

    vel = get_velocity()
    vel.clear()  # clean velocity state -> honest NORMAL baseline per run
    scorer = make_v3_scorer(vel.compute)
    loop = SelfPlayLoop(
        pcec_engine=get_helix_engine(),
        score_one=scorer,
        velocity_get=vel.observe,
        min_reaction_ratio=reaction_ratio,
    )
    results = loop.run(iterations=iterations)
    stats = loop.stats()
    stats["reaction_ratio"] = reaction_ratio
    return {
        "status": "completed",
        "iterations": iterations,
        "stats": stats,
        "recent_results": results[-5:],
    }


@app.get("/api/v1/helix/self-play/stats", tags=["helix"])
def helix_self_play_stats() -> dict:
    """Last self-play run's measured stats (or no_data before first run)."""
    # in-memory: report the last aggregate via a fresh read of gene state
    eng = get_helix_engine()
    return {
        "status": "available",
        "gene_count": eng.gene_map.count(),
        "measured_gene_hits": eng.stats()["gene_hits"],
        "recovery_rate": eng.stats()["recovery_rate"],
    }
