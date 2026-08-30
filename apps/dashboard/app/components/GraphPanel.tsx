"use client";

// GraphPanel — Layer 2 graph store. Renders the REAL local graph pipeline
// (nodes/edges/fraud edges per temporal snapshot, GNN config) plus an HONEST
// Neo4j connectivity card: offline locally until `make ingest-graph` runs.
// No invented graph data — everything comes from /api/v1/graph/status.

import { useCallback, useEffect, useState } from "react";
import { fetchGraphStatus, type GraphStatus } from "../lib/api";

function fmt(n: number | null): string {
  if (n == null) return "—";
  return n.toLocaleString("en-US");
}

function compact(n: number | null): string {
  if (n == null) return "—";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

export default function GraphPanel() {
  const [graph, setGraph] = useState<GraphStatus | null>(null);
  const [err, setErr] = useState("");

  const refresh = useCallback(async () => {
    try {
      setGraph(await fetchGraphStatus());
      setErr("");
    } catch (e) {
      setErr(e instanceof Error ? e.message : "graph API unreachable");
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 15000);
    return () => clearInterval(id);
  }, [refresh]);

  const p = graph?.pipeline;
  const snapshots = p?.snapshots ?? [];
  const maxEdges = Math.max(1, ...snapshots.map((s) => s.n_edges ?? 0));
  const maxFraud = Math.max(1, ...snapshots.map((s) => s.n_fraud ?? 0));
  const neo4j = graph?.neo4j;

  return (
    <div className="panel">
      <div className="panel-head">
        <h2 className="panel-title">Graph store · Layer 2</h2>
        <span className={`pill ${neo4j?.reachable ? "ok" : "alert"}`}>
          {neo4j?.reachable ? "NEO4J ONLINE" : "NEO4J OFFLINE · LOCAL PIPELINE"}
        </span>
      </div>
      <p className="panel-sub">
        Temporal knowledge graph built from the same leakage-safe parquet
        splits: Customer → Merchant purchases with amount/channel, bucketed
        into monthly snapshots. The dashboard renders the local pipeline;
        Neo4j adds live Cypher on top when running.
      </p>

      {err && <p className="empty">graph API unreachable: {err}</p>}
      {!graph && !err && <p className="empty">Loading graph status…</p>}
      {graph && (
        <>
          <div className="stat-grid graph-stats">
            <div className="stat">
              <span className="stat-k">Merchants</span>
              <span className="stat-v">{compact(p?.n_merchants ?? null)}</span>
            </div>
            <div className="stat">
              <span className="stat-k">Customers</span>
              <span className="stat-v">{compact(p?.n_customers ?? null)}</span>
            </div>
            <div className="stat">
              <span className="stat-k">Cards</span>
              <span className="stat-v">{compact(p?.n_cards ?? null)}</span>
            </div>
            <div className="stat">
              <span className="stat-k">Edges (all snapshots)</span>
              <span className="stat-v">{compact(p?.total_edges ?? null)}</span>
            </div>
            <div className="stat">
              <span className="stat-k">Fraud edges</span>
              <span className="stat-v">{fmt(p?.total_fraud_edges ?? null)}</span>
            </div>
            <div className="stat">
              <span className="stat-k">Snapshots</span>
              <span className="stat-v">{fmt(p?.n_snapshots ?? null)}</span>
            </div>
          </div>

          {snapshots.length > 0 && (
            <div className="graph-trend">
              <div className="trend-label">
                <span>edges per snapshot (months {p?.month_range?.[0]}–{p?.month_range?.[1]})</span>
                <span className="muted">fraud edges (orange)</span>
              </div>
              <div className="trend-bars">
                {snapshots.map((s) => (
                  <div className="trend-col" key={s.month_idx} title={`M${s.month_idx}: ${fmt(s.n_edges)} edges, ${fmt(s.n_fraud)} fraud`}>
                    <div className="trend-fraud" style={{ height: `${((s.n_fraud ?? 0) / maxFraud) * 100}%` }} />
                    <div className="trend-edge" style={{ height: `${((s.n_edges ?? 0) / maxEdges) * 100}%` }} />
                  </div>
                ))}
              </div>
            </div>
          )}

          {p && Array.isArray(p.top_merchants) && p.top_merchants.length > 0 && (
            <div className="top-m">
              <div className="trend-label">
                <span>hottest confirmed-fraud merchants</span>
                <span className="muted">{p.top_merchants_source}</span>
              </div>
              <div className="audit-row audit-head">
                <span>merchant_id</span>
                <span>episodes</span>
                <span>confirmed fraud</span>
                <span>missed fraud</span>
                <span className="mono">share</span>
              </div>
              {p.top_merchants.slice(0, 8).map((m) => (
                <div className="audit-row" key={m.merchant_id}>
                  <span className="mono" title={m.merchant_id}>
                    {m.merchant_id}
                  </span>
                  <span>{m.txns.toLocaleString()}</span>
                  <span>{m.confirmed_fraud.toLocaleString()}</span>
                  <span>{m.missed_fraud.toLocaleString()}</span>
                  <span className="mono">
                    {m.txns > 0
                      ? `${((m.confirmed_fraud / m.txns) * 100).toFixed(2)}%`
                      : "—"}
                  </span>
                </div>
              ))}
            </div>
          )}

          {graph.gnn && (
            <div className="gnn-row">
              <div className="gnn-head">
                <span className="tag">Temporal GNN · Layer 3</span>
                <span className="muted">
                  {graph.gnn.architecture} · {fmt(graph.gnn.params ?? null)} params ·{" "}
                  {graph.gnn.epochs} epochs · {graph.gnn.fit_seconds}s on {graph.gnn.device_used}
                </span>
              </div>
              <div className="gnn-metrics">
                <span>
                  val ROC <b>{(graph.gnn.metrics_validation?.roc_auc ?? 0).toFixed(4)}</b>
                </span>
                <span>
                  test ROC <b>{(graph.gnn.metrics_test_locked?.roc_auc ?? 0).toFixed(4)}</b>
                </span>
                <span className="muted">
                  not row-aligned to the event split — fused scores require the
                  Kaggle aligned regeneration (docs/FUSION_KAGGLE_RUNBOOK.md)
                </span>
              </div>
            </div>
          )}
        </>
      )}

      <div className={`neo4j-card ${neo4j?.reachable ? "ok" : "off"}`}>
        <span className="neo4j-title">
          {neo4j?.reachable ? "● Neo4j reachable" : "○ Neo4j not reachable"}
        </span>
        <span className="muted">
          {neo4j?.reachable
            ? `${neo4j.url} — live graph queries available.`
            : `bolt://localhost:7687 refused — expected offline. Start Neo4j and run "make ingest-graph" to load the 24.39M-edge graph, then this card flips online and live Cypher becomes available.`}
        </span>
      </div>
    </div>
  );
}