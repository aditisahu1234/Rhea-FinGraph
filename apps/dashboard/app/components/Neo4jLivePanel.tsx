"use client";

// Neo4jLivePanel — live Cypher console for the dashboard. Presents the
// whitelisted Neo4j queries (read-only, safe), runs one through the backend
// proxy /api/v1/graph/cypher, and renders the result with the shared
// force-directed visualizer. Shows an honest OFFLINE state (with the exact
// setup hint) until Neo4j is reachable — it never fabricates graph data.

import { useEffect, useState } from "react";
import { runGraphCypher, type CypherResult } from "../lib/api";
import ForceGraphCanvas from "./ForceGraphCanvas";

const QUERIES: { key: string; label: string }[] = [
  { key: "overview", label: "Connected web (customers ↔ merchants)" },
  { key: "hot_merchants", label: "Highest fraud-rate merchants" },
  { key: "cards_of_customers", label: "Customers and their cards" },
  { key: "fraud_edges", label: "Confirmed-fraud purchase edges" },
];

export default function Neo4jLivePanel() {
  const [query, setQuery] = useState("overview");
  const [result, setResult] = useState<CypherResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  async function run() {
    setBusy(true);
    setErr("");
    try {
      setResult(await runGraphCypher(query));
    } catch (e) {
      setErr(e instanceof Error ? e.message : "cypher request failed");
    } finally {
      setBusy(false);
    }
  }

  // Auto-run the default query once on mount so the live Neo4j graph is
  // visible without a click (the backend proxy + Neo4j are already up).
  useEffect(() => {
    run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const online = result?.online === true;

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", marginTop: 12 }}>
        <select
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={{ flex: 1, minWidth: 220 }}
        >
          {QUERIES.map((q) => (
            <option key={q.key} value={q.key}>{q.label}</option>
          ))}
        </select>
        <button className="pill ok" onClick={run} disabled={busy || !query}>
          {busy ? "Querying…" : "Run query"}
        </button>
        <span className={`pill ${online ? "ok" : "alert"}`}>
          {online ? "NEO4J LIVE" : "NEO4J OFFLINE"}
        </span>
      </div>

      {online && result && (
        <p className="muted small" style={{ marginTop: 8 }}>
          <strong>{result.label}</strong> — {result.n_nodes} nodes · {result.n_edges} edges · via{" "}
          <code>{result.source}</code>
        </p>
      )}
      {!online && result && (
        <div className="neo4j-card off" style={{ marginTop: 10 }}>
          <span className="neo4j-title">○ Neo4j not reachable</span>
          <span className="muted">{result.hint ?? result.detail}</span>
        </div>
      )}
      {err && <p className="error">{err}</p>}

      {result && online && result.nodes.length > 0 && (
        <>
          <ForceGraphCanvas
            nodes={result.nodes}
            edges={result.edges}
            emptyText="query returned no graph"
          />
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginTop: 6, fontSize: "0.75rem", color: "#64748b" }}>
            <span><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: "50%", background: "#3b82f6", marginRight: 4 }} />customer</span>
            <span><span style={{ display: "inline-block", width: 10, height: 10, transform: "rotate(45deg)", background: "#f59e0b", marginRight: 4 }} />merchant</span>
            <span><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: "50%", background: "#22c55e", marginRight: 4 }} />card</span>
            <span><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: "50%", background: "#ef4444", marginRight: 4 }} />high fraud-rate merchant</span>
          </div>
        </>
      )}

      {online && result && result.cypher && (
        <pre className="mono" style={{ marginTop: 10, fontSize: "0.7rem", overflow: "auto" }}>
          {result.cypher}
        </pre>
      )}
    </div>
  );
}
