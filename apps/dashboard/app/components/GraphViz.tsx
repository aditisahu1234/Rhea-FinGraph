"use client";

// GraphViz — renders the local temporal graph snapshot as a live
// force-directed canvas. Data always comes from /api/v1/graph/sample (real
// local snapshot); nothing invented. Rendering is delegated to the shared
// ForceGraphCanvas so the local view and the live Neo4j view look identical.

import { useEffect, useState } from "react";
import { fetchGraphSample, type GraphSample } from "../lib/api";
import ForceGraphCanvas from "./ForceGraphCanvas";

export default function GraphViz() {
  const [graph, setGraph] = useState<GraphSample | null>(null);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(true);

  async function load() {
    setLoading(true);
    setErr("");
    try {
      setGraph(await fetchGraphSample(140));
    } catch (e) {
      setErr(e instanceof Error ? e.message : "graph sample unreachable");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8 }}>
        <span className="muted small">
          {graph
            ? `${graph.n_nodes} nodes · ${graph.n_edges} edges · ${graph.source_snapshot}`
            : "local temporal graph snapshot"}
        </span>
        <button className="pill" onClick={load} disabled={loading}>
          {loading ? "loading…" : "Re-layout"}
        </button>
      </div>
      {err && <p className="error">{err}</p>}
      {loading && !graph && <p className="muted small">building graph layout…</p>}
      <ForceGraphCanvas
        nodes={(graph?.nodes ?? []).map((n) => ({ ...n }))}
        edges={(graph?.edges ?? []).map((e) => ({ ...e }))}
        emptyText="no graph data to render"
      />
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginTop: 6, fontSize: "0.75rem", color: "#64748b" }}>
        <span><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: "50%", background: "#3b82f6", marginRight: 4 }} />customer</span>
        <span><span style={{ display: "inline-block", width: 10, height: 10, transform: "rotate(45deg)", background: "#f59e0b", marginRight: 4 }} />merchant</span>
        <span><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: "50%", background: "#22c55e", marginRight: 4 }} />card</span>
        <span><span style={{ display: "inline-block", width: 10, height: 10, borderRadius: "50%", background: "#ef4444", marginRight: 4 }} />confirmed-fraud merchant</span>
      </div>
      <p className="muted small">{graph?.note}</p>
    </div>
  );
}
