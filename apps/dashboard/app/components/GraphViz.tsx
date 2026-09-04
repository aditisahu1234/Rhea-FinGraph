"use client";

// GraphViz — interactive force-directed rendering of the temporal fraud
// knowledge graph (Layer 2). Fully self-contained: a lightweight physics
// layout runs on a <canvas> with no external graph library. Data comes from
// /api/v1/graph/sample (real local snapshot) — nothing invented.
//
// Node color by type: customer = blue, merchant = amber, card = green.
// Merchants flagged `fraud` in the payload are red/triangular as a clear red
// flag. Edges: purchased = solid, has_card = dashed.

import { useEffect, useRef, useState } from "react";
import { fetchGraphSample, type GraphSample } from "../lib/api";

interface PNode {
  id: string;
  type: string;
  label: string;
  fraud?: boolean;
  x: number;
  y: number;
  vx: number;
  vy: number;
}

const COLORS: Record<string, string> = {
  customer: "#3b82f6",
  merchant: "#f59e0b",
  card: "#22c55e",
};

export default function GraphViz() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [graph, setGraph] = useState<GraphSample | null>(null);
  const [err, setErr] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const nodesRef = useRef<PNode[]>([]);
  const edgesRef = useRef<GraphSample["edges"]>([]);

  async function load() {
    setLoading(true);
    setErr("");
    try {
      const g = await fetchGraphSample(140);
      setGraph(g);
      edgesRef.current = g.edges;
      // seed positions on a ring
      const n = g.nodes.length;
      nodesRef.current = g.nodes.map((nd, i) => {
        const ang = (i / Math.max(1, n)) * Math.PI * 2;
        const r = 90 + 20 * Math.sin(i);
        return {
          ...nd,
          x: 260 + r * Math.cos(ang),
          y: 220 + r * Math.sin(ang),
          vx: 0,
          vy: 0,
        };
      });
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

  // physics + draw loop
  useEffect(() => {
    const canvas = canvasRef.current as HTMLCanvasElement | null;
    if (!canvas) return;
    const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;
    const W = canvas.width;
    const H = canvas.height;
    let raf = 0;
    let tick = 0;

    function step() {
      tick += 1;
      const nodes = nodesRef.current;
      const edges = edgesRef.current;
      if (nodes.length) {
        // repulsion
        for (let i = 0; i < nodes.length; i++) {
          for (let j = i + 1; j < nodes.length; j++) {
            const dx = nodes[j].x - nodes[i].x;
            const dy = nodes[j].y - nodes[i].y;
            const d2 = dx * dx + dy * dy + 1e-6;
            const f = 2400 / d2;
            const d = Math.sqrt(d2);
            nodes[i].vx -= (f * dx) / d;
            nodes[i].vy -= (f * dy) / d;
            nodes[j].vx += (f * dx) / d;
            nodes[j].vy += (f * dy) / d;
          }
        }
        // springs
        for (const e of edges) {
          const a = nodes.find((x) => x.id === e.source);
          const b = nodes.find((x) => x.id === e.target);
          if (!a || !b) continue;
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const d = Math.sqrt(dx * dx + dy * dy) + 1e-6;
          const target = 70;
          const f = 0.02 * (d - target);
          const fx = (f * dx) / d;
          const fy = (f * dy) / d;
          a.vx += fx;
          a.vy += fy;
          b.vx -= fx;
          b.vy -= fy;
        }
        // integrate + damping
        for (const n of nodes) {
          n.vx *= 0.85;
          n.vy *= 0.85;
          n.x += n.vx;
          n.y += n.vy;
          // soft bounds
          n.x = Math.max(20, Math.min(W - 20, n.x));
          n.y = Math.max(20, Math.min(H - 20, n.y));
        }
      }

      // draw
      ctx.clearRect(0, 0, W, H);
      // edges
      for (const e of edges) {
        const a = nodes.find((x) => x.id === e.source);
        const b = nodes.find((x) => x.id === e.target);
        if (!a || !b) continue;
        ctx.strokeStyle = a.type === b.type ? "#cbd5e1" : "#94a3b8";
        ctx.globalAlpha = 0.55;
        ctx.lineWidth = 1;
        if (e.kind === "has_card") ctx.setLineDash([4, 4]);
        else ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
      // nodes
      for (const n of nodes) {
        const r = n.type === "customer" ? 5 : n.type === "card" ? 4.5 : 6;
        const col = n.fraud ? "#ef4444" : COLORS[n.type] || "#94a3b8";
        const isSel = selected === n.id;
        ctx.beginPath();
        if (n.type === "merchant") {
          // triangle for merchants
          ctx.moveTo(n.x, n.y - r);
          ctx.lineTo(n.x + r, n.y + r);
          ctx.lineTo(n.x - r, n.y + r);
          ctx.closePath();
        } else {
          ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
        }
        ctx.fillStyle = col;
        ctx.globalAlpha = n.fraud ? 0.95 : 0.8;
        ctx.fill();
        ctx.globalAlpha = 1;
        if (isSel) {
          ctx.strokeStyle = "#0f172a";
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      }
      // labels for selected + fraud
      for (const n of nodes) {
        if (n.id === selected || n.fraud) {
          ctx.fillStyle = n.fraud ? "#ef4444" : "#0f172a";
          ctx.font = "11px ui-monospace, monospace";
          ctx.fillText(n.label, n.x + 7, n.y + 3);
        }
      }

      if (tick < 500) raf = requestAnimationFrame(step);
    }
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [selected]);

  function onCanvasClick(e: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    let best: PNode | null = null;
    let bestD = 14;
    for (const n of nodesRef.current) {
      const d = Math.hypot(n.x - mx, n.y - my);
      if (d < bestD) {
        bestD = d;
        best = n;
      }
    }
    setSelected(best ? best.id : null);
  }

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
      <div style={{ position: "relative" }}>
        <canvas
          ref={canvasRef}
          width={560}
          height={460}
          onClick={onCanvasClick}
          style={{
            width: "100%",
            height: "auto",
            border: "1px solid #e2e8f0",
            borderRadius: 8,
            background: "linear-gradient(180deg,#f8fafc,#eef2f7)",
            cursor: "pointer",
          }}
        />
        {loading && (
          <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", color: "#64748b" }}>
            building graph layout…
          </div>
        )}
      </div>
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
