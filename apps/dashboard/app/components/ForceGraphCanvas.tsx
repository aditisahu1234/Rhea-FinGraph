"use client";

// ForceGraphCanvas — reusable force-directed canvas renderer for live graph
// data (local snapshots OR Neo4j Cypher). Pure renderer: takes nodes+edges as
// props, runs a lightweight physics layout on a <canvas>, animates, and lets
// the user click a node to inspect it. No external graph library.

import { useEffect, useRef, useState } from "react";

export interface FGNode {
  id: string;
  type: string;
  label: string;
  fraud?: boolean;
}
export interface FGEdge {
  source: string;
  target: string;
  kind?: string;
  is_fraud?: boolean;
}

interface PNode extends FGNode {
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

export default function ForceGraphCanvas({
  nodes,
  edges,
  height = 380,
  emptyText = "no graph data to render",
}: {
  nodes: FGNode[];
  edges: FGEdge[];
  height?: number;
  emptyText?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const nodesRef = useRef<PNode[]>([]);
  const edgesRef = useRef<FGEdge[]>(edges);
  const [selected, setSelected] = useState<string | null>(null);

  // rebuild physics model whenever new data arrives
  useEffect(() => {
    edgesRef.current = edges;
    const n = nodes.length;
    nodesRef.current = nodes.map((nd, i) => {
      const ang = (i / Math.max(1, n)) * Math.PI * 2;
      const r = 80 + 18 * Math.sin(i * 1.7);
      return {
        ...nd,
        x: 240 + r * Math.cos(ang),
        y: 180 + r * Math.sin(ang),
        vx: 0,
        vy: 0,
      };
    });
    setSelected(null);
  }, [nodes, edges]);

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
      const ns = nodesRef.current;
      const es = edgesRef.current;
      if (ns.length) {
        for (let i = 0; i < ns.length; i++) {
          for (let j = i + 1; j < ns.length; j++) {
            const dx = ns[j].x - ns[i].x;
            const dy = ns[j].y - ns[i].y;
            const d2 = dx * dx + dy * dy + 1e-6;
            const f = 2800 / d2;
            const d = Math.sqrt(d2);
            ns[i].vx -= (f * dx) / d;
            ns[i].vy -= (f * dy) / d;
            ns[j].vx += (f * dx) / d;
            ns[j].vy += (f * dy) / d;
          }
        }
        for (const e of es) {
          const a = ns.find((x) => x.id === e.source);
          const b = ns.find((x) => x.id === e.target);
          if (!a || !b) continue;
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const d = Math.sqrt(dx * dx + dy * dy) + 1e-6;
          const target = 55;
          const f = 0.04 * (d - target);
          const fx = (f * dx) / d;
          const fy = (f * dy) / d;
          a.vx += fx;
          a.vy += fy;
          b.vx -= fx;
          b.vy -= fy;
        }
        for (const n of ns) {
          n.vx *= 0.85;
          n.vy *= 0.85;
          n.x += n.vx;
          n.y += n.vy;
          n.x = Math.max(14, Math.min(W - 14, n.x));
          n.y = Math.max(14, Math.min(H - 14, n.y));
        }
      }

      ctx.clearRect(0, 0, W, H);
      for (const e of es) {
        const a = ns.find((x) => x.id === e.source);
        const b = ns.find((x) => x.id === e.target);
        if (!a || !b) continue;
        ctx.globalAlpha = e.is_fraud ? 0.9 : 0.5;
        ctx.strokeStyle = e.is_fraud ? "#ef4444" : "#94a3b8";
        ctx.lineWidth = e.is_fraud ? 1.6 : 1;
        ctx.setLineDash(e.kind === "has_card" ? [4, 4] : []);
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;

      for (const n of ns) {
        const r = n.type === "merchant" ? 6 : 4.5;
        const col = n.fraud ? "#ef4444" : COLORS[n.type] || "#94a3b8";
        ctx.beginPath();
        if (n.type === "merchant") {
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
        if (selected === n.id) {
          ctx.strokeStyle = "#0f172a";
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      }
      for (const n of ns) {
        if (n.id === selected || n.fraud) {
          ctx.fillStyle = n.fraud ? "#ef4444" : "#0f172a";
          ctx.font = "11px ui-monospace, monospace";
          ctx.fillText(n.label, n.x + 7, n.y + 3);
        }
      }
      if (tick < 400) raf = requestAnimationFrame(step);
    }
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [selected, height]);

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

  if (nodes.length === 0) {
    return (
      <div style={{
        height, width: "100%", border: "1px solid #e2e8f0", borderRadius: 8,
        display: "grid", placeItems: "center", color: "#94a3b8",
        background: "#f8fafc", fontSize: "0.82rem",
      }}>
        {emptyText}
      </div>
    );
  }

  return (
    <canvas
      ref={canvasRef}
      width={560}
      height={Math.max(height, 200)}
      onClick={onCanvasClick}
      style={{
        width: "100%", height: "auto", border: "1px solid #e2e8f0",
        borderRadius: 8, background: "linear-gradient(180deg,#f8fafc,#eef2f7)",
        cursor: "pointer",
      }}
    />
  );
}
