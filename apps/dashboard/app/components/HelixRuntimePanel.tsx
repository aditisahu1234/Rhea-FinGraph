"use client";

// HelixRuntimePanel — Helix Runtime: PCEC 6-stage repair engine + Gene Map.
// Shows the system's self-healing loop as a live, measurable entity:
//   * Gene Map — SQLite+RL knowledge base of repair genes (Q-values,
//     success/failure counts), the "immune system".
//   * Measured recovery + gene-hit rates from real repairs (never the generic
//     99.9% marketing claim).
//   * "Trigger a failure" runs a scripted flaky error through the REAL PCEC
//     loop and stores the winning strategy as a gene; "Reset" clears it.

import { useCallback, useEffect, useState } from "react";
import {
  fetchHelixGenes,
  fetchHelixStatus,
  resetHelixGenes,
  runHelixDemoAttack,
  type HelixGenes,
  type HelixStatus,
} from "../lib/api";

function pct(v: number | null): string {
  if (v == null) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

export default function HelixRuntimePanel() {
  const [status, setStatus] = useState<HelixStatus | null>(null);
  const [genes, setGenes] = useState<HelixGenes | null>(null);
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const refresh = useCallback(async () => {
    try {
      const [s, g] = await Promise.all([fetchHelixStatus(), fetchHelixGenes()]);
      setStatus(s);
      setGenes(g);
      setErr("");
    } catch (e) {
      setErr(e instanceof Error ? e.message : "helix API unreachable");
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 8000);
    return () => clearInterval(id);
  }, [refresh]);

  async function trigger() {
    setBusy(true);
    try {
      const d = (await runHelixDemoAttack()) as {
        message?: string;
        stats?: { gene_count?: number };
      };
      setMsg(d.message ?? "PCEC repair executed.");
      await refresh();
    } catch (e) {
      setErr(e instanceof Error ? e.message : "demo failed");
    } finally {
      setBusy(false);
    }
  }

  async function reset() {
    setBusy(true);
    try {
      const d = (await resetHelixGenes()) as { message?: string };
      setMsg(d.message ?? "Gene Map reset.");
      await refresh();
    } catch (e) {
      setErr(e instanceof Error ? e.message : "reset failed");
    } finally {
      setBusy(false);
    }
  }

  const s = status;

  return (
    <div className="panel">
      <div className="panel-head">
        <h2 className="panel-title">Helix Runtime · PCEC + Gene Map</h2>
        <span className={`pill ${(genes?.count ?? 0) > 0 ? "ok" : "alert"}`}>
          {s?.status === "active" ? "ACTIVE · SELF-HEALING" : "OFFLINE"}
        </span>
      </div>
      <p className="panel-sub">
        The self-healing loop. Every time an operation fails, the 6-stage
        PCEC engine (Perceive → Construct → Evaluate → Commit → Verify →
        Gene) repairs it and stores the winning strategy as a{" "}
        <strong>gene</strong> so the identical failure resolves instantly next
        time. Recovery and gene-hit rates below are{" "}
        <strong>measured from real repairs here</strong> — not claimed.
      </p>

      {err && <p className="stream-err">{err}</p>}

      <div className="stream-grid">
        <div className="stream-block">
          <span className="stream-k">genes learned</span>
          <span className="stream-v">{genes?.count ?? 0}</span>
        </div>
        <div className="stream-block">
          <span className="stream-k">repairs run</span>
          <span className="stream-v">{s?.repair_attempts ?? 0}</span>
        </div>
        <div className="stream-block">
          <span className="stream-k">recovery rate</span>
          <span className={`stream-v ${(s?.recovery_rate ?? 0) >= 0.9 ? "ok" : ""}`}>
            {pct(s?.recovery_rate ?? null)}
          </span>
        </div>
        <div className="stream-block">
          <span className="stream-k">gene-hit rate</span>
          <span className="stream-v">{pct(s?.gene_hit_rate ?? null)}</span>
        </div>
      </div>

      <div className="stream-row" style={{ marginTop: 10 }}>
        <button className="score-btn" onClick={trigger} disabled={busy}>
          🧪 Trigger a failure (PCEC heals it)
        </button>
        <button className="score-btn" onClick={reset} disabled={busy}>
          🔄 Reset Gene Map
        </button>
      </div>
      {msg && <p className="stream-sub">{msg}</p>}

      {genes && genes.genes.length > 0 && (
        <div className="gene-list" style={{ marginTop: 12 }}>
          <div className="trend-label">
            <span>learned repair genes</span>
            <span className="muted">ordered by Q-value</span>
          </div>
          <table style={{ width: "100%", borderCollapse: "collapse", marginTop: 6 }}>
            <thead>
              <tr>
                <th className="mono small" style={{ textAlign: "left", borderBottom: "1px solid var(--line)", padding: "4px 6px" }}>strategy</th>
                <th className="mono small" style={{ textAlign: "right", borderBottom: "1px solid var(--line)", padding: "4px 6px" }}>Q</th>
                <th className="mono small" style={{ textAlign: "right", borderBottom: "1px solid var(--line)", padding: "4px 6px" }}>uses</th>
                <th className="mono small" style={{ textAlign: "right", borderBottom: "1px solid var(--line)", padding: "4px 6px" }}>success</th>
              </tr>
            </thead>
            <tbody>
              {genes.genes.map((g) => (
                <tr key={g.error_signature}>
                  <td className="mono small" style={{ padding: "4px 6px", borderBottom: "1px solid var(--line)" }}>
                    {JSON.stringify(g.repair_strategy)}
                  </td>
                  <td className="mono small" style={{ textAlign: "right", padding: "4px 6px", borderBottom: "1px solid var(--line)" }}>{g.q_value.toFixed(3)}</td>
                  <td className="mono small" style={{ textAlign: "right", padding: "4px 6px", borderBottom: "1px solid var(--line)" }}>{g.total_uses}</td>
                  <td className="mono small" style={{ textAlign: "right", padding: "4px 6px", borderBottom: "1px solid var(--line)" }}>{g.success_rate == null ? "—" : `${(g.success_rate * 100).toFixed(0)}%`}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {s && s.recent_repairs.length > 0 && (
        <div style={{ marginTop: 12 }}>
          <div className="trend-label"><span>recent repairs</span></div>
          {s.recent_repairs.map((r, i) => (
            <div key={i} className="audit-row" style={{ justifyContent: "space-between" }}>
              <span className="mono small">{r.error_type}</span>
              <span className="mono small">{JSON.stringify(r.strategy)}</span>
              <span className={`small ${r.success ? "ok" : "alert"}`}>
                {r.success ? "✔" : "✖"}
                {r.gene_hit ? " · gene" : ""}
              </span>
            </div>
          ))}
        </div>
      )}

      <p className="muted small" style={{ marginTop: 10 }}>
        The generic Helix runtime reports 99.9% recovery &lt;1ms fixes; this
        panel shows <em>this system&apos;s</em> measured counts and rates from
        real repairs, never borrowed stats.
      </p>
    </div>
  );
}
