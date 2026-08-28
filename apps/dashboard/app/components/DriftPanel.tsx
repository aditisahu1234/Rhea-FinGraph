"use client";

// DriftPanel — Helix Layer 5 self-healing memory. Shows the retrain trigger
// and per-feature PSI / mean-shift drift so the operator sees *why* the model
// may be decaying, per feature, rather than a flat aggregate.

import type { HelixDriftReport } from "../lib/api";

function psiClass(psi: number | null) {
  if (psi == null) return "";
  if (psi > 0.25) return "hot";
  if (psi > 0.1) return "warm";
  return "ok";
}

export default function DriftPanel({ report }: { report: HelixDriftReport | null }) {
  if (!report) {
    return (
      <div className="panel">
        <h2 className="panel-title">Helix memory · Layer 5</h2>
        <p className="empty">No drift report loaded.</p>
      </div>
    );
  }
  const features = [...(report.features ?? [])].sort(
    (a, b) => Math.abs(b.psi ?? 0) - Math.abs(a.psi ?? 0)
  );

  return (
    <div className="panel">
      <div className="panel-head">
        <h2 className="panel-title">Helix memory · Layer 5</h2>
        <span
          className={`pill ${report.trigger === "YES" ? "alert" : "ok"}`}
        >
          {report.trigger === "YES" ? "RETRAIN TRIGGERED" : "STABLE"}
        </span>
      </div>
      <p className="panel-sub">
        Self-healing memory watches per-feature distribution drift. Aggregate
        score drift looked flat (mean ≈0.006) while the input migrated — this
        panel surfaces the per-feature culprit.
      </p>

      <div className="drift-bars">
        {features.slice(0, 8).map((f) => (
          <div className="drift-row" key={f.feature}>
            <span className="drift-name" title={f.feature}>
              {f.feature}
            </span>
            <div className="drift-track">
              <div
                className={`drift-bar ${psiClass(f.psi)}`}
                style={{ width: `${Math.min(100, Math.abs(f.psi ?? 0) * 14)}%` }}
              />
            </div>
            <span className="drift-psi">{f.psi?.toFixed(2)}</span>
            <span className="drift-z" title="standardized mean shift">
              z {f.z?.toFixed(1)}
            </span>
          </div>
        ))}
      </div>

      {report.culprits.length > 0 && (
        <div className="culprits">
          <span className="culprit-label">culprits</span>
          {report.culprits.slice(0, 6).map((c) => (
            <span className="tag" key={c}>
              {c}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
