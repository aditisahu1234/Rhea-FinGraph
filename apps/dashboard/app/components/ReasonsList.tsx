"use client";

// ReasonsList — renders SHAP top contributing features as horizontal bars.
// Positive (increases risk) bars go right in red, negative (reduces) left in
// teal, contextual reasons render as plain rows.

import type { RiskReason } from "../lib/api";

function barStyle(v: number, maxAbs: number) {
  const width = (Math.abs(v) / Math.max(maxAbs, 1e-6)) * 100;
  const fill = v > 0 ? "var(--neg)" : v < 0 ? "var(--pos)" : "var(--muted)";
  const justify = v > 0 ? "flex-start" : "flex-end";
  return { width: `${width}%`, background: fill, justify };
}

export default function ReasonsList({ reasons }: { reasons: RiskReason[] }) {
  const shap = reasons.filter((r) => r.magnitude != null);
  const maxAbs = Math.max(...shap.map((r) => Math.abs(r.magnitude ?? 0)), 1e-6);
  const contextual = reasons.filter((r) => r.magnitude == null);

  return (
    <div className="reasons">
      {shap.map((r, i) => (
        <div className="reason" key={`${r.feature}-${i}`}>
          <div className="reason-head">
            <span className="reason-feature">
              {r.feature}
              <span className="reason-dir">
                {r.direction === "increases_risk" ? "↑ risk" : "↓ risk"}
              </span>
            </span>
            <span className="reason-mag">
              {r.magnitude != null && r.magnitude > 0 ? "+" : ""}
              {r.magnitude?.toFixed(3)}
            </span>
          </div>
          <div className="reason-track">
            <div className="reason-bar" style={barStyle(r.magnitude ?? 0, maxAbs)} />
          </div>
        </div>
      ))}
      {contextual.map((r, i) => (
        <div className="reason contextual" key={`c-${i}`}>
          <span className="reason-feature">{r.feature}</span>
          <span className="detail">{r.detail}</span>
        </div>
      ))}
    </div>
  );
}
