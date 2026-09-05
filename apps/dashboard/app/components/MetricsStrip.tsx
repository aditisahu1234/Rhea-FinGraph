"use client";

// MetricsStrip — model KPIs (val ROC-AUC, test ROC-AUC, AP) plus preparedness.

import type { ModelStatus } from "../lib/api";

function fmt(v: number | undefined) {
  if (v == null) return "—";
  return v.toFixed(4).replace(/^0/, "");
}

export default function MetricsStrip({ status }: { status: ModelStatus | null }) {
  if (!status || !status.ready) {
    return (
      <div className="metrics">
        <div className="metric pill-warn">
          <span className="metric-v">NO MODEL</span>
          <span className="metric-k">No trained model on disk</span>
        </div>
      </div>
    );
  }
  const v = status.metrics_validation ?? {};
  const t = status.metrics_test_locked ?? {};
  return (
    <div className="metrics">
      <div className="metric">
        <span className="metric-v">{fmt(v["roc_auc"] as number)}</span>
        <span className="metric-k">val ROC-AUC</span>
      </div>
      <div className="metric">
        <span className="metric-v">{fmt(t["roc_auc"] as number)}</span>
        <span className="metric-k">test ROC-AUC</span>
      </div>
      <div className="metric">
        <span className="metric-v">
          {fmt((t["average_precision"] as number) ?? (t["ap"] as number))}
        </span>
        <span className="metric-k">test avg precision</span>
      </div>
      <div className="metric">
        <span className="metric-v">{status.backend ?? "—"}</span>
        <span className="metric-k">backend</span>
      </div>
      <div className="metric">
        <span className="metric-v">
          {status.training_rows != null
            ? (status.training_rows / 1e6).toFixed(1) + "M"
            : "—"}
        </span>
        <span className="metric-k">training rows</span>
      </div>
    </div>
  );
}
