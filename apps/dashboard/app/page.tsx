"use client";

// Rhea FinGraph dashboard — Layer 0 vertical slice.
// Live transaction scoring + SHAP explainability + Helix (Layer 5) drift.

import { useEffect, useState } from "react";
import { API_BASE, fetchHelixDrift, fetchModelStatus } from "./lib/api";
import type { HelixDriftReport, ModelStatus } from "./lib/api";
import Scorer from "./components/Scorer";
import DriftPanel from "./components/DriftPanel";
import MetricsStrip from "./components/MetricsStrip";

const layers = [
  ["2 · Graph store", "Neo4j · 24.39M edges · 30 snapshots"],
  ["3 · Temporal GNN", "trained on Kaggle T4 · 46K params"],
  ["4 · Ensemble risk", "XGBoost + AE + SHAP/LIME explainability"],
  ["5 · Helix memory", "per-feature drift + retrain trigger"],
];

export default function Home() {
  const [status, setStatus] = useState<ModelStatus | null>(null);
  const [drift, setDrift] = useState<HelixDriftReport | null>(null);
  const [apiUp, setApiUp] = useState<boolean | null>(null);

  async function refresh() {
    try {
      const [s, d] = await Promise.all([
        fetchModelStatus(),
        fetchHelixDrift(),
      ]);
      setStatus(s);
      setDrift(d);
      setApiUp(true);
    } catch {
      setApiUp(false);
    }
  }

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 10000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo">RF</span>
          <div>
            <h1>Rhea FinGraph</h1>
            <p className="tagline">
              Defense-only · Temporal graph intelligence for merchant fraud
              decisions
            </p>
          </div>
        </div>
        <div className="status-chip">
          <span
            className={`dot ${apiUp === false ? "down" : "up"}`}
          />
          {apiUp === false ? "API offline" : "API online"}
        </div>
      </header>

      <MetricsStrip status={status} />

      <div className="pool">
        <section className="panel score-panel">
          <div className="panel-head">
            <h2 className="panel-title">Live scoring</h2>
            <span className="muted">Layer 0 · {API_BASE}</span>
          </div>
          <Scorer />
        </section>

        <section className="panel">
          <DriftPanel report={drift} />
        </section>
      </div>

      <section className="layers">
        {layers.map(([k, v]) => (
          <div className="layer" key={k}>
            <span className="layer-k">{k}</span>
            <span className="layer-v">{v}</span>
          </div>
        ))}
      </section>

      <footer className="foot muted">
        Rhea FinGraph · defense-only fraud intelligence · every decision is
        auditable and never executes a payment.
      </footer>
    </div>
  );
}
