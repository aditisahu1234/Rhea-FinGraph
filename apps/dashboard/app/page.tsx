"use client";

// Rhea FinGraph dashboard — Layer 0 vertical slice.
// Live transaction scoring + SHAP explainability + Helix (Layer 5) drift.

import { useEffect, useState } from "react";
import {
  API_BASE,
  fetchAuditHealth,
  fetchAuditRecent,
  fetchAuditSummary,
  fetchAuditVerify,
  fetchHelixDrift,
  fetchModelStatus,
} from "./lib/api";
import type {
  AuditHealth,
  AuditRecord,
  AuditSummary,
  AuditVerify,
  HelixDriftReport,
  ModelStatus,
} from "./lib/api";
import Scorer from "./components/Scorer";
import DriftPanel from "./components/DriftPanel";
import AuditPanel from "./components/AuditPanel";
import StreamingPanel from "./components/StreamingPanel";
import HealingPanel from "./components/HealingPanel";
import GraphPanel from "./components/GraphPanel";
import ModelRacePanel from "./components/ModelRacePanel";
import ModelSwitcherPanel from "./components/ModelSwitcherPanel";
import MetricsStrip from "./components/MetricsStrip";
import BusinessImpactPanel from "./components/BusinessImpactPanel";
import FinancialImpactCard from "./components/FinancialImpactCard";
import RazorpayDemoPanel from "./components/RazorpayDemoPanel";
import AttackSimulatorPanel from "./components/AttackSimulatorPanel";
import OutcomePanel from "./components/OutcomePanel";

const layers = [
  ["0 · Live scoring", "score every event · explain + audit · never executes"],
  ["1 · Streaming velocity", "rolling 1h/24h/7d · strictly-past · Redis/in-mem"],
  ["2 · Graph store", "Neo4j · 24.39M edges · 30 snapshots"],
  ["3 · Temporal GNN", "trained on Kaggle T4 · 46K params"],
  ["4 · Ensemble risk", "XGBoost + AE + SHAP/LIME explainability"],
  ["5 · Helix memory", "drift + failure memory + auto-retrain queue"],
  ["6 · Audit ledger", "tamper-evident decisions · hash-chained"],
];

export default function Home() {
  const [status, setStatus] = useState<ModelStatus | null>(null);
  const [drift, setDrift] = useState<HelixDriftReport | null>(null);
  const [auditHealth, setAuditHealth] = useState<AuditHealth | null>(null);
  const [auditSummary, setAuditSummary] = useState<AuditSummary | null>(null);
  const [auditVerify, setAuditVerify] = useState<AuditVerify | null>(null);
  const [auditRecent, setAuditRecent] = useState<AuditRecord[]>([]);
  const [apiUp, setApiUp] = useState<boolean | null>(null);

  async function refresh() {
    try {
      const [s, d, ah, asm, av, ar] = await Promise.all([
        fetchModelStatus(),
        fetchHelixDrift(),
        fetchAuditHealth(),
        fetchAuditSummary(),
        fetchAuditVerify(),
        fetchAuditRecent(8),
      ]);
      setStatus(s);
      setDrift(d);
      setAuditHealth(ah);
      setAuditSummary(asm);
      setAuditVerify(av);
      setAuditRecent(ar);
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

      <BusinessImpactPanel />

      <FinancialImpactCard />

      <div className="pool">
        <section className="panel score-panel">
          <div className="panel-head">
            <h2 className="panel-title">Live scoring</h2>
            <span className="muted">Layer 0 · {API_BASE}</span>
          </div>
          <Scorer />
        </section>

        <section className="panel">
          <AuditPanel
            health={auditHealth}
            summary={auditSummary}
            verify={auditVerify}
            recent={auditRecent}
          />
        </section>
      </div>

      <RazorpayDemoPanel />

      <AttackSimulatorPanel />

      <OutcomePanel />

      <section className="panel">
        <StreamingPanel />
      </section>

      <section className="panel">
        <GraphPanel />
      </section>

      <section className="panel">
        <ModelRacePanel />
      </section>

      <section className="panel">
        <ModelSwitcherPanel />
      </section>

      <section className="panel">
        <HealingPanel />
      </section>

      <section className="panel">
        <DriftPanel report={drift} />
      </section>

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
