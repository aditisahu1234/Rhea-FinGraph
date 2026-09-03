"use client";

// AttackSimulatorPanel — LIMITATION #7: adaptive-attack scenarios scored
// through the REAL engine (velocity -> v3 XGBoost -> raw margin). The before/
// after headline is the real raw model margin, where the model's velocity
// discrimination lives (calibrated proba is compressed 650x by
// calibration_scale_pos_weight). No number is invented.

import { useEffect, useState } from "react";
import {
  fetchAttackScenarios,
  runAttackSimulation,
  type AttackScenarios,
  type AttackSimulation,
} from "../lib/api";

function fmtRaw(v: number | null | undefined): string {
  if (v == null) return "—";
  return `${v >= 0 ? "+" : ""}${v.toFixed(3)}`;
}

export default function AttackSimulatorPanel() {
  const [meta, setMeta] = useState<AttackScenarios | null>(null);
  const [scenario, setScenario] = useState("VELOCITY_ATTACK");
  const [sim, setSim] = useState<AttackSimulation | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  useEffect(() => {
    fetchAttackScenarios().then(setMeta).catch((e) => setErr(String(e)));
  }, []);

  async function run(key: string) {
    setScenario(key);
    setLoading(true);
    setErr("");
    try {
      setSim(await runAttackSimulation(key));
    } catch (e) {
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="panel">
      <div className="panel-head">
        <h2 className="panel-title">Attack-scenario simulator</h2>
        <p className="panel-sub">
          Fraudsters adapt to a static rule — the defence is behavioural
          velocity. Each stream is scored through the REAL hero model (v3); the
          raw model margin is the honest risk reaction.
        </p>
      </div>

      {err && <p className="error">{err}</p>}

      {meta && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
          {meta.scenarios.map((s) => (
            <button
              key={s.key}
              onClick={() => run(s.key)}
              disabled={loading}
              className={s.key === scenario ? "pill ok" : "pill"}
            >
              {s.title}
            </button>
          ))}
        </div>
      )}

      {sim && (
        <div style={{ marginTop: "12px" }}>
          <p className="panel-sub">{sim.description}</p>
          <div className="impact-strip" style={{ marginTop: "10px" }}>
            <div className="impact-cell">
              <span className="muted small">Raw margin BEFORE</span>
              <span className="impact-num mono">{fmtRaw(sim.raw_margin_before)}</span>
            </div>
            <div className="impact-cell">
              <span className="muted small">Raw margin AFTER</span>
              <span className="impact-num mono">{fmtRaw(sim.raw_margin_after)}</span>
            </div>
            <div className="impact-cell">
              <span className="muted small">Δ velocity reaction</span>
              <span className="impact-num mono">{fmtRaw(sim.delta_raw_margin)}</span>
            </div>
            <div className="impact-cell">
              <span className="muted small">Model</span>
              <span className="impact-num mono" style={{ fontSize: "0.75rem" }}>
                {sim.model_used}
              </span>
            </div>
          </div>
          <p className="muted small" style={{ marginTop: "6px" }}>
            {sim.calibration_note}
          </p>

          <table className="audit-row" style={{ width: "100%", marginTop: "10px" }}>
            <thead>
              <tr className="audit-head">
                <th>#</th>
                <th>Amount (INR)</th>
                <th>Action</th>
                <th>Raw margin</th>
                <th>Calibrated</th>
              </tr>
            </thead>
            <tbody>
              {sim.steps.map((s) => (
                <tr key={s.index}>
                  <td>{s.index}</td>
                  <td>₹{s.amount_inr.toLocaleString("en-IN")}</td>
                  <td>
                    <span className="pill">{s.action}</span>
                  </td>
                  <td className="mono">{fmtRaw(s.raw_margin)}</td>
                  <td className="mono">{s.risk.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted small" style={{ marginTop: "8px" }}>
            {sim.honesty}
          </p>
        </div>
      )}
    </section>
  );
}
