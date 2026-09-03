"use client";

// OutcomePanel — LIMITATION #10: outcome / chargeback simulator. Two modes:
//   * verified — the REAL locked-test P&L (fraud prevented ₹31,018,572.48,
//     missed chargeback loss ₹1,184,238, honest false-positive disclosure of
//     2,337,330 legitimate holds). No number invented.
//   * synthetic — score a stream through the real hero model, apply your
//     chargeback ground-truth, and aggregate the honest per-event P&L.

import { useEffect, useState } from "react";
import {
  runAttackOutcome,
  type AttackOutcome,
  type SyntheticOutcome,
  type VerifiedOutcome,
} from "../lib/api";

function inrCr(v: number | null | undefined): string {
  if (v == null) return "—";
  return `₹${(v / 1_00_00_000).toFixed(2)} Cr`;
}

function inrFull(v: number | null | undefined): string {
  if (v == null) return "—";
  return `₹${v.toLocaleString("en-IN")}`;
}

export default function OutcomePanel() {
  const [mode, setMode] = useState<"verified" | "synthetic">("verified");
  const [scenario, setScenario] = useState("VELOCITY_ATTACK");
  const [fraudFrom, setFraudFrom] = useState(3);
  const [outcome, setOutcome] = useState<AttackOutcome | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  useEffect(() => {
    runAttackOutcome({ mode: "verified" }).then(setOutcome).catch((e) => setErr(String(e)));
  }, []);

  async function run() {
    setLoading(true);
    setErr("");
    try {
      if (mode === "verified") {
        setOutcome(await runAttackOutcome({ mode: "verified" }));
      } else {
        setOutcome(
          await runAttackOutcome({
            mode: "synthetic",
            scenario,
            fraud_from: Number(fraudFrom),
          })
        );
      }
    } catch (e) {
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  }

  const verified = outcome?.mode === "verified" ? (outcome as VerifiedOutcome) : null;
  const synth = outcome?.mode === "synthetic" ? (outcome as SyntheticOutcome) : null;

  return (
    <section className="panel">
      <div className="panel-head">
        <h2 className="panel-title">Outcome · chargeback simulator</h2>
        <p className="panel-sub">
          A decision is only half the story. Given the real chargeback outcome,
          what did we actually save? Verified shows the locked-test P&amp;L;
          synthetic scores a stream and applies a fraud outcome you choose.
        </p>
      </div>

      <div style={{ display: "flex", gap: "8px", alignItems: "center", flexWrap: "wrap" }}>
        <button
          className={mode === "verified" ? "pill ok" : "pill"}
          onClick={() => setMode("verified")}
        >
          Verified (locked test)
        </button>
        <button
          className={mode === "synthetic" ? "pill ok" : "pill"}
          onClick={() => setMode("synthetic")}
        >
          Synthetic stream
        </button>
        {mode === "synthetic" && (
          <>
            <select
              value={scenario}
              onChange={(e) => setScenario(e.target.value)}
            >
              <option value="VELOCITY_ATTACK">Velocity attack</option>
              <option value="AMOUNT_SPIKE">Amount spike</option>
              <option value="NORMAL">Normal</option>
            </select>
            <label className="muted small">
              fraud from event #
              <input
                type="number"
                min={0}
                value={fraudFrom}
                onChange={(e) => setFraudFrom(Number(e.target.value))}
                style={{ width: "48px", marginLeft: "6px" }}
              />
            </label>
            <button className="pill ok" onClick={run} disabled={loading}>
              {loading ? "running…" : "Run"}
            </button>
          </>
        )}
      </div>

      {err && <p className="error">{err}</p>}

      {verified && (
        <div style={{ marginTop: "12px" }}>
          <div className="impact-strip">
            <div className="impact-cell">
              <span className="muted small">Fraud prevented</span>
              <span className="impact-num">{inrCr(verified.fraud_prevented_value)}</span>
            </div>
            <div className="impact-cell">
              <span className="muted small">Missed chargeback loss</span>
              <span className="impact-num" style={{ color: "#c0392b" }}>
                {inrFull(verified.missed_fraud_value)}
              </span>
            </div>
            <div className="impact-cell">
              <span className="muted small">Legitimate holds (FP)</span>
              <span className="impact-num" style={{ fontSize: "0.85rem" }}>
                {verified.false_positive_legit_holds.toLocaleString("en-IN")}
              </span>
            </div>
            <div className="impact-cell">
              <span className="muted small">Net protected</span>
              <span className="impact-num">{inrCr(verified.net_protected_value)}</span>
            </div>
          </div>
          <p className="muted small">{verified.false_positive_note}</p>
          <p className="muted small">
            Caught {verified.frauds_caught}/{verified.frauds_total} frauds ·
            recall by amount {(verified.recall_by_amount * 100).toFixed(1)}% ·
            protecting {inrFull(verified.per_month_protected_inr)}/month
          </p>
          <p className="muted small" style={{ opacity: 0.8 }}>
            {verified.parity_note}. {verified.honesty}
          </p>
        </div>
      )}

      {synth && (
        <div style={{ marginTop: "12px" }}>
          <div className="impact-strip">
            <div className="impact-cell">
              <span className="muted small">Fraud prevented</span>
              <span className="impact-num">{inrFull(synth.pnl.fraud_prevented_value)}</span>
            </div>
            <div className="impact-cell">
              <span className="muted small">Missed fraud</span>
              <span className="impact-num" style={{ color: "#c0392b" }}>
                {inrFull(synth.pnl.missed_fraud_value)}
              </span>
            </div>
            <div className="impact-cell">
              <span className="muted small">False-positive cost</span>
              <span className="impact-num">{inrFull(synth.pnl.false_positive_cost)}</span>
            </div>
            <div className="impact-cell">
              <span className="muted small">Net</span>
              <span className="impact-num">{inrFull(synth.pnl.net_protected_value)}</span>
            </div>
          </div>
          <table className="audit-row" style={{ width: "100%", marginTop: "10px" }}>
            <thead>
              <tr className="audit-head">
                <th>Event</th>
                <th>Amount</th>
                <th>Model action</th>
                <th>Ground truth</th>
                <th>Classification</th>
              </tr>
            </thead>
            <tbody>
              {synth.pnl.rows.map((r) => (
                <tr key={r.transaction_id}>
                  <td>{r.transaction_id}</td>
                  <td>₹{r.amount_inr.toLocaleString("en-IN")}</td>
                  <td>
                    <span className="pill">{r.action}</span>
                  </td>
                  <td>{r.outcome}</td>
                  <td>{r.classification}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted small" style={{ opacity: 0.8 }}>{synth.honesty}</p>
        </div>
      )}
    </section>
  );
}
