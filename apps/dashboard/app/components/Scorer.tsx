"use client";

// Scorer — a live transaction scoring form wired to the FastAPI /score
// endpoint. Lets a pitch audience type in a transaction and watch the model
// decide in real time, with SHAP reasons.

import { useState } from "react";
import {
  scoreTransaction,
  type RiskDecision,
  type ScorePayload,
} from "../lib/api";
import DecisionGauge from "./DecisionGauge";
import ReasonsList from "./ReasonsList";

const EMPTY: ScorePayload = {
  transaction_id: "",
  event_time: new Date().toISOString().slice(0, 16),
  customer_id: "",
  card_id: "",
  merchant_id: "",
  merchant_category_code: "5411",
  amount: "",
  payment_channel: "swipe",
};

export default function Scorer() {
  const [form, setForm] = useState<ScorePayload>(EMPTY);
  const [decision, setDecision] = useState<RiskDecision | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const set = (k: keyof ScorePayload) => (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => setForm((f) => ({ ...f, [k]: e.target.value }));

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const payload: ScorePayload = { ...form };
      payload.event_time = new Date(form.event_time).toISOString();
      const d = await scoreTransaction(payload);
      setDecision(d);
    } catch (err) {
      setError(err instanceof Error ? err.message : "scoring failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="scorer">
      <form className="score-form" onSubmit={onSubmit}>
        <div className="field-row">
          <label>
            Transaction ID
            <input value={form.transaction_id} onChange={set("transaction_id")}
              placeholder="tx_8f2a" required />
          </label>
          <label>
            Amount (USD)
            <input value={form.amount} onChange={set("amount")}
              placeholder="249.00" type="number" step="0.01" min="0" required />
          </label>
        </div>
        <div className="field-row">
          <label>
            Customer ID
            <input value={form.customer_id} onChange={set("customer_id")}
              placeholder="cust_9201" required />
          </label>
          <label>
            Card ID
            <input value={form.card_id} onChange={set("card_id")}
              placeholder="card_4412" required />
          </label>
        </div>
        <div className="field-row">
          <label>
            Merchant ID
            <input value={form.merchant_id} onChange={set("merchant_id")}
              placeholder="merch_9921" required />
          </label>
          <label>
            MCC
            <input value={form.merchant_category_code ?? ""}
              onChange={set("merchant_category_code")} placeholder="5411" />
          </label>
        </div>
        <div className="field-row">
          <label>
            Payment channel
            <select value={form.payment_channel ?? ""} onChange={set("payment_channel")}>
              <option value="swipe">Swipe (card present)</option>
              <option value="chip">Chip (card present)</option>
              <option value="online">Online (card not present)</option>
            </select>
          </label>
          <label>
            Event time
            <input value={form.event_time} onChange={set("event_time")} type="datetime-local" />
          </label>
        </div>
        <button className="score-btn" disabled={loading}>
          {loading ? "Scoring…" : "Score transaction"}
        </button>
        {error && <p className="error">{error}</p>}
      </form>

      {decision && (
        <div className="score-result">
          <DecisionGauge probability={decision.fraud_probability}
            action={decision.action} />
          <div className="result-body">
            <div className="result-head">
              <h3>
                Decision · <span className="muted">{decision.transaction_id}</span>
                {decision.is_cold_start ? (
                  <span className="pill demo-cold">COLD-START RULE ROUTE</span>
                ) : null}
              </h3>
              <span className="muted">
                {decision.model_version} ·{" "}
                {decision.processed_at ? new Date(decision.processed_at).toLocaleTimeString() : ""}
              </span>
            </div>
            <div className="result-sec">
              Security action:{" "}
              <b className="sec-ta">{decision.security_action ?? "REVIEW"}</b>
              {decision.reasons_human && decision.reasons_human.length > 0 ? (
                <ul className="demo-reasons">
                  {decision.reasons_human.map((r, i) => (
                    <li key={`h${i}`}>{r}</li>
                  ))}
                </ul>
              ) : null}
            </div>
            <ReasonsList reasons={decision.reasons} />
          </div>
        </div>
      )}
    </div>
  );
}
