"use client";

// RazorpayDemoPanel — LIMITATION #2 fix.
// Wraps the existing FINGRAPH inference engine in a Razorpay-style lifecycle:
// create test order -> payment event -> velocity -> XGBoost -> SHAP ->
// ALLOW / REVIEW / HOLD -> webhook -> audit. Demonstrable live against the
// real API without rebuilding the model or using live Razorpay keys.

import { useState } from "react";
import {
  createDemoOrder,
  payDemoOrder,
  sendRazorpayWebhook,
  type DemoOrder,
  type DemoWebhook,
  type WebhookResponse,
} from "../lib/api";

const MERCHANTS = [
  { id: "TerraMart-5311", label: "TerraMart · MCC 5311" },
  { id: "FurniCasa-5712", label: "FurniCasa · MCC 5712" },
  { id: "GoGrocer-5411", label: "GoGrocer · MCC 5411" },
  { id: "AirWings-3722", label: "AirWings · MCC 3722" },
];

const ACTION_TONE: Record<string, string> = {
  allow: "demo-allow",
  review: "demo-review",
  hold: "demo-hold",
};

export default function RazorpayDemoPanel() {
  const [amount, setAmount] = useState("1999.00");
  const [merchant, setMerchant] = useState(MERCHANTS[0].id);
  const [order, setOrder] = useState<DemoOrder | null>(null);
  const [hook, setHook] = useState<DemoWebhook | null>(null);
  const [wh, setWh] = useState<WebhookResponse | null>(null);
  const [whAmount, setWhAmount] = useState("199900"); // paise
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  async function create() {
    setErr("");
    setBusy(true);
    try {
      const o = await createDemoOrder(amount, merchant);
      setOrder(o);
      setHook(null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "create order failed");
    } finally {
      setBusy(false);
    }
  }

  async function pay() {
    if (!order) return;
    setErr("");
    setBusy(true);
    try {
      const h = await payDemoOrder(order.order_id);
      setHook(h);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "pay failed");
    } finally {
      setBusy(false);
    }
  }

  async function fireWebhook() {
    setErr("");
    setBusy(true);
    try {
      const r = await sendRazorpayWebhook({
        order_id: `order_wh_${Date.now()}`,
        payment_id: `pay_wh_${Date.now()}`,
        amount: Number(whAmount) || 0,
        currency: "INR",
        customer: { id: "C-WH-9001" },
        card: { id: "K-WH-9001" },
        merchant: { id: merchant },
        method: "card",
      });
      setWh(r);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "webhook failed");
    } finally {
      setBusy(false);
    }
  }

  const ra = hook?.risk_assessment;

  return (
    <div className="panel">
      <div className="panel-head">
        <h2 className="panel-title">Razorpay payment · risk demo</h2>
        <span className="pill ok">TEST-MODE ADAPTER · NO LIVE KEYS</span>
      </div>
      <p className="panel-sub">
        A Razorpay-style flow through the existing engine: create a test
        order, then the payment event enters FINGRAPH → velocity → XGBoost →
        SHAP → ALLOW/REVIEW/HOLD → webhook → audit. Reuses the exact same
        scoring path as live traffic; nothing is mocked at the model layer.
      </p>

      <div className="demo-controls">
        <label>
          Amount (₹)
          <input
            type="text"
            value={amount}
            onChange={(e) => setAmount(e.target.value)}
          />
        </label>
        <label>
          Merchant
          <select value={merchant} onChange={(e) => setMerchant(e.target.value)}>
            {MERCHANTS.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </select>
        </label>
        <button onClick={create} disabled={busy}>
          {busy ? "…" : "Create test order"}
        </button>
        <button onClick={pay} disabled={busy || !order}>
          {busy ? "…" : "Score payment"}
        </button>
      </div>

      {err && <p className="empty">error: {err}</p>}

      {order && (
        <div className="demo-order">
          <span className="mono">order {order.order_id}</span>
          <span>· {order.amount_inr} INR</span>
          <span>
            {" "}
            · event {String(order.event.transaction_id).slice(0, 12)}
          </span>
        </div>
      )}

      {hook && ra && (
        <div className={`demo-verdict ${ACTION_TONE[ra.action] ?? ""}`}>
          <div className="demo-big">
            <b>{ra.action.toUpperCase()}</b>
            <span className="muted">
              {" "}
              · risk {ra.fraud_probability.toFixed(4)} · {ra.fraud_verdict}
            </span>
          </div>
          <div className="muted small">
            security action{" "}
            <b className="demo-sec">{ra.security_action ?? "—"}</b>
            {ra.is_cold_start ? (
              <span className="pill demo-cold">COLD-START RULE ROUTE</span>
            ) : null}
          </div>
          <div className="muted small">
            model {ra.model_version} · webhook {hook.event}
          </div>
          {ra.reasons_human && ra.reasons_human.length > 0 && (
            <ul className="demo-reasons">
              {ra.reasons_human.map((r, i) => (
                <li key={`h${i}`}>{r}</li>
              ))}
            </ul>
          )}
          <ul className="demo-reasons">
            {ra.top_reasons.map((r, i) => (
              <li key={i}>
                <span className="mono">{r.feature}</span> — {r.detail}
              </li>
            ))}
          </ul>
          <div className="muted small">
            audited: {hook.audit.transaction_id} · decision auditable{" "}
            {String(hook.audit.decision_auditable)}
          </div>
        </div>
      )}

      <div className="demo-webhook">
        <h3 className="subhead">…or simulate a payment webhook</h3>
        <div className="demo-controls">
          <label>
            Amount (paise)
            <input
              type="text"
              value={whAmount}
              onChange={(e) => setWhAmount(e.target.value)}
            />
          </label>
          <button onClick={fireWebhook} disabled={busy}>
            {busy ? "…" : "Send webhook"}
          </button>
        </div>
        {wh && (
          <div className={`demo-verdict ${ACTION_TONE[wh.risk.decision] ?? ""}`}>
            <div className="demo-big">
              <b>{wh.risk.decision.toUpperCase()}</b>
              <span className="muted">
                {" "}
                · risk {wh.risk.fraud_probability.toFixed(4)} ·{" "}
                {wh.risk.verdict}
              </span>
            </div>
            <div className="muted small">
              model {wh.risk.model_version} · webhook-to-merchant{" "}
              <b>{wh.webhook_to_merchant}</b> · security action{" "}
              <b className="demo-sec">{wh.risk.security_action}</b>
              {wh.risk.is_cold_start ? (
                <span className="pill demo-cold">COLD-START RULE ROUTE</span>
              ) : null}
            </div>
            {wh.risk.reasons_human && wh.risk.reasons_human.length > 0 && (
              <ul className="demo-reasons">
                {wh.risk.reasons_human.map((r, i) => (
                  <li key={`w${i}`}>{r}</li>
                ))}
              </ul>
            )}
            <div className="muted small">
              audited: {wh.audit.transaction_id} · decision auditable{" "}
              {String(wh.audit.decision_auditable)}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
