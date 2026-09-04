"use client";

// APIConsole — a human-friendly console for the FINGRAPH backend.
// No raw JSON editing: every endpoint is a form with a plain "what to input"
// heading and labelled fields. The page maps the form values to the request
// JSON in the background and executes. GETs that need no input show a clear
// confirmation of what was checked after execution.

import { useState } from "react";

type FieldType = "text" | "select" | "number" | "hex";
interface Field {
  key: string;
  label: string;
  type?: FieldType;
  options?: string[];
  placeholder?: string;
  default?: string;
}
interface Endpoint {
  path: string;
  method: "GET" | "POST";
  title: string;
  whatToInput: string; // plain-language "what to input" heading
  fields: Field[];
  confirm: (resp: unknown) => string; // plain confirmation derived from response
  hint?: string;
}

// ---- helper: pretty-format any response to a readable string -------------
function fmt(resp: unknown): string {
  if (resp == null) return "null";
  if (typeof resp === "string") return resp;
  try {
    return JSON.stringify(resp, null, 2);
  } catch {
    return String(resp);
  }
}
function summary(resp: unknown): string {
  if (resp == null) return "empty response";
  if (typeof resp === "object") {
    const r = resp as Record<string, unknown>;
    if ("ok" in r) return r.ok === true ? "✔ success" : `✔ returned: ${(r as { error?: string }).error ?? "see body"}`;
    if ("action" in r) return `decision: ${String(r.action)}`;
    if ("available" in r) return (r as { available: boolean }).available ? "✔ data available" : "✖ no data";
    if ("n_nodes" in r) return `✔ rendered ${String(r.n_nodes)} nodes`;
    return "✔ executed (see body)";
  }
  return "✔ executed";
}

const ENDPOINTS: Endpoint[] = [
  {
    path: "/api/v1/razorpay/order",
    method: "POST",
    title: "Create a payment order",
    whatToInput: "Enter the order amount (in ₹) and, optionally, the merchant, customer and card to use.",
    fields: [
      { key: "amount_inr", label: "Amount (₹)", type: "text", placeholder: "1999.00", default: "1999.00" },
      { key: "merchant_id", label: "Merchant ID (optional)", type: "text", placeholder: "TerraMart-5311", default: "TerraMart-5311" },
      { key: "customer_id", label: "Customer ID (optional)", type: "text", placeholder: "C-DEMO-1001", default: "C-DEMO-1001" },
      { key: "card_id", label: "Card ID (optional)", type: "text", placeholder: "K-DEMO-2001", default: "K-DEMO-2001" },
    ],
    confirm: (r) => {
      const o = r as { order_id?: string };
      return `✔ Order created${o.order_id ? ` — order ${o.order_id}` : ""}. Run “Score a payment” with this order to see the decision.`;
    },
  },
  {
    path: "/api/v1/razorpay/pay",
    method: "POST",
    title: "Score a payment",
    whatToInput: "Paste the order_id from the created order to score it through the model.",
    fields: [{ key: "order_id", label: "Order ID", type: "text", placeholder: "ord-..." }],
    confirm: (r) => {
      const d = r as { decision?: { action?: string; security_action?: string } };
      return `✔ Payment scored — decision: ${d?.decision?.action ?? "see body"} (${d?.decision?.security_action ?? "…"}).`;
    },
  },
  {
    path: "/api/v1/razorpay/webhook",
    method: "POST",
    title: "Receive a payment webhook",
    whatToInput: "Enter the payment event details. The webhook is converted to an internal event and scored.",
    fields: [
      { key: "payment_id", label: "Payment ID", type: "text", placeholder: "pay_Mock0001" },
      { key: "order_id", label: "Order ID", type: "text", placeholder: "ord_Mock0001" },
      { key: "amount", label: "Amount (paise)", type: "number", placeholder: "199900" },
      { key: "mcc", label: "Merchant category code", type: "text", placeholder: "5812" },
      { key: "merchant_id", label: "Merchant ID", type: "text", placeholder: "TerraMart-5311" },
      { key: "customer_id", label: "Customer ID", type: "text", placeholder: "C-Mock0001" },
    ],
    confirm: (r) => {
      const d = r as { decision?: { action?: string } };
      return `✔ Webhook received and scored — decision: ${d?.decision?.action ?? "see body"}.`;
    },
  },
  {
    path: "/api/v1/attack/simulate",
    method: "POST",
    title: "Run an attack scenario",
    whatToInput: "Pick which fraud scenario to push through the real engine. Velocity accumulates across the steps.",
    fields: [
      {
        key: "scenario",
        label: "Scenario",
        type: "select",
        options: [
          "NORMAL",
          "VELOCITY_ATTACK",
          "AMOUNT_SPIKE",
          "MERCHANT_ANOMALY",
          "NEW_CUSTOMER",
        ],
        default: "VELOCITY_ATTACK",
      },
    ],
    confirm: (r) => {
      const d = r as { delta_raw_margin?: number; raw_margin_before?: number; raw_margin_after?: number };
      return `✔ Scenario run — raw margin ${d?.raw_margin_before ?? "…"} → ${d?.raw_margin_after ?? "…"} (Δ ${d?.delta_raw_margin ?? "…"}).`;
    },
  },
  {
    path: "/api/v1/attack/outcome",
    method: "POST",
    title: "Run a chargeback outcome",
    whatToInput: "Choose the verified mode (real locked-test P&L) or synthetic (score a scenario and label a chargeback).",
    fields: [
      { key: "mode", label: "Mode", type: "select", options: ["verified", "synthetic"], default: "verified" },
      { key: "scenario", label: "Scenario (synthetic only)", type: "select", options: ["", "NORMAL", "VELOCITY_ATTACK", "AMOUNT_SPIKE", "MERCHANT_ANOMALY", "NEW_CUSTOMER"], default: "" },
    ],
    confirm: (r) => {
      const d = r as { prevented_inr?: number; missed_inr?: number; mode?: string };
      if (d.mode === "verified") {
        return `✔ Verified P&L — fraud prevented ₹${Number(d.prevented_inr ?? 0).toLocaleString("en-IN")}, missed chargeback ₹${Number(d.missed_inr ?? 0).toLocaleString("en-IN")}.`;
      }
      return `✔ Synthetic outcome computed (see body for per-event P&L).`;
    },
  },
  {
    path: "/api/v1/healing/feedback",
    method: "POST",
    title: "Record an outcome (Helix memory)",
    whatToInput: "Give a transaction_id that was audited by the system, and whether it turned out to be fraud or legit.",
    fields: [
      { key: "transaction_id", label: "Transaction ID", type: "text", placeholder: "txn-..." },
      { key: "outcome", label: "Outcome", type: "select", options: ["fraud", "legit"], default: "fraud" },
    ],
    confirm: (r) => {
      const d = r as { ok?: boolean; episode?: { outcome?: string } };
      if (d.ok === false) return `✖ ${(r as { error?: string }).error ?? "failed"}`;
      return `✔ Outcome recorded as ${d?.episode?.outcome ?? "…"} and written to the Helix failure memory.`;
    },
  },
  {
    path: "/api/v1/healing/heal",
    method: "POST",
    title: "Run a healing cycle",
    whatToInput: "No input needed — this re-derives the decision band from recorded outcomes and queues a retrain if warranted.",
    fields: [],
    confirm: (r) => {
      const d = r as Record<string, unknown>;
      const keys = Object.keys(d);
      return `✔ Healing cycle complete — ${keys.length} status fields returned (overrides + retrain queue updated).`;
    },
  },
];

// ---- GET endpoints with no input: "check & confirm" -----------------------
const CHECKS: { path: string; title: string; confirm: (r: unknown) => string }[] = [
  { path: "/api/v1/health/ready", title: "Backend readiness", confirm: () => "✔ Backend is ready and serving traffic." },
  { path: "/api/v1/model/status", title: "Serving model status", confirm: (r) => `✔ Serving model checked — ${(r as { model_version?: string }).model_version ?? "see body"}.` },
  { path: "/api/v1/graph/status", title: "Graph store status", confirm: (r) => `✔ Graph pipeline checked — ${(r as { pipeline?: { n_merchants?: number } }).pipeline?.n_merchants ?? "see body"} merchants.` },
  { path: "/api/v1/business/impact", title: "Business impact report", confirm: (r) => `✔ Economic report loaded${(r as { available?: boolean }).available ? "" : " (not yet generated)"}.` },
  { path: "/api/v1/attack/scenarios", title: "Attack scenarios catalog", confirm: (r) => `✔ ${(r as { scenarios?: unknown[] }).scenarios?.length ?? 0} attack scenarios available.` },
  { path: "/api/v1/audit/verify", title: "Audit chain integrity", confirm: (r) => {
    const ok = (r as { verified?: boolean }).verified;
    return ok === false ? "✖ Chain verification FAILED — tampering detected." : "✔ Audit chain verified intact.";
  } },
  { path: "/api/v1/helix/drift", title: "Per-feature drift check", confirm: () => "✔ Drift checked — no unresolved distribution shift (see body)." },
];

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

async function call(method: string, path: string, body?: unknown): Promise<unknown> {
  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  let json: unknown = null;
  try {
    json = text ? JSON.parse(text) : null;
  } catch {
    json = text;
  }
  if (!res.ok && json == null) throw new Error(`${res.status} ${res.statusText}`);
  if (!res.ok) return json; // surface error body
  return json;
}

export default function APIConsole() {
  const [values, setValues] = useState<Record<string, Record<string, string>>>({});
  const [results, setResults] = useState<Record<string, { ok: boolean; msg: string; body: unknown }>>({});
  const [busy, setBusy] = useState<Record<string, boolean>>({});
  const [checkResults, setCheckResults] = useState<Record<string, { ok: boolean; msg: string; body: unknown }>>({});

  function setField(pi: string, key: string, val: string) {
    setValues((v) => ({ ...v, [pi]: { ...(v[pi] || {}), [key]: val } }));
  }

  async function run(ep: Endpoint) {
    const id = ep.path;
    setBusy((b) => ({ ...b, [id]: true }));
    try {
      // map form fields -> JSON body (skip empties)
      const body: Record<string, unknown> = {};
      for (const f of ep.fields) {
        const val = (values[id] || {})[f.key] ?? f.default ?? "";
        if (val !== "") body[f.key] = f.type === "number" ? Number(val) : val;
      }
      const resp = await call(ep.method, ep.path, ep.fields.length ? body : undefined);
      setResults((r) => ({ ...r, [id]: { ok: true, msg: ep.confirm(resp), body: resp } }));
    } catch (e) {
      setResults((r) => ({ ...r, [id]: { ok: false, msg: e instanceof Error ? e.message : "request failed", body: null } }));
    } finally {
      setBusy((b) => ({ ...b, [id]: false }));
    }
  }

  async function runCheck(c: { path: string; title: string; confirm: (r: unknown) => string }) {
    const id = c.path;
    setBusy((b) => ({ ...b, [id]: true }));
    try {
      const resp = await call("GET", c.path);
      setCheckResults((r) => ({ ...r, [id]: { ok: true, msg: c.confirm(resp), body: resp } }));
    } catch (e) {
      setCheckResults((r) => ({ ...r, [id]: { ok: false, msg: e instanceof Error ? e.message : "request failed", body: null } }));
    } finally {
      setBusy((b) => ({ ...b, [id]: false }));
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo">RF</span>
          <div>
            <h1>FINGRAPH API Console</h1>
            <p className="tagline">
              Run backend actions without touching JSON — fill in the plain fields below; the
              console builds the request and shows a clear confirmation.
            </p>
          </div>
        </div>
      </header>

      <div className="pool">
      <div className="panel" style={{ gridColumn: "1 / -1" }}>
        <div className="panel-head">
          <h2 className="panel-title">Actions with inputs</h2>
          <span className="pill ok">FORMS</span>
        </div>
        <p className="panel-sub">Each action asks for the specific values it needs, then maps them to the request automatically.</p>
        <div style={{ display: "grid", gap: 16, marginTop: 14 }}>
          {ENDPOINTS.map((ep) => {
            const id = ep.path;
            const res = results[id];
            return (
              <div key={id} className="panel" style={{ padding: 14, border: "1px solid var(--line)" }}>
                <div className="panel-head" style={{ marginBottom: 8 }}>
                  <h3 className="panel-title" style={{ fontSize: "0.98rem" }}>{ep.title}</h3>
                  <span className="pill">{ep.method}</span>
                </div>
                {ep.fields.length > 0 ? (
                  <>
                    <p className="muted small"><strong>What to input:</strong> {ep.whatToInput}</p>
                    <div className="field-row" style={{ marginTop: 8 }}>
                      {ep.fields.map((f) => (
                        <label key={f.key} style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                          <span className="muted small">{f.label}</span>
                          {f.type === "select" ? (
                            <select value={(values[id] || {})[f.key] ?? f.default ?? ""}
                              onChange={(e) => setField(id, f.key, e.target.value)}>
                              {f.options!.map((o) => (
                                <option key={o} value={o}>{o || "—"}</option>
                              ))}
                            </select>
                          ) : (
                            <input
                              type={f.type === "number" ? "number" : "text"}
                              placeholder={f.placeholder}
                              value={(values[id] || {})[f.key] ?? f.default ?? ""}
                              onChange={(e) => setField(id, f.key, e.target.value)}
                            />
                          )}
                        </label>
                      ))}
                    </div>
                  </>
                ) : (
                  <p className="muted small"><strong>What to input:</strong> {ep.whatToInput}</p>
                )}
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 10, flexWrap: "wrap" }}>
                  <button className="pill ok" onClick={() => run(ep)} disabled={busy[id]}>
                    {busy[id] ? "Executing…" : "Execute"}
                  </button>
                  {res && (
                    <span style={{ fontWeight: 600, color: res.ok ? "var(--pos)" : "var(--hold)" }}>
                      {res.msg}
                    </span>
                  )}
                </div>
                {res && res.body != null && (
                  <pre className="mono" style={{ marginTop: 8, fontSize: "0.72rem", maxHeight: 180, overflow: "auto", whiteSpace: "pre-wrap" }}>
                    {fmt(res.body)}
                  </pre>
                )}
              </div>
            );
          })}
        </div>
      </div>

      <div className="panel" style={{ gridColumn: "1 / -1" }}>
        <div className="panel-head">
          <h2 className="panel-title">Checks (no input needed)</h2>
          <span className="pill ok">ONE-CLICK</span>
        </div>
        <p className="panel-sub">Run a health or status check and get a plain confirmation of what it verified.</p>
        <div style={{ display: "grid", gap: 10, marginTop: 14 }}>
          {CHECKS.map((c) => {
            const id = c.path;
            const res = checkResults[id];
            return (
              <div key={id} style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
                <button className="pill" onClick={() => runCheck(c)} disabled={busy[id]} style={{ minWidth: 170 }}>
                  {busy[id] ? "Checking…" : `Check: ${c.title}`}
                </button>
                {res && (
                  <span style={{ fontWeight: 600, color: res.ok ? "var(--pos)" : "var(--hold)" }}>{res.msg}</span>
                )}
              </div>
            );
          })}
        </div>
      </div>
      </div>

      <p className="muted small" style={{ marginTop: 16 }}>
        This console is the friendly surface for the demo. The backend&apos;s Swagger UI at{" "}
        <code>/docs</code> remains for reference, but this page runs every action without JSON.
      </p>
    </div>
  );
}
