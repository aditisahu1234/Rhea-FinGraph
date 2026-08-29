"use client";

// HealingPanel — Helix v2 self-healing memory. Shows what the system
// *remembers* (decision outcomes that turned out to be fraud) and what the
// healing cycle did about it: merchant hot-lists, threshold overrides, retrain
// queue. The "Run healing cycle" button executes one real heal() pass.

import { useCallback, useEffect, useState } from "react";
import {
  fetchHealingMemory,
  fetchHealingStatus,
  runHealingCycle,
  sendFeedback,
  type HealingMemory,
  type HealingStatus,
} from "../lib/api";

type FeedbackOutcome = "fraud" | "legit";

export default function HealingPanel() {
  const [status, setStatus] = useState<HealingStatus | null>(null);
  const [memory, setMemory] = useState<HealingMemory | null>(null);
  const [actions, setActions] = useState<string[]>([]);
  const [feedbackTx, setFeedbackTx] = useState("");
  const [feedbackOutcome, setFeedbackOutcome] = useState<FeedbackOutcome>("fraud");
  const [feedbackMsg, setFeedbackMsg] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const refresh = useCallback(async () => {
    try {
      const [s, m] = await Promise.all([fetchHealingStatus(), fetchHealingMemory()]);
      setStatus(s);
      setMemory(m);
      setErr("");
    } catch (e) {
      setErr(e instanceof Error ? e.message : "healing API unreachable");
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 8000);
    return () => clearInterval(id);
  }, [refresh]);

  async function heal() {
    setBusy(true);
    try {
      const report = (await runHealingCycle()) as Record<string, unknown>;
      setActions((report.actions as string[]) ?? []);
      await refresh();
    } catch (e) {
      setErr(e instanceof Error ? e.message : "heal failed");
    } finally {
      setBusy(false);
    }
  }

  async function submitFeedback() {
    if (!feedbackTx.trim()) return;
    setBusy(true);
    try {
      const res = (await sendFeedback(feedbackTx.trim(), feedbackOutcome)) as {
        ok: boolean;
        error?: string;
      };
      setFeedbackMsg(res.ok ? `recorded ${feedbackOutcome} for ${feedbackTx}` : `error: ${res.error ?? ""}`);
      setFeedbackTx("");
      await refresh();
    } catch (e) {
      setFeedbackMsg(e instanceof Error ? e.message : "feedback failed");
    } finally {
      setBusy(false);
    }
  }

  const s = status?.memory;
  const overrides = status?.threshold_overrides ?? {};

  return (
    <div className="panel">
      <div className="panel-head">
        <h2 className="panel-title">Self-healing · Helix v2</h2>
        <span className={`pill ${(s?.failures ?? 0) > 0 ? "alert" : "ok"}`}>
          {s ? `${s.failures} failures remembered` : "OFFLINE"}
        </span>
      </div>
      <p className="panel-sub">
        Failure memory records every confirmed outcome against an audited
        decision. A healing cycle turns that memory into actions: merchant
        hot-lists, threshold overrides, and retrain requests.
      </p>

      {err && <p className="stream-err">{err}</p>}

      {s && (
        <div className="stream-grid">
          <div className="stream-block">
            <span className="stream-k">episodes</span>
            <span className="stream-v">{s.episodes}</span>
          </div>
          <div className="stream-block">
            <span className="stream-k">missed fraud</span>
            <span className="stream-v stream-hot">{s.missed_fraud}</span>
          </div>
          <div className="stream-block">
            <span className="stream-k">false holds</span>
            <span className="stream-v">{s.false_hold}</span>
          </div>
          <div className="stream-block">
            <span className="stream-k">miss rate</span>
            <span className="stream-v">{s.miss_rate.toFixed(3)}</span>
          </div>
          <div className="stream-block">
            <span className="stream-k">hot merchants</span>
            <span className="stream-v">{s.hot_merchants}</span>
          </div>
          <div className="stream-block">
            <span className="stream-k">retrain queue</span>
            <span className="stream-v">{status?.retrain_queue_len ?? 0}</span>
          </div>
        </div>
      )}

      {memory && memory.hot_merchants.length > 0 && (
        <div className="culprits">
          <span className="culprit-label">hot merchants</span>
          {memory.hot_merchants.slice(0, 6).map((m) => {
            const mid = String(m.merchant_id);
            return (
              <span className="tag" key={mid} title={`${m.failures} failures`}>
                {mid}
              </span>
            );
          })}
        </div>
      )}

      {Object.keys(overrides).length > 0 && (
        <div className="culprits">
          <span className="culprit-label">thresholds</span>
          {Object.entries(overrides).map(([k, v]) => (
            <span className="tag" key={k}>
              {k}={typeof v === "number" ? v.toExponential(3) : String(v)}
            </span>
          ))}
        </div>
      )}

      {actions.length > 0 && (
        <ul className="heal-actions">
          {actions.map((a, i) => (
            <li key={i}>{a}</li>
          ))}
        </ul>
      )}

      <div className="stream-row">
        <button className="score-btn" onClick={heal} disabled={busy}>
          Run healing cycle
        </button>
        <input
          className="stream-input"
          placeholder="transaction_id (audited)"
          value={feedbackTx}
          onChange={(e) => setFeedbackTx(e.target.value)}
        />
        <select
          className="stream-input"
          value={feedbackOutcome}
          onChange={(e) => setFeedbackOutcome(e.target.value as FeedbackOutcome)}
        >
          <option value="fraud">fraud</option>
          <option value="legit">legit</option>
        </select>
        <button
          className="score-btn"
          onClick={submitFeedback}
          disabled={busy || !feedbackTx.trim()}
        >
          Record outcome
        </button>
      </div>
      {feedbackMsg && <p className="stream-sub">{feedbackMsg}</p>}
    </div>
  );
}