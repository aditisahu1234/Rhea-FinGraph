// api.ts — client-side data access for the Rhea FinGraph dashboard.
// Reads API_BASE from a build-time env var, defaulting to localhost:8000.

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export interface RiskReason {
  feature: string;
  direction: "increases_risk" | "reduces_risk" | "context";
  detail: string;
  magnitude: number | null;
}

export interface RiskDecision {
  transaction_id: string;
  model_version: string;
  fraud_probability: number;
  action: "allow" | "review" | "hold";
  reasons: RiskReason[];
  is_model_ready: boolean;
  processed_at: string | null;
  // LIMITATION #3 — concrete payment-security action + readable summary.
  security_action?: "APPROVE" | "REQUEST_STEP_UP" | "DECLINE";
  reasons_human?: string[];
  // LIMITATION #4 — cold-start routing flag (conservative rule engine).
  is_cold_start?: boolean;
}

export interface FeatureDrift {
  feature: string;
  ref_mean: number | null;
  obs_mean: number | null;
  psi: number | null;
  z: number | null;
}

export interface HelixDriftReport {
  trigger: "YES" | "NO";
  score: number | null;
  n_features: number;
  n_culprits: number;
  culprits: string[];
  reasons: string[];
  features: FeatureDrift[];
}

export interface ModelStatus {
  ready: boolean;
  model_version: string;
  backend: string | null;
  trained_at: string | null;
  training_rows: number | null;
  thresholds: Record<string, number> | null;
  metrics_validation: Record<string, number> | null;
  metrics_test_locked: Record<string, number> | null;
}

// ---- Layer 2: graph store ------------------------------------------------

export interface Neo4jStatus {
  reachable: boolean;
  detail: string;
  url: string;
}

export interface GraphSnapshotRow {
  month_idx: number;
  n_edges: number | null;
  n_fraud: number | null;
}

export interface TopMerchantRow {
  merchant_id: string;
  txns: number;
  failures: number;
  missed_fraud: number;
  confirmed_fraud: number;
}

export interface GraphPipeline {
  source: string;
  n_customers: number | null;
  n_merchants: number | null;
  n_cards: number | null;
  n_snapshots: number | null;
  month_range: [number, number] | null;
  bucket_months: number | null;
  total_edges: number | null;
  total_fraud_edges: number | null;
  snapshots: GraphSnapshotRow[];
  top_merchants: TopMerchantRow[];
  top_merchants_source: string | null;
}

export interface GnnSummary {
  architecture: string | null;
  params: number | null;
  epochs: number | null;
  fit_seconds: number | null;
  device_used: string | null;
  best_val_auc: number | null;
  metrics_validation: Record<string, number> | null;
  metrics_test_locked: Record<string, number> | null;
}

export interface GraphStatus {
  neo4j: Neo4jStatus;
  pipeline: GraphPipeline;
  gnn: GnnSummary | null;
}

export function fetchGraphStatus(): Promise<GraphStatus> {
  return getJSON<GraphStatus>("/api/v1/graph/status");
}

// ---- Layer 4: model fight card -------------------------------------------

export interface RaceModel {
  name: string;
  label: string;
  backend: string | null;
  feature_set: string | null;
  training_rows: number | null;
  created_at: string | null;
  val_roc: number | null;
  test_roc: number | null;
  test_ap: number | null;
  test_action_counts: Record<string, number> | null;
  caught_frauds_by_action: Record<string, number> | null;
  role: "serving" | "promotion-candidate" | "candidate";
  // LIMITATION #6 — hero vs future-ensemble positioning.
  is_hero?: boolean;
  is_research?: boolean;
}

export interface ModelRacePositioning {
  hero_model: string;
  hero_note: string;
  hero_metrics: Record<string, number | null>;
  research_as_future_ensemble: string;
}

export interface ModelRace {
  models: RaceModel[];
  serving_name: string;
  gate_report: RepairGateReport | null;
  positioning?: ModelRacePositioning;
}

export function fetchModelRace(): Promise<ModelRace> {
  return getJSON<ModelRace>("/api/v1/model/race");
}

// ---- Concept-drift auto-switch --------------------------------------------

export interface SwitchAlert {
  detector: string;
  window: string | number;
  observed: number;
  baseline: number;
  message: string;
}

export interface SwitchDecision {
  triggered: boolean;
  reason: string;
  from_model: string | null;
  to_model: string | null;
  source: string;
  alerts: SwitchAlert[];
}

export interface DriftWindowRow {
  month: string;
  rows: number;
  mean_score: number;
  z_mean_score: number;
  psi: number;
  ewma_mean_score: number;
  cusum_stat: number;
  fraud_rate?: number;
}

export interface ModelSwitcherStatus {
  serving_model: string;
  last_decision: SwitchDecision | null;
  drift_report: {
    reference: Record<string, unknown>;
    detectors: Record<string, unknown>;
    windows: DriftWindowRow[];
    alerts: Record<string, string>;
  } | null;
}

export function fetchModelSwitcherStatus(): Promise<ModelSwitcherStatus> {
  return getJSON<ModelSwitcherStatus>("/api/v1/model/switcher/status");
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`GET ${path} -> ${res.status}`);
  return (await res.json()) as T;
}

export function fetchModelStatus(): Promise<ModelStatus> {
  return getJSON<ModelStatus>("/api/v1/model/status");
}

// ---- Payment-relevant business operating point (LIMITATION #1) --------

export interface BusinessImpact {
  available: boolean;
  model?: string;
  split?: string;
  parity?: {
    roc_auc_recomputed: number;
    ap_recomputed: number;
    matches_recorded_config: boolean;
  };
  assumptions?: Record<string, unknown>;
  totals?: { rows: number; frauds: number; fraud_amount_inr: number };
  actions?: Record<string, number>;
  caught_by_action?: Record<
    string,
    { count: number; amount_inr: number }
  >;
  protection?: {
    frauds_caught: number;
    recall_by_count: number;
    recall_by_amount: number;
    fraud_amount_caught_inr: number;
    fraud_amount_missed_inr: number;
    per_month_protected_inr: number;
    per_month_missed_inr: number;
  };
  top_mcc_by_fraud_amount?: { mcc: string; fraud_amount_inr: number }[];
  ato_evidence?: Record<string, Record<string, number | null>>;
}

export function fetchBusinessImpact(): Promise<BusinessImpact> {
  return getJSON<BusinessImpact>("/api/v1/business/impact");
}

// ---- Financial-impact headline cards (sprint Hour 2-3) ------------------

export interface ImpactSummary {
  available: boolean;
  total_protected_inr: number | null;
  monthly_protected_inr: number | null;
  fraud_amount_blocked_rate: number | null;
  fraud_events_blocked_rate: number | null;
  total_fraud_inr?: number | null;
  missed_inr?: number | null;
  model?: string;
  split?: string;
}

export function fetchImpactSummary(): Promise<ImpactSummary> {
  return getJSON<ImpactSummary>("/api/v1/impact/summary");
}

// ---- Payment demo adapter (LIMITATION #2) -----------------------------

export interface DemoOrder {
  order_id: string;
  amount_inr: string;
  currency: string;
  status: string;
  event: Record<string, unknown>;
}

export interface DemoWebhook {
  event: string;
  order: { order_id: string; amount_inr: string; currency: string };
  risk_assessment: {
    model_version: string;
    fraud_probability: number;
    action: "allow" | "review" | "hold";
    security_action?: "APPROVE" | "REQUEST_STEP_UP" | "DECLINE";
    fraud_verdict: string;
    is_cold_start?: boolean;
    reasons_human?: string[];
    top_reasons: {
      feature: string;
      direction: string;
      detail: string;
      magnitude: number | null;
    }[];
  };
  audit: {
    transaction_id: string;
    decision_auditable: boolean;
    processed_at: string | null;
  };
}

export async function createDemoOrder(
  amountInr: string,
  merchantId: string
): Promise<DemoOrder> {
  const res = await fetch(`${API_BASE}/api/v1/payment/order`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ amount_inr: amountInr, merchant_id: merchantId }),
  });
  if (!res.ok) throw new Error(`createOrder -> ${res.status}`);
  return (await res.json()) as DemoOrder;
}

export async function payDemoOrder(orderId: string): Promise<DemoWebhook> {
  const res = await fetch(`${API_BASE}/api/v1/payment/pay`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ order_id: orderId }),
  });
  if (!res.ok) throw new Error(`pay -> ${res.status}`);
  return (await res.json()) as DemoWebhook;
}

export function fetchPaymentFlow(): Promise<{ flow: string[]; endpoints: Record<string, string> }> {
  return getJSON("/api/v1/payment/flow");
}

// ---- Simulate a Payment webhook (sprint Hour 0-1) ----------------------

export interface WebhookRisk {
  model_version: string;
  fraud_probability: number;
  decision: "allow" | "review" | "hold";
  security_action: "APPROVE" | "REQUEST_STEP_UP" | "DECLINE";
  is_cold_start: boolean;
  reasons_human: string[];
  verdict: string;
}

export interface WebhookResponse {
  received: boolean;
  order_id: string | null;
  payment_id: string | null;
  risk: WebhookRisk;
  webhook_to_merchant: string;
  audit: {
    transaction_id: string;
    decision_auditable: boolean;
    processed_at: string | null;
  };
}

export async function sendPaymentWebhook(
  payload: Record<string, unknown>
): Promise<WebhookResponse> {
  const res = await fetch(`${API_BASE}/api/v1/payment/webhook`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(`webhook -> ${res.status}: ${JSON.stringify(err)}`);
  }
  return (await res.json()) as WebhookResponse;
}

// ---- Payment synthetic event (LIMITATION #5: contract != training set) --

export interface PaymentEventResponse {
  received: boolean;
  decision: {
    fraud_probability: number;
    action: "allow" | "review" | "hold";
    security_action: "APPROVE" | "REQUEST_STEP_UP" | "DECLINE";
    verdict: string;
    is_cold_start: boolean;
    reasons_human: string[];
    model_version: string;
  };
  mapping: {
    payment_id: string;
    canonical_channel: string;
    model_feature_used: boolean;
    model_features: string[];
    future_signals_not_model_inputs: string[];
    amount_inr: number;
    currency: string;
  };
  future_signals_not_model_inputs: string[];
  audit: {
    transaction_id: string;
    decision_auditable: boolean;
    processed_at: string | null;
  };
}

export async function sendPaymentEvent(
  payload: Record<string, unknown>
): Promise<PaymentEventResponse> {
  const res = await fetch(`${API_BASE}/api/v1/payment/event`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(`event -> ${res.status}: ${JSON.stringify(err)}`);
  }
  return (await res.json()) as PaymentEventResponse;
}

export function fetchHelixDrift(): Promise<HelixDriftReport> {
  return getJSON<HelixDriftReport>("/api/v1/helix/drift");
}

export interface ScorePayload {
  transaction_id: string;
  event_time: string;
  customer_id: string;
  card_id: string;
  merchant_id: string;
  merchant_category_code?: string | null;
  amount: string;
  payment_channel?: string | null;
  device_id?: string | null;
  ip_hash?: string | null;
  merchant_city?: string | null;
  merchant_state?: string | null;
  merchant_country?: string | null;
}

export async function scoreTransaction(
  payload: ScorePayload
): Promise<RiskDecision> {
  const res = await fetch(`${API_BASE}/api/v1/transactions/score`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`score -> ${res.status}`);
  return (await res.json()) as RiskDecision;
}

// ---- Layer 6: compliance audit + observability --------------------------

export interface AuditRecord {
  id: string;
  event_type: string;
  payload: Record<string, unknown>;
  prev_hash: string;
  hash: string;
  audited_at: number | null;
  seq: number | null;
}

export interface AuditHealth {
  healthy: boolean;
  backend: string;
  buffered: number;
  total: number;
}

export interface AuditSummary {
  total: number;
  backend: string;
  buffered: number;
  valid: boolean;
  verified_records: number;
  store_healthy: boolean;
}

export interface AuditVerify {
  valid: boolean;
  records: number;
  first_broken_index: number | null;
  backend: string;
  store_healthy: boolean;
  buffered: number;
}

export function fetchAuditHealth(): Promise<AuditHealth> {
  return getJSON<AuditHealth>("/api/v1/audit/health");
}

export function fetchAuditRecent(limit = 10): Promise<AuditRecord[]> {
  return getJSON<AuditRecord[]>(`/api/v1/audit/recent?limit=${limit}`);
}

export function fetchAuditSummary(): Promise<AuditSummary> {
  return getJSON<AuditSummary>("/api/v1/audit/summary");
}

export function fetchAuditVerify(): Promise<AuditVerify> {
  return getJSON<AuditVerify>("/api/v1/audit/verify");
}

// ---- Layer 1: streaming velocity store --------------------------------

export interface StreamingHealth {
  layer: string;
  read_contract: string;
  healthy: boolean;
  backend: string;
  observations: number;
  total_flowed_keys: number | null;
  entries?: Record<string, number>;
}

export interface StreamingWindow {
  count: number;
  amount: number;
}

export interface StreamingSnapshot {
  entity: string;
  id: string;
  windows: Record<string, StreamingWindow>;
  priors: Record<string, number>;
}

export function fetchStreamingHealth(): Promise<StreamingHealth> {
  return getJSON<StreamingHealth>("/api/v1/streaming/health");
}

export function fetchStreamingSnapshot(
  entity: string,
  entityId: string
): Promise<StreamingSnapshot> {
  return getJSON<StreamingSnapshot>(
    `/api/v1/streaming/snapshot?entity=${encodeURIComponent(entity)}&entity_id=${encodeURIComponent(entityId)}`
  );
}

// ---- Layer 5 v2: Helix self-healing memory -----------------------------

export interface HealingMemoryStats {
  episodes: number;
  failures: number;
  missed_fraud: number;
  false_hold: number;
  miss_rate: number;
  false_hold_rate: number;
  hot_merchants: number;
  durable: boolean;
  durable_file: string;
  recent: Record<string, unknown>[];
}

export interface HealingMemory {
  stats: HealingMemoryStats;
  merchant_rollup: Record<string, Record<string, unknown>>;
  hot_merchants: Record<string, unknown>[];
}

export interface RepairGateReport {
  verdict: string;
  serving?: { roc_auc?: number; top5k_caught?: number };
  repair?: { roc_auc?: number; top5k_caught?: number };
  slice?: { rows?: number; frauds?: number };
}

export interface HealingStatus {
  memory: HealingMemoryStats;
  drift: { trigger: string; score: number | null } | null;
  threshold_overrides: Record<string, number>;
  retrain_queue_len: number;
  last_retrain_request: Record<string, unknown> | null;
  hot_merchants: Record<string, unknown>[];
  heal_report_exists: boolean;
  gate_report: RepairGateReport | null;
}

export function fetchHealingMemory(): Promise<HealingMemory> {
  return getJSON<HealingMemory>("/api/v1/healing/memory");
}

export function fetchHealingStatus(): Promise<HealingStatus> {
  return getJSON<HealingStatus>("/api/v1/healing/status");
}

export async function runHealingCycle(): Promise<Record<string, unknown>> {
  const res = await fetch(`${API_BASE}/api/v1/healing/heal`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(`heal -> ${res.status}`);
  return (await res.json()) as Record<string, unknown>;
}

export async function sendFeedback(
  transactionId: string,
  outcome: "fraud" | "legit"
): Promise<Record<string, unknown>> {
  const res = await fetch(`${API_BASE}/api/v1/healing/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ transaction_id: transactionId, outcome }),
  });
  if (!res.ok) throw new Error(`feedback -> ${res.status}`);
  return (await res.json()) as Record<string, unknown>;
}

// ---- LIMITATION #7: attack-scenario simulator -----------------------------
export interface AttackScenarioMeta {
  key: string;
  title: string;
  description: string;
  n_events: number;
  channel: string;
}

export interface AttackScenarios {
  source: string;
  honesty: string;
  scenarios: AttackScenarioMeta[];
}

export interface AttackStep {
  index: number;
  amount_inr: number;
  risk: number;
  raw_margin: number;
  action: string;
  model_version: string;
  is_cold_start: boolean;
}

export interface AttackSimulation {
  scenario: string;
  title: string;
  description: string;
  n_events: number;
  risk_before: number | null;
  risk_after: number | null;
  delta_risk: number | null;
  raw_margin_before: number | null;
  raw_margin_after: number | null;
  delta_raw_margin: number | null;
  calibration_note: string;
  model_used: string | null;
  honesty: string;
  steps: AttackStep[];
}

export function fetchAttackScenarios(): Promise<AttackScenarios> {
  return getJSON<AttackScenarios>("/api/v1/attack/scenarios");
}

export async function runAttackSimulation(
  scenario: string
): Promise<AttackSimulation> {
  const res = await fetch(`${API_BASE}/api/v1/attack/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scenario }),
  });
  if (!res.ok) throw new Error(`attack/simulate -> ${res.status}`);
  return (await res.json()) as AttackSimulation;
}

// ---- LIMITATION #10: outcome / chargeback simulator -----------------------
export interface OutcomePnlRow {
  transaction_id: string;
  action: string;
  outcome: string;
  amount_inr: number;
  classification: string;
  protected_value: number;
  missed_value: number;
  false_positive_cost: number;
}

export interface OutcomePnl {
  n: number;
  fraud_prevented_value: number;
  missed_fraud_value: number;
  false_positive_cost: number;
  net_protected_value: number;
  prevented_count: number;
  missed_count: number;
  false_positive_count: number;
  by_class: Record<string, number>;
  rows: OutcomePnlRow[];
}

export interface VerifiedOutcome {
  mode: "verified";
  model: string;
  split: string;
  as_of: string;
  parity_note: string;
  fraud_prevented_value: number;
  missed_fraud_value: number;
  false_positive_legit_holds: number;
  false_positive_note: string;
  frauds_total: number;
  frauds_caught: number;
  recall_by_count: number;
  recall_by_amount: number;
  per_month_protected_inr: number;
  net_protected_value: number;
  net_protected_note: string;
  honesty: string;
}

export interface SyntheticOutcome {
  mode: "synthetic";
  scenario: string;
  title: string;
  description: string;
  fraud_from: number;
  n_events: number;
  model_used: string;
  pnl: OutcomePnl;
  honesty: string;
}

export type AttackOutcome = VerifiedOutcome | SyntheticOutcome;

export async function runAttackOutcome(payload: Record<string, unknown>): Promise<AttackOutcome> {
  const res = await fetch(`${API_BASE}/api/v1/attack/outcome`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`attack/outcome -> ${res.status}`);
  return (await res.json()) as AttackOutcome;
}

// ---- Graph visualization sample (LIMITATION Layer 2) ----------------------
export interface GraphSampleNode {
  id: string;
  type: "customer" | "merchant" | "card";
  label: string;
  fraud?: boolean;
}

export interface GraphSampleEdge {
  source: string;
  target: string;
  kind: "purchased" | "has_card";
}

export interface GraphSample {
  source_snapshot: string;
  n_nodes: number;
  n_edges: number;
  node_types: string[];
  n_fraud_marked: number;
  nodes: GraphSampleNode[];
  edges: GraphSampleEdge[];
  note: string;
}

export async function fetchGraphSample(
  maxNodes?: number
): Promise<GraphSample> {
  const q = maxNodes ? `?max_nodes=${maxNodes}` : "";
  return getJSON<GraphSample>(`/api/v1/graph/sample${q}`);
}

// ---- Live Neo4j Cypher gateway -------------------------------------------
export interface CypherNode {
  id: string;
  type: string;
  label: string;
  fraud?: boolean;
}
export interface CypherEdge {
  source: string;
  target: string;
  kind: string;
  is_fraud?: boolean;
}
export interface CypherResult {
  online: boolean;
  query: string;
  label: string;
  source: string;
  n_nodes: number;
  n_edges: number;
  nodes: CypherNode[];
  edges: CypherEdge[];
  cypher?: string;
  detail?: string;
  hint?: string;
}
export async function runGraphCypher(
  query: string,
  limit?: number
): Promise<CypherResult> {
  const res = await fetch(`${API_BASE}/api/v1/graph/cypher`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, limit: limit ?? 100 }),
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`${res.status}: ${t}`);
  }
  return res.json();
}

// ---- Helix Runtime (PCEC + Gene Map) --------------------------------------
export interface HelixGene {
  error_signature: string;
  repair_strategy: Record<string, unknown>;
  success_count: number;
  failure_count: number;
  total_uses: number;
  q_value: number;
  success_rate: number | null;
  last_used: string;
}
export interface HelixRepair {
  error_signature: string;
  error_type: string;
  strategy: Record<string, unknown>;
  success: boolean;
  gene_hit: boolean;
  timestamp: number;
}
export interface HelixStatus {
  status: string;
  mode: string;
  gene_count: number;
  repair_attempts: number;
  recovery_rate: number | null;
  gene_hit_rate: number | null;
  recent_repairs: HelixRepair[];
}
export interface HelixGenes {
  genes: HelixGene[];
  count: number;
}
export async function fetchHelixStatus(): Promise<HelixStatus> {
  return getJSON<HelixStatus>("/api/v1/helix/status");
}
export async function fetchHelixGenes(): Promise<HelixGenes> {
  return getJSON<HelixGenes>("/api/v1/helix/genes");
}
export async function runHelixDemoAttack(): Promise<Record<string, unknown>> {
  const res = await fetch(`${API_BASE}/api/v1/helix/demo-error`, { method: "POST" });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`${res.status}: ${t}`);
  }
  return res.json();
}
export async function resetHelixGenes(): Promise<Record<string, unknown>> {
  const res = await fetch(`${API_BASE}/api/v1/helix/reset`, { method: "POST" });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`${res.status}: ${t}`);
  }
  return res.json();
}
