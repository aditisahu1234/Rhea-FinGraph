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
}

export interface ModelRace {
  models: RaceModel[];
  serving_name: string;
  gate_report: RepairGateReport | null;
}

export function fetchModelRace(): Promise<ModelRace> {
  return getJSON<ModelRace>("/api/v1/model/race");
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`GET ${path} -> ${res.status}`);
  return (await res.json()) as T;
}

export function fetchModelStatus(): Promise<ModelStatus> {
  return getJSON<ModelStatus>("/api/v1/model/status");
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
