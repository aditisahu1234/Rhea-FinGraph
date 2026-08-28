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
