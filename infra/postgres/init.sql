CREATE TABLE IF NOT EXISTS audit_events (
    id UUID PRIMARY KEY,
    occurred_at TIMESTAMPTZ NOT NULL,
    event_type TEXT NOT NULL,
    transaction_id TEXT,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS audit_events_occurred_at_idx
    ON audit_events (occurred_at DESC);

CREATE INDEX IF NOT EXISTS audit_events_transaction_id_idx
    ON audit_events (transaction_id);

CREATE TABLE IF NOT EXISTS model_decisions (
    id UUID PRIMARY KEY,
    transaction_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('allow', 'review', 'hold')),
    fraud_probability DOUBLE PRECISION NOT NULL,
    reasons JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS model_decisions_transaction_id_idx
    ON model_decisions (transaction_id);

-- Layer 6: tamper-evident compliance audit ledger (hash chain).
CREATE TABLE IF NOT EXISTS audit_ledger (
    seq        BIGSERIAL PRIMARY KEY,
    id         UUID NOT NULL,
    event_type TEXT NOT NULL,
    payload    JSONB NOT NULL DEFAULT '{}'::jsonb,
    prev_hash  TEXT NOT NULL,
    hash       TEXT NOT NULL,
    audited_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS audit_ledger_seq_idx
    ON audit_ledger (seq DESC);
