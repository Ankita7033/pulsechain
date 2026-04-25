-- ============================================================
-- PulseChain TimescaleDB Schema
-- Run automatically on container first boot via docker-entrypoint-initdb.d
-- ============================================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ============================================================
-- 1. signal_events  — raw ingested + enriched events
-- ============================================================
CREATE TABLE IF NOT EXISTS signal_events (
    id              TEXT PRIMARY KEY,
    signal_type     TEXT NOT NULL,
    signal_value    DOUBLE PRECISION NOT NULL,
    source          TEXT NOT NULL,
    metadata        JSONB DEFAULT '{}',
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Convert to hypertable for time-series queries
SELECT create_hypertable('signal_events', 'ingested_at',
    if_not_exists => TRUE,
    migrate_data  => TRUE
);

CREATE INDEX IF NOT EXISTS idx_signal_events_source  ON signal_events (source, ingested_at DESC);
CREATE INDEX IF NOT EXISTS idx_signal_events_meta    ON signal_events USING GIN (metadata);

-- ============================================================
-- 2. signal_baselines  — rolling stats for z-score computation
-- ============================================================
CREATE TABLE IF NOT EXISTS signal_baselines (
    id          SERIAL PRIMARY KEY,
    source      TEXT NOT NULL,
    region      TEXT NOT NULL,
    mean_value  DOUBLE PRECISION,
    std_dev     DOUBLE PRECISION,
    sample_count INTEGER DEFAULT 0,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (source, region)
);

-- Seed with default baselines so z-score node has data from day 1
INSERT INTO signal_baselines (source, region, mean_value, std_dev, sample_count) VALUES
    ('wastewater',    'northeast', 45000,  12000,  30),
    ('wastewater',    'southeast', 38000,  9500,   30),
    ('wastewater',    'midwest',   42000,  11000,  30),
    ('wastewater',    'west',      35000,  8000,   30),
    ('pharmacy',      'northeast', 52.0,   15.0,   30),
    ('pharmacy',      'southeast', 48.0,   13.0,   30),
    ('pharmacy',      'midwest',   50.0,   14.0,   30),
    ('pharmacy',      'west',      46.0,   12.0,   30),
    ('search_trends', 'northeast', 50.0,   12.0,   30),
    ('search_trends', 'southeast', 47.0,   11.0,   30),
    ('search_trends', 'midwest',   49.0,   13.0,   30),
    ('search_trends', 'west',      45.0,   10.0,   30),
    ('absenteeism',   'northeast', 0.08,   0.03,   30),
    ('absenteeism',   'southeast', 0.07,   0.025,  30),
    ('absenteeism',   'midwest',   0.075,  0.028,  30),
    ('absenteeism',   'west',      0.065,  0.022,  30),
    ('ed_triage',     'northeast', 45.0,   12.0,   30),
    ('ed_triage',     'southeast', 42.0,   11.0,   30),
    ('ed_triage',     'midwest',   44.0,   11.5,   30),
    ('ed_triage',     'west',      40.0,   10.0,   30)
ON CONFLICT (source, region) DO NOTHING;

-- ============================================================
-- 3. risk_assessments  — Bayesian fusion outputs
-- ============================================================
CREATE TABLE IF NOT EXISTS risk_assessments (
    assessment_id   TEXT PRIMARY KEY,
    region          TEXT NOT NULL,
    region_name     TEXT,
    risk_score      DOUBLE PRECISION NOT NULL DEFAULT 0,
    alert_tier      INTEGER NOT NULL DEFAULT 0,
    signal_breakdown JSONB DEFAULT '[]',
    ai_narrative    TEXT,
    assessed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('risk_assessments', 'assessed_at',
    if_not_exists => TRUE,
    migrate_data  => TRUE
);

CREATE INDEX IF NOT EXISTS idx_risk_region ON risk_assessments (region, assessed_at DESC);

-- ============================================================
-- 4. alert_dispatches  — every alert sent + SLA tracking
-- ============================================================
CREATE TABLE IF NOT EXISTS alert_dispatches (
    dispatch_id     TEXT PRIMARY KEY,
    assessment_id   TEXT,
    alert_tier      INTEGER NOT NULL DEFAULT 0,
    region          TEXT NOT NULL,
    channel         TEXT NOT NULL DEFAULT 'none',
    status          TEXT NOT NULL DEFAULT 'sent',
    message_preview TEXT,
    sent_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by TEXT
);

SELECT create_hypertable('alert_dispatches', 'sent_at',
    if_not_exists => TRUE,
    migrate_data  => TRUE
);

CREATE INDEX IF NOT EXISTS idx_dispatch_status ON alert_dispatches (status, sent_at DESC);

-- ============================================================
-- 5. audit_log  — immutable append-only audit trail
-- ============================================================
CREATE TABLE IF NOT EXISTS audit_log (
    log_id          TEXT PRIMARY KEY,
    workflow_id     TEXT,
    execution_id    TEXT,
    node_name       TEXT,
    action          TEXT,
    decision_tier   INTEGER DEFAULT 0,
    operator_id     TEXT DEFAULT 'system',
    metadata        JSONB DEFAULT '{}',
    logged_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('audit_log', 'logged_at',
    if_not_exists => TRUE,
    migrate_data  => TRUE
);

-- ============================================================
-- 6. Continuous aggregate: hourly signal summary (optional)
-- ============================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS signal_hourly
WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('1 hour', ingested_at) AS bucket,
        source,
        metadata->>'region' AS region,
        AVG(signal_value)   AS avg_value,
        MAX(signal_value)   AS max_value,
        COUNT(*)            AS event_count
    FROM signal_events
    GROUP BY bucket, source, metadata->>'region'
WITH NO DATA;