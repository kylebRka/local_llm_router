PRAGMA foreign_keys = ON;

-- One request can have several decisions in future retries. The current CLI makes one.
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('interactive', 'legacy')),
    text TEXT NOT NULL CHECK (length(trim(text)) > 0)
);

CREATE TABLE IF NOT EXISTS routing_decisions (
    id INTEGER PRIMARY KEY,
    request_id INTEGER NOT NULL REFERENCES requests(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    predicted_model_label TEXT NOT NULL CHECK (predicted_model_label IN ('E2B','E4B','12B')),
    selected_model_label TEXT NOT NULL CHECK (selected_model_label IN ('E2B','E4B','12B')),
    selected_model_id TEXT NOT NULL,
    category TEXT NOT NULL,
    difficulty INTEGER NOT NULL CHECK (difficulty BETWEEN 0 AND 9),
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    margin REAL NOT NULL CHECK (margin BETWEEN 0 AND 1),
    probabilities_json TEXT NOT NULL,
    route_reason TEXT NOT NULL,
    explanation TEXT NOT NULL,
    classifier_sha256 TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisions_request ON routing_decisions(request_id);

CREATE TABLE IF NOT EXISTS responses (
    id INTEGER PRIMARY KEY,
    decision_id INTEGER NOT NULL UNIQUE REFERENCES routing_decisions(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('success','error','legacy_unknown')),
    text TEXT,
    error_message TEXT,
    raw_json TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    response_id INTEGER PRIMARY KEY REFERENCES responses(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    total_seconds REAL,
    routing_seconds REAL,
    explanation_seconds REAL,
    model_prepare_seconds REAL,
    model_request_seconds REAL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    reasoning_tokens INTEGER,
    ttft_seconds REAL,
    tokens_per_second REAL,
    raw_stats_json TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY,
    response_id INTEGER NOT NULL UNIQUE REFERENCES responses(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    rating INTEGER NOT NULL CHECK (rating IN (-1,1)),
    expected_model_label TEXT CHECK (expected_model_label IN ('E2B','E4B','12B')),
    comment TEXT NOT NULL DEFAULT ''
);

-- Feedback may propose a label. Only explicit review makes it training data.
CREATE TABLE IF NOT EXISTS training_labels (
    id INTEGER PRIMARY KEY,
    request_id INTEGER NOT NULL UNIQUE REFERENCES requests(id) ON DELETE CASCADE,
    model_label TEXT NOT NULL CHECK (model_label IN ('E2B','E4B','12B')),
    source TEXT NOT NULL CHECK (source IN ('feedback','manual')),
    source_feedback_id INTEGER REFERENCES feedback(id),
    status TEXT NOT NULL CHECK (status IN ('pending','approved','rejected')),
    created_at TEXT NOT NULL,
    reviewed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_labels_status ON training_labels(status, model_label);

CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('candidate','promoted','rejected')),
    candidate_path TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    real_train_count INTEGER NOT NULL,
    real_test_count INTEGER NOT NULL,
    synthetic_count INTEGER NOT NULL,
    promoted_at TEXT
);
