-- 上一个发布版本（v0.14.0）的完整 schema 快照，供启动路径回归使用。
-- 用途：tests/test_database_migration.py 用它在「旧库 + 新代码」上跑真实 init_db()，
-- 任何新增列/索引只要顺序写错（例如在 _SCHEMA 里引用 migrate() 才补的列）都会在这里炸。
-- 发布新版本后如需更新此快照，请另存为 schema_v<新版本>.sql，保留旧快照作为升级链一环。
CREATE TABLE IF NOT EXISTS admins (
    username TEXT PRIMARY KEY,
    display_name TEXT NOT NULL DEFAULT '',
    password_hash TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    display_name TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1,
    total_calls INTEGER NOT NULL DEFAULT 0,
    failed_calls INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS user_api_keys (
    key TEXT PRIMARY KEY,
    username TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT 'default',
    allowed_models TEXT NOT NULL DEFAULT '["*"]',
    enabled INTEGER NOT NULL DEFAULT 1,
    total_calls INTEGER NOT NULL DEFAULT 0,
    failed_calls INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS providers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    provider_type TEXT NOT NULL DEFAULT 'openai',
    api_base TEXT NOT NULL DEFAULT '',
    api_key TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1,
    provider_options TEXT NOT NULL DEFAULT '{}',
    upstream_headers TEXT NOT NULL DEFAULT '{}',
    request_timeout INTEGER NOT NULL DEFAULT 120,
    retry_count INTEGER NOT NULL DEFAULT 0,
    retry_backoff REAL NOT NULL DEFAULT 0.5,
    force_chat_completions INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS provider_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_name TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT '',
    preprocessor TEXT NOT NULL DEFAULT '',
    image_generation TEXT NOT NULL DEFAULT '',
    responses_status TEXT NOT NULL DEFAULT 'unknown',
    responses_checked_at TEXT NOT NULL DEFAULT '',
    responses_expires_at TEXT NOT NULL DEFAULT '',
    responses_streaming INTEGER NOT NULL DEFAULT 0,
    responses_streaming_status TEXT NOT NULL DEFAULT 'unknown',
    responses_tool_types TEXT NOT NULL DEFAULT '[]',
    responses_error TEXT NOT NULL DEFAULT '',
    capabilities TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (provider_id) REFERENCES providers(id) ON DELETE CASCADE,
    UNIQUE(provider_id, model_id)
);

CREATE INDEX IF NOT EXISTS idx_provider_models_model_id ON provider_models(model_id);

CREATE TABLE IF NOT EXISTS preprocessors (
    id TEXT PRIMARY KEY,
    api_base TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    api_key TEXT NOT NULL DEFAULT '',
    timeout INTEGER NOT NULL DEFAULT 120,
    max_images INTEGER NOT NULL DEFAULT 10,
    prompt TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1,
    max_tokens INTEGER NOT NULL DEFAULT 2048,
    created_at TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS image_generators (
    id TEXT PRIMARY KEY,
    backend_type TEXT NOT NULL DEFAULT 'existing_model',
    provider_model TEXT NOT NULL DEFAULT '',
    api_base TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    api_key TEXT NOT NULL DEFAULT '',
    timeout INTEGER NOT NULL DEFAULT 180,
    workflow TEXT NOT NULL DEFAULT '{}',
    workflow_mapping TEXT NOT NULL DEFAULT '{}',
    poll_interval REAL NOT NULL DEFAULT 1.0,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS routing_rules (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT 'New Rule',
    enabled INTEGER NOT NULL DEFAULT 1,
    username TEXT NOT NULL DEFAULT '',
    api_key_pattern TEXT NOT NULL DEFAULT '',
    match_model TEXT NOT NULL DEFAULT '',
    match_scope TEXT NOT NULL DEFAULT 'any',
    target_model TEXT NOT NULL DEFAULT '',
    target_provider TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS fallback_policies (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT 'New Fallback Policy',
    enabled INTEGER NOT NULL DEFAULT 1,
    match_provider TEXT NOT NULL DEFAULT '',
    match_model TEXT NOT NULL DEFAULT '*',
    triggers TEXT NOT NULL DEFAULT '{}',
    chain TEXT NOT NULL DEFAULT '[]',
    attempt_timeout INTEGER NOT NULL DEFAULT 60,
    created_at TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS global_stats (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL DEFAULT '0'
);

CREATE TABLE IF NOT EXISTS request_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    model TEXT NOT NULL,
    username TEXT NOT NULL DEFAULT '',
    success INTEGER NOT NULL DEFAULT 1,
    tokens INTEGER NOT NULL DEFAULT 0,
    request_kind TEXT NOT NULL DEFAULT '',
    image_model TEXT NOT NULL DEFAULT '',
    image_count INTEGER NOT NULL DEFAULT 0,
    image_bytes INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_req_ts ON request_records(timestamp);
-- 历史统计会按 model / username 分组；缺索引时 SQLite 退化为临时 B 树（P6）。
CREATE INDEX IF NOT EXISTS idx_req_ts_model ON request_records(timestamp, model);
CREATE INDEX IF NOT EXISTS idx_req_ts_user ON request_records(timestamp, username);
-- user_api_keys 按 username 关联查询时避免全表扫描（P6）。
CREATE INDEX IF NOT EXISTS idx_user_api_keys_username ON user_api_keys(username);
CREATE TABLE IF NOT EXISTS request_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    username TEXT NOT NULL DEFAULT '',
    api_key TEXT NOT NULL DEFAULT '',
    requested_model TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    provider TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT '',
    stream INTEGER NOT NULL DEFAULT 0,
    tokens INTEGER NOT NULL DEFAULT 0,
    request_body TEXT,
    response_body TEXT,
    details TEXT NOT NULL DEFAULT '{}',
    error TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_reqlog_ts ON request_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_reqlog_endpoint ON request_logs(endpoint);
CREATE TABLE IF NOT EXISTS model_registry (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    url TEXT NOT NULL DEFAULT '',
    fetched_at TEXT NOT NULL DEFAULT '',
    payload TEXT NOT NULL DEFAULT '[]'
);
