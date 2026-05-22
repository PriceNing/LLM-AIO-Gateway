#!/bin/bash
set -e

CONFIG_FILE="${LLM_GATEWAY_CONFIG:-/app/data/config.json}"
DATA_DIR="$(dirname "$CONFIG_FILE")"

mkdir -p "$DATA_DIR"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "First run: creating default config at $CONFIG_FILE"
    cat > "$CONFIG_FILE" <<EOF
{
    "host": "0.0.0.0",
    "port": 8000,
    "database": "$DATA_DIR/data.db",
    "logging": {
        "enabled": true,
        "level": "INFO",
        "log_dir": "logs",
        "retention_days": 30,
        "console": false
    },
    "defaults": {
        "max_tokens": 16384,
        "temperature": 0.7,
        "tool_only_limit": 20,
        "min_image_max_tokens": 2000,
        "session_ttl_hours": 12,
        "request_log_max": 200,
        "reasoning_cache_ttl": 1800,
        "reasoning_cache_max_size": 1000,
        "tool_only_turns_ttl": 600,
        "tool_only_turns_max_size": 2000,
        "image_cache_max_size": 500
    }
}
EOF
fi

exec python main.py
