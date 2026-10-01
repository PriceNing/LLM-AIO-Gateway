#!/bin/bash
set -e

CONFIG_FILE="${LLM_GATEWAY_CONFIG:-/app/data/config.json}"
DATA_DIR="$(dirname "$CONFIG_FILE")"

mkdir -p "$DATA_DIR"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "First run: creating default config at $CONFIG_FILE"
    # 只写引导项（监听/端口/数据库/日志目录）。defaults 段一律不在这里手抄：
    # 代码里的 default_config() 会在加载时补齐缺失键，手抄只会造成第三处真源
    # 并随版本漂移（历史上已经漏掉 max_request_body_bytes / reasoning_max_tokens 等）。
    # 需要改运行参数时用管理页“设置”，或直接编辑本文件后点“重新加载”。
    cat > "$CONFIG_FILE" <<EOF
{
    "host": "0.0.0.0",
    "port": 8000,
    "reload": false,
    "database": "$DATA_DIR/data.db",
    "image_result_dir": "generated_images",
    "logging": {
        "enabled": true,
        "level": "INFO",
        "log_dir": "logs",
        "retention_days": 30,
        "console": false
    }
}
EOF
fi

exec python main.py
