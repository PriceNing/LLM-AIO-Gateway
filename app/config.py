import os
import json
from pathlib import Path
from typing import Optional


def default_config() -> dict:
    return {
        "host": "0.0.0.0",
        "port": 8000,
        "reload": False,
        "database": "data.db",
        "image_result_dir": "generated_images",
        # 生产环境请改为具体来源列表，例如 ["https://gateway.example.com"]。
        "cors_allow_origins": ["*"],
        "logging": {
            "enabled": True,
            "level": "INFO",
            "log_dir": "logs",
            "retention_days": 30,
            "console": False
        },
        "defaults": {
            "max_tokens": 16384,
            # 客户端未指定输出上限、且模型声明 supports_reasoning 时注入的上限。
            # 上游把思考内容与最终答案计入同一个 completion 预算，16384 容易被思考吃光。
            "reasoning_max_tokens": 32768,
            "temperature": 0.7,
            "max_request_body_bytes": 134217728,
            "tool_only_limit": 20,
            "min_image_max_tokens": 2000,
            "litellm_request_timeout": 120,
            "same_target_retry_limit": 1,
            # 流式块间空闲超时：只在**已向客户端产出首个可见输出之后**计时，0 = 关闭。
            # 为什么需要单独一个键：socket 层的读超时（provider.request_timeout 透传给
            # httpx）对“冷 prefill 静默”与“解码期间块间静默”是同一个旋钮，本地长上下文
            # 引擎需要 300 秒容忍 prefill，同一个值用于解码阶段意味着真死流也要 5 分钟
            # 才发现（2026-10-03 单槽上游楔死事故的误杀侧）。0 为默认：不改变现有部署行为。
            "stream_idle_timeout_seconds": 0,
            "session_ttl_hours": 12,
            "login_attempt_limit": 10,
            "login_attempt_window_seconds": 300,
            "login_lockout_seconds": 900,
            "login_attempt_max_identities": 10000,
            "request_log_max": 200,
            "storage_maintenance_interval_seconds": 60,
            "request_log_capture_payloads": True,
            "request_log_redact_fields": [
                "api_key", "authorization", "cookie", "password", "secret", "token"
            ],
            "reasoning_cache_ttl": 1800,
            "reasoning_cache_max_size": 1000,
            "tool_only_turns_ttl": 600,
            "tool_only_turns_max_size": 2000,
            "image_cache_max_size": 500,
            # 生图预算（会话级时间窗口）：代码一直在读这三个键，之前未在默认值里声明，
            # 导致它们在配置文件里“隐形”、无法被发现与设置。
            "image_generation_budget_limit": 20,
            "image_generation_budget_ttl": 3600,
            "image_generation_budget_max_size": 1000,
            "image_result_ttl_seconds": 86400,
            "image_result_max_files": 500,
            "image_preview_enabled": True,
            "image_preview_max_dimension": 1280,
            "image_preview_max_source_pixels": 40000000,
            "image_preview_quality": 82,
            "image_preview_max_bytes": 800000,
            "image_preview_inline_limit": 4,
            "image_generation_max_retries": 2,
            "image_generation_retry_base_seconds": 1.0,
            "image_generation_max_retry_delay_seconds": 30.0,
            "image_generation_batch_concurrency": 1,
            "image_generation_batch_timeout_seconds": 2400,
            "image_generation_result_max_bytes": 26214400,
            "image_download_allow_private_hosts": False,
            # 局域网自建推理服务是正常用法，默认放行私网上游；元数据地址始终禁止。
            "allow_private_upstream_hosts": True,
            "image_generation_idempotency_ttl_seconds": 300,
            "image_generation_idempotency_max_entries": 64,
            "responses_capability_supported_ttl": 604800,
            "responses_capability_unsupported_ttl": 21600,
            "responses_capability_transient_ttl": 300,
            "responses_capability_probe_timeout": 8,
            "responses_capability_probe_max_output_tokens": 16,
            "model_registry_enabled": True,
            "model_registry_url": "https://openrouter.ai/api/v1/models",
            "model_registry_ttl_seconds": 604800,
            "repair_tool_leaks": True,
            "anthropic_thinking_budget_tokens": 1024,
        }
    }


class ConfigManager:
    """Server-level config only. Data storage is in SQLite (app.database)."""

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path or os.environ.get("LLM_GATEWAY_CONFIG", "config.json"))
        self.config: dict = {}
        self._loaded = False

    def load(self) -> None:
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = default_config()
            self.save()
        self._fill_defaults()
        self._loaded = True

    def _fill_defaults(self) -> None:
        base = default_config()
        for key in base:
            self.config.setdefault(key, base[key])
        for section in ("logging", "defaults"):
            self.config.setdefault(section, base[section])
            if not isinstance(self.config[section], dict):
                self.config[section] = dict(base[section])
                continue
            for key in base[section]:
                self.config[section].setdefault(key, base[section][key])

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        tmp_path.replace(self.path)

    def patch(self, section: str, values: dict | None = None, remove: tuple[str, ...] = ()) -> dict:
        """只改指定键并落盘；返回磁盘上该 section 的最新内容。

        故意不直接 ``save()`` 内存里的 self.config：``load()`` 会把全部内置默认值
        补齐到内存，整体回写会把它们固化进文件，此后网关升级带来的新默认值对这些
        键永久失效（``_fill_defaults`` 对已存在的键不做 setdefault）。因此本函数从
        磁盘重读原文件、只动指定的键、原子写回，保留管理员手改的其它内容。

        "section=\"\"" 表示顶层键。文件不存在时从空对象开始；磁盘 JSON 损坏时直接
        报错退出，绝不把坏文件“修好”成只有本次改动键的文件。
        """
        values = dict(values or {})
        if not values and not remove:
            return dict(self._section_on_disk(section, filled=False))

        on_disk: dict = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                raise ValueError("config.json must contain a JSON object")
            on_disk = loaded

        if section:
            block = on_disk.get(section)
            if not isinstance(block, dict):
                block = {}
            for key in remove:
                block.pop(key, None)
            block.update(values)
            on_disk[section] = block
        else:
            for key in remove:
                on_disk.pop(key, None)
            on_disk.update(values)

        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(on_disk, f, indent=2, ensure_ascii=False)
        tmp_path.replace(self.path)

        # 内存同步：单进程部署，写盘成功后立即让 get_default 读到新值。
        if section:
            current = self.config.get(section)
            if not isinstance(current, dict):
                current = {}
            for key in remove:
                current.pop(key, None)
            current.update(values)
            self.config[section] = current
        else:
            for key in remove:
                self.config.pop(key, None)
            self.config.update(values)
        self._fill_defaults()

        return dict(self._section_on_disk(section, filled=False))

    def _section_on_disk(self, section: str, *, filled: bool) -> dict:
        """读磁盘（或内存）的指定 section，用于区分“文件里写了”与“内置默认”。"""
        if not filled and self.path.exists():
            try:
                with self.path.open("r", encoding="utf-8") as f:
                    loaded = json.load(f)
            except (OSError, json.JSONDecodeError):
                loaded = {}
            source = loaded if isinstance(loaded, dict) else {}
        else:
            source = self.config
        if not section:
            return dict(source)
        block = source.get(section)
        return dict(block) if isinstance(block, dict) else {}

    def file_status(self) -> dict:
        """配置文件路径与可写性，供“诊断”页与写入失败时给出可操作提示。"""
        path = self.path
        exists = path.exists()
        writable = path.parent.is_dir() and (not exists or os.access(path, os.W_OK))
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        return {
            "path": str(path),
            "exists": bool(exists),
            "writable": bool(writable),
            "mtime": mtime,
        }


_config_manager: Optional[ConfigManager] = None


def load_config(path: Optional[str] = None, force_reload: bool = False) -> ConfigManager:
    global _config_manager
    if force_reload or _config_manager is None:
        _config_manager = ConfigManager(path)
        _config_manager.load()
    return _config_manager


def get_config() -> ConfigManager:
    global _config_manager
    if _config_manager is None:
        return load_config()
    return _config_manager


def reload_config() -> ConfigManager:
    """重读当前配置文件（路径不变），供 /admin/config/reload 使用。

    不能图省事调 ``load_config(force_reload=True)``：那个入口不传路径时会回退到
    ``config.json`` 默认路径，在测试与非常规部署下会加载到另一个文件。
    """
    manager = get_config()
    manager.load()
    return manager


def get_default(key: str, fallback=None):
    """Read a value from config.json defaults, returning fallback when missing."""
    try:
        cfg = get_config()
        defaults = cfg.config.get("defaults")
        if isinstance(defaults, dict):
            return defaults.get(key, fallback)
        return fallback
    except (KeyError, TypeError, AttributeError):
        return fallback


def defaults_written_on_disk() -> dict:
    """磁盘文件 defaults 段里显式写了的键。

    用于区分“内置默认值”与“管理员写过”：设置页据此展示来源，并知道“重置为默认”
    该删哪些键。
    """
    return get_config()._section_on_disk("defaults", filled=False)
