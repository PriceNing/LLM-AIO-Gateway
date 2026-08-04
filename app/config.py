import os
import json
from pathlib import Path
from typing import Optional


def default_config() -> dict:
    return {
        "host": "0.0.0.0",
        "port": 8000,
        "database": "data.db",
        "image_result_dir": "generated_images",
        "logging": {
            "enabled": True,
            "level": "INFO",
            "log_dir": "logs",
            "retention_days": 30,
            "console": False
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
            "image_cache_max_size": 500,
            "image_result_ttl_seconds": 86400,
            "image_result_max_files": 500,
            "image_preview_enabled": True,
            "image_preview_max_dimension": 1280,
            "image_preview_quality": 82,
            "image_preview_max_bytes": 800000,
            "responses_capability_supported_ttl": 604800,
            "responses_capability_unsupported_ttl": 21600,
            "responses_capability_transient_ttl": 300,
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
