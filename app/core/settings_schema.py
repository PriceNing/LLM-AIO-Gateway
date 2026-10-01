"""设置项 schema：由 ``default_config()`` 生成，配置文件与界面不漂移。

设计约束：

* 条目来自 ``config.default_config()["defaults"]`` 的全部键，新增配置键自动出现在
  界面与 API 上，不需要再维护第二份清单。
* 类型由默认值推断；``_META`` 只补充单位、范围、分组、风险条件与提示文案键。
* 校验在服务端强制执行（范围、类型、URL 安全边界），界面只做提示。绝不允许写入
  schema 之外的键——那等于给配置文件开任意写入口。
* 危险项是"值相关"的判定（例如把请求体上限设为 0 等于禁用），不是静态标记。
"""

from __future__ import annotations

from typing import Any

from app.config import default_config

# 分组顺序即界面展示顺序；label 是界面 i18n 键。
GROUPS: list[tuple[str, str]] = [
    ("output_budget", "settings.group.outputBudget"),
    ("sampling", "settings.group.sampling"),
    ("inbound", "settings.group.inbound"),
    ("reasoning", "settings.group.reasoning"),
    ("tools", "settings.group.tools"),
    ("upstream", "settings.group.upstream"),
    ("logs", "settings.group.logs"),
    ("image_preview", "settings.group.imagePreview"),
    ("image_generation", "settings.group.imageGeneration"),
    ("responses_native", "settings.group.responsesNative"),
    ("registry", "settings.group.registry"),
    ("security", "settings.group.security"),
    ("maintenance", "settings.group.maintenance"),
]

# 单位决定界面如何格式化（bytes/tokens 这类数字人眼不可读，必须换算显示）。
_META: dict[str, dict] = {
    # -- 输出预算 --
    "max_tokens": {
        "group": "output_budget", "unit": "tokens", "min": 0, "max": 1_000_000,
        "hint": "settings.hint.maxTokens",
    },
    "reasoning_max_tokens": {
        "group": "output_budget", "unit": "tokens", "min": 0, "max": 1_000_000,
        "hint": "settings.hint.reasoningMaxTokens",
    },
    "min_image_max_tokens": {
        "group": "output_budget", "unit": "tokens", "min": 0, "max": 1_000_000,
        "hint": "settings.hint.minImageMaxTokens",
    },
    "temperature": {
        "group": "sampling", "unit": "ratio", "min": 0.0, "max": 2.0,
    },
    # -- 入站体积 --
    "max_request_body_bytes": {
        "group": "inbound", "unit": "bytes", "min": 0, "max": 512 * 1024 * 1024,
        "hint": "settings.hint.maxRequestBodyBytes",
        "danger": {"when": "lte", "value": 0, "key": "settings.danger.bodyLimitOff"},
    },
    # -- 推理连续性与缓存 --
    "reasoning_cache_ttl": {"group": "reasoning", "unit": "seconds", "min": 10, "max": 86_400},
    "reasoning_cache_max_size": {"group": "reasoning", "unit": "count", "min": 10, "max": 100_000},
    "tool_only_turns_ttl": {"group": "reasoning", "unit": "seconds", "min": 10, "max": 86_400},
    "tool_only_turns_max_size": {"group": "reasoning", "unit": "count", "min": 10, "max": 100_000},
    "tool_only_limit": {
        "group": "tools", "unit": "count", "min": 0, "max": 1000,
        "hint": "settings.hint.toolOnlyLimit",
        "danger": {"when": "lte", "value": 0, "key": "settings.danger.toolOnlyLimitOff"},
    },
    "repair_tool_leaks": {
        "group": "tools", "unit": "bool",
        "hint": "settings.hint.repairToolLeaks",
        "danger": {"when": "false", "key": "settings.danger.toolLeaksOff"},
    },
    # -- 上游调用 --
    "litellm_request_timeout": {"group": "upstream", "unit": "seconds", "min": 5, "max": 3600},
    "same_target_retry_limit": {"group": "upstream", "unit": "count", "min": 0, "max": 3},
    "anthropic_thinking_budget_tokens": {
        "group": "upstream", "unit": "tokens", "min": 1024, "max": 128_000,
        "hint": "settings.hint.thinkingBudget",
    },
    # -- 日志 --
    "request_log_max": {
        "group": "logs", "unit": "count", "min": 1, "max": 100_000,
        "hint": "settings.hint.requestLogMax",
    },
    "request_log_capture_payloads": {
        "group": "logs", "unit": "bool",
        "danger": {"when": "false", "key": "settings.danger.payloadCaptureOff"},
    },
    "request_log_redact_fields": {
        "group": "logs", "unit": "list", "max_items": 64, "max_item_len": 64,
        # 空列表 = 不脱敏，api_key/token 会明文进 request_logs。
        "danger": {"when": "empty", "key": "settings.danger.redactEmpty"},
    },
    "storage_maintenance_interval_seconds": {"group": "maintenance", "unit": "seconds", "min": 5, "max": 86_400},
    # -- 图像预览与结果 --
    "image_preview_enabled": {"group": "image_preview", "unit": "bool"},
    "image_preview_max_dimension": {"group": "image_preview", "unit": "pixels", "min": 256, "max": 4096},
    "image_preview_max_source_pixels": {"group": "image_preview", "unit": "count", "min": 100_000, "max": 200_000_000},
    "image_preview_quality": {"group": "image_preview", "unit": "count", "min": 40, "max": 95},
    "image_preview_max_bytes": {"group": "image_preview", "unit": "bytes", "min": 65_536, "max": 4 * 1024 * 1024},
    "image_preview_inline_limit": {"group": "image_preview", "unit": "count", "min": 1, "max": 20},
    "image_result_ttl_seconds": {"group": "image_preview", "unit": "seconds", "min": 60, "max": 2_592_000},
    "image_result_max_files": {"group": "image_preview", "unit": "count", "min": 1, "max": 100_000},
    "image_cache_max_size": {"group": "image_preview", "unit": "count", "min": 1, "max": 100_000},
    # -- 图像生成 --
    "image_generation_max_retries": {"group": "image_generation", "unit": "count", "min": 0, "max": 10},
    "image_generation_retry_base_seconds": {"group": "image_generation", "unit": "seconds", "min": 0.1, "max": 60},
    "image_generation_max_retry_delay_seconds": {"group": "image_generation", "unit": "seconds", "min": 1, "max": 300},
    "image_generation_batch_concurrency": {"group": "image_generation", "unit": "count", "min": 1, "max": 8},
    "image_generation_batch_timeout_seconds": {"group": "image_generation", "unit": "seconds", "min": 30, "max": 7200},
    "image_generation_result_max_bytes": {"group": "image_generation", "unit": "bytes", "min": 65_536, "max": 128 * 1024 * 1024},
    "image_generation_idempotency_ttl_seconds": {"group": "image_generation", "unit": "seconds", "min": 30, "max": 86_400},
    "image_generation_idempotency_max_entries": {"group": "image_generation", "unit": "count", "min": 1, "max": 1000},
    "image_generation_budget_limit": {"group": "image_generation", "unit": "count", "min": 0, "max": 1000},
    "image_generation_budget_ttl": {"group": "image_generation", "unit": "seconds", "min": 60, "max": 86_400},
    "image_generation_budget_max_size": {"group": "image_generation", "unit": "count", "min": 10, "max": 100_000},
    # -- 原生 Responses 能力探测 --
    "responses_capability_supported_ttl": {"group": "responses_native", "unit": "seconds", "min": 60, "max": 2_592_000},
    "responses_capability_unsupported_ttl": {"group": "responses_native", "unit": "seconds", "min": 60, "max": 2_592_000},
    "responses_capability_transient_ttl": {"group": "responses_native", "unit": "seconds", "min": 10, "max": 86_400},
    "responses_capability_probe_timeout": {"group": "responses_native", "unit": "seconds", "min": 1, "max": 120},
    "responses_capability_probe_max_output_tokens": {"group": "responses_native", "unit": "tokens", "min": 1, "max": 1024},
    # -- 在线能力库 --
    "model_registry_enabled": {"group": "registry", "unit": "bool"},
    "model_registry_url": {"group": "registry", "type": "url", "unit": "url", "max_len": 2048},
    "model_registry_ttl_seconds": {"group": "registry", "unit": "seconds", "min": 60, "max": 2_592_000},
    # -- 安全与会话 --
    "allow_private_upstream_hosts": {
        "group": "security", "unit": "bool",
        "hint": "settings.hint.allowPrivateUpstream",
        "danger": {"when": "true", "key": "settings.danger.privateUpstream"},
    },
    "image_download_allow_private_hosts": {
        "group": "security", "unit": "bool",
        "danger": {"when": "true", "key": "settings.danger.privateDownload"},
    },
    "session_ttl_hours": {"group": "security", "unit": "hours", "min": 1, "max": 720},
    "login_attempt_limit": {"group": "security", "unit": "count", "min": 1, "max": 1000},
    "login_attempt_window_seconds": {"group": "security", "unit": "seconds", "min": 10, "max": 86_400},
    "login_lockout_seconds": {"group": "security", "unit": "seconds", "min": 10, "max": 86_400},
    "login_attempt_max_identities": {"group": "security", "unit": "count", "min": 100, "max": 1_000_000},
}

_DEFAULT_GROUP = "maintenance"

# 顶层（非 defaults）配置：只读展示，改动需要重启。lifespan 里 init_db / init_logging
# 只跑一次，热改这些键只会制造"文件与运行时不一致"。
RESTART_REQUIRED_TOP_LEVEL = (
    "host", "port", "reload", "database", "image_result_dir", "cors_allow_origins", "logging",
)


def _infer_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return "string_list"
    if isinstance(value, dict):
        return "json_object"
    return "string"


def _defaults() -> dict:
    return default_config()["defaults"]


def _build_schema() -> dict[str, dict]:
    entries: dict[str, dict] = {}
    for key, value in _defaults().items():
        meta = _META.get(key, {})
        kind = _infer_type(value)
        entry = {
            "key": key,
            "type": meta.get("type", kind),
            "default": value,
            "group": meta.get("group", _DEFAULT_GROUP),
            "unit": meta.get("unit", "text" if kind in ("string", "string_list", "json_object") else "count"),
            "min": meta.get("min"),
            "max": meta.get("max"),
            "max_items": meta.get("max_items"),
            "max_item_len": meta.get("max_item_len"),
            "max_len": meta.get("max_len"),
            "hint": meta.get("hint", ""),
            "danger": meta.get("danger"),
            # 所有 defaults 键都已做到写入即生效（固化在对象里的值由 runtime hook 推送）。
            "hot": True,
            "declared": key in _META,
        }
        entries[key] = entry
    return entries


_SCHEMA_CACHE: dict[str, dict] | None = None


def schema() -> dict[str, dict]:
    """返回 {key: 条目}（类型/默认值/范围/单位/分组/提示/风险条件）。

    进程级缓存：条目只由 `default_config()` 与 `_META` 决定，两者都是静态表；
    `validate_values` 会对每个键调一次，不缓存就是 N+1 次全量构建。返回值按只读
    对待，调用方需要修改时自己复制。
    """
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        _SCHEMA_CACHE = _build_schema()
    return _SCHEMA_CACHE


def group_order() -> list[tuple[str, str]]:
    return list(GROUPS)


def _danger_reason(entry: dict, value: Any) -> str | None:
    rule = entry.get("danger")
    if not isinstance(rule, dict):
        return None
    when = rule.get("when")
    try:
        if when == "true":
            hit = value is True
        elif when == "false":
            hit = value is False
        elif when == "lte":
            hit = float(value) <= float(rule.get("value", 0))
        elif when == "gte":
            hit = float(value) >= float(rule.get("value", 0))
        elif when == "empty":
            # 空集合/空字符串：对“脱敏字段列表”这类键，清空等于关闭保护。
            hit = isinstance(value, (list, tuple, dict, str)) and len(value) == 0
        else:
            hit = False
    except (TypeError, ValueError):
        hit = False
    return rule.get("key") if hit else None


class SettingsValidationError(ValueError):
    """一条或多条设置项校验失败；消息面向管理员，包含键名与原因。"""


def _check_url(key: str, value: str) -> str:
    from app.services.url_guard import UnsafeUpstreamURL, validate_upstream_url

    text = str(value or "").strip()
    if not text:
        raise SettingsValidationError(f"{key}: URL 不能为空")
    try:
        return validate_upstream_url(text, field=key)
    except UnsafeUpstreamURL as exc:
        raise SettingsValidationError(f"{key}: {exc}") from exc


def validate(key: str, value: Any) -> Any:
    """把提交值归一化为可写入配置的形式；非法输入抛 SettingsValidationError。

    bool 是 int 的子类：int 字段必须显式拒绝 True/False，否则 {"max_tokens": true}
    会写成 1。
    """
    entry = schema().get(key)
    if entry is None:
        raise SettingsValidationError(f"{key}: 不是已知设置项")
    kind = entry["type"]

    if kind == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        raise SettingsValidationError(f"{key}: 需要布尔值")

    if kind in ("int", "float"):
        if isinstance(value, bool):
            raise SettingsValidationError(f"{key}: 需要数字，不接受 true/false")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise SettingsValidationError(f"{key}: 需要数字") from None
        if kind == "int":
            if number != int(number):
                raise SettingsValidationError(f"{key}: 需要整数")
            number = int(number)
        low, high = entry.get("min"), entry.get("max")
        if low is not None and number < low:
            raise SettingsValidationError(f"{key}: 不得小于 {low}")
        if high is not None and number > high:
            raise SettingsValidationError(f"{key}: 不得大于 {high}")
        return number

    if kind == "string":
        text = str(value if value is not None else "").strip()
        limit = entry.get("max_len") or 512
        if len(text) > limit:
            raise SettingsValidationError(f"{key}: 长度不得超过 {limit}")
        return text

    if kind == "url":
        return _check_url(key, value)

    if kind == "string_list":
        if isinstance(value, str):
            items = [item.strip() for item in value.split(",")]
        elif isinstance(value, list):
            items = [str(item).strip() for item in value]
        else:
            raise SettingsValidationError(f"{key}: 需要字符串数组")
        items = [item for item in items if item]
        limit = entry.get("max_items") or 64
        item_len = entry.get("max_item_len") or 128
        if len(items) > limit:
            raise SettingsValidationError(f"{key}: 最多 {limit} 项")
        if any(len(item) > item_len for item in items):
            raise SettingsValidationError(f"{key}: 单项长度不得超过 {item_len}")
        # 去重但保持顺序：脱敏字段列表里重复项没有意义，静默去重比写入脏数据好。
        seen: set[str] = set()
        unique = []
        for item in items:
            if item.lower() not in seen:
                seen.add(item.lower())
                unique.append(item)
        return unique

    # json_object：默认值里没有这种键，出现时按受限 JSON 对象透传。
    if not isinstance(value, dict):
        raise SettingsValidationError(f"{key}: 需要 JSON 对象")
    if len(str(value)) > 4096:
        raise SettingsValidationError(f"{key}: JSON 对象过大")
    return value


def validate_values(values: dict) -> dict:
    """整批校验：全部合法才返回，任何一条失败都抛出聚合错误（不做半批写入）。"""
    if not isinstance(values, dict):
        raise SettingsValidationError("values 必须是 JSON 对象")
    normalized: dict[str, Any] = {}
    errors: list[str] = []
    for key, value in values.items():
        try:
            normalized[str(key)] = validate(str(key), value)
        except SettingsValidationError as exc:
            errors.append(str(exc))
    if errors:
        raise SettingsValidationError("；".join(errors))
    return normalized


def known_keys() -> list[str]:
    return sorted(schema())


def danger_for(key: str, value: Any) -> str | None:
    entry = schema().get(key)
    return _danger_reason(entry, value) if entry else None
