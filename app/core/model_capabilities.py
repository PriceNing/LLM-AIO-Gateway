"""模型能力元数据：内置家族表、归一化与合并。

下游 harness（OpenCode/Codex 类客户端）通过 GET /v1/models 获取模型清单时，
顺带拿到 context_window / vision / tools 等能力信息。数据来源三层，优先级：

    内置模型家族启发式  <  上游 /models 元数据透传  <  管理员手动覆盖

上游元数据与管理员覆盖持久化在 provider_models.capabilities（JSON）；
内置表不落库，输出时兜底合并，升级网关即可刷新启发式而不覆盖用户数据。
"""

from typing import Any

import re

# 能力字段白名单：context_window/max_output_tokens 为正整数，
# supports_vision/supports_tools 为布尔，input_modalities 为字符串列表，
# pricing 为 {prompt|completion|image: 字符串数字}（OpenRouter 口径，$/M tokens）。
_INT_KEYS = ("context_window", "max_output_tokens")
_BOOL_KEYS = ("supports_vision", "supports_tools")
_CAPABILITY_KEYS = _INT_KEYS + _BOOL_KEYS + ("input_modalities", "pricing")

# 异常/被篡改的上游数据不得直出客户端：超出合理上限的值丢弃（保持"未知"）。
_INT_LIMITS = {"context_window": 100_000_000, "max_output_tokens": 10_000_000}

_BOOL_TRUE = frozenset({"true", "1", "yes", "on"})
_BOOL_FALSE = frozenset({"false", "0", "no", "off"})


def _parse_bool(value: Any) -> bool | None:
    """严格布尔解析：只接受 True/False/1/0 及常见字符串形式。

    无法识别时返回 None（调用方丢弃该键，保持"未知"），绝不用 bool()
    强转——那会把上游的 "false"/"0"/"no" 变成 True，将纯文本模型
    广告成支持视觉，诱导客户端发图后被上游 400。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _BOOL_TRUE:
            return True
        if lowered in _BOOL_FALSE:
            return False
    return None


def normalize_capabilities(raw: Any) -> dict:
    """校验并归一化能力字典；非法字段丢弃，非法值忽略。"""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for key in _INT_KEYS:
        value = raw.get(key)
        if value is None or isinstance(value, bool):
            continue  # bool 是 int 子类，True 会变成 1，必须显式拒绝
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        # .get 兜底：将来 _INT_KEYS 加键而漏配限额时不得 KeyError 打挂解析路径
        limit = _INT_LIMITS.get(key, 100_000_000)
        if parsed <= 0 or parsed > limit:
            continue
        out[key] = parsed
    for key in _BOOL_KEYS:
        if key in raw and raw[key] is not None:
            parsed = _parse_bool(raw[key])
            if parsed is not None:
                out[key] = parsed
    modalities = raw.get("input_modalities")
    if isinstance(modalities, list):
        cleaned = [str(m) for m in modalities if isinstance(m, (str, int))]
        if cleaned:
            out["input_modalities"] = cleaned
    pricing = raw.get("pricing")
    if isinstance(pricing, dict):
        cleaned_pricing = {
            str(k): str(v)
            for k, v in pricing.items()
            # bool 是 int 子类：{"prompt": True} 不得变成 "True"
            if isinstance(v, (str, int, float)) and not isinstance(v, bool)
            and str(k) in ("prompt", "completion", "image", "request")
        }
        if cleaned_pricing:
            out["pricing"] = cleaned_pricing
    return out


def merge_capabilities(*sources: dict | None) -> dict:
    """后面的来源覆盖前面的（仅覆盖显式给出的键）。"""
    merged: dict[str, Any] = {}
    for source in sources:
        if isinstance(source, dict):
            for key, value in source.items():
                if key in _CAPABILITY_KEYS and value is not None:
                    merged[key] = value
    return merged


# ── 内置模型家族启发式 ─────────────────────────────────────────────
# 只收录高置信度的知名家族；数值为公开文档口径的近似值，仅作 harness 提示，
# 管理员覆盖与上游透传永远优先。顺序：先特例后泛化（首个命中生效）。
_BUILTIN_FAMILIES: list[tuple[tuple[str, ...], dict]] = [
    # Anthropic
    (("claude-3-5-haiku",), {"context_window": 200000, "max_output_tokens": 8192, "supports_vision": True, "supports_tools": True}),
    (("claude-3", "claude-4", "claude-sonnet", "claude-opus", "claude-haiku"), {"context_window": 200000, "max_output_tokens": 4096, "supports_vision": True, "supports_tools": True}),
    (("claude-2",), {"context_window": 100000, "supports_vision": False, "supports_tools": False}),
    # OpenAI
    (("gpt-4o",), {"context_window": 128000, "max_output_tokens": 16384, "supports_vision": True, "supports_tools": True}),
    (("gpt-4.1",), {"context_window": 1047552, "max_output_tokens": 32768, "supports_vision": True, "supports_tools": True}),
    (("gpt-4-turbo", "gpt-4-turbo-preview"), {"context_window": 128000, "max_output_tokens": 4096, "supports_vision": True, "supports_tools": True}),
    (("gpt-6",), {"context_window": 1050000, "max_output_tokens": 128000, "supports_vision": True, "supports_tools": True}),
    (("gpt-5",), {"max_output_tokens": 128000, "supports_vision": True, "supports_tools": True}),
    (("o1", "o3", "o4-mini"), {"context_window": 200000, "max_output_tokens": 100000, "supports_vision": True, "supports_tools": True}),
    (("gpt-3.5",), {"context_window": 16383, "max_output_tokens": 4096, "supports_vision": False, "supports_tools": True}),
    # Google
    (("gemini-1.5-pro", "gemini-1.5-ultra"), {"context_window": 2000000, "max_output_tokens": 8192, "supports_vision": True, "supports_tools": True}),
    (("gemini-1.5-flash",), {"context_window": 1000000, "max_output_tokens": 8192, "supports_vision": True, "supports_tools": True}),
    (("gemini-2.0", "gemini-2.5"), {"context_window": 1000000, "max_output_tokens": 8192, "supports_vision": True, "supports_tools": True}),
    (("gemini",), {"context_window": 1000000, "supports_vision": True, "supports_tools": True}),
    # DeepSeek（在线注册表不可用时的离线兜底；数据可能滞后，以注册表/管理员覆盖为准）
    (("deepseek-flash", "deepseek-v4-flash-vision"), {"context_window": 1048576, "max_output_tokens": 943718, "supports_vision": True, "supports_tools": True}),
    (("deepseek-v4",), {"context_window": 1048576, "max_output_tokens": 393216, "supports_vision": False, "supports_tools": True}),
    (("deepseek-reasoner", "deepseek-r1"), {"context_window": 163840, "max_output_tokens": 32768, "supports_vision": False, "supports_tools": True}),
    (("deepseek",), {"context_window": 163840, "max_output_tokens": 16384, "supports_vision": False, "supports_tools": True}),
    # Qwen
    (("qwen-vl", "qwen2-vl", "qwen2.5-vl", "qwen3-vl"), {"supports_vision": True, "supports_tools": True}),
    (("qwen2.5-coder", "qwen3-coder"), {"context_window": 128000, "max_output_tokens": 8192, "supports_vision": False, "supports_tools": True}),
    (("qwen-max", "qwen-plus", "qwen-turbo", "qwen2.5", "qwen3"), {"context_window": 128000, "max_output_tokens": 8192, "supports_vision": False, "supports_tools": True}),
    # Meta / Mistral / 其他开源家族
    (("llama-3.1", "llama-3.3", "llama3.1", "llama3.3"), {"context_window": 128000, "max_output_tokens": 4096, "supports_vision": False, "supports_tools": True}),
    (("llama-3.2-vision",), {"context_window": 128000, "supports_vision": True, "supports_tools": True}),
    (("mistral-large",), {"context_window": 128000, "supports_vision": False, "supports_tools": True}),
    (("pixtral",), {"context_window": 128000, "supports_vision": True, "supports_tools": True}),
    (("minicpm-v", "llava", "internvl", "glm-4v"), {"supports_vision": True, "supports_tools": False}),
    (("kimi-k2",), {"context_window": 128000, "max_output_tokens": 8192, "supports_vision": False, "supports_tools": True}),
    (("moonshot-v1",), {"context_window": 128000, "supports_vision": False, "supports_tools": True}),
    (("glm-4-plus", "glm-4-air", "glm-4-flash"), {"context_window": 128000, "supports_vision": False, "supports_tools": True}),
]

# 明确非对话模型：不给任何能力提示
_NON_CHAT_MARKERS = ("embedding", "rerank", "reranker", "whisper", "tts", "audio", "image-", "dall-e", "stable-diffusion", "flux", "sora")


# 短标记（无连字符且 ≤4 字符，如 "o1"/"o3"/"tts"/"sora"）用词边界匹配，
# 避免 "sao10k" 含 "o1" 这类子串误命中；长标记/含连字符的保持子串匹配。
_SHORT_MARKER_RE: dict[str, "re.Pattern[str]"] = {}


def _marker_hit(marker: str, text: str) -> bool:
    if "-" not in marker and len(marker) <= 4:
        pattern = _SHORT_MARKER_RE.get(marker)
        if pattern is None:
            pattern = re.compile(rf"(?<![a-z0-9]){re.escape(marker)}(?![a-z0-9])")
            _SHORT_MARKER_RE[marker] = pattern
        return pattern.search(text) is not None
    return marker in text


def builtin_capabilities(model_id: str, model_name: str = "") -> dict:
    """按模型名启发式匹配内置能力表；未知模型返回空 dict（不猜测）。

    注意：内置表只是离线兜底，数据可能滞后；权威来源是在线注册表
    （services/model_registry.py）与管理员覆盖。
    """
    text = f"{model_id or ''} {model_name or ''}".lower()
    if any(_marker_hit(marker, text) for marker in _NON_CHAT_MARKERS):
        return {}
    for markers, caps in _BUILTIN_FAMILIES:
        if any(_marker_hit(marker, text) for marker in markers):
            return dict(caps)
    return {}


def resolve_model_capabilities(model: dict, remote: dict | None = None) -> dict:
    """合并能力来源：内置家族表 < 在线注册表 < 已存储（上游透传 + 管理员覆盖）。

    ``model`` 为 _model_from_row 产物，``capabilities`` 键为持久化 JSON；
    ``remote`` 为 services.model_registry.registry_lookup 的在线查询结果。
    """
    stored = model.get("capabilities")
    if not isinstance(stored, dict):
        stored = {}
    stored_values = {k: v for k, v in stored.items() if k in _CAPABILITY_KEYS}
    base = builtin_capabilities(str(model.get("id") or ""), str(model.get("name") or ""))
    # 读取侧再归一化一次：DB 里可能残留旧版本（bool 强转/无上限）写入的
    # 异常值，注册表 payload 也是按拉取当时的规则序列化的；只靠写入侧
    # 校验会被历史数据/直接改库绕过，异常值将直出客户端。
    return normalize_capabilities(merge_capabilities(base, remote if isinstance(remote, dict) else {}, stored_values))


def capabilities_for_client_entry(caps: dict) -> dict:
    """把能力字典投影为 /v1/models 条目字段（未知字段不输出）。

    客户端契约：supports_vision/supports_tools 只在做正向声明时输出，
    "字段缺失 = 未知或不支持"。管理员选"不支持"的语义是不向客户端
    声明该能力（而非输出显式 false），与既有 harness 兼容行为一致。
    """
    entry: dict[str, Any] = {}
    if caps.get("context_window"):
        entry["context_window"] = caps["context_window"]
        # OpenRouter 风格别名，部分 harness 读这个名字
        entry["context_length"] = caps["context_window"]
    if caps.get("max_output_tokens"):
        entry["max_output_tokens"] = caps["max_output_tokens"]
    if caps.get("supports_vision"):
        entry["supports_vision"] = True
    if caps.get("supports_tools"):
        entry["supports_tools"] = True
    if caps.get("input_modalities"):
        entry["input_modalities"] = list(caps["input_modalities"])
    if caps.get("pricing"):
        entry["pricing"] = dict(caps["pricing"])
    return entry
