"""设置 schema 的契约测试。

核心承诺：条目由 default_config() 生成，配置文件 / 界面 / API 不会互相漂移；
校验在服务端强制，界面只是提前提示。
"""

import pytest

from app.config import default_config
from app.core import settings_schema
from app.core.settings_schema import SettingsValidationError, schema, validate, validate_values


def test_schema_covers_every_default_key():
    """新增配置键自动进入设置页；漏掉一个就是界面隐藏了一个真实生效的开关。"""
    declared = set(default_config()["defaults"])
    assert set(schema()) == declared
    assert "max_request_body_bytes" in declared
    # 曾经"隐形"的键：代码在读但没在默认值里声明过
    assert {"image_generation_budget_limit", "image_generation_budget_ttl",
            "image_generation_budget_max_size"} <= declared


def test_schema_entries_are_complete_and_hot():
    for key, entry in schema().items():
        assert entry["key"] == key
        assert entry["type"] in ("int", "float", "bool", "string", "url", "string_list", "json_object")
        assert entry["group"], key
        assert entry["unit"], key
        # 固化在对象里的值已由 core.runtime_config 的钩子推送，不再有"需重启"的 defaults 键
        assert entry["hot"] is True, key
        assert entry["default"] == default_config()["defaults"][key]


def test_schema_groups_match_declared_groups():
    known = {gid for gid, _ in settings_schema.GROUPS} | {"maintenance"}
    for key, entry in schema().items():
        assert entry["group"] in known, key
    # 每个分组标签都是 i18n 键，界面按它取文案
    for gid, label in settings_schema.GROUPS:
        assert label.startswith("settings.group.")
        assert gid in known


def test_type_inference():
    assert schema()["max_tokens"]["type"] == "int"
    assert schema()["temperature"]["type"] == "float"
    assert schema()["repair_tool_leaks"]["type"] == "bool"
    assert schema()["model_registry_url"]["type"] == "url"
    assert schema()["request_log_redact_fields"]["type"] == "string_list"


def test_int_rejects_bool_because_bool_is_int_subclass():
    with pytest.raises(SettingsValidationError):
        validate("max_tokens", True)
    assert validate("max_tokens", 2048) == 2048
    assert validate("max_tokens", "2048") == 2048


def test_int_range_enforced():
    assert validate("temperature", 0) == 0.0
    with pytest.raises(SettingsValidationError, match="不得大于"):
        validate("temperature", 2.5)
    with pytest.raises(SettingsValidationError, match="不得小于"):
        validate("login_attempt_limit", 0)
    with pytest.raises(SettingsValidationError, match="需要整数"):
        validate("max_tokens", 1.5)


def test_zero_means_disable_and_is_allowed_where_declared():
    """0 在多个键上有"关闭"语义，不能当非法值挡掉，也不能静默改回默认。"""
    assert validate("max_request_body_bytes", 0) == 0
    assert validate("max_tokens", 0) == 0
    assert validate("tool_only_limit", 0) == 0
    # request_log_max 例外：0 会让内存滚动日志为空，界面与 API 都不允许
    with pytest.raises(SettingsValidationError):
        validate("request_log_max", 0)


def test_bool_parsing():
    assert validate("repair_tool_leaks", False) is False
    assert validate("repair_tool_leaks", "true") is True
    assert validate("repair_tool_leaks", "False") is False
    with pytest.raises(SettingsValidationError):
        validate("repair_tool_leaks", 1)


def test_string_list_normalisation():
    value = validate("request_log_redact_fields", ["api_key", " authorization ", "", "api_key"])
    assert value == ["api_key", "authorization"]
    assert validate("request_log_redact_fields", "a, b ,a") == ["a", "b"]
    with pytest.raises(SettingsValidationError):
        validate("request_log_redact_fields", [str(index) for index in range(100)])


def test_url_validation_goes_through_url_guard():
    assert validate("model_registry_url", "https://openrouter.ai/api/v1/models").startswith("https://")
    for bad in ("http://169.254.169.254/latest/meta-data/", "file:///etc/passwd", "ftp://x/y", ""):
        with pytest.raises(SettingsValidationError):
            validate("model_registry_url", bad)


def test_unknown_key_rejected():
    """schema 之外的键一律拒绝：否则等于给配置文件开任意写入口。"""
    with pytest.raises(SettingsValidationError, match="不是已知设置项"):
        validate("arbitrary_key", 1)
    with pytest.raises(SettingsValidationError):
        validate_values({"max_tokens": 1, "nope": 2})


def test_validate_values_is_all_or_nothing():
    """一批里有一条非法就整批报错，不做半批写入。"""
    with pytest.raises(SettingsValidationError) as exc:
        validate_values({"max_tokens": 99_999_999, "temperature": 2.5})
    message = str(exc.value)
    assert "max_tokens" in message
    assert "temperature" in message
    normalized = validate_values({"max_tokens": 2048, "temperature": 0.5})
    assert normalized == {"max_tokens": 2048, "temperature": 0.5}


def test_danger_rules_are_value_dependent():
    assert settings_schema.danger_for("max_request_body_bytes", 128 * 1024 * 1024) is None
    assert settings_schema.danger_for("max_request_body_bytes", 0) == "settings.danger.bodyLimitOff"
    assert settings_schema.danger_for("allow_private_upstream_hosts", True) == "settings.danger.privateUpstream"
    assert settings_schema.danger_for("allow_private_upstream_hosts", False) is None
    assert settings_schema.danger_for("request_log_capture_payloads", False) == "settings.danger.payloadCaptureOff"
    assert settings_schema.danger_for("repair_tool_leaks", True) is None
    assert settings_schema.danger_for("tool_only_limit", 0) == "settings.danger.toolOnlyLimitOff"
    # 非数字输入不得让判定抛异常
    assert settings_schema.danger_for("max_request_body_bytes", "junk") is None


def test_empty_list_danger_rule():
    """脱敏字段清空 = 关闭保护，凭据会明文进请求日志，必须跟其他风险项同样弹确认。"""
    assert settings_schema.danger_for("request_log_redact_fields", ["api_key"]) is None
    assert settings_schema.danger_for("request_log_redact_fields", []) == "settings.danger.redactEmpty"
    # 清空是合法输入（不是校验错误），只是需要确认
    assert validate("request_log_redact_fields", []) == []


def test_danger_keys_have_i18n_labels():
    for key, entry in schema().items():
        rule = entry.get("danger")
        if isinstance(rule, dict):
            assert rule["key"].startswith("settings.danger."), key
            assert rule["when"] in ("lte", "gte", "true", "false", "empty"), key


def test_known_keys_sorted():
    keys = settings_schema.known_keys()
    assert keys == sorted(keys)
    assert len(keys) == len(schema())


def test_schema_is_cached():
    """validate_values 对每个键调一次 schema()，不缓存就是 N+1 次全量构建。"""
    assert schema() is schema()


def test_every_schema_i18n_key_has_a_translation():
    """服务端下发的 hint / danger / group 键必须在 app.js 里有译文。

    这些键是运行时拼出来的，界面无法靠静态扫描发现漏翻译；漏一个就直出裸键名。
    """
    from pathlib import Path

    import app as app_pkg

    js = (Path(app_pkg.__file__).parent / "web" / "static" / "app.js").read_text(encoding="utf-8")
    required = {label for _, label in settings_schema.GROUPS}
    required.add("settings.group.other")
    for entry in schema().values():
        if entry["hint"]:
            required.add(entry["hint"])
        rule = entry.get("danger")
        if isinstance(rule, dict) and rule.get("key"):
            required.add(rule["key"])
    missing = sorted(key for key in required if f"'{key}'" not in js)
    assert not missing, f"app.js 缺少设置页译文：{missing}"
