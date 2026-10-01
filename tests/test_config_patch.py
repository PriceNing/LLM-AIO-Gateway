"""config.json 写回层的行为测试。

核心契约：设置页只能"改指定的键"，绝不能把内置默认整表固化进文件——否则
`_fill_defaults()` 的 setdefault 对这些键永久失效，网关升级带来的新默认值
再也跟不上（文件方案最容易踩的坑）。
"""

import json

import pytest

from app.config import (
    ConfigManager,
    default_config,
    defaults_written_on_disk,
    get_config,
    get_default,
    load_config,
)


@pytest.fixture
def gateway_config(tmp_path):
    """一个"最小文件"的配置：defaults 段只显式写了 max_tokens。"""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "host": "127.0.0.1",
        "port": 8000,
        "database": str(tmp_path / "data.db"),
        "defaults": {"max_tokens": 1000},
    }), encoding="utf-8")
    cfg = load_config(str(path), force_reload=True)
    yield cfg
    load_config(str(tmp_path / "teardown.json"), force_reload=True)


def _disk(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_patch_writes_only_touched_keys(gateway_config):
    gateway_config.patch("defaults", {"temperature": 0.3})
    on_disk = _disk(gateway_config.path)["defaults"]
    assert on_disk == {"max_tokens": 1000, "temperature": 0.3}
    # 关键回归：内置默认没有被整表固化进文件
    assert "max_request_body_bytes" not in on_disk
    assert len(on_disk) < len(default_config()["defaults"])


def test_patch_takes_effect_immediately(gateway_config):
    gateway_config.patch("defaults", {"temperature": 0.3})
    assert get_default("temperature", 0.7) == 0.3
    assert get_config().config["defaults"]["temperature"] == 0.3


def test_patch_preserves_external_hand_edits(gateway_config):
    """SSH 手改磁盘后，界面写另一个键不得覆盖它（patch 前重读磁盘）。"""
    on_disk = _disk(gateway_config.path)
    on_disk["defaults"]["tool_only_limit"] = 5
    gateway_config.path.write_text(json.dumps(on_disk), encoding="utf-8")
    # 内存里还停留在旧内容，模拟"管理员改了文件但进程没重载"
    gateway_config.patch("defaults", {"temperature": 0.1})
    saved = _disk(gateway_config.path)["defaults"]
    assert saved["tool_only_limit"] == 5
    assert saved["temperature"] == 0.1


def test_patch_remove_deletes_key(gateway_config):
    gateway_config.patch("defaults", {"max_tokens": 4096})
    assert _disk(gateway_config.path)["defaults"]["max_tokens"] == 4096
    # 重置为默认 = 删键，而不是写回内置默认值
    gateway_config.patch("defaults", {}, remove=("max_tokens",))
    assert "max_tokens" not in _disk(gateway_config.path)["defaults"]
    assert get_default("max_tokens", 16384) == 16384


def test_patch_noop_does_not_touch_file(gateway_config):
    before = gateway_config.path.read_text(encoding="utf-8")
    gateway_config.patch("defaults", {})
    assert gateway_config.path.read_text(encoding="utf-8") == before


def test_patch_refuses_corrupt_file_without_clobbering(gateway_config):
    gateway_config.path.write_text("{ this is not json", encoding="utf-8")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        gateway_config.patch("defaults", {"temperature": 0.2})
    # 坏文件必须原样保留，不能被"修好"成只剩本次改动键的文件
    assert gateway_config.path.read_text(encoding="utf-8") == "{ this is not json"


def test_patch_creates_missing_defaults_section(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"host": "127.0.0.1"}), encoding="utf-8")
    cfg = load_config(str(path), force_reload=True)
    cfg.patch("defaults", {"temperature": 0.5})
    assert _disk(path)["defaults"] == {"temperature": 0.5}
    assert _disk(path)["host"] == "127.0.0.1"


def test_patch_top_level_section(gateway_config):
    gateway_config.patch("", {"port": 9100})
    assert _disk(gateway_config.path)["port"] == 9100


def test_file_status_reports_path_and_writability(gateway_config):
    status = gateway_config.file_status()
    assert status["path"] == str(gateway_config.path)
    assert status["exists"] is True
    assert status["writable"] is True
    assert status["mtime"] > 0


def test_file_status_missing_file(tmp_path):
    cfg = ConfigManager(str(tmp_path / "absent.json"))
    status = cfg.file_status()
    assert status["exists"] is False


def test_defaults_written_on_disk_excludes_builtin_fill(gateway_config):
    written = defaults_written_on_disk()
    assert written == {"max_tokens": 1000}
    # 内存里是补齐过的（get_default 才能拿到全部键），两者不能混为一谈
    assert len(gateway_config.config["defaults"]) > len(written)


def test_default_config_declares_all_budget_keys():
    defaults = default_config()["defaults"]
    # 这三个键代码一直在读，之前没在默认值里声明，属于"隐形配置"
    assert defaults["image_generation_budget_limit"] == 20
    assert defaults["image_generation_budget_ttl"] == 3600
    assert defaults["image_generation_budget_max_size"] == 1000
    assert defaults["reasoning_max_tokens"] == 32768
    assert defaults["max_request_body_bytes"] == 128 * 1024 * 1024
