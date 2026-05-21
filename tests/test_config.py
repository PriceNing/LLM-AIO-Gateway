import pytest
from app.config import ConfigManager


@pytest.fixture
def temp_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"host": "0.0.0.0", "port": 8000, "database": "test.db", "logging": {"enabled": false, "level": "INFO", "log_dir": "logs", "retention_days": 30, "console": false}}', encoding="utf-8")
    yield path


def test_config_manager_load(temp_config):
    manager = ConfigManager(str(temp_config))
    manager.load()
    assert manager.config["host"] == "0.0.0.0"
    assert manager.config["port"] == 8000
    assert manager.config["database"] == "test.db"


def test_missing_config_is_created(tmp_path):
    path = tmp_path / "new-config.json"
    manager = ConfigManager(str(path))
    manager.load()
    assert path.exists()
    assert manager.config["host"] == "0.0.0.0"


def test_defaults_filled(tmp_path):
    path = tmp_path / "partial.json"
    path.write_text('{"host": "127.0.0.1"}', encoding="utf-8")
    manager = ConfigManager(str(path))
    manager.load()
    assert manager.config["host"] == "127.0.0.1"
    assert manager.config["port"] == 8000  # default
    assert "logging" in manager.config
