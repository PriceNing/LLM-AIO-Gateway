import pytest
from app.config import ConfigManager
from app.services.logger import LogManager


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


def test_logger_recreates_missing_log_dir(tmp_path):
    log_root = tmp_path / "logs"
    mgr = LogManager()
    mgr.configure({
        "enabled": True,
        "level": "INFO",
        "log_dir": str(log_root),
        "retention_days": 30,
        "console": False,
    })
    day_dir = log_root / __import__("datetime").datetime.utcnow().strftime("%Y-%m-%d")
    assert day_dir.exists()
    for item in day_dir.iterdir():
        item.unlink()
    day_dir.rmdir()
    log_root.rmdir()

    logger = mgr.get_logger("app")
    logger.info("hello")

    assert day_dir.exists()
    assert (day_dir / "app.log").exists()
