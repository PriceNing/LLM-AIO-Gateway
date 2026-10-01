"""把配置当前值推进到"结构上定死"的运行时对象。

大多数 `defaults` 键是每次调用现读（`get_default`），写完立即生效。但有几类值不是
读出来的，而是被固化在对象里：

* `core.state` 的 TTLDict —— ttl / max_size 在构造时定死；
* `router.proxy` 的内存请求日志 —— `deque.maxlen` 只读，只能重建；
* `services.lite_llm` —— `litellm.request_timeout` 是全局模块属性。

这些对象注册一个 reconfigure 钩子到这里，配置写入（管理页 PUT）或
`/admin/config/reload` 之后统一调用 `apply_runtime_config()` 推送新值。钩子在导入时
注册，因此本模块不得导入任何业务模块——反向导入会形成循环。

单进程部署（`main.py` 的 uvicorn 无 workers、Dockerfile 单容器）下，写进程就是读进程，
不需要 TTL 或版本号广播。
"""

from __future__ import annotations

from typing import Callable

from app.config import get_config
from app.services.logger import get_logger

_app_log = get_logger("app")

# 钩子签名：接收整份配置 dict，自行取需要的段并推送进自己的对象。
ReconfigureHook = Callable[[dict], None]

_HOOKS: list[tuple[str, ReconfigureHook]] = []


def register_runtime_hook(name: str, hook: ReconfigureHook) -> None:
    """注册一个 reconfigure 钩子；同名重复注册以最后一次为准（便于测试与热重载）。"""
    _HOOKS[:] = [(existing, fn) for existing, fn in _HOOKS if existing != name]
    _HOOKS.append((name, hook))


def registered_hooks() -> list[str]:
    return [name for name, _ in _HOOKS]


def apply_runtime_config() -> dict[str, str]:
    """把当前配置推送到所有已注册对象，返回每个钩子的执行结果说明。

    单个钩子失败不影响其它钩子：配置界面必须能报告"哪些子系统没跟上"，而不是
    让一半的值静默停留在旧状态。
    """
    config = get_config().config
    results: dict[str, str] = {}
    for name, hook in list(_HOOKS):
        try:
            hook(config)
            results[name] = "applied"
        except Exception as exc:  # noqa: BLE001 - 钩子失败必须降级为报告，不能中断请求
            results[name] = f"failed: {type(exc).__name__}: {exc}"
            _app_log.warning("[runtime_config] hook '%s' failed: %s", name, exc)
    return results
