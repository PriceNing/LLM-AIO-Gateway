"""入站请求体大小限制。

网关会代理含 data URI 图片的多模态请求，请求体天然偏大；但没有任何上限时，
单个有效 API Key 就能用任意大的 JSON body 把进程内存打满（见「当前问题.md」S4）。
这里同时覆盖两种客户端：

* 带 ``Content-Length`` 的请求：读取前直接拒绝，不产生任何缓冲开销。
* 分块（chunked）请求：在 ``receive`` 通道上累计字节数，超限即中断。
"""

from __future__ import annotations

import json

from starlette.responses import Response

from app.config import get_default

# 多模态请求包含 base64 图片，默认给到 32 MiB；可通过 config.json 调整。
DEFAULT_MAX_REQUEST_BODY_BYTES = 32 * 1024 * 1024

_BODY_LIMIT_METHODS = frozenset({"POST", "PUT", "PATCH"})


class RequestBodyTooLarge(Exception):
    """Raised when a streamed request body crosses the configured limit."""


def max_request_body_bytes() -> int:
    """Resolve the configured limit, clamped to a sane range."""
    try:
        configured = int(get_default("max_request_body_bytes", DEFAULT_MAX_REQUEST_BODY_BYTES))
    except (TypeError, ValueError):
        configured = DEFAULT_MAX_REQUEST_BODY_BYTES
    return max(1024, configured)


def _oversized_response(limit: int) -> Response:
    payload = {
        "error": {
            "message": f"Request body too large. Limit is {limit} bytes.",
            "type": "invalid_request_error",
            "code": "request_body_too_large",
        }
    }
    return Response(
        content=json.dumps(payload),
        status_code=413,
        media_type="application/json",
    )


def _declared_length(scope: dict) -> int | None:
    for name, value in scope.get("headers") or []:
        if name == b"content-length":
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


class RequestBodyLimitMiddleware:
    """Pure-ASGI body cap; avoids BaseHTTPMiddleware buffering of the request."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("method") not in _BODY_LIMIT_METHODS:
            await self.app(scope, receive, send)
            return

        limit = max_request_body_bytes()
        declared = _declared_length(scope)
        if declared is not None and declared > limit:
            await _oversized_response(limit)(scope, receive, send)
            return

        state = {"sent": False}
        consumed = 0

        async def limited_receive():
            nonlocal consumed
            message = await receive()
            if message.get("type") == "http.request":
                consumed += len(message.get("body") or b"")
                if consumed > limit:
                    raise RequestBodyTooLarge(f"request body exceeded {limit} bytes")
            return message

        async def tracked_send(message):
            if message.get("type") == "http.response.start":
                state["sent"] = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except RequestBodyTooLarge:
            if state["sent"]:
                # 响应已开始，无法再改写状态码；向上抛出交由服务端断开连接。
                raise
            await _oversized_response(limit)(scope, receive, send)
