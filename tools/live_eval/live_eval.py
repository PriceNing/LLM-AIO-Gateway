#!/usr/bin/env python
"""
Live conformance evaluator for an already deployed LLM AIO Gateway.

This script intentionally talks to a real gateway and real upstream models. It is
not part of the default pytest suite because it can spend tokens and depends on
the deployed server configuration.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import sys
import time
import traceback
import urllib.error
import urllib.request
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def probe_image_base64() -> str:
    """生成 64x64 RGB 渐变 PNG（仅标准库）。

    注意：很多上游（如 DeepSeek）会把 1x1 占位图判为无效图片并报 400，
    探针图必须是真实可解码、尺寸合理的图像。
    """
    width = height = 64
    rows = b"".join(
        b"\x00" + b"".join(bytes(((x * 4) % 256, (y * 4) % 256, 128)) for x in range(width))
        for y in range(height)
    )
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(rows, 9))
        + _png_chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode()


PROBE_IMAGE_PNG_B64 = probe_image_base64()

# expect 值中属于"能力探针"的用例：上游拒绝时判定为 unsupported 而非网关故障
CAPABILITY_EXPERTS = {
    "chat_tool",
    "messages_tool",
    "responses_tool",
}

# 不得判为“能力不支持”的 4xx：401/403=网关侧凭据/授权问题（冒烟配置错误），
# 404/405/410=上游“模型/端点不存在”类存在性错误（配置问题，网关代理路径自身不产生 404），
# 408=超时，429=瞬时限流。只有 400/422 这类“请求内容被拒”才是能力缺失信号。
UNSUPPORTED_REJECTION_EXCLUDED = (401, 403, 404, 405, 408, 410, 429)


def is_unsupported_rejection(status: int | None, expect: str, capability_probe: bool) -> bool:
    """仅上游明确以 4xx 拒绝能力探针时计为 unsupported；5xx/429 等仍是故障。"""
    if status is None or not (400 <= status <= 499):
        return False
    if status in UNSUPPORTED_REJECTION_EXCLUDED:
        return False
    return expect in CAPABILITY_EXPERTS or capability_probe

@dataclass
class CaseResult:
    name: str
    endpoint: str
    ok: bool
    score: float
    status: int | None = None
    latency_ms: int = 0
    request_id: str = ""
    response_id: str = ""
    summary: str = ""
    error: str = ""
    response_excerpt: str = ""
    log_entries: list[dict[str, Any]] = field(default_factory=list)
    judge: dict[str, Any] = field(default_factory=dict)
    # pass/fail 计入网关得分；skip（能力未声明）与 unsupported（上游拒绝能力探针）不计入
    verdict: str = "fail"
    # 上游协议断言：provider_type 可推得的预期上游（多个用 | 分隔）与实际观测到的上游
    expected_upstream: str = ""
    observed_upstream: str = ""


@dataclass
class ModelResult:
    model: str
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def scored_cases(self) -> list[CaseResult]:
        return [case for case in self.cases if case.verdict in ("pass", "fail")]

    @property
    def score(self) -> float:
        scored = self.scored_cases
        if not scored:
            return 1.0
        return round(sum(case.score for case in scored) / len(scored), 2)

    @property
    def passed(self) -> int:
        return sum(1 for case in self.cases if case.verdict == "pass")

    @property
    def no_signal(self) -> bool:
        """全部用例被 skip/unsupported：本次测试对该模型没有任何有效信号。"""
        return bool(self.cases) and not self.scored_cases


# 客户端端点 -> 日志里的 endpoint 名
ENDPOINT_LOG_NAME = {
    "/v1/chat/completions": "chat_completions",
    "/v1/completions": "completions",
    "/v1/messages": "messages",
    "/v1/responses": "responses",
}

# 上游协议三列
UPSTREAM_PROTOCOLS = ("chat_completions", "messages", "responses")

# 可达的 客户端×上游 组合（原生 responses 上游仅 /responses 端点可触发）
REACHABLE_CELLS: tuple[tuple[str, str], ...] = (
    ("chat_completions", "chat_completions"),
    ("chat_completions", "messages"),
    ("completions", "chat_completions"),
    ("completions", "messages"),
    ("messages", "chat_completions"),
    ("messages", "messages"),
    ("responses", "chat_completions"),
    ("responses", "messages"),
    ("responses", "responses"),
)


def build_coverage_matrix(results: list[ModelResult]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {
        ep: {up: 0 for up in UPSTREAM_PROTOCOLS}
        for ep in ("chat_completions", "completions", "messages", "responses")
    }
    for result in results:
        for case in result.cases:
            if not case.observed_upstream:
                continue
            ep = ENDPOINT_LOG_NAME.get(case.endpoint, case.endpoint)
            row = matrix.setdefault(ep, {up: 0 for up in UPSTREAM_PROTOCOLS})
            row[case.observed_upstream] = row.get(case.observed_upstream, 0) + 1
    return matrix


def missing_reachable_cells(matrix: dict[str, dict[str, int]]) -> list[tuple[str, str]]:
    return [cell for cell in REACHABLE_CELLS if not matrix.get(cell[0], {}).get(cell[1])]


def print_coverage_matrix(matrix: dict[str, dict[str, int]]) -> None:
    print("\nProtocol coverage matrix (client endpoint -> observed upstream protocol):")
    print(f"  {'client \\ upstream':24}" + "".join(f"{up:>18}" for up in UPSTREAM_PROTOCOLS))
    for ep in ("chat_completions", "completions", "messages", "responses"):
        row = matrix.get(ep, {})
        cells = "".join(f"{(str(row.get(up, 0)) if row.get(up) else '-'):>18}" for up in UPSTREAM_PROTOCOLS)
        print(f"  {ep:24}{cells}")
    missing = missing_reachable_cells(matrix)
    if missing:
        print("  uncovered reachable cells: " + ", ".join(f"{ep}->{up}" for ep, up in missing))
        print("  (需配置对应 provider_type / 支持原生 Responses 的模型才能覆盖满 9 格)")


def expected_upstream_for(endpoint_name: str, provider_type: str) -> str:
    """由 provider_type 推导预期上游；"|" 表示多个均可接受。无法得知时返回空。"""
    if provider_type == "anthropic":
        return "messages"
    if provider_type == "openai":
        if endpoint_name == "responses":
            # 原生 supported 时走 responses，否则降级 chat_completions，两者均为合法路径
            return "responses|chat_completions"
        return "chat_completions"
    return ""


def observed_upstream_from_logs(entries: list[dict[str, Any]], endpoint_name: str) -> str:
    """从 case 关联的请求日志中取最新一条匹配端点的上游协议。"""
    best: dict[str, Any] | None = None
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("endpoint") != endpoint_name:
            continue
        if best is None or str(entry.get("full_time", "")) >= str(best.get("full_time", "")):
            best = entry
    if best is None:
        return ""
    details = best.get("details") if isinstance(best.get("details"), dict) else {}
    mode = str(best.get("responses_mode") or details.get("responses_mode") or "")
    upstream = str(best.get("upstream_endpoint") or details.get("upstream_endpoint") or "")
    if mode == "native" or upstream == "responses":
        return "responses"
    if upstream in ("chat_completions", "messages"):
        return upstream
    return ""


class GatewayClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float,
        admin_username: str = "",
        admin_password: str = "",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.admin_username = admin_username
        self.admin_password = admin_password
        self.admin_token = ""

    def request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        admin: bool = False,
        stream: bool = False,
    ) -> tuple[int, Any, int]:
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if admin:
            token = self.ensure_admin_token()
            headers["Authorization"] = f"Bearer {token}"
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = None
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self.base_url + path, data=data, headers=headers, method=method)
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                status = resp.status
                if stream:
                    payload = read_sse_stream(resp, self.timeout, started)
                    latency_ms = int((time.perf_counter() - started) * 1000)
                    return status, payload, latency_ms
                raw = resp.read()
                latency_ms = int((time.perf_counter() - started) * 1000)
                text = raw.decode("utf-8", errors="replace")
                try:
                    return status, json.loads(text), latency_ms
                except json.JSONDecodeError:
                    return status, text, latency_ms
        except urllib.error.HTTPError as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                payload: Any = json.loads(raw)
            except json.JSONDecodeError:
                payload = raw
            return exc.code, payload, latency_ms

    def ensure_admin_token(self) -> str:
        if self.admin_token:
            return self.admin_token
        if not self.admin_username or not self.admin_password:
            return ""
        status, payload, _ = self.request(
            "POST",
            "/auth/login",
            {"username": self.admin_username, "password": self.admin_password},
            admin=False,
        )
        if status == 200 and isinstance(payload, dict):
            self.admin_token = str(payload.get("token", ""))
        return self.admin_token

    def get_models(self) -> dict[str, dict[str, Any]]:
        """返回 model_id -> 能力元数据（supports_vision / supports_tools 等）。"""
        status, payload, _ = self.request("GET", "/v1/models")
        if status != 200 or not isinstance(payload, dict):
            raise RuntimeError(f"/v1/models failed: HTTP {status} {payload!r}")
        models: dict[str, dict[str, Any]] = {}
        for item in payload.get("data", []):
            if not isinstance(item, dict) or not item.get("id"):
                continue
            caps = {
                key: item.get(key)
                for key in ("supports_vision", "supports_tools", "context_window", "max_output_tokens")
                if key in item
            }
            models[str(item["id"])] = caps
        return models

    def get_provider_types(self) -> dict[str, str]:
        """model_id -> provider_type（需管理员凭据；不可用时返回空表）。"""
        if not (self.admin_username and self.admin_password):
            return {}
        status, payload, _ = self.request("GET", "/admin/models", admin=True)
        if status != 200 or not isinstance(payload, dict):
            return {}
        out: dict[str, str] = {}
        for item in payload.get("models", []):
            if isinstance(item, dict) and item.get("id"):
                out[str(item["id"])] = str(item.get("provider_type") or "")
        return out

    def reset_responses_capability(self, model_id: str) -> int | None:
        """清除指定模型的 Responses 能力探测缓存，使本次冒烟对原生路径做真实探测。

        返回受影响行数；无管理员凭据或端点不可用时返回 None。
        """
        if not (self.admin_username and self.admin_password):
            return None
        status, payload, _ = self.request(
            "POST", "/admin/models/responses-capability/reset", {"model": model_id}, admin=True,
        )
        if status == 200 and isinstance(payload, dict):
            return int(payload.get("reset", 0) or 0)
        return None

    def get_recent_logs(self, model: str, since_full_time: str = "") -> list[dict[str, Any]]:
        if not (self.admin_username and self.admin_password):
            return []
        status, payload, _ = self.request("GET", "/admin/stats", admin=True)
        if status != 200 or not isinstance(payload, dict):
            return []
        entries = []
        for entry in payload.get("request_log", []):
            if not isinstance(entry, dict):
                continue
            if entry.get("requested_model") != model and entry.get("model") != model:
                continue
            if since_full_time and str(entry.get("full_time", "")) < since_full_time:
                continue
            entries.append(entry)
        return entries[:20]


def parse_sse(text: str) -> dict[str, Any]:
    events = []
    done = False
    for block in text.replace("\r\n", "\n").split("\n\n"):
        if not block.strip():
            continue
        event_name = ""
        data_lines = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if not data_lines:
            continue
        data = "\n".join(data_lines)
        if data == "[DONE]":
            done = True
            events.append({"event": event_name or "done", "data": "[DONE]"})
            continue
        try:
            parsed: Any = json.loads(data)
        except json.JSONDecodeError:
            parsed = data
        events.append({"event": event_name, "data": parsed})
    return {"events": events, "done": done, "raw_excerpt": text[:2000]}


def read_sse_stream(resp, timeout: float, started: float) -> dict[str, Any]:
    events = []
    done = False
    raw_parts = []
    block_lines = []
    first_event_ms = 0

    def flush_block() -> None:
        nonlocal done, first_event_ms, block_lines
        if not block_lines:
            return
        event_name = ""
        data_lines = []
        for raw_line in block_lines:
            line = raw_line.rstrip("\r\n")
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
        block_lines = []
        if not data_lines:
            return
        data = "\n".join(data_lines)
        if first_event_ms == 0:
            first_event_ms = int((time.perf_counter() - started) * 1000)
        if data == "[DONE]":
            done = True
            events.append({"event": event_name or "done", "data": "[DONE]"})
            return
        try:
            parsed: Any = json.loads(data)
        except json.JSONDecodeError:
            parsed = data
        events.append({"event": event_name, "data": parsed})

    while True:
        if timeout and time.perf_counter() - started > timeout:
            raise TimeoutError(f"SSE stream exceeded {timeout:.1f}s")
        raw = resp.readline()
        if not raw:
            flush_block()
            break
        line = raw.decode("utf-8", errors="replace")
        if sum(len(part) for part in raw_parts) < 2000:
            raw_parts.append(line)
        if line in ("\n", "\r\n"):
            flush_block()
            if done:
                break
            continue
        block_lines.append(line)

    return {
        "events": events,
        "done": done,
        "raw_excerpt": "".join(raw_parts)[:2000],
        "first_event_ms": first_event_ms,
    }


def compact_json(value: Any, limit: int = 1200) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def text_from_chat(payload: Any) -> str:
    try:
        return str(payload["choices"][0]["message"].get("content") or "")
    except Exception:
        return ""


def text_from_completion(payload: Any) -> str:
    try:
        return str(payload["choices"][0].get("text") or "")
    except Exception:
        return ""


def text_from_messages(payload: Any) -> str:
    try:
        parts = payload.get("content", [])
        return "\n".join(str(part.get("text", "")) for part in parts if part.get("type") == "text")
    except Exception:
        return ""


def text_from_responses(payload: Any) -> str:
    chunks = []
    for item in payload.get("output", []) if isinstance(payload, dict) else []:
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if part.get("type") == "output_text":
                chunks.append(str(part.get("text", "")))
    return "\n".join(chunks)


def extract_response_id(payload: Any) -> str:
    return str(payload.get("id", "")) if isinstance(payload, dict) else ""


def has_tool_call_chat(payload: Any) -> bool:
    try:
        return bool(payload["choices"][0]["message"].get("tool_calls"))
    except Exception:
        return False


def has_tool_call_messages(payload: Any) -> bool:
    try:
        return any(part.get("type") == "tool_use" for part in payload.get("content", []))
    except Exception:
        return False


def has_tool_call_responses(payload: Any) -> bool:
    try:
        return any(item.get("type") == "function_call" for item in payload.get("output", []))
    except Exception:
        return False


def has_stream_events(parsed: Any, names: set[str]) -> bool:
    if not isinstance(parsed, dict):
        return False
    for event in parsed.get("events", []):
        data = event.get("data")
        if isinstance(data, dict) and data.get("type") in names:
            return True
        if event.get("event") in names:
            return True
    return False


def skip_case(name: str, endpoint: str, reason: str) -> CaseResult:
    """未声明对应能力时不真实请求，记录为 skip，不计入得分。"""
    return CaseResult(name=name, endpoint=endpoint, ok=False, score=0.0, summary=reason, verdict="skip")


def make_case(
    *,
    client: GatewayClient,
    model: str,
    name: str,
    endpoint: str,
    body: dict[str, Any],
    expect: str,
    stream: bool = False,
    capability_probe: bool = False,
    provider_type: str = "",
) -> CaseResult:
    request_id = f"live-eval-{int(time.time())}-{abs(hash((model, name))) % 100000}"
    body = dict(body)
    body["model"] = model
    body.setdefault("temperature", 0)
    # 预算字段按协议区分：max_tokens 是 Chat 风格字段，发到原生 /responses 会被
    # 上游以 400 拒绝（tools 请求尤其严格），还会连带把能力缓存打成 unknown。
    # 预算取 512（Chat/messages）/ 2048（responses）：推理模型的 reasoning 也吃
    # completion 预算，太小会把 content 截成空串误判为网关故障（deepseek 续轮
    # 实测 512 会被 thinking 吃满导致 incomplete/max_output_tokens 空输出）。
    if endpoint == "/v1/responses":
        budget = body.pop("max_tokens", 2048)
        body.setdefault("max_output_tokens", budget)
    else:
        body.setdefault("max_tokens", 512)
    body.setdefault("metadata", {})
    if isinstance(body["metadata"], dict):
        body["metadata"]["live_eval_request_id"] = request_id
    started = time.strftime("%Y-%m-%dT%H:%M:%S")

    try:
        status, payload, latency_ms = client.request("POST", endpoint, body, stream=stream)
        ok, score, summary = evaluate_payload(status, payload, expect)
        verdict = "pass" if ok else "fail"
        # 能力探针被上游以 4xx 拒绝才判 unsupported；5xx/504 是网关/基础设施故障，
        # 429 是瞬时限流，401/403 在本网关语义下是服务端凭据问题，均不得洗成“模型不支持”。
        if not ok and is_unsupported_rejection(status, expect, capability_probe):
            verdict = "unsupported"
            summary += " (upstream rejected capability probe)"
        log_entries = client.get_recent_logs(model, started)
        endpoint_name = ENDPOINT_LOG_NAME.get(endpoint, endpoint)
        expected = expected_upstream_for(endpoint_name, provider_type)
        observed = observed_upstream_from_logs(log_entries, endpoint_name) if verdict == "pass" else ""
        if verdict == "pass" and expected and observed and observed not in expected.split("|"):
            # 实际走了错误适配器 = 路由/协议层回归，必须硬失败
            verdict = "fail"
            ok = False
            score = 0.0
            summary += f" [upstream mismatch: expected {expected}, got {observed}]"
        return CaseResult(
            name=name,
            endpoint=endpoint,
            ok=ok,
            score=score,
            status=status,
            latency_ms=latency_ms,
            request_id=request_id,
            response_id=extract_response_id(payload),
            summary=summary,
            verdict=verdict,
            expected_upstream=expected,
            observed_upstream=observed,
            response_excerpt=compact_json(payload),
            log_entries=log_entries,
        )
    except Exception as exc:
        return CaseResult(
            name=name,
            endpoint=endpoint,
            ok=False,
            score=0.0,
            request_id=request_id,
            verdict="fail",
            error=f"{type(exc).__name__}: {exc}",
            response_excerpt=traceback.format_exc(limit=4),
            log_entries=client.get_recent_logs(model, started),
        )


def evaluate_payload(status: int, payload: Any, expect: str) -> tuple[bool, float, str]:
    if status < 200 or status >= 300:
        return False, 0.0, f"HTTP {status}"
    if expect == "chat_text":
        text = text_from_chat(payload)
        return bool(text.strip()), 1.0 if text.strip() else 0.4, f"chat text chars={len(text)}"
    if expect == "completion_text":
        text = text_from_completion(payload)
        return bool(text.strip()), 1.0 if text.strip() else 0.4, f"completion text chars={len(text)}"
    if expect == "messages_text":
        text = text_from_messages(payload)
        return bool(text.strip()), 1.0 if text.strip() else 0.4, f"messages text chars={len(text)}"
    if expect == "responses_text":
        text = text_from_responses(payload)
        return bool(text.strip()) and bool(extract_response_id(payload)), 1.0 if text.strip() else 0.4, f"responses text chars={len(text)}"
    if expect == "chat_tool":
        ok = has_tool_call_chat(payload)
        return ok, 1.0 if ok else 0.2, "chat tool_call present" if ok else "chat tool_call missing"
    if expect == "messages_tool":
        ok = has_tool_call_messages(payload)
        return ok, 1.0 if ok else 0.2, "messages tool_use present" if ok else "messages tool_use missing"
    if expect == "responses_tool":
        ok = has_tool_call_responses(payload)
        return ok, 1.0 if ok else 0.2, "responses function_call present" if ok else "responses function_call missing"
    if expect == "chat_stream":
        ok = isinstance(payload, dict) and payload.get("done") and has_stream_events(payload, {"chat.completion.chunk"})
        if not ok and isinstance(payload, dict):
            ok = payload.get("done") and bool(payload.get("events"))
        return bool(ok), 1.0 if ok else 0.2, "chat stream ended" if ok else "chat stream invalid"
    if expect == "messages_stream":
        ok = has_stream_events(payload, {"message_start", "message_stop"}) or has_stream_events(payload, {"message_stop"})
        return bool(ok), 1.0 if ok else 0.2, "messages stream events present" if ok else "messages stream invalid"
    if expect == "responses_stream":
        ok = has_stream_events(payload, {"response.created", "response.completed"})
        return bool(ok), 1.0 if ok else 0.2, "responses stream events present" if ok else "responses stream invalid"
    return True, 1.0, "HTTP success"


def build_cases(
    client: GatewayClient,
    model: str,
    include_multimodal: bool,
    include_stream: bool,
    caps: dict[str, Any] | None = None,
    ignore_capabilities: bool = False,
    provider_type: str = "",
) -> list[CaseResult]:
    caps = caps or {}
    gate = not ignore_capabilities
    supports_tools = bool(caps.get("supports_tools"))
    supports_vision = bool(caps.get("supports_vision"))
    cases: list[CaseResult] = []
    cases.append(make_case(
        client=client,
        model=model,
        provider_type=provider_type,
        name="chat_multi_turn",
        endpoint="/v1/chat/completions",
        expect="chat_text",
        body={
            "messages": [
                {"role": "system", "content": "You are a concise API conformance test assistant."},
                {"role": "user", "content": "Remember the word quartz. Reply only: remembered."},
                {"role": "assistant", "content": "remembered"},
                {"role": "user", "content": "What word did I ask you to remember? Reply with the word only."},
            ],
        },
    ))
    cases.append(make_case(
        client=client,
        model=model,
        provider_type=provider_type,
        name="completions_text",
        endpoint="/v1/completions",
        expect="completion_text",
        body={"prompt": "Return exactly one short sentence about automated endpoint testing."},
    ))
    cases.append(make_case(
        client=client,
        model=model,
        provider_type=provider_type,
        name="messages_multi_turn",
        endpoint="/v1/messages",
        expect="messages_text",
        body={
            "system": "You are a concise API conformance test assistant.",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Remember the number 2468. Reply ok."}]},
                {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
                {"role": "user", "content": [{"type": "text", "text": "What number did I ask you to remember? Reply with digits only."}]},
            ],
        },
    ))
    first_response = make_case(
        client=client,
        model=model,
        provider_type=provider_type,
        name="responses_first_turn",
        endpoint="/v1/responses",
        expect="responses_text",
        body={
            "instructions": "You are a concise API conformance test assistant.",
            "input": "Remember the token azure-42. Reply ok.",
        },
    )
    cases.append(first_response)
    previous_id = first_response.response_id
    response_body: dict[str, Any] = {"input": "What token did I ask you to remember? Reply with the token only."}
    if previous_id:
        response_body["previous_response_id"] = previous_id
    cases.append(make_case(
        client=client,
        model=model,
        provider_type=provider_type,
        name="responses_followup",
        endpoint="/v1/responses",
        expect="responses_text",
        body=response_body,
    ))
    tool_probes = (
        ("chat_tool_call", "/v1/chat/completions", "chat_tool", {
            "messages": [{"role": "user", "content": "Call the lookup_order tool for order_id A123."}],
            "tools": [tool_schema_chat()],
            "tool_choice": {"type": "function", "function": {"name": "lookup_order"}},
        }),
        ("messages_tool_call", "/v1/messages", "messages_tool", {
            "messages": [{"role": "user", "content": "Use lookup_order for order_id A123."}],
            "tools": [tool_schema_anthropic()],
            "tool_choice": {"type": "tool", "name": "lookup_order"},
        }),
        ("responses_tool_call", "/v1/responses", "responses_tool", {
            "input": "Use lookup_order for order_id A123.",
            "tools": [tool_schema_responses()],
            "tool_choice": {"type": "function", "name": "lookup_order"},
        }),
    )
    if gate and not supports_tools:
        for name, endpoint, _expect, _body in tool_probes:
            cases.append(skip_case(name, endpoint, "skipped: model does not advertise supports_tools"))
    else:
        for name, endpoint, expect, body in tool_probes:
            cases.append(make_case(
                client=client,
                model=model,
                provider_type=provider_type,
                name=name,
                endpoint=endpoint,
                expect=expect,
                body=body,
            ))
    if include_multimodal:
        if gate and not supports_vision:
            for name, endpoint in (
                ("chat_multimodal", "/v1/chat/completions"),
                ("messages_multimodal", "/v1/messages"),
                ("responses_multimodal", "/v1/responses"),
            ):
                cases.append(skip_case(name, endpoint, "skipped: model does not advertise supports_vision"))
        else:
            image_url = "data:image/png;base64," + PROBE_IMAGE_PNG_B64
            cases.append(make_case(
                client=client,
                model=model,
                provider_type=provider_type,
                name="chat_multimodal",
                endpoint="/v1/chat/completions",
                expect="chat_text",
                capability_probe=True,
                body={
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": "Describe this image in five words or fewer."},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ]}],
                    "max_tokens": 2000,
                },
            ))
            cases.append(make_case(
                client=client,
                model=model,
                provider_type=provider_type,
                name="messages_multimodal",
                endpoint="/v1/messages",
                expect="messages_text",
                capability_probe=True,
                body={
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": "Describe this image in five words or fewer."},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PROBE_IMAGE_PNG_B64}},
                    ]}],
                    "max_tokens": 2000,
                },
            ))
            cases.append(make_case(
                client=client,
                model=model,
                provider_type=provider_type,
                name="responses_multimodal",
                endpoint="/v1/responses",
                expect="responses_text",
                capability_probe=True,
                body={
                    "input": [{"role": "user", "content": [
                        {"type": "input_text", "text": "Describe this image in five words or fewer."},
                        {"type": "input_image", "image_url": image_url},
                    ]}],
                    "max_tokens": 2000,
                },
            ))
    if include_stream:
        cases.append(make_case(
            client=client,
            model=model,
            provider_type=provider_type,
            name="chat_stream",
            endpoint="/v1/chat/completions",
            expect="chat_stream",
            stream=True,
            body={"messages": [{"role": "user", "content": "Stream a two word greeting."}], "stream": True},
        ))
        cases.append(make_case(
            client=client,
            model=model,
            provider_type=provider_type,
            name="messages_stream",
            endpoint="/v1/messages",
            expect="messages_stream",
            stream=True,
            body={"messages": [{"role": "user", "content": "Stream a two word greeting."}], "stream": True},
        ))
        cases.append(make_case(
            client=client,
            model=model,
            provider_type=provider_type,
            name="responses_stream",
            endpoint="/v1/responses",
            expect="responses_stream",
            stream=True,
            body={"input": "Stream a two word greeting.", "stream": True},
        ))
    return cases


def tool_schema_chat() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up an order by ID.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    }


def tool_schema_anthropic() -> dict[str, Any]:
    return {
        "name": "lookup_order",
        "description": "Look up an order by ID.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    }


def tool_schema_responses() -> dict[str, Any]:
    return {
        "type": "function",
        "name": "lookup_order",
        "description": "Look up an order by ID.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    }


def run_judge(client: GatewayClient, judge_model: str, result: ModelResult) -> None:
    if not judge_model:
        return
    rubric = {
        "model": result.model,
        "score": result.score,
        "cases": [
            {
                "name": case.name,
                "verdict": case.verdict,
                "expected_upstream": case.expected_upstream,
                "observed_upstream": case.observed_upstream,
                "score": case.score,
                "status": case.status,
                "summary": case.summary,
                "error": case.error,
                "logs": case.log_entries[:3],
                "response_excerpt": case.response_excerpt[:600],
            }
            for case in result.cases
        ],
    }
    body = {
        "model": judge_model,
        "temperature": 0,
        "max_tokens": 800,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are judging live LLM gateway endpoint conformance. "
                    "Return compact JSON with keys overall_score, verdict, critical_failures, notes. "
                    "Score 0-100. Focus on endpoint compatibility, stream/tool/multimodal behavior, and logs."
                ),
            },
            {"role": "user", "content": json.dumps(rubric, ensure_ascii=False)},
        ],
    }
    status, payload, latency_ms = client.request("POST", "/v1/chat/completions", body)
    judge_text = text_from_chat(payload)
    judge_payload: dict[str, Any] = {"status": status, "latency_ms": latency_ms, "raw": judge_text[:2000]}
    try:
        start = judge_text.find("{")
        end = judge_text.rfind("}")
        if start >= 0 and end > start:
            judge_payload["parsed"] = json.loads(judge_text[start:end + 1])
    except Exception:
        pass
    for case in result.cases:
        case.judge = judge_payload


def write_reports(results: list[ModelResult], output_dir: Path, base_url: str, matrix: dict[str, dict[str, int]] | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    payload = {
        "base_url": base_url,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "coverage_matrix": matrix or {},
        "models": [
            {
                "model": result.model,
                "score": result.score,
                "no_signal": result.no_signal,
                "passed": result.passed,
                "scored": len(result.scored_cases),
                "skipped": sum(1 for case in result.cases if case.verdict == "skip"),
                "unsupported": sum(1 for case in result.cases if case.verdict == "unsupported"),
                "total": len(result.cases),
                "cases": [asdict(case) for case in result.cases],
            }
            for result in results
        ],
    }
    json_path = output_dir / f"live-eval-{stamp}.json"
    md_path = output_dir / f"live-eval-{stamp}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Live Gateway Evaluation",
        "",
        f"- Base URL: `{payload['base_url']}`",
        f"- Created at: `{payload['created_at']}`",
        "",
        "| Model | Score | Passed | Scored | Skipped | Unsupported | Total |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in payload["models"]:
        score_cell = "N/A" if model.get("no_signal") else f"{model['score']:.2f}"
        lines.append(
            f"| `{model['model']}` | {score_cell} | {model['passed']} | {model['scored']} | "
            f"{model['skipped']} | {model['unsupported']} | {model['total']} |"
        )
    for model in payload["models"]:
        lines.extend(["", f"## {model['model']}", "", "| Case | Verdict | Upstream (expected -\u003e observed) | Score | HTTP | Latency | Summary |", "|---|---|---|---:|---:|---:|---|"])
        for case in model["cases"]:
            summary = (case["summary"] or case["error"] or "").replace("|", "\\|")
            upstream = f"{case.get('expected_upstream') or '?'} -> {case.get('observed_upstream') or '?'}".replace("|", "\\|")
            lines.append(
                f"| `{case['name']}` | {case['verdict']} | {upstream} | {case['score']:.2f} | {case.get('status') or ''} | "
                f"{case.get('latency_ms', 0)}ms | {summary} |"
            )
        judge = model["cases"][0].get("judge") if model["cases"] else None
        if judge:
            lines.extend(["", "Judge:", "", "```json", json.dumps(judge.get("parsed") or judge, ensure_ascii=False, indent=2), "```"])
    matrix = payload.get("coverage_matrix") or {}
    if matrix:
        lines.extend(["", "## Protocol coverage matrix", "", "| client \\ upstream | " + " | ".join(UPSTREAM_PROTOCOLS) + " |", "|---|" + "---:|" * len(UPSTREAM_PROTOCOLS)])
        for ep in ("chat_completions", "completions", "messages", "responses"):
            row = matrix.get(ep, {})
            lines.append(f"| `{ep}` | " + " | ".join(str(row.get(up) or "-") for up in UPSTREAM_PROTOCOLS) + " |")
        missing = [cell for cell in REACHABLE_CELLS if not matrix.get(cell[0], {}).get(cell[1])]
        if missing:
            lines.extend(["", "Uncovered reachable cells: " + ", ".join(f"`{ep}->{up}`" for ep, up in missing)])
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live endpoint evaluation against a deployed LLM AIO Gateway.")
    parser.add_argument(
        "--config",
        default=os.getenv(
            "LLM_AIO_LIVE_CONFIG",
            "tools/live_eval/live-eval.config.json",
        ),
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--admin-username", default=None)
    parser.add_argument("--admin-password", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--model", action="append", default=None, help="Model ID to test. Can be repeated. Defaults to all /v1/models.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-multimodal", action="store_true")
    parser.add_argument("--skip-stream", action="store_true")
    parser.add_argument(
        "--ignore-capabilities",
        action="store_true",
        help="Probe tools/multimodal even when the model does not advertise the capability.",
    )
    parser.add_argument(
        "--require-matrix",
        action="store_true",
        help="Exit non-zero unless every reachable client x upstream cell is covered by this run.",
    )
    parser.add_argument(
        "--require-signal",
        action="store_true",
        help="Exit non-zero when any model ends up with zero scored cases (all skipped/unsupported).",
    )
    parser.add_argument(
        "--keep-capability-cache",
        action="store_true",
        help="Do not reset the Responses capability probe cache before testing (keeps stale native decisions).",
    )
    args = parser.parse_args(argv)

    config = load_live_eval_config(args.config, explicit="--config" in argv)
    args.base_url = pick_setting(args.base_url, config, "base_url", "LLM_AIO_LIVE_BASE_URL", "http://localhost:8000")
    args.api_key = pick_setting(args.api_key, config, "api_key", "LLM_AIO_LIVE_API_KEY", "")
    args.admin_username = pick_setting(args.admin_username, config, "admin_username", "LLM_AIO_ADMIN_USERNAME", "")
    args.admin_password = pick_setting(args.admin_password, config, "admin_password", "LLM_AIO_ADMIN_PASSWORD", "")
    args.judge_model = pick_setting(args.judge_model, config, "judge_model", "LLM_AIO_JUDGE_MODEL", "")
    args.limit = int(pick_setting(args.limit, config, "limit", "LLM_AIO_LIVE_LIMIT", 0) or 0)
    args.timeout = float(pick_setting(args.timeout, config, "timeout", "LLM_AIO_LIVE_TIMEOUT", 120) or 120)
    args.output_dir = pick_setting(args.output_dir, config, "output_dir", "LLM_AIO_LIVE_OUTPUT_DIR", "reports/live-eval")
    if args.model is None:
        configured_models = config.get("models", config.get("model", []))
        if isinstance(configured_models, str):
            args.model = [configured_models]
        elif isinstance(configured_models, list):
            args.model = [str(item) for item in configured_models if item]
        else:
            args.model = []
    if not args.skip_multimodal:
        args.skip_multimodal = bool(config.get("skip_multimodal", False))
    if not args.skip_stream:
        args.skip_stream = bool(config.get("skip_stream", False))
    if not args.ignore_capabilities:
        args.ignore_capabilities = bool(config.get("ignore_capabilities", False))
    if not args.require_matrix:
        args.require_matrix = bool(config.get("require_matrix", False))
    if not args.require_signal:
        args.require_signal = bool(config.get("require_signal", False))
    if not args.keep_capability_cache:
        args.keep_capability_cache = bool(config.get("keep_capability_cache", False))
    return args


def load_live_eval_config(path_value: str, *, explicit: bool) -> dict[str, Any]:
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.exists():
        if explicit:
            raise SystemExit(f"Config file not found: {path}")
        return {}
    try:
        raw = path.read_bytes()
        encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
        payload = json.loads(raw.decode(encoding))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid JSON config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Config file must contain a JSON object: {path}")
    return payload


def pick_setting(cli_value: Any, config: dict[str, Any], key: str, env_name: str, default: Any) -> Any:
    if cli_value not in (None, ""):
        return cli_value
    if key in config and config[key] not in (None, ""):
        return config[key]
    env_value = os.getenv(env_name)
    if env_value not in (None, ""):
        return env_value
    return default


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if not args.api_key:
        print("Missing --api-key or LLM_AIO_LIVE_API_KEY.", file=sys.stderr)
        return 2
    client = GatewayClient(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
        admin_username=args.admin_username,
        admin_password=args.admin_password,
    )
    advertised: dict[str, dict[str, Any]] = {}
    try:
        advertised = client.get_models()
    except Exception as exc:
        if not args.model:
            print(f"Cannot list models: {exc}", file=sys.stderr)
            return 2
        print(f"Warning: /v1/models failed ({exc}); capability gating disabled", file=sys.stderr)
    models = args.model or list(advertised)
    if args.limit:
        models = models[:args.limit]
    if not models:
        print("No models found.", file=sys.stderr)
        return 1

    print(f"Testing {len(models)} model(s) from {args.base_url}")
    provider_types = client.get_provider_types()
    if not provider_types:
        print("Note: no admin credentials; upstream-protocol assertions disabled (matrix will rely on observed logs).")
    results = []
    for index, model in enumerate(models, 1):
        print(f"\n[{index}/{len(models)}] {model}")
        provider_type = provider_types.get(model, "")
        if provider_type == "openai" and not args.keep_capability_cache:
            reset_rows = client.reset_responses_capability(model)
            if reset_rows:
                print(f"  capability: Responses probe cache cleared ({reset_rows} row(s)); native path re-evaluated this run")
            elif reset_rows == 0:
                print("  WARN: capability reset matched 0 rows (provider_models row missing?); native decision may be stale")
        result = ModelResult(model=model)
        result.cases = build_cases(
            client,
            model,
            include_multimodal=not args.skip_multimodal,
            include_stream=not args.skip_stream,
            caps=advertised.get(model, {}),
            ignore_capabilities=args.ignore_capabilities,
            provider_type=provider_type,
        )
        run_judge(client, args.judge_model, result)
        results.append(result)
        marks = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP ", "unsupported": "UNSPT"}
        for case in result.cases:
            mark = marks[case.verdict]
            print(f"  {mark:5} {case.name:22} {case.score:.2f} {case.latency_ms:5d}ms {case.summary or case.error}")
        skipped = sum(1 for case in result.cases if case.verdict == "skip")
        unsupported = sum(1 for case in result.cases if case.verdict == "unsupported")
        extra = ""
        if skipped or unsupported:
            extra = f", {skipped} skipped, {unsupported} unsupported"
        if result.no_signal:
            print(f"  SCORE N/A (0 scored{extra})  <-- 本次对该模型无任何有效信号，勿视为健康")
        else:
            print(f"  SCORE {result.score:.2f} ({result.passed}/{len(result.scored_cases)} scored{extra})")
        if skipped + unsupported > len(result.scored_cases):
            print("  WARN: 超过半数探针未形成计分信号（skip/unsupported）；能力元数据可能失真，建议 --ignore-capabilities 复验")

    matrix = build_coverage_matrix(results)
    print_coverage_matrix(matrix)
    write_reports(results, Path(args.output_dir), args.base_url, matrix)
    failed = sum(1 for result in results for case in result.cases if case.verdict == "fail")
    if failed:
        return 1
    if args.require_signal:
        silent = [result.model for result in results if result.no_signal]
        if silent:
            print("\n--require-signal: models with zero scored cases: " + ", ".join(silent))
            return 1
    if args.require_matrix:
        missing = missing_reachable_cells(matrix)
        if missing:
            print("\n--require-matrix: uncovered reachable cells: " + ", ".join(f"{ep}->{up}" for ep, up in missing))
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
