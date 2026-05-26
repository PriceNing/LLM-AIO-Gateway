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
import sys
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PNG_1X1_RED = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


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


@dataclass
class ModelResult:
    model: str
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def score(self) -> float:
        if not self.cases:
            return 0.0
        return round(sum(case.score for case in self.cases) / len(self.cases), 2)

    @property
    def passed(self) -> int:
        return sum(1 for case in self.cases if case.ok)


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
                raw = resp.read()
                latency_ms = int((time.perf_counter() - started) * 1000)
                text = raw.decode("utf-8", errors="replace")
                if stream:
                    return status, parse_sse(text), latency_ms
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

    def get_models(self) -> list[str]:
        status, payload, _ = self.request("GET", "/v1/models")
        if status != 200 or not isinstance(payload, dict):
            raise RuntimeError(f"/v1/models failed: HTTP {status} {payload!r}")
        models = []
        for item in payload.get("data", []):
            if isinstance(item, dict) and item.get("id"):
                models.append(str(item["id"]))
        return models

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


def make_case(
    *,
    client: GatewayClient,
    model: str,
    name: str,
    endpoint: str,
    body: dict[str, Any],
    expect: str,
    stream: bool = False,
) -> CaseResult:
    request_id = f"live-eval-{int(time.time())}-{abs(hash((model, name))) % 100000}"
    body = dict(body)
    body["model"] = model
    body.setdefault("temperature", 0)
    body.setdefault("max_tokens", 160)
    body.setdefault("metadata", {})
    if isinstance(body["metadata"], dict):
        body["metadata"]["live_eval_request_id"] = request_id
    started = time.strftime("%Y-%m-%dT%H:%M:%S")

    try:
        status, payload, latency_ms = client.request("POST", endpoint, body, stream=stream)
        ok, score, summary = evaluate_payload(status, payload, expect)
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
            response_excerpt=compact_json(payload),
            log_entries=client.get_recent_logs(model, started),
        )
    except Exception as exc:
        return CaseResult(
            name=name,
            endpoint=endpoint,
            ok=False,
            score=0.0,
            request_id=request_id,
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


def build_cases(client: GatewayClient, model: str, include_multimodal: bool, include_stream: bool) -> list[CaseResult]:
    cases: list[CaseResult] = []
    cases.append(make_case(
        client=client,
        model=model,
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
        name="completions_text",
        endpoint="/v1/completions",
        expect="completion_text",
        body={"prompt": "Return exactly one short sentence about automated endpoint testing."},
    ))
    cases.append(make_case(
        client=client,
        model=model,
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
        name="responses_followup",
        endpoint="/v1/responses",
        expect="responses_text",
        body=response_body,
    ))
    cases.append(make_case(
        client=client,
        model=model,
        name="chat_tool_call",
        endpoint="/v1/chat/completions",
        expect="chat_tool",
        body={
            "messages": [{"role": "user", "content": "Call the lookup_order tool for order_id A123."}],
            "tools": [tool_schema_chat()],
            "tool_choice": {"type": "function", "function": {"name": "lookup_order"}},
        },
    ))
    cases.append(make_case(
        client=client,
        model=model,
        name="messages_tool_call",
        endpoint="/v1/messages",
        expect="messages_tool",
        body={
            "messages": [{"role": "user", "content": "Use lookup_order for order_id A123."}],
            "tools": [tool_schema_anthropic()],
            "tool_choice": {"type": "tool", "name": "lookup_order"},
        },
    ))
    cases.append(make_case(
        client=client,
        model=model,
        name="responses_tool_call",
        endpoint="/v1/responses",
        expect="responses_tool",
        body={
            "input": "Use lookup_order for order_id A123.",
            "tools": [tool_schema_responses()],
            "tool_choice": {"type": "function", "name": "lookup_order"},
        },
    ))
    if include_multimodal:
        image_url = "data:image/png;base64," + PNG_1X1_RED
        cases.append(make_case(
            client=client,
            model=model,
            name="chat_multimodal",
            endpoint="/v1/chat/completions",
            expect="chat_text",
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
            name="messages_multimodal",
            endpoint="/v1/messages",
            expect="messages_text",
            body={
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "Describe this image in five words or fewer."},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_1X1_RED}},
                ]}],
                "max_tokens": 2000,
            },
        ))
        cases.append(make_case(
            client=client,
            model=model,
            name="responses_multimodal",
            endpoint="/v1/responses",
            expect="responses_text",
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
            name="chat_stream",
            endpoint="/v1/chat/completions",
            expect="chat_stream",
            stream=True,
            body={"messages": [{"role": "user", "content": "Stream a two word greeting."}], "stream": True},
        ))
        cases.append(make_case(
            client=client,
            model=model,
            name="messages_stream",
            endpoint="/v1/messages",
            expect="messages_stream",
            stream=True,
            body={"messages": [{"role": "user", "content": "Stream a two word greeting."}], "stream": True},
        ))
        cases.append(make_case(
            client=client,
            model=model,
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
                "ok": case.ok,
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


def write_reports(results: list[ModelResult], output_dir: Path, base_url: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    payload = {
        "base_url": base_url,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "models": [
            {
                "model": result.model,
                "score": result.score,
                "passed": result.passed,
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
        "| Model | Score | Passed | Total |",
        "|---|---:|---:|---:|",
    ]
    for model in payload["models"]:
        lines.append(f"| `{model['model']}` | {model['score']:.2f} | {model['passed']} | {model['total']} |")
    for model in payload["models"]:
        lines.extend(["", f"## {model['model']}", "", "| Case | OK | Score | HTTP | Latency | Summary |", "|---|---:|---:|---:|---:|---|"])
        for case in model["cases"]:
            ok = "yes" if case["ok"] else "no"
            summary = (case["summary"] or case["error"] or "").replace("|", "\\|")
            lines.append(
                f"| `{case['name']}` | {ok} | {case['score']:.2f} | {case.get('status') or ''} | "
                f"{case.get('latency_ms', 0)}ms | {summary} |"
            )
        judge = model["cases"][0].get("judge") if model["cases"] else None
        if judge:
            lines.extend(["", "Judge:", "", "```json", json.dumps(judge.get("parsed") or judge, ensure_ascii=False, indent=2), "```"])
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live endpoint evaluation against a deployed LLM AIO Gateway.")
    parser.add_argument("--config", default=os.getenv("LLM_AIO_LIVE_CONFIG", "tools/live-eval.config.json"))
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
    args = parser.parse_args(argv)

    config = load_live_eval_config(args.config, explicit="--config" in argv)
    args.base_url = pick_setting(args.base_url, config, "base_url", "LLM_AIO_LIVE_BASE_URL", "http://192.168.75.100:8000")
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
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
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
    models = args.model or client.get_models()
    if args.limit:
        models = models[:args.limit]
    if not models:
        print("No models found.", file=sys.stderr)
        return 1

    print(f"Testing {len(models)} model(s) from {args.base_url}")
    results = []
    for index, model in enumerate(models, 1):
        print(f"\n[{index}/{len(models)}] {model}")
        result = ModelResult(model=model)
        result.cases = build_cases(
            client,
            model,
            include_multimodal=not args.skip_multimodal,
            include_stream=not args.skip_stream,
        )
        run_judge(client, args.judge_model, result)
        results.append(result)
        for case in result.cases:
            mark = "PASS" if case.ok else "FAIL"
            print(f"  {mark:4} {case.name:22} {case.score:.2f} {case.latency_ms:5d}ms {case.summary or case.error}")
        print(f"  SCORE {result.score:.2f} ({result.passed}/{len(result.cases)})")

    write_reports(results, Path(args.output_dir), args.base_url)
    failed = sum(1 for result in results for case in result.cases if not case.ok)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
