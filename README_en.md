<p align="center">
  <a href="README.md">简体中文</a> |
  <a href="README_en.md">English</a>
</p>

# LLM AIO Gateway - All-in-One LLM API Gateway

LLM AIO Gateway is a unified FastAPI gateway for OpenAI-compatible and Anthropic-compatible LLM providers. It exposes OpenAI Chat Completions, legacy Completions, Anthropic Messages, and OpenAI Responses endpoints through one service, with routing, API-key management, reasoning continuity, tool-call repair, and vision-model image preprocessing.

The current proxy core is built around a provider-neutral internal representation: every client protocol is normalized into `InternalRequest` / `InternalMessage`, processed by shared policy logic that produces a structured `RoutingDecision`, sent through an upstream adapter, then rendered back to the requested client protocol.

## Features

| Feature | Description |
|---|---|
| Unified protocol gateway | Supports `/chat/completions`, `/completions`, `/messages`, `/responses`, and `/models`, mounted at both root and `/v1` paths. |
| OpenAI and Anthropic providers | OpenAI-compatible providers go through liteLLM; Anthropic-compatible providers use a direct Anthropic Messages adapter from the internal IR. |
| Shared IR pipeline | Routing, preprocessing, reasoning cache, tool repair, and circuit-breaking run once on internal messages instead of duplicated endpoint-specific conversions. |
| Structured routing | The policy layer returns `RoutingDecision` with requested/resolved/target model, target provider, matched rule, and reason. |
| Vision model injection | Images can be described by a configured vision model, then replaced with text so text-only models can handle visual context. |
| Image-generation gateway | Supports `/images/generations`, Codex `/responses` tool bridging, short-lived originals, compressed previews, batches, and image usage statistics. |
| Tool-call reliability | Preserves tool IDs across protocol conversions, repairs malformed tool JSON, and includes a tool-only loop circuit breaker. |
| Reasoning continuity | Caches and replays `reasoning_content` for DeepSeek-style thinking models across multi-turn tool flows. |
| Web admin panel | Manage providers, users, API keys, routing rules, model preprocessors, and usage stats. |
| SQLite storage | Providers, users, keys, routing rules, stats, and request records are stored in `data.db`. |

## Quick Start

### Requirements

- Python 3.10+
- pip

### Manual Install

```bash
pip install -r requirements.txt
python main.py
```

The service starts on `http://localhost:8000` by default. On first startup it creates `config.json` and `data.db` if they do not exist.

### Docker

Use the published GHCR image:

```bash
docker pull ghcr.io/pricening/llm-aio-gateway:latest
docker run -d --name llm-aio-gateway \
  -p 8000:8000 \
  -v llm-aio-data:/app/data \
  -v llm-aio-logs:/app/logs \
  ghcr.io/pricening/llm-aio-gateway:latest
```

Or build locally from source:

```bash
docker compose up -d
```

## First-Time Setup

1. Open `http://localhost:8000`.
2. Create the first admin account.
3. Add an upstream provider in the admin panel.
4. Refresh provider models or add models manually.
5. Create a user and generate an API key.
6. Call the gateway with `Authorization: Bearer sk-aio-...`.

## Provider Types

| Provider type | Upstream style | Gateway behavior |
|---|---|---|
| `openai` | OpenAI-compatible chat/completions APIs | Internal request is projected to OpenAI chat shape and sent through liteLLM. |
| `anthropic` | Anthropic-compatible Messages API | Internal request is projected to native Anthropic Messages shape and sent directly to `{api_base}/v1/messages`. |

Use model IDs as either `provider/model` or simple `model`. The composite form selects a specific provider. Simple names resolve to the first enabled provider/model match.

## API Endpoints

All proxy endpoints are available at both root and `/v1` paths.

| Endpoint | Protocol | Notes |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI Chat Completions | Chat, tools, streaming, images. Used by OpenCode/OpenWebUI-style clients. |
| `POST /v1/completions` | OpenAI legacy Completions | `prompt` is wrapped into an internal user message, then rendered back as `choices[0].text`. |
| `POST /v1/messages` | Anthropic Messages | Claude Code-compatible Messages API, tools, streaming, images. |
| `POST /v1/responses` or `/responses` | OpenAI Responses | Codex-compatible Responses API, tools, streaming, previous response IDs. |
| `POST /v1/images/generations` or `/images/generations` | OpenAI Images | Sends OpenAI Images-compatible requests to the globally configured image backend. |
| `GET /v1/models` | OpenAI Models | Lists models allowed for the caller's API key. |

### Chat Completions Example

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-aio-xxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "provider/model-name",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 1024
  }'
```

### Completions Example

```bash
curl http://localhost:8000/v1/completions \
  -H "Authorization: Bearer sk-aio-xxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "provider/model-name",
    "prompt": "Write a short poem about spring.",
    "max_tokens": 200
  }'
```

### Anthropic Messages Example

```bash
curl http://localhost:8000/v1/messages \
  -H "Authorization: Bearer sk-aio-xxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "provider/model-name",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
  }'
```

### Responses Example

```bash
curl http://localhost:8000/v1/responses \
  -H "Authorization: Bearer sk-aio-xxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "provider/model-name",
    "input": "Hello"
  }'
```

## Vision Model Injection

Vision injection lets text-only models handle image input. When enabled for the requested model, the gateway describes current-turn images with a configured vision model, strips original image data, and injects descriptions into the conversation.

Add preprocessors in the admin panel under Vision Model Injection, then enable preprocessing for target models there. Preprocessor definitions and model toggles are stored in SQLite, not `config.json`. The decision is based on the originally requested model, not the routed target model.

## Image Generation

Image generation is separate from vision-model injection. Select one global image backend in the Image Generation admin page, then enable image generation for each chat model that may use it. If a user can access model A and image generation is enabled for model A, requests through A may use the global image backend; the backend image model does not need separate inclusion in that user's chat-model allow-list.

This release supports existing provider models, external OpenAI Images-compatible backends, and ComfyUI. ComfyUI accepts an API-format workflow JSON. The admin UI analyzes its nodes and inputs, suggests positive/negative prompt, width, height, seed, steps, CFG, batch-size, and output mappings, and exposes dropdowns instead of requiring manually typed node IDs. The gateway submits workflows through `/prompt`, polls `/history/{prompt_id}`, and downloads results from `/view`. Codex can invoke image generation through the `/responses` tool bridge or `/images/generations`. The gateway stores short-lived originals and returns bounded previews to clients.

Enabling image generation on a model grants a capability; it does not require every request to generate an image. `/responses` exposes image generation as an optional tool, and the global image backend is contacted only when the model invokes that tool (or the client explicitly forces the standard `image_generation` tool). The gateway does not rewrite ordinary text, code, or terminal-tool calls into image requests. Intent checks read only the latest explicit user message, never system instructions or tool output.

ComfyUI accepts either a regular UI workflow or one exported with “Save (API Format)”. For a regular workflow, the gateway reconstructs the API prompt from `nodes`, `links`, and `widgets_values` before mapping inputs; API Format remains the most stable and exact representation. The workflow must include at least one image output node, normally `SaveImage` or `PreviewImage`. A remote ComfyUI instance must also listen on an address reachable by the gateway.

The admin UI can also list saved workflows through ComfyUI's userdata API and load a selected JSON. When “Analyze Workflow” is clicked, a regular saved workflow is converted to API Format automatically and its detected mappings are displayed.

Regular UI workflow conversion is best-effort: standard nodes normally convert correctly, while custom nodes may use nonstandard widget serialization. If a converted value or mapping is wrong, export the workflow with “Save (API Format)”. Initial model loading can be substantially slower than later jobs. `timeout` is the total workflow wait time and `poll_interval` controls history polling. With a `batch_size` mapping the gateway submits one batch workflow; without it, requested images are submitted as separate jobs. Public assistant text contains only concise original links and compressed previews, never private bridge markers or agent instructions.

## Routing Rules

Routing rules can transparently redirect requests by username, API-key substring, and requested model pattern. The first matching enabled rule wins.

Routing runs in the shared policy layer and is represented as a `RoutingDecision`. Logs include requested model, resolved model, target model, target provider, matched rule, and reason, making route debugging explicit.

Rule structure:

```json
{
  "name": "route-example",
  "enabled": true,
  "username": "",
  "api_key_pattern": "",
  "match_model": "MiniMax-M2*",
  "match_scope": "any",
  "target_model": "target-model",
  "target_provider": "target-provider"
}
```

`match_scope` controls which form of the requested model ID the rule matches:

- `any` (default): match the requested model as-is; a simple alias also matches the model component of a composite `provider/model` request.
- `unqualified`: match only simple model names (no `/` in the requested model).
- `qualified`: match only composite `provider/model` requests.

Routing rules only describe active routing. Passive fallback is configured separately in `fallback_policies`: match the routed provider/model plus a failure trigger such as `timeout`, `connection_error`, `http_429`, or `http_5xx`, then try the configured fallback chain. The admin UI provides a dedicated fallback policy editor, so users do not need to write JSON in a routing rule.

The admin API includes `POST /admin/routing-rules/dry-run` to inspect which active routing rule would match, and `POST /admin/fallback-policies/dry-run` to inspect which fallback chain would activate for a given provider, model, and failure type. Provider cards also expose a health check that probes `/models` availability, latency, and model count.

## Configuration

`config.json` contains server-level settings. Changes require a service restart. The complete set of options lives in the `defaults` block of `config.example.json`.

Important defaults:

| Key | Default | Description |
|---|---:|---|
| `max_tokens` | 16384 | Used when the client omits `max_tokens` and `max_completion_tokens`. |
| `temperature` | 0.7 | Default temperature. |
| `max_request_body_bytes` | 33554432 | Inbound request body limit (32 MiB); larger bodies return 413. |
| `litellm_request_timeout` | 120 | liteLLM upstream call timeout. |
| `tool_only_limit` | 20 | Tool-only loop circuit breaker threshold. |
| `min_image_max_tokens` | 2000 | Minimum max tokens for requests containing images. |
| `session_ttl_hours` | 12 | Admin session lifetime. |
| `login_attempt_limit` | 10 | Failed admin-login attempts before lockout. |
| `login_attempt_window_seconds` | 300 | Window used to count login attempts. |
| `login_lockout_seconds` | 900 | Lockout duration after exceeding the attempt limit. |
| `login_attempt_max_identities` | 10000 | Maximum number of admin-login throttle identities retained in memory. |
| `request_log_max` | 200 | Rolling request-log entries kept in memory. |
| `storage_maintenance_interval_seconds` | 60 | Background storage cleanup interval. |
| `request_log_capture_payloads` | true | Store request/response bodies; disable to retain metadata only. |
| `request_log_redact_fields` | `[api_key, authorization, ...]` | Redacted fields when capturing request/response bodies. |
| `reasoning_cache_ttl` | 1800 | Reasoning cache TTL in seconds. |
| `reasoning_cache_max_size` | 1000 | Reasoning cache capacity. |
| `tool_only_turns_ttl` | 600 | Tool-only counter TTL in seconds. |
| `tool_only_turns_max_size` | 2000 | Tool-only counter capacity. |
| `image_cache_max_size` | 500 | Image description cache capacity. |
| `image_result_ttl_seconds` | 86400 | Retention time for generated originals. |
| `image_result_max_files` | 500 | Maximum files retained in the original-image directory. |
| `image_preview_enabled` | true | Generate inline preview thumbnails. |
| `image_preview_max_dimension` | 1280 | Longest edge of generated previews. |
| `image_preview_max_source_pixels` | 40000000 | Maximum source pixels accepted when downscaling. |
| `image_preview_quality` | 82 | JPEG quality for previews. |
| `image_preview_max_bytes` | 800000 | Target byte limit for each inline preview. |
| `image_preview_inline_limit` | 4 | Maximum inline previews attached to one response. |
| `image_generation_max_retries` | 2 | Retries against the image backend. |
| `image_generation_retry_base_seconds` | 1.0 | Base backoff between image-generation retries. |
| `image_generation_max_retry_delay_seconds` | 30.0 | Maximum backoff between image-generation retries. |
| `image_generation_batch_concurrency` | 1 | Concurrency within one image batch. |
| `image_generation_batch_timeout_seconds` | 2400 | Total wait time for one image batch. |
| `image_generation_result_max_bytes` | 26214400 | Maximum bytes accepted for one upstream image. |
| `image_download_allow_private_hosts` | false | Allow image-result URL downloads from private networks. |
| `allow_private_upstream_hosts` | true | Allow private upstream addresses (cloud metadata endpoints are always blocked). |
| `image_generation_idempotency_ttl_seconds` | 300 | TTL of image-generation idempotency keys. |
| `image_generation_idempotency_max_entries` | 64 | Maximum image-generation idempotency keys. |
| `responses_capability_supported_ttl` | 604800 | Positive native-Responses capability probe cache TTL. |
| `responses_capability_unsupported_ttl` | 21600 | Negative native-Responses capability probe cache TTL. |
| `responses_capability_transient_ttl` | 300 | Transient native-Responses capability probe cache TTL. |
| `anthropic_thinking_budget_tokens` | 1024 | Anthropic extended-thinking budget. |

## Safety And Limits

- Inbound bodies are capped by `RequestBodyLimitMiddleware` (`max_request_body_bytes`).
- Upstream URLs pass through `app/services/url_guard.py`: non-http(s), cloud metadata, and reserved IP ranges are rejected; private networks are governed by `allow_private_upstream_hosts`.
- Admin login throttling lives in `app/security.py` and is tuned by `login_attempt_*`.
- The Anthropic and native Responses adapters reuse the shared HTTP connection pool in `app/services/http_pool.py`.

## Architecture Summary

```text
Client endpoint
  -> protocol ingress
  -> internal IR
  -> shared policy layer (RoutingDecision, preprocessing, reasoning, tool repair)
  -> adapter selection:
       - OpenAI-compatible providers: native Responses or Chat Completions
       - Anthropic-compatible providers: direct Anthropic Messages
  -> internal output / output events
  -> protocol egress
```

OpenAI-compatible providers default to Chat Completions. The native `/responses` path is only attempted when the capability probe cache allows it and no Codex-owned tool/state marker forces Chat compatibility; otherwise the gateway falls back to the Chat path. Anthropic-compatible providers always project from the IR to native Anthropic Messages.

This design keeps endpoint-specific protocol details at ingress/egress while routing, preprocessing, reasoning cache, tool repair, and adapter selection operate on one internal format.

Main code boundaries:

| Module | Responsibility |
|---|---|
| `app/router/proxy.py` | FastAPI endpoints, auth, provider resolution, adapter dispatch, non-streaming request stats, native Responses capability probing and downgrade. |
| `app/protocols/ingress.py` | Converts `/chat/completions`, `/completions`, `/messages`, and `/responses` request bodies into internal IR. |
| `app/core/policy.py` | Routing decisions, message normalization, preprocessing hook, reasoning injection, tool argument repair, tool-only limit. |
| `app/core/state.py` | TTL caches, reasoning cache, tool-only counter, response-chain cache. |
| `app/core/streaming.py` | Streaming event metering, reasoning storage, tool-only counting, stream error rendering, stats callback. |
| `app/core/body_limit.py` | `RequestBodyLimitMiddleware` capping inbound request bodies (default 32 MiB). |
| `app/core/images.py` | Data URI extraction, image-content detection, and OpenAI image-content normalization. |
| `app/core/image_intent.py` | Image-generation intent check (`is_image_generation_intent`, `latest_user_text`). |
| `app/core/image_bridge.py` | Codex `/responses` image-tool discovery, invocation parsing, and asset handoff. |
| `app/core/image_results.py` | Original storage, preview compression, and capability-token downloads. |
| `app/core/image_batch.py` | Image batch coordination and short-lived idempotent reuse. |
| `app/core/outcome.py` | Request status (ok / degraded / partial / fail / rejected / cancelled) and stats counters. |
| `app/adapters/openai.py` | Internal request -> OpenAI chat kwargs. |
| `app/adapters/openai_streaming.py` | OpenAI/liteLLM chunks -> internal output events. |
| `app/adapters/anthropic.py` | Internal request -> direct Anthropic Messages call -> internal output. |
| `app/adapters/anthropic_streaming.py` | Anthropic SSE -> internal output events. |
| `app/adapters/responses.py` | Internal request -> native OpenAI Responses call and SSE pass-through. |
| `app/adapters/imagegen.py` | OpenAI Images-compatible backends, parameter compatibility, retries, and result downloads. |
| `app/adapters/comfyui.py` | ComfyUI adapter. |
| `app/protocols/egress.py` | Renders internal output back into Chat, Completions, Messages, and Responses protocols. |
| `app/services/lite_llm.py` | OpenAI-compatible liteLLM wrapper only, plus minimal reasoning compatibility patches. |
| `app/services/http_pool.py` | Shared HTTP connection pool reused by the Anthropic and native Responses adapters. |
| `app/services/url_guard.py` | Upstream URL validation (SSRF / cloud-metadata protection). |
| `app/db/routing.py` | Routing-rule migrations and CRUD helpers. |
| `app/db/fallback.py` | Fallback-policy migrations and CRUD helpers (including `attempt_timeout`). |
| `app/db/request_logs.py` | Request-log CRUD and inspection helpers. |

## Testing

```bash
pytest tests/ -q
```

Expected current result: `745 passed`.

Live smoke matrix:

- Claude Code -> `/messages`
- Codex -> `/responses`
- OpenCode -> `/chat/completions`
- curl -> `/completions`
- At least one OpenAI-compatible provider and one Anthropic-compatible provider.

## License

MIT. See the `LICENSE` file.
