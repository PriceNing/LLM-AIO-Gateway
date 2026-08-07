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

This release supports existing provider models and external OpenAI Images-compatible backends. ComfyUI is reserved as a future interchangeable backend and is not implemented yet. Codex can invoke image generation through the `/responses` tool bridge or `/images/generations`. The gateway stores short-lived originals and returns bounded previews to clients.

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
  "target_model": "target-model",
  "target_provider": "target-provider"
}
```

Routing rules only describe active routing. Passive fallback is configured separately in `fallback_policies`: match the routed provider/model plus a failure trigger such as `timeout`, `connection_error`, `http_429`, or `http_5xx`, then try the configured fallback chain. The admin UI provides a dedicated fallback policy editor, so users do not need to write JSON in a routing rule.

The admin API includes `POST /admin/routing-rules/dry-run` to inspect which active routing rule would match, and `POST /admin/fallback-policies/dry-run` to inspect which fallback chain would activate for a given provider, model, and failure type. Provider cards also expose a health check that probes `/models` availability, latency, and model count.

## Configuration

`config.json` contains server-level settings. Changes require a service restart.

Important defaults:

| Key | Default | Description |
|---|---:|---|
| `max_tokens` | 16384 | Used when the client omits `max_tokens` and `max_completion_tokens`. |
| `temperature` | 0.7 | Default temperature. |
| `tool_only_limit` | 20 | Tool-only loop circuit breaker threshold. |
| `min_image_max_tokens` | 2000 | Minimum max tokens for requests containing images. |
| `reasoning_cache_ttl` | 1800 | Reasoning cache TTL in seconds. |
| `reasoning_cache_max_size` | 1000 | Reasoning cache capacity. |
| `tool_only_turns_ttl` | 600 | Tool-only counter TTL in seconds. |
| `tool_only_turns_max_size` | 2000 | Tool-only counter capacity. |
| `image_cache_max_size` | 500 | Image description cache capacity. |
| `request_log_capture_payloads` | true | Store request/response bodies; disable to retain metadata only. |
| `login_attempt_max_identities` | 10000 | Maximum number of admin-login throttle identities retained in memory. |
| `image_result_ttl_seconds` | 86400 | Retention time for generated originals. |
| `image_preview_max_bytes` | 800000 | Target byte limit for each inline preview. |
| `image_generation_batch_concurrency` | 1 | Concurrency within one image batch. |
| `image_generation_result_max_bytes` | 26214400 | Maximum bytes accepted for one upstream image. |
| `image_download_allow_private_hosts` | false | Allow image-result URL downloads from private networks. |

## Architecture Summary

```text
Client endpoint
  -> protocol ingress
  -> internal IR
  -> shared policy layer (RoutingDecision, preprocessing, reasoning, tool repair)
  -> OpenAI/liteLLM adapter or direct Anthropic adapter
  -> internal output
  -> protocol egress
```

This design keeps endpoint-specific protocol details at ingress/egress while routing, preprocessing, reasoning cache, tool repair, and adapter selection operate on one internal format.

Main code boundaries:

| Module | Responsibility |
|---|---|
| `app/router/proxy.py` | FastAPI endpoints, auth, provider resolution, adapter dispatch, non-streaming request stats. |
| `app/protocols/ingress.py` | Converts `/chat/completions`, `/completions`, `/messages`, and `/responses` request bodies into internal IR. |
| `app/core/policy.py` | Routing decisions, message normalization, preprocessing hook, reasoning injection, tool argument repair, tool-only limit. |
| `app/core/state.py` | TTL caches, reasoning cache, tool-only counter, response-chain cache. |
| `app/core/streaming.py` | Streaming event metering, reasoning storage, tool-only counting, stream error rendering, stats callback. |
| `app/core/images.py` | Data URI extraction, image-content detection, and OpenAI image-content normalization. |
| `app/adapters/imagegen.py` | OpenAI Images-compatible backends, parameter compatibility, retries, and result downloads. |
| `app/core/image_bridge.py` | Codex `/responses` image-tool discovery, invocation parsing, and asset handoff. |
| `app/core/image_results.py` | Original storage, preview compression, and capability-token downloads. |
| `app/core/image_batch.py` | Image batch coordination and short-lived idempotent reuse. |
| `app/adapters/` | Sends internal requests to OpenAI/liteLLM or direct Anthropic Messages and converts responses to internal output/events. |
| `app/protocols/egress.py` | Renders internal output back into Chat, Completions, Messages, and Responses protocols. |
| `app/services/lite_llm.py` | OpenAI-compatible liteLLM wrapper only, plus minimal reasoning compatibility patches. |

## Testing

```bash
pytest tests/ -q
```

Expected current result: `602 passed`.

Live smoke matrix:

- Claude Code -> `/messages`
- Codex -> `/responses`
- OpenCode -> `/chat/completions`
- curl -> `/completions`
- At least one OpenAI-compatible provider and one Anthropic-compatible provider.

## License

MIT. See the `LICENSE` file.
