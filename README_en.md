<p align="center">
  <a href="README.md">简体中文</a> |
  <a href="README_en.md">English</a>
</p>

# LLM AIO Gateway - All-in-One LLM API Gateway

LLM AIO Gateway is a unified FastAPI gateway for OpenAI-compatible and Anthropic-compatible LLM providers. It exposes OpenAI Chat Completions, legacy Completions, Anthropic Messages, and OpenAI Responses endpoints through one service, with routing, API-key management, reasoning continuity, tool-call repair, and vision-model image preprocessing.

The current proxy core is built around a provider-neutral internal representation: every client protocol is normalized into `InternalRequest` / `InternalMessage`, processed by shared policy logic, sent through an upstream adapter, then rendered back to the requested client protocol.

## Features

| Feature | Description |
|---|---|
| Unified protocol gateway | Supports `/chat/completions`, `/completions`, `/messages`, `/responses`, and `/models`, mounted at both root and `/v1` paths. |
| OpenAI and Anthropic providers | OpenAI-compatible providers go through liteLLM; Anthropic-compatible providers use a direct Anthropic Messages adapter from the internal IR. |
| Shared IR pipeline | Routing, preprocessing, reasoning cache, tool repair, and circuit-breaking run once on internal messages instead of duplicated endpoint-specific conversions. |
| Vision model injection | Images can be described by a configured vision model, then replaced with text so text-only models can handle visual context. |
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

Configure preprocessors in `config.json`:

```json
{
  "preprocessors": {
    "my-vision": {
      "api_base": "http://127.0.0.1:8080/v1",
      "model": "MiniCPM-V-4.6",
      "api_key": "",
      "timeout": 60,
      "max_images": 20,
      "max_tokens": 1024,
      "prompt": "Please describe this image in detail.",
      "enabled": true
    }
  }
}
```

Then enable preprocessing for the target model in the admin panel. The decision is based on the originally requested model, not the routed target model.

## Routing Rules

Routing rules can transparently redirect requests by username, API-key substring, and requested model pattern. The first matching enabled rule wins.

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

## Architecture Summary

```text
Client endpoint
  -> protocol ingress
  -> internal IR
  -> shared policy layer
  -> OpenAI/liteLLM adapter or direct Anthropic adapter
  -> internal output
  -> protocol egress
```

This design keeps endpoint-specific protocol details at ingress/egress while routing, preprocessing, reasoning cache, tool repair, and adapter selection operate on one internal format.

## Testing

```bash
pytest tests/ -q
```

Expected current result: `290 passed`.

Live smoke matrix:

- Claude Code -> `/messages`
- Codex -> `/responses`
- OpenCode -> `/chat/completions`
- curl -> `/completions`
- At least one OpenAI-compatible provider and one Anthropic-compatible provider.

## License

MIT. See the `LICENSE` file.
