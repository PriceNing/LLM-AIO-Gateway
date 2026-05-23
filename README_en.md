<p align="center">
  <a href="README.md">简体中文</a> |
  <a href="README_en.md">English</a>
</p>

# LLM AIO Gateway · All-in-One LLM API Gateway

> A unified OpenAI / Anthropic / Responses API gateway with built-in vision model injection — let any text-only model "see" images.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE)

---

## Features

| Feature | Description |
|---|---|
| 🖼️ **Vision Model Injection** | **Core feature** — automatically send user-uploaded images to a vision model for description, allowing any non-multimodal model (DeepSeek, GLM, etc.) to "see" image content, saving the cost of upgrading to multimodal models |
| 🔄 Triple-Protocol Proxy | Simultaneously supports OpenAI Chat Completions, Anthropic Messages, and OpenAI Responses protocols |
| 🎯 Transparent Routing | Routing rules based on user / API key / requested model name, with `*` wildcard support for seamless upstream model switching |
| ⚡ Tool Call Circuit Breaker | Automatically strips `tools` when consecutive tool-only calls exceed the threshold, preventing infinite loops |
| 📊 Web Admin Panel | Built-in SPA management UI for providers, users, API keys, routing rules, and real-time usage statistics |
| 🔑 Multi-Tenant | User + API key authentication system with per-key model whitelists and usage tracking |
| 🧠 Context-Aware | Transparently handles multi-turn caching and replay of reasoning/thinking content without client intervention |

---

## 🖼️ Core Feature: Vision Model Injection

### The Problem

Most high-performance, low-cost models (e.g., DeepSeek-V3, MiniMax-M*) **do not support image input**, forcing you to upgrade to multimodal models that cost twice as much. Vision model injection lets you keep using cheap text-only models while gaining image understanding capabilities.

### How It Works

```
User sends an image              Gateway auto-processes              Model receives text
───────┴────────    ───────────────┴───────────────    ───────────┴──────────
  "What is this?"    ① Extract image → vision model describes it     "What is this?
   + 📷 photo.jpg      ② Description injected into conversation       [Image #1: An orange cat
                      ③ Strip image, send plain text                   sitting on a windowsill...]"
```

1. The user sends a mixed text-image message to `/chat/completions` or `/messages` as usual
2. The gateway detects images, calls the configured vision model (e.g., MiniCPM-V, Qwen-VL) to generate text descriptions
3. Image content is replaced with `[Image #N: description text]` and sent to the target text model
4. The model response is forwarded back to the client — **entirely transparent to the client**

### Configuration

**Step 1** — Configure a vision model (preprocessor) in `config.json`, or via the Web UI under "Vision Model Injection → Add Preprocessor":

```json
{
  "preprocessors": {
    "my-vision": {
      "api_base": "http://127.0.0.1:8080/v1",
      "model": "MiniCPM-V-4.6",
      "api_key": "sk-xxx",
      "timeout": 60,
      "max_images": 20,
      "max_tokens": 1024,
      "prompt": "Please describe this image in detail.",
      "enabled": true
    }
  }
}
```

**Step 2** — Toggle the feature on for target models in the admin panel:

- Go to **Vision Model Injection** → click the toggle switch for the target model (on/off)
- Or use the API: `PUT /admin/models/preprocessor` `{"model_id": "provider/model-name", "enabled": true}`

**Done.** All subsequent image requests to that model will automatically have descriptions injected.

> Vision injection decisions are based on the **model name requested by the user**, not the routed target model, ensuring that routing rule changes do not affect injection behavior.

### Recommended Local Vision Models

The following models can be deployed locally (vLLM / llama.cpp / Ollama) as vision injection frontends:

| Model | HuggingFace | Highlights |
|------|------------|------|
| **MiniCPM-V 4.6** | [openbmb/MiniCPM-V-4.6](https://huggingface.co/openbmb/MiniCPM-V-4.6) | **Fast** — only 1B params (SigLIP2-400M + Qwen3.5-0.8B), CPU inference reaches 35+ tokens/s, ideal for lightweight scenarios |
| **Qwen3.6-35B-A3B** | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | **High quality** — 35B total / 3B active, 256-expert MoE + 27-layer ViT, native 262K context, best description quality |

---

You can also use online multimodal models as preprocessors for other models to achieve vision injection.

## 🚀 Quick Start

### Requirements

- Python 3.10+
- pip

### Docker Deployment (Recommended)

```bash
git clone https://github.com/PriceNing/LLM-AIO-Gateway.git
cd LLM-AIO-Gateway
docker compose up -d
```

On first startup, `config.json` and `data.db` are auto-generated in the `./data` directory. The service runs at `http://localhost:8000`.

To update, pull the latest code and rebuild:

```bash
git pull
docker compose up -d --build
```

### Manual Install

```bash
git clone https://github.com/PriceNing/LLM-AIO-Gateway.git
cd LLM-AIO-Gateway
pip install -r requirements.txt
python main.py
```

On first startup, `config.json` (with default settings) and `data.db` (SQLite database) are auto-generated in the project directory.

After startup, visit `http://localhost:8000`:

- **Admin Panel**: Open in browser; prompted to create an admin account on first visit
- **API Endpoints**: `/chat/completions`, `/messages`, `/responses`, `/models`

### Add a Provider

1. Log into the admin panel → **Providers** → Add Provider
2. Fill in name, API Base URL, upstream API Key, and type (OpenAI-compatible / Anthropic-compatible)
3. Click **Refresh** to auto-discover available models from the remote `/models` endpoint
4. **User Management** → Add User → Generate an API Key
5. Start calling:

```bash
curl http://localhost:8000/chat/completions \
  -H "Authorization: Bearer sk-aio-xxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{"model": "provider/model-name", "messages": [{"role": "user", "content": "Hello"}]}'
```

---

## 📖 API Endpoints

All proxy endpoints are mounted at both the root path and under the `/v1/` prefix. Both forms are equivalent:

| Endpoint | Protocol | Description |
|---|---|---|
| `POST /chat/completions` | OpenAI Chat | Streaming / non-streaming chat completions |
| `POST /completions` | OpenAI Text | Text completions |
| `POST /messages` | Anthropic | Streaming / non-streaming messages (with tool calls) |
| `POST /responses` | OpenAI Responses | Codex CLI / OpenAI Responses format |
| `GET /models` | OpenAI | Available model list |

> Root path example: `http://localhost:8000/chat/completions`
> `/v1/` prefix example: `http://localhost:8000/v1/chat/completions`

### Admin API

| Endpoint | Description |
|---|---|
| `POST /auth/login` | Admin login |
| `GET/POST /admin/providers` | Provider management |
| `POST /admin/providers/{id}/refresh` | Auto-discover models from remote `/models` |
| `GET/POST /admin/users` | User management |
| `POST/PUT/DELETE /admin/users/{username}/api-keys` | API key management |
| `GET/POST /admin/routing-rules` | Routing rule management |
| `GET /admin/stats` | Usage statistics |
| `PUT /admin/models/preprocessor` | Toggle vision injection for a model |

---

## ⚙️ Configuration Reference

`config.json` is auto-generated on first startup. A restart is required for changes to take effect.

### Top-Level Settings

| Key | Default | Description |
|---|---|---|
| `host` | `"0.0.0.0"` | Listen address |
| `port` | `8000` | Listen port |
| `database` | `"data.db"` | SQLite database path |

### `defaults` Section

| Key | Default | Description |
|---|---|---|
| `max_tokens` | `16384` | Default max_tokens when client omits it |
| `temperature` | `0.7` | Default temperature |
| `tool_only_limit` | `20` | Consecutive tool-call circuit breaker threshold |
| `min_image_max_tokens` | `2000` | Lower bound for max_tokens on requests containing images |
| `session_ttl_hours` | `12` | Admin session TTL |
| `request_log_max` | `200` | In-memory request log retention count |
| `reasoning_cache_ttl` | `1800` | Reasoning cache TTL (seconds) |
| `reasoning_cache_max_size` | `1000` | Max reasoning cache entries |
| `tool_only_turns_ttl` | `600` | Tool call counter TTL (seconds) |
| `tool_only_turns_max_size` | `2000` | Max tool call counter entries |
| `image_cache_max_size` | `500` | Max image description cache entries |

### `preprocessors` Section

Configure vision models (preprocessors) for image description. Each entry includes `api_base`, `model`, `api_key`, `timeout`, `max_images`, `prompt`, `max_tokens`, `enabled`, and other fields.

### Environment Variables

| Variable | Description |
|---|---|
| `LLM_GATEWAY_CONFIG` | Override the `config.json` file path |

---

## 🏗️ Architecture

```
Client (Claude Code / Codex / OpenCode / OpenWebUI / curl)
       │
       ▼
┌─────────────────────────────┐
│     LLM AIO Gateway :8000   │
│  ┌──────────┐ ┌───────────┐ │
│  │ proxy    │ │ admin/    │ │
│  │ router   │ │ auth      │ │
│  └────┬─────┘ └─────┬─────┘ │
│       │             │       │
│  ┌────┴─────────────┴────┐  │
│  │   liteLLM + patches   │  │
│  │   vision injector     │  │
│  │   routing engine      │  │
│  └───────────┬───────────┘  │
│              │              │
│  ┌───────────┴───────────┐  │
│  │    SQLite data.db     │  │
│  │    TTLDict caches     │  │
│  └───────────────────────┘  │
└─────────────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│  Upstream LLM Providers      │
└──────────────────────────────┘
```

- **Storage**: SQLite (providers, users, API keys, routing rules, statistics)
- **Caches**: In-process `TTLDict` (thread-safe, TTL + capacity-based eviction)
- **Config**: `config.json` auto-generated + `LLM_GATEWAY_CONFIG` env var to override path
- **Streaming**: Background thread bridges synchronous liteLLM → async SSE, automatic upstream cancellation on client disconnect

---

## ✅ Tested Platforms

The following platforms have been verified in production.

### Upstream Providers

| Provider | Protocol Type | Tested Models |
|--------|---------|-----------|
| **DeepSeek** | OpenAI-compatible / Anthropic-compatible | V4-Flash, V4-Pro |
| **MiniMax** | OpenAI-compatible / Anthropic-compatible | M2.7 |
| **OpenCode Go** | OpenAI-compatible | hy3-preview, kimi-2.6 |

### Downstream Agents / Clients

| Agent | Endpoints Used | Notes |
|--------|---------|------|
| **Claude Code CLI** | `/messages` | Native Anthropic protocol, supports multi-turn tool call relay |
| **OpenCode CLI** | `/chat/completions`, `/responses` | Open-source coding agent, vision injection + routing rule integration; for multimodal support, declare `modalities: {input: ["text", "image"]}` in `opencode.json` for the target model |
| **Codex Desktop** | `/responses` | OpenAI Codex desktop client, transparent access to any model |
| **OpenWebUI** | `/chat/completions` | OpenWebUI web chat |

### OpenCode Multimodal Configuration

OpenCode CLI assumes custom models do not support image input by default. To use vision model injection with OpenCode, declare multimodal capability for the target model in `opencode.json`:

```json
"models": {
    "your-model": {
        "name": "your-model",
        "limit": {"output": 4096, "context": 128000},
        "modalities": {"input": ["text", "image"], "output": ["text"]}
    }
}
```

---

## 📝 License

MIT — see the [LICENSE](LICENSE) file.
