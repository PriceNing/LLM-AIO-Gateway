# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project overview

LLM AIO Gateway — an all-in-one unified API gateway that proxies multiple LLM providers behind OpenAI/Anthropic-compatible endpoints. Built with FastAPI + liteLLM. **Data stored in SQLite (`data.db`)**; server-level settings in `config.json`.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Start dev server (port 8000, auto-reload)
python main.py

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_config.py -v

# Run a single test by name
pytest tests/test_admin.py::test_list_providers -v
```

## Architecture

```
main.py                         # FastAPI app, mounts routers at /v1 and root level
app/
  config.py                     # Server-level config: host, port, database path, logging, preprocessors
  database.py                   # SQLite schema + CRUD: providers, users, API keys, routing rules, admin accounts, stats
  models.py                     # Pydantic models (ProviderCreate/Update, ChatCompletionRequest, StatsResponse)
  security.py                   # Password hashing (pbkdf2), API key generation, in-memory session tokens
  router/
    auth.py                     # /auth — admin setup/login/logout/session (async)
    admin.py                    # /admin — provider/user/key/routing-rule/preprocessor CRUD, model refresh, stats (async)
    proxy.py                    # All proxy endpoints: /v1/chat/completions, /v1/completions, /v1/models, /v1/messages, /responses
  services/
    lite_llm.py                 # liteLLM wrapper: provider resolution, model name mapping, 4 monkey-patches, completion calls
    discovery.py                # Remote model discovery via provider /models endpoints
    preprocessing.py            # Image preprocessing: turn-aware vision description + image stripping
    logger.py                   # Structured logging: date-based dirs, per-channel levels, liteLLM log capture
  web/static/                   # SPA admin UI (index.html, app.js, styles.css)
tests/                          # pytest + FastAPI TestClient (251 test cases)
```

## Data storage

- **SQLite (`data.db`)**: providers, provider_models, users, user_api_keys, admins, routing_rules, global_stats. Managed by `app/database.py`.
- **`config.json`**: server-level settings — host, port, database path, logging config, `defaults` section for tunable server parameters, and `preprocessors` section for vision model configuration. Overridable via env `LLM_GATEWAY_CONFIG`.
- `config.json` 修改需手动重启服务（uvicorn `reload=True` 只监听 `.py` 文件变更）。
- **`defaults` section keys**: `max_tokens` (16384), `temperature` (0.7), `tool_only_limit` (20), `min_image_max_tokens` (2000), `session_ttl_hours` (12), `request_log_max` (200), `reasoning_cache_ttl` (1800), `reasoning_cache_max_size` (1000), `tool_only_turns_ttl` (600), `tool_only_turns_max_size` (2000), `image_cache_max_size` (500). Accessed via `get_default(key, fallback)` from `app.config`.

## Key details

- **`max_tokens` default:** When the client omits both `max_tokens` and `max_completion_tokens` from the request body, the value from `config.json` → `defaults.max_tokens` is used (default 16384). Previously hardcoded to 4096 which truncated long outputs (e.g. HTML generation). All 4 endpoints (`/v1/chat/completions`, `/v1/completions`, `/v1/messages`, `/responses`) support `max_completion_tokens` as an alternative parameter name.
- **ModelId type:** `database.py` defines `ModelId` (immutable, `__slots__`-based) that encapsulates `provider/model` composite format parsing. All model identifiers flow through the codebase as `ModelId` instances, accessed via `.provider_id`, `.model_name`, `.composite`, `.is_composite` attributes. No manual `"/"` splitting except inside `ModelId.parse()`. `ModelId.__eq__` supports 4-direction comparison (composite↔composite, composite↔simple, simple↔composite, simple↔simple), enabling `parse_model_id(model) in allowed_models` patterns.
- **Auth model:** Admin accounts in SQLite `admins` table with pbkdf2-hashed passwords. In-memory session tokens in `security.py` (`_sessions` dict, 12-hour TTL, daemon cleanup every 5 minutes). End-users with API keys (`sk-aio-` prefix) in `user_api_keys` table, each having per-key `allowed_models` list and per-key usage stats.
- **Provider routing:**
  - `provider_type` is `openai` or `anthropic`.
  - liteLLM maps model names: Anthropic-type providers → `anthropic/<model>`, OpenAI-compatible with custom api_base → `openai/<model>`, OpenAI-hosted (api.openai.com/azure.com) → bare `<model>`.
  - **Anthropic passthrough:** For `provider_type: "anthropic"` providers (e.g. DeepSeek via `https://api.deepseek.com/anthropic`), the gateway bypasses liteLLM entirely and makes direct HTTP calls to `{api_base}/messages` with native Anthropic-format request bodies. This avoids double-formatting (OpenAI↔Anthropic) which loses `tool_use` IDs and breaks multi-turn tool calls. Used in both streaming (`_stream_anthropic_passthrough`) and non-streaming (`_anthropic_passthrough`) paths. `/responses` endpoint also uses passthrough for non-streaming Anthropic providers (streaming keeps liteLLM path because the passthrough outputs Anthropic SSE, not Responses SSE).
  - `_openai_messages_to_anthropic(messages, system_prompt)` converts Chat Completions format to Anthropic native format for the passthrough. Returns `(anthropic_messages, combined_system_text)`.
  - `_attr(obj, key, default)` — safe attribute/key accessor that works with both dicts (`.get()`) and objects (`getattr()`). Used where values may come from either `_anthropic_passthrough` (returns dict message) or liteLLM (returns object).
  - When the upstream returns an error, the passthrough functions parse the response body and forward the exact error message to the client (e.g. `"Upstream: this model does not support image input"`) rather than a generic status code.
- **Model discovery:** `POST /admin/providers/{id}/refresh` probes `{api_base}/models` with multiple auth header formats. New models are appended (never removed).
- **Auth/admin endpoints are async:** `auth.py` and `admin.py` use `async def` to avoid blocking FastAPI's default threadpool. Sync endpoints would compete with liteLLM calls (`to_thread.run_sync`) for the same threadpool.
- **Server restart on Windows:** zombie processes on port 8000 persist after `psutil.kill()`. Use `taskkill //F //IM python.exe` to clean up all Python processes before starting the server.
- **Route mounting:** ALL proxy routes are mounted at both `/v1/...` and root level (`/chat/completions`, `/completions`, `/models`, `/messages`, `/responses`). Both prefixes work identically.

## Routing rules

User-defined rules in SQLite `routing_rules` table that transparently redirect model requests. Evaluated at the start of all four endpoints (`/v1/chat/completions`, `/v1/completions`, `/v1/messages`, `/responses`).

```
Rule structure: {name, enabled, username, api_key_pattern, match_model, target_model, target_provider}
```

Matching logic (`_apply_routing_rules` in proxy.py):
1. `username`: empty = match all, or exact match
2. `api_key_pattern`: empty = match all, or substring match
3. `match_model`: supports `*` wildcard (e.g. `MiniMax-M2*`), matched against the requested model name
4. First matching rule wins — returns `(target_model, target_provider)`

Admin CRUD: `GET/POST /admin/routing-rules`, `PUT/DELETE /admin/routing-rules/{rule_id}`.

**Important:** Routing rules do NOT affect the preprocessor decision — that is always based on the model the user requested (see Image preprocessing below).

## Image preprocessing pipeline

Two-phase design (configured via `config.json` → `preprocessors.{id}` and DB `provider_models.preprocessor`):

**Phase 1 — Description decision (based on requested model):**
- `_maybe_preprocess()` looks up the **requested model** (before routing) in `provider_models.preprocessor`. If the requested model has `preprocessor` set, the first enabled preprocessor from `config.json` is used to generate image descriptions.
- If the requested model does NOT have a preprocessor enabled, images pass through unchanged. This ensures routing transparency: changing the target model via routing rules does not alter whether image descriptions are generated.

**Phase 2 — Image stripping (inside `preprocess_messages`):**
- `_strip_all_images()` removes ALL image content (image_url, input_image, Anthropic image blocks, embedded data URIs) from the entire message array, replacing them with `"[image: removed]"` placeholders. Handles nested images inside `tool_result` blocks. Converts all-text arrays to plain strings for backward compatibility.
- Only runs when a preprocessor is active (as part of `preprocess_messages`).
- **Image replacement format:** For current-turn images with descriptions, the replaced text is `"[Image #N at HH:MM:SS]: <description>"` (no `[image: removed]` prefix, to avoid confusing the model). History images (before `new_turn_start`) get `"[image: removed]"` without description.

**Turn-boundary detection** (`preprocess_messages` in preprocessing.py):
- Only images in the **current turn** are described. The function finds `new_turn_start` by scanning backward for the last assistant message with text content but no tool_calls (previous turn's final reply), or a user message containing `<image_description>` tags.
- Images in history (before `new_turn_start`) are only stripped, never re-described. This prevents old image descriptions from contaminating new turn context.

**Same-turn deduplication:**
- Within a turn, identical images appearing in both user messages and tool_results are deduplicated by MD5 cache key, preventing redundant description generation.

**Description cache:**
- In-memory dict (`preprocessing.py`), keyed by `md5(image_data)[:16]`, max 500 entries, FIFO eviction. Avoids repeated vision calls for the same image.

**Fallback messages (no 500 errors):**
- Vision model unavailable: `"[image: vision model unavailable or timed out]"`
- HTTP error: `"[image: vision model HTTP {status_code}]"`
- Empty response: `"[image: vision model returned empty response]"`
- Generic failure: `"[image: could not be described]"`

**`_reasoning_cache` invalidation:** After preprocessing modifies messages, `_reasoning_cache` is dropped for that conversation to prevent stale visual reasoning from polluting the new context.

## Streaming architecture

- **`_iter_stream_async(stream_func)`** bridges sync liteLLM generators to async. Runs the generator iteration in a daemon thread, feeds chunks through `queue.Queue` to the event loop with `queue.get(timeout=0.01)` + `await anyio.sleep(0)` to yield control. This prevents one streaming request from blocking other concurrent requests.
- **Force-interrupt:** If the background thread is stuck in a blocking read (e.g. llamacpp prefill taking minutes), uses `ctypes.PyThreadState_SetAsyncExc` to forcibly raise an exception in the thread after stream_gen.close().
- **Four streaming generators:**
  - `_stream_chat` — /v1/chat/completions SSE (OpenAI format)
  - `_stream_responses` — /responses SSE (Codex Responses API format)
  - `_stream_completions` — /v1/completions SSE (text completion format)
  - `_stream_anthropic_messages` — /v1/messages SSE (Anthropic format, for non-Anthropic providers; Anthropic providers use `_stream_anthropic_passthrough`)
- **Cancellation:** `_iter_stream_async` uses a `threading.Event` cancel signal. When the client disconnects, the async generator's `finally` block sets `cancel`, the background thread checks it between chunks and exits early, avoiding wasted LLM provider bandwidth.

## Anthropic Messages API (/v1/messages)

Bidirectional conversion between Anthropic and OpenAI formats:
- `_anthropic_to_openai_messages()`: Converts Anthropic `tool_use`/`tool_result` blocks to OpenAI `tool_calls`/`tool` role messages, preserving image content. Handles nested images in tool_result content blocks. Converts Anthropic `system` array format (content blocks with `cache_control`) to plain string for non-Anthropic providers.
- `_anthropic_content_to_openai()`: Converts Anthropic content blocks to OpenAI format. **Filters empty text blocks** (`text: ""`) to prevent upstream rejections (Moonshot returns "text content is empty").
- `_openai_to_anthropic_content()`: Converts OpenAI assistant message (text + tool_calls) back to Anthropic content blocks.
- `_map_stop_reason()`: Maps OpenAI finish_reason to Anthropic stop_reason.
- If the target provider is Anthropic-native (provider_type: "anthropic"), these conversions are skipped and messages pass through directly via the Anthropic passthrough.
- **Tool result fallback:** When all text parts in a tool_result are empty, content falls back to `"(tool output)"` to avoid empty-content rejections.

## Responses API (/responses — Codex/OpenAI compatible)

Converts OpenAI Responses API format to Chat Completions format:
- `_convert_responses_input()`: Handles `message`, `function_call`, `function_call_output`, `reasoning` (skipped), `input_image` (attached to nearest preceding user message). Preserves `reasoning_content` from assistant messages and function_call items for DeepSeek multi-turn continuity.
- `_convert_responses_tools()`: Converts Responses API flat tool format to Chat Completions nested format. Filters out Codex-specific types (`web_search`, `custom`). Strips OpenAI-specific fields (`strict`, `additionalProperties`) that non-OpenAI providers reject.
- `_stream_responses()`: Generates Responses API SSE events — `response.created`, `response.in_progress`, `response.output_item.added`, `content_part.added`, `output_text.delta`, `function_call_arguments.delta`, `response.output_item.done`, `response.completed`. On error, yields `type: "error"` event, then `response.completed` with `status: "failed"` and `status_details.error.message` (Codex CLI reads errors from this field).
- MiniMax `tool_choice="auto"` is stripped before forwarding (rejected with error 2013).

## Message normalization

`_normalize_messages()` merges consecutive same-role messages (for `system` and `user` roles) into a single message. Many providers (MiniMax, DeepSeek) require strict user/assistant/tool alternation and reject consecutive same-role messages. Called in `/v1/chat/completions` and `/responses` endpoints. Does NOT merge `assistant` or `tool` roles to preserve tool_use/tool_result adjacency.

## Stateful caches (in-process memory)

All caches use `TTLDict` (defined in proxy.py) — a thread-safe dict with `threading.Lock`, TTL-based lazy eviction, and max-size cap. Atomic `increment()`/`reset()` methods avoid get-then-set races.

| Cache | TTL | Max size | Purpose |
|-------|-----|----------|---------|
| `_reasoning_cache` | 30 min | 1000 | Stores DeepSeek `reasoning_content` per conversation, injected on next turn |
| `_tool_only_turns` | 10 min | 2000 | Counts consecutive tool-only responses per conversation for circuit breaker |

Both caches keyed by `_conversation_cache_key(api_key, messages)` which hashes the first user message (MD5 16-char hex) to isolate different conversations sharing the same API key.

**Not shared across workers:** if running with `--workers > 1`, these caches are per-process. For multi-worker deployments, external storage (Redis) would be needed.

## Circuit breaker

`TOOL_ONLY_LIMIT` (from `config.json` → `defaults.tool_only_limit`, default 20): after N consecutive assistant responses that have tool_calls but no text content (for the same conversation), the system strips `tools` and `tool_choice` from subsequent requests. Counter is reset on any response with text content or on error. Counter uses `TTLDict.increment()` and `TTLDict.reset()` for thread-safe atomic updates. Checked in `_stream_chat`, `chat_completions` non-streaming, and `_stream_responses`.

## Think-tag stripping

DeepSeek thinking mode wraps reasoning content in `<think>...</think>` tags inside the text content. The gateway strips these inline:
- `_extract_and_strip_think()`: Extracts thinking from a text buffer, returns (remaining_buffer, extracted_thinking). Handles nested `<think>` blocks.
- `_strip_think_tags()`: Simple regex removal of `<think>...</think>` from final text.
- Used in `_stream_chat` and non-streaming `chat_completions`. Text is buffered until `</think>` is seen, then only post-think content is emitted to the client. If no `<think>` tags are found, the buffer is flushed immediately.

## Tool-call sanitization

`_fix_tool_args()` and `_sanitize_args()` fix malformed JSON in tool-call arguments where MiniMax emits bare JavaScript `undefined` values (e.g. `url:undefined`), which are invalid JSON. Applied in both streaming and non-streaming responses across all endpoints.

## DeepSeek reasoning_content

DeepSeek thinking mode requires `reasoning_content` to be echoed back in multi-turn conversations. The gateway handles this transparently:

- **Injection** (before LLM call): `_reasoning_cache` is checked, cached content injected into all assistant messages missing it. Applies to both `/v1/chat/completions` and `/responses` endpoints.
- **Storage** (after LLM response): `reasoning_content` from the response is cached. Applies to streaming (`_stream_chat`, `_stream_responses`) and non-streaming (`chat_completions`).
- `_convert_responses_input()` also preserves `reasoning_content` from incoming assistant messages and function_call items (Codex may attach it to either).

## Upstream error mapping

`_friendly_error_msg()` in proxy.py maps known upstream error patterns to Chinese-friendly messages, applied to all 8 error paths (4 streaming + 4 non-streaming). Unmatched errors pass through unchanged. Logs always record the original error for debugging.

Current mappings:
| Pattern | Friendly message |
|---------|-----------------|
| `output new_sensitive (1027)` | 内容被上游安全策略拦截（输出端） |
| `input new_sensitive` | 内容被上游安全策略拦截（输入端） |
| `No endpoints found that support image input` | 该模型不支持图像输入，请在管理面板开启图像预处理 |
| `content_filter` | 内容被上游安全策略拦截 |
| `content_policy_violation` | 内容违反上游使用策略 |
| `safety_rating` | 内容未通过上游安全评级 |

Add new patterns to `_UPSTREAM_ERROR_MAP` list in proxy.py.

## liteLLM monkey-patches (4 total in `lite_llm.py`)

1. **`id` coercion + `reasoning_content` field:** Patches `AnthropicResponse`, `AnthropicResponseContentBlockToolUse`, `ModelResponse`, `ChatCompletionChunk` to coerce `id` to `str` (MiniMax Anthropic endpoint returns integer). Adds `reasoning_content` field to liteLLM's `Message` and `Delta` Pydantic models via `model_fields` injection + `model_rebuild(force=True)`.

2. **`reasoning_content` preservation:** Wraps `litellm.utils.convert_to_model_response_object` to inject `reasoning_content` from raw API response dict into each choice's `message` object — liteLLM's original function only copies `content`, `role`, `function_call`, `tool_calls`.

3. **Force-disable thinking for non-Anthropic endpoints:** Patches `AnthropicChatCompletion.completion` to inject `thinking={"type": "disabled"}` when `api_base` is not `api.anthropic.com`. Without this, DeepSeek's Anthropic endpoint enables thinking mode, liteLLM's streaming adapter drops thinking blocks, causing `reasoning_content` loss and "thinking must be passed back to the API" errors on the next turn.

4. **`reasoning_content` → `thinking_blocks` conversion:** Patches `anthropic_messages_pt` in liteLLM's prompt templates factory to synthesize `thinking_blocks` from `reasoning_content` before liteLLM processes messages. Without this, liteLLM drops injected reasoning_content during OpenAI→Anthropic message conversion.

## `clean_params()` in lite_llm.py

Strips None values, stream_options, and empty extra_body from parameters before passing to liteLLM. Prevents provider-specific 400 errors (e.g. MiniMax rejects stream_options, and some providers reject empty extra_body).

## Code conventions

- **Imports at top:** no `import sys` or `import time` inside function bodies. All imports at module level.
- **User/key extraction:** every endpoint extracts `username = user.get("username", "legacy")` and `api_key_value = api_key.get("key", "")` once at the top. Never use `user["username"]` (may KeyError) or `user.get("username")` (returns None) directly in conditional checks.
- **Tool calls serialization:** liteLLM streaming returns ChatCompletionDeltaToolCall Pydantic objects in delta.tool_calls. These must be converted to plain dicts via 	c.model_dump(exclude_none=True) before json.dumps(). Never assign liteLLM/openai types directly into dicts that will be JSON-serialized. Applies to _stream_chat(), non-streaming chat_completions(), and _stream_anthropic_messages() in proxy.py.
- **JavaScript U+2028/U+2029:** LINE SEPARATOR (U+2028) and PARAGRAPH SEPARATOR (U+2029) are invisible Unicode characters that JavaScript treats as line terminators. Always use ` ` and ` ` escape sequences in JS source, never the literal characters.
- **Clipboard API:** `navigator.clipboard.writeText()` is only available in secure contexts (HTTPS/localhost). `app.js` `copyText()` has a fallback using `document.execCommand('copy')` for HTTP remote access.
