# Changelog

All notable changes to LLM AIO Gateway will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] —— 发版时将本段重命名为 [0.13.0] 并补日期；在此之前任何对外面（UI 版本号、tag、Release）不得出现 0.13.0

### Added
- **Per-conversation image-generation budget**: image generation cost is now controlled post-hoc by a per-conversation budget (`state.charge_image_generation_budget`, default 20 images/hour), replacing the removed pre-flight NLP intent gating.
- **Responses protocol modules extracted**: `app/protocols/responses_features.py` (client wire-format feature analysis, computed at the protocol boundary and written into `InternalRequest.metadata`) and `app/services/responses_capability.py` (native-Responses upstream capability state: probing, TTL freshness, negative caching, degradation) — both lifted out of `router/proxy.py` so endpoint code stays thin.

### Changed
- **Image bridge is now purely model-driven**: the gateway always injects the bridge tool for image-capable models (same `should_inject_image_bridge` gate for chat and responses; future /messages reuses it) and never gates on NLP intent. All correction / forced `tool_choice` / empty-response 502 logic is removed; if the model does not call the tool the response passes through untouched.
- **Bridge tool renamed** `llm_aio_image_generation` → `image_generation` (the hosted-tool name models already know from training corpora).
- **Chat streaming with the bridge is now true streaming**: text is forwarded token-by-token; if the model calls the bridge tool, the gateway executes image generation at end-of-stream and appends the result as a continuation of the same SSE stream (with a short "generating" notice). Non-bridge requests are unaffected. Chat image results are rendered as download URLs only (no inline base64) to avoid bloating conversation history.
- Request-log hygiene: the bridge's interim "running" log row is overwritten by the final log (shared `log_id`); mixed tool rounds keep `finish_reason=tool_calls` consistency.
- Admin UI: image-bridge related settings surfaced.

### 更新内容（中文）
- **每会话生图预算**：生图成本现由每会话预算后置控制（`state.charge_image_generation_budget`，默认 20 张/小时），取代移除的预检 NLP 意图门控。
- **Responses 协议模块抽离**：`app/protocols/responses_features.py`（客户端 wire 格式特征分析，在协议边界计算并写入 `InternalRequest.metadata`）与 `app/services/responses_capability.py`（native Responses 上游能力状态：探测、TTL 新鲜度、负向缓存、降级判断）——均从 `router/proxy.py` 迁出，端点代码保持精简。
- **生图桥接改为纯模型驱动**：对具备生图能力的模型恒定注入桥接工具（chat/responses 同一入口 `should_inject_image_bridge`，未来 /messages 直接复用），不再做任何 NLP 意图检测；删除全部 correction / 强制 tool_choice / 空响应 502 逻辑，模型不调用工具就原样 passthrough。
- **桥接工具改名** `llm_aio_image_generation` → `image_generation`（与宿主工具同名，模型训练语料中已知，提高调用命中率）。
- **Chat 流式桥接现为真流式**：文本逐 token 转发；模型调用桥接工具时，网关在流结束时执行生图并把结果作为同一 SSE 流的续轮追加（带简短"生成中"提示）。非桥接请求不受影响。Chat 图像结果仅渲染为下载 URL（不内联 base64），避免膨胀对话历史。
- 请求日志卫生：桥接临时"running"日志行被最终日志覆盖（共享 `log_id`）；混合工具轮保持 `finish_reason=tool_calls` 一致性。
- 管理 UI：暴露图像桥接相关设置。

### Tests
- `test_chat_image_bridge.py`（+211）+ `test_responses_native.py`（+63）+ `test_image_intent.py`（+62）+ `test_image_generation.py`（精简）；全量 **995 passed**。

## [0.12.3] - 2026-09-21

### Added
- **Image-generation bridge now shared across protocol endpoints**: the model-driven image-generation orchestration loop was lifted out of the `/responses` endpoint into a protocol-agnostic module (`app/core/image_orchestration.py`) operating on the internal representation, with all endpoint/client-specific behavior injected as callables. `/chat/completions` now supports the same image-generation bridge in non-streaming form, so image generation works uniformly across protocols.
- `app/core/image_intent.py` now exposes a pure-string `is_image_generation_intent_text()` helper, decoupling intent detection from the request object.
- **Note**: the Anthropic `/messages` endpoint does not yet support the image-generation bridge (not wired in this release).

### Changed
- `tools/live_eval/live_eval.py` image probes updated to cover the new cross-protocol bridge.

### 更新内容（中文）
- **图像生成桥接跨协议共享**：把「模型驱动生图」编排循环从 `/responses` 端点抽到协议无关模块（`app/core/image_orchestration.py`），基于内部表示（IR）运作，端点/客户端特定行为以可调用注入。`/chat/completions` 现以非流式形式支持同一图像生成桥接，图像生成在各协议端点行为统一。
- `app/core/image_intent.py` 现暴露纯字符串函数 `is_image_generation_intent_text()`，把意图检测与请求对象解耦。
- **说明**：Anthropic `/messages` 端点暂未支持图像生成桥接（本次未接入）。
- `tools/live_eval/live_eval.py` 图像探针更新，覆盖新的跨协议桥接。

### Tests
- `test_chat_image_bridge.py`（新，240 行）+ `test_live_eval_probes.py`（+74）+ `test_image_intent.py`（+36）；全量 **996 passed**。

## [0.12.2] - 2026-09-21

### Fixed
- Chat stream adapters no longer mask silent truncation: if the upstream closes without a `finish_reason` and produced no answer (text/tool calls; reasoning-only counts as none), `iter_openai_chat_output_events` raises a confirmed-upstream error instead of synthesizing `message_done("stop")`. The error path records a failure and sends an error SSE so clients can retry (SuperGrok 2026-09-16 watchdog incident). Streams that already produced answer text/tool output without a finish chunk keep the legacy synthesized completion. Empty streams still participate in fallback / same-target degenerate retry.

### 更新内容（中文）
- 修复 Chat 流式适配器静默截断：上游不发 `finish_reason` 且无回答（正文/工具；仅 reasoning 不算）时改为报确认的上游失败并下发 error SSE，不再合成 `message_done("stop")` 记成功（SuperGrok 2026-09-16 服务端掐流事故）。已有正文/工具但缺 finish 的流仍走旧合成完成。空流仍可走 fallback / 同目标退化重试。

## [0.12.1] - 2026-09-16

### Added
- **`supports_reasoning` capability field**: `GET /v1/models` now also advertises whether a model supports reasoning. Sourced from builtin high-confidence families (o-series, gpt-5/6, claude-4, full deepseek line, qwen3, gemini-2.x, kimi-k2, ...), upstream/registry extraction (from `supported_parameters` containing `reasoning`/`include_reasoning`), normalized on write and read, positive-declaration only, with a three-state admin editor and a 🧠 badge in the panel.
- **Leaked tool-call rescue layer** (`core/tool_leak.py`, new module): some local inference frameworks (llama.cpp etc.) probabilistically fail to parse template-native XML tool calls into structured `tool_calls`, leaving raw markup in the message body; harnesses like Codex then treat the turn as plain text and end it (observed ~5% on llama.cpp b10884, causing hangs). On the non-stream IR normalization point this layer does a conservative, schema-driven rescue — only when (1) the turn declared tools, (2) the whole body is exactly 1..N well-formed, adjacent tool-call blocks, (3) every tool name matches the declared set, (4) every parameter name/type validates against the schema, and (5) block count/depth are within limits. Any single condition failing → pass through unchanged (prefer under-repair over mis-repair). Hits are counted as `tool_leak_repaired` in details + the `tool_calls` log channel; streaming is detect-only (`tool_leak_detected`) to avoid buffering that hurts TTFT. New config `repair_tool_leaks` (default on); wired into 5 non-stream return points + 5 stream call sites.

### Changed
- `discovery._pick_positive_int` now rejects bools so capability ints stay ints.

### 更新内容（中文）
- **`supports_reasoning` 能力字段**：`GET /v1/models` 现额外广播模型是否支持推理。来源：内置高置信家族（o 系、gpt-5/6、claude-4、deepseek 全系、qwen3、gemini-2.x、kimi-k2 等）、上游/注册表提取（`supported_parameters` 含 `reasoning`/`include_reasoning`）、写入与读取都归一化、只做正向声明，管理面板三态编辑 + 🧠 徽章。
- **泄漏工具调用抢救层**（`core/tool_leak.py`，新模块）：部分本地推理框架（llama.cpp 等）会概率性无法把模板原生 XML 工具调用解析成结构化 `tool_calls`，原始标记文本留在正文里；Codex 等 harness 收到后误判回合结束而卡死（llama.cpp b10884 实测 ~5%）。在非流式 IR 规范化点做保守、schema 驱动的抢救——仅当（1）本轮声明了工具、（2）正文整体恰好是 1..N 个良构且紧邻的工具调用块、（3）每个工具名命中声明集、（4）每个参数名/类型通过 schema 校验、（5）块数/深度在上限内，才把正文替换为结构化工具调用；任何一条不满足即原样透传（宁可漏修，绝不误伤）。命中记 `tool_leak_repaired` 进 details + `tool_calls` 日志通道；流式仅检测（`tool_leak_detected`）不做修改，避免缓冲伤 TTFT。新配置 `repair_tool_leaks`（默认开），接入 5 个非流式返回点 + 5 个流式调用点。

### Tests
- `test_tool_leak.py`（17 用例：事故同构样本、误伤防护矩阵、端点集成、流式检测、开关）+ `test_model_capabilities.py`（+4）；全量 **968 passed**。

## [0.12.0] - 2026-09-15

### Added
- **Model capability metadata system**: `GET /v1/models` now advertises per-model capabilities (context window, max output tokens, vision/tool support, input modalities, pricing), merged from four layers — builtin family heuristics, an online OpenRouter-style registry (persisted in the DB, TTL refresh with failure backoff, SSRF-guarded fetch), upstream `/models` passthrough, and admin overrides (`PUT /admin/models/capabilities`). Capability booleans are positive-declaration only; every value is normalized on both write and read paths.
- Client-visible upstream error status-code mapping via a single source of truth `classify_for_client()` in `core/text.py`: authoritative upstream status wins (4xx kept as-is), upstream 401/403 -> 502 with a dedicated gateway-credentials message, 408 / isinstance-level timeout evidence -> 504, upstream 5xx and non-4xx authoritative statuses -> 502, unclassifiable -> 500 (reserved for gateway bugs). Text heuristics can never veto an authoritative status nor imply a specific 4xx/429.
- `POST /admin/models/responses-capability/reset`: clears the native Responses probe cache per provider/model (also accepts bare model names cross-provider), giving ops and live-eval a deterministic re-probe hook.
- `InternalOutputMessage.request_details`: dedicated field for per-request upstream metadata (`upstream_endpoint`, fallback info), so liteLLM non-stream successes finally record it (previously silently dropped because `raw` is a `ModelResponse` object).
- Live-eval (`tools/live_eval/live_eval.py`): capability-gated probes from `/v1/models` metadata (skip without sending), four-state verdicts (pass/fail/skip/unsupported, 4xx-only rejections), per-case client x upstream protocol assertions with a 9-cell coverage matrix, `--ignore-capabilities` / `--require-matrix` / `--require-signal` / `--keep-capability-cache`, real 64x64 probe PNG (1x1 placeholders are rejected as invalid by upstreams), reasoning-model-safe probe budgets (512 tokens for Chat/Messages, 2048 via the protocol-correct `max_output_tokens` for Responses).
- Error-mapping baseline gate: `tools/scripts/check_error_mapping.py` (diff corpus / required assertions / hardcoded-status whitelist / doc count consistency) plus `tests/test_error_mapping_gate.py` so checks 1-3 run with every `pytest` invocation.
- README (zh/en): new "client-visible error status codes" contract section; AGENTS/CLAUDE: error-mapping invariants recorded.
- Tests: `test_request_details` / `test_upstream_error_status` / `test_live_eval_probes` / `test_error_mapping_gate`, image-backend status mapping parametrization; full test suite at **947 passed**.

### Fixed
- A native Responses request carrying tools that receives an authoritative 4xx (e.g. DeepSeek thinking mode rejecting a forced `tool_choice`, or an upstream rejecting Codex-style `custom` tools) now records a **tool-shape-level negative capability** (`responses_tools_status`/`responses_tools_expires_at` on `provider_models`, cleared by the admin reset endpoint) instead of demoting the model-wide native capability: subsequent tool-carrying `/responses` requests route straight to the Chat compatibility path — critical for client-owned tools where a silent downgrade after the fact is disallowed — while text/stream traffic stays on native. A native success with tools clears the negative. (Regression guard: `test_responses_native.py`.)
- `responses_error` capability-cache entries and the native-fallback warning now carry the full upstream response body (via `error_detail_for_log`), so rejection reasons are readable without digging through `request_logs`.

### Changed
- **Behavior change (external contract)**: Anthropic adapter paths no longer pass upstream 401/403 through and no longer echo raw upstream error text (including SSE `error` event messages) to clients; raw text now goes to server logs only and clients receive a safe message plus `request_id`.
- Proxy and image-generation endpoints no longer hardcode `status_code=500/502` for upstream failures — all route through the mapping function.
- `run-live-eval.bat` quick mode no longer pins a stale model id (uses `--limit 1`).

### 更新内容（中文）
- **模型能力元数据系统**：`GET /v1/models` 现广播每模型能力（上下文窗口、最大输出 token、视觉/工具支持、输入模态、定价），四层合并——内置家族启发式、在线 OpenRouter 式注册表（持久化、TTL 刷新+失败退避、SSRF 防护抓取）、上游 `/models` 透传、管理员覆盖（`PUT /admin/models/capabilities`）。能力布尔只做正向声明；所有值在写入与读取路径都归一化。
- 新增客户端可见上游错误状态码映射，单一事实来源 `classify_for_client()`：权威状态码优先（4xx 保留）、上游 401/403 → 502（独立凭据文案，不伪装成客户端 key 失效）、408/真实超时 → 504、上游 5xx 及非 4xx 权威状态 → 502、无法归类 → 500（保留给网关内部错）；文本启发式不得否决权威状态码或推出具体 4xx/429。
- 新增 `POST /admin/models/responses-capability/reset`：按 provider/模型清除原生 Responses 探测缓存（支持裸模型名跨 provider），为运维与冒烟提供确定性重探钩子。
- `InternalOutputMessage` 新增 `request_details` 专用字段：修复 liteLLM 非流式成功请求丢失 `upstream_endpoint` 等元数据的观测缺口（旧版往 `raw` 对象挂 dict 属性静默失败）。
- live-eval 冒烟工具：能力元数据门控探针、四态判定（pass/fail/skip/unsupported）、逐用例客户端×上游协议断言与 9 格覆盖矩阵、真实 64×64 探针图、推理模型友好的探针预算（Chat/Messages 512；Responses 改用协议合法的 `max_output_tokens` 并提至 2048）、四个新开关。
- 修复：含 tools 的原生 Responses 请求收到权威 4xx（如 thinking 模式拒绝强制 tool_choice、上游拒绝 Codex 式 custom 工具）时，改为记录**工具形态级负向能力**（`provider_models` 新增 `responses_tools_status`/`responses_tools_expires_at`，重置端点一并清除）：后续带工具请求直接走 Chat 兼容路径（client-owned 工具不允许事后静默降级，这点关键），文本/流式继续原生；带工具的原生成功会解除负向记录（`test_responses_native.py` 防回归）。
- 修复：能力缓存的 `responses_error` 与降级 warning 现包含上游响应体（`error_detail_for_log`），拒绝原因无需再翻 `request_logs`。
- 收口基线固化：`tools/scripts/check_error_mapping.py` 四项检查 + pytest 门禁用例；README/AGENTS/CLAUDE 写入错误映射契约。
- **对外行为变更**：Anthropic 路径上游 401/403 不再透传（改 502），上游错误原文（含 SSE error 事件）不再出现在客户端响应中，仅落服务端日志。
- 全量测试 **947 passed**。

## [0.11.0] - 2026-09-08

### Changed
- Comprehensive hardening and fixes across the gateway based on code review, covering the protocol layer, database, proxy orchestration, and image pipeline.
- Database migration idempotency fixed to avoid re-execution or inconsistent state.
- Admin pagination interaction and request-log statistics corrected for accurate, consistent data.
- Tightened protocol boundary handling (OpenAI/Anthropic/Responses) for more robust cross-protocol conversion.

### Added
- Admin UI responsive/mobile support: tablet and phone breakpoints (900px/600px), touch-scroll optimizations, and small-screen layout adjustments in `styles.css`.
- Extensive regression tests (test_database_admin_fixes / test_protocol_adapter_fixes / test_proxy_image_fixes) strengthening coverage of protocol adapters, database management, and the image pipeline.

## [0.10.3] - 2026-09-08

### Added
- Tool-history sanitization: removes malformed tool-call/tool-result pairs that strict Chat upstreams reject.
- Same-target first-byte retry: on a first-byte upstream failure (e.g. connection error), retry the same target once before falling back.

### Changed
- Connection-error classification walks the exception chain to detect real connection errors (httpx.ConnectError / NetworkError / ConnectionError), avoiding misclassifying generic RuntimeError as retryable.

## [0.10.2] - 2026-09-08

### Added
- Streaming performance metrics: request logs now record time-to-first-token (TTFT) and streaming generation duration for upstream performance visibility.

### Changed
- Image-intent detection strips Codex XML envelopes (`<environment_context>`, `<thread_title>`, etc.) to correctly identify real user image intent.
- Responses `thread_title` turns are recognized as system turns so title-generation and other meta requests no longer trigger the image bridge.
- Admin UI Chinese localization: unified UI copy (Fallback policies, Dry Run, API Key, etc.) into Chinese.

### Fixed
- Plural image-intent keywords (posters/images/avatars) now match correctly, restoring image generation for plural requests.

## [0.10.1] - 2026-09-08

### Fixed
- Hardened the native Responses capability probe so upstream support is detected more reliably, avoiding misclassification that caused unnecessary fallback or failures.
- Improved fallback handling for Responses requests: native Responses failures now fall back to the OpenAI Chat path more reliably.
- Fixed tool-history related errors: handles serialization/replay anomalies in Responses tool-call history, reducing 4xx failures.

### Changed
- Synced README and usage docs with 0.10.0 features, config options, and the full endpoint inventory.

## [0.10.0] - 2026-09-08

### Added
- Request body size limit (default 32 MiB, configurable) to prevent oversized JSON bodies from exhausting gateway memory.
- Shared upstream connection pool: Anthropic and native Responses adapters reuse TCP/TLS connections to direct upstreams; reference counting keeps in-flight streaming responses alive.
- Upstream URL validation (SSRF guard) rejecting non-http(s) schemes, cloud metadata endpoints, and reserved addresses; private addresses allowed by default for self-hosted inference.
- Codex-compatible image generation through `/responses` and `/images/generations`, with batch execution, short-lived originals, compressed previews, idempotent retries, and image statistics.
- Configurable request-log payload capture and structured secret redaction.
- Admin login attempt throttling and image-result download network/size safeguards.

### Changed
- Production startup now disables Uvicorn reload unless `reload` is explicitly enabled.
- Provider model discovery follows paginated model lists before replacing stored models.
- Live evaluation tools and examples use UTF-8 and their current `tools/live_eval/` paths.

## [0.9.0] - 2026-08-04

### Added
- Global image-generation backend configuration and per-chat-model image-generation switches.
- Generated-image request logging and usage statistics in the admin interface.

## [0.8.4] - 2026-08-02

### Changed
- Routing, streaming fallback, and Responses compatibility fixes.

## [0.1.0] - 2026-06-01

### Added
- FastAPI gateway exposing OpenAI Chat Completions, legacy Completions, Anthropic Messages, and OpenAI Responses endpoints at root and `/v1` paths.
- Protocol-neutral internal request/output representation with edge adapters for OpenAI/liteLLM and direct Anthropic Messages.
- Routing rules, fallback policies, image preprocessing, reasoning continuity, tool-call repair, and tool-only circuit breaking.
- Admin SPA for managing providers, API keys, users, routing rules, fallback policies, and live stats.
- Standalone (green) distribution pipeline using python-build-standalone and a Tkinter launcher.
- `tools/scripts/bump_version.py` to bump versions and keep documentation snippets in sync.
- GitHub Actions release workflow that cross-builds Windows/macOS/Linux green packages on tag push.

### Changed
- `main.py` reads the FastAPI version from `app.__version__` instead of a hard-coded constant.
- Self-update check in the launcher reads from the GitHub Releases API.

### Notes
- The single source of truth for the project version is `app/__init__.py.__version__`.
- Use `python tools/scripts/bump_version.py <new-version>` to cut a new release.
