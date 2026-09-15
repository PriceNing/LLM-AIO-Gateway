# Live Gateway Evaluation

`tools/live_eval/live_eval.py` runs real requests against an already deployed LLM AIO Gateway and its configured upstream providers. It is intentionally separate from the normal pytest suite because it can spend tokens, depends on live provider health, and may exercise every model visible to the supplied API key.

## Quick Start

```powershell
Copy-Item tools/live_eval/live-eval.config.example.json tools/live_eval/live-eval.config.local.json
# Edit tools/live_eval/live-eval.config.local.json and fill in base_url/api_key.
python tools/live_eval/live_eval.py --limit 2
```

Reports are written to `reports/live-eval/` as JSON and Markdown.

## Useful Options

```powershell
# Test specific models only
python tools/live_eval/live_eval.py --model provider/model-a --model provider/model-b

# Use another config file
python tools/live_eval/live_eval.py --config local-live-eval.json

# Skip token-expensive capability probes
python tools/live_eval/live_eval.py --skip-multimodal --skip-stream

# Probe tools/multimodal even when the model does not advertise the capability
python tools/live_eval/live_eval.py --ignore-capabilities

# Fail unless every reachable client x upstream cell is covered by this run
python tools/live_eval/live_eval.py --require-matrix

# Fail when any model ends with zero scored cases (everything skipped/unsupported)
python tools/live_eval/live_eval.py --require-signal

# Keep the gateway's Responses capability probe cache as-is (by default the tool
# clears it per openai-type model so the native path is re-evaluated every run)
python tools/live_eval/live_eval.py --keep-capability-cache

# Include admin dashboard request logs in each case result
# Fill admin_username/admin_password in live-eval.config.local.json.
python tools/live_eval/live_eval.py --model provider/model-a

# Ask a configured gateway model to judge the run summary
# Fill judge_model in live-eval.config.local.json.
python tools/live_eval/live_eval.py --model provider/model-a
```

Default config file: `tools/live_eval/live-eval.config.json`. The Windows launcher prefers `live-eval.config.local.json` when present.

Example:

```json
{
  "base_url": "http://localhost:8000",
  "api_key": "sk-aio-...",
  "admin_username": "admin",
  "admin_password": "password",
  "judge_model": "provider/judge-model",
  "models": ["provider/model-a"],
  "limit": 0,
  "timeout": 120,
  "output_dir": "reports/live-eval",
  "skip_multimodal": false,
  "skip_stream": false,
  "ignore_capabilities": false,
  "require_matrix": false,
  "require_signal": false,
  "keep_capability_cache": false
}
```

Command-line arguments override the JSON file. Environment variables are still supported as a fallback for automation.

Config fields:

| Variable | Meaning |
|---|---|
| `base_url` | Gateway base URL. Defaults to `http://localhost:8000`. |
| `api_key` | User API key used for `/v1/models` and proxy endpoints. Required. |
| `admin_username` | Optional admin username for `/admin/stats` logs. |
| `admin_password` | Optional admin password for `/admin/stats` logs. |
| `judge_model` | Optional model ID used as an AI judge through `/v1/chat/completions`. |
| `models` | Optional list of model IDs. Empty means all models returned by `/v1/models`. |
| `limit` | Optional maximum number of models to test. |
| `timeout` | Per-request timeout in seconds. Defaults to `120`. |
| `output_dir` | Report output directory. Defaults to `reports/live-eval`. |
| `skip_multimodal` | Skip image probes when true. |
| `skip_stream` | Skip SSE probes when true. |
| `ignore_capabilities` | Probe tools/multimodal regardless of advertised capabilities when true. |
| `require_matrix` | Exit non-zero unless every reachable client x upstream cell is covered. |
| `require_signal` | Exit non-zero when any model has zero scored cases this run. |
| `keep_capability_cache` | Do not reset the Responses probe cache before openai-type models (see Safety Notes). |

## What It Tests

Capability probes (tool calls, multimodal) are gated on the `supports_tools` /
`supports_vision` metadata advertised by `/v1/models`: a model that does not advertise a
capability records `skip` cases without sending any request, so the run does not burn
tokens on probes the contract says are unsupported. Use `--ignore-capabilities` to force
the probes anyway. Note this trusts the gateway's own advertised metadata — if that
metadata is wrong, capability regressions can slip through as `SKIP`; when a model shows
mostly skips, re-run with `--ignore-capabilities` to verify the metadata itself.

With admin credentials configured, every case additionally asserts the **actual upstream
protocol**: the tool reads `upstream_endpoint`/`responses_mode` from the admin request log
and compares it against the protocol implied by the model's `provider_type` (anthropic ->
`messages`; openai -> `chat_completions`; the `/responses` endpoint may legitimately use
either native `responses` or the Chat compatibility path). A wrong adapter is a hard FAIL.
The run ends with a client x upstream coverage matrix; `--require-matrix` turns uncovered
reachable cells into a non-zero exit code.

For each model returned by `/v1/models`, the script probes:

- OpenAI Chat Completions multi-turn text.
- Legacy OpenAI Completions text.
- Anthropic Messages multi-turn text.
- OpenAI Responses first turn and follow-up with `previous_response_id`.
- Tool-call forcing on Chat Completions, Messages, and Responses.
- Multimodal image input on Chat Completions, Messages, and Responses unless skipped.
- Streaming on Chat Completions, Messages, and Responses unless skipped.

The built-in score is structural: HTTP status, response shape, non-empty text, expected tool-call blocks, and expected SSE events. It does not claim the model is semantically good. If `LLM_AIO_JUDGE_MODEL` is set, the judge model receives each model's case summaries, response excerpts, and optional admin logs, then returns a second-pass verdict.

## Interpreting Results

Every case gets one of four verdicts:

- `pass`: endpoint and expected response structure worked (counted in the score).
- `fail`: gateway-side failure — HTTP error or timeout on text/stream probes, empty output, or missing expected shape (counted in the score).
- `skip`: the capability was not advertised by `/v1/models`; no request was sent (not counted).
- `unsupported`: a capability probe (tool/multimodal) was rejected by the upstream with
  HTTP **4xx** — an upstream/model limitation, not a gateway failure (not counted).
  Only request-content rejections (400/422-style) qualify: 5xx/504 stay `fail`
  (gateway/infra), 429 stays `fail` (transient rate limit), 401/403 stay `fail`
  (smoke-test key misconfigured), and 404/405/410 stay `fail` (model/endpoint does
  not exist upstream — a configuration error, never a capability gap). A model
  advertising `supports_vision` while its upstream rejects images usually points at a
  capability-metadata issue worth investigating separately.

Case scores run from `0.0` to `1.0` (`1.0` worked; `0.4` returned but text was empty or incomplete; `0.2` returned but the expected shape was missing; `0.0` failed). A model score averages only `pass`/`fail` cases, and the process exits non-zero only when at least one `fail` case exists.

A model advertising `supports_vision` while its upstream rejects images usually points at a capability-metadata issue worth investigating separately. The report is most useful for comparing the same gateway configuration over time and catching regressions in adapter compatibility.

## Safety Notes

- By default, before testing each openai-type model the tool calls
  `POST /admin/models/responses-capability/reset` to clear that model's Responses probe
  cache on the **target gateway** (a side effect: the next `/responses` request performs
  a real native probe). Use `--keep-capability-cache` / `keep_capability_cache` to skip this.
- Start with `--limit 1` or explicit `--model` while tuning prompts.
- Use `--skip-multimodal` if image preprocessing invokes an expensive vision model.
- Use a dedicated test API key with restricted allowed models when possible.
- Avoid running against production keys during heavy traffic unless you intentionally want a production smoke test.
