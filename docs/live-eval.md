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
  "skip_stream": false
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

## What It Tests

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

Scores are per case from `0.0` to `1.0`; a model score is the average of its cases.

- `1.0`: endpoint and expected response structure worked.
- `0.4`: endpoint returned but text was empty or incomplete.
- `0.2`: endpoint returned but the expected capability shape was missing.
- `0.0`: HTTP failure, timeout, or exception.

Tool and multimodal failures should be read as capability failures, not always gateway failures. Some upstream models simply do not support those features. The report is most useful for comparing the same gateway configuration over time and catching regressions in adapter compatibility.

## Safety Notes

- Start with `--limit 1` or explicit `--model` while tuning prompts.
- Use `--skip-multimodal` if image preprocessing invokes an expensive vision model.
- Use a dedicated test API key with restricted allowed models when possible.
- Avoid running against production keys during heavy traffic unless you intentionally want a production smoke test.
