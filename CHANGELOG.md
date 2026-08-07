# Changelog

All notable changes to LLM AIO Gateway will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
