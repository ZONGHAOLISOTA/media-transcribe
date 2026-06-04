# AGENTS.md

Orientation for AI coding agents (Claude Code, Codex, Cursor, …) working in this repo.
Read this first — it's the fast path to being productive and not breaking things.

## What this is

`media-transcribe` is a **standalone media-to-text CLI** (video/audio → cleaned text),
plus a **thin Claude Code plugin** that drives the CLI. Two layers, hard boundary:

- **The tool** — `src/media_transcribe/` — does all the work: ffmpeg audio extraction →
  ASR engine → transcript cleaning, batched and resumable.
- **The adapter** — `plugins/claude-code/` — is a thin wrapper: it recognizes intent,
  checks prerequisites, calls the CLI, and summarizes. **It contains no transcription
  logic.** Never move extraction / ASR / cleaning into the skill or command.

## Repo map

| Path | What |
| --- | --- |
| `src/media_transcribe/cli.py` | CLI entry (`media-transcribe`): arg parsing, orchestration, exit codes |
| `src/media_transcribe/batch.py` | discovery, output-path mapping, collision pre-flight, resume, run manifest |
| `src/media_transcribe/media.py` | ffmpeg extract/normalize to 16 kHz mono WAV |
| `src/media_transcribe/clean.py` | transcript cleaning: dedupe + context-echo strip + filler tail |
| `src/media_transcribe/engines/base.py` | `Engine` interface + registry |
| `src/media_transcribe/engines/qwen_mlx.py` | the one built-in engine (`mlx-qwen3-asr --stdout-only`) |
| `src/media_transcribe/engines/__init__.py` | registers built-ins + discovers entry-point engines |
| `plugins/claude-code/` | the thin Claude Code adapter (a `transcribe` skill + command) |
| `tests/` | unit tests; use a `FakeEngine`, so no model / Apple Silicon needed |

## Run & test

```bash
pip install -e ".[dev]"                      # or: uv tool install .
pytest                                        # fast; no model needed (FakeEngine)
media-transcribe ./media --lang zh --dry-run  # preview a run without transcribing
```

## Invariants — do not break these

1. **The adapter stays thin.** Skill/command only orchestrate the CLI; zero ASR logic there.
2. **Engines return raw text.** Cleaning lives in `clean.py`, never inside an engine.
3. **`Engine.transcribe` receives a normalized 16 kHz mono WAV.** Don't re-extract in engines.
4. **Output mirrors the input tree** (`<out>/<relpath>.txt`); a collision pre-flight aborts on duplicate targets (exit 2).
5. **Resumable:** a non-empty target `.txt` is skipped unless `--overwrite`.
6. **Sequential** processing in v0.1 (no `--jobs`).
7. **Exit codes:** `0` ok · `1` some files failed · `2` usage/precondition error.
8. **Python 3.10+**, but keep modules import-safe on 3.9 (no runtime `X | Y` unions; guard `entry_points(group=…)`).
9. Bad config / missing files print a concise error and exit `2` — never a raw traceback.

## Add an engine

Implement `engines/base.Engine` (`name`, `is_available`, `transcribe`). Register it as a
built-in (one import line in `engines/__init__.py`) **or** as a third-party package via the
`media_transcribe.engines` entry-point group — see `docs/ENGINE_DEVELOPMENT.md`.

## House rules

- Small, focused modules; match the surrounding style.
- Tests must pass before you declare done; add tests for new behavior.
- If you change a contract (flags, exit codes, output layout), update README **and** tests together.
- This project is reviewed (often via an external review pass) — prefer correctness and clear
  contracts over cleverness.
