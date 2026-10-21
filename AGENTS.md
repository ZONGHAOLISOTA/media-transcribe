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
| `src/media_transcribe/cli.py` | CLI entry (`media-transcribe`): grouped `--help` + examples, option validation, orchestration, exit codes |
| `src/media_transcribe/batch.py` | discovery, output-path mapping, collision pre-flight, resume, run manifest |
| `src/media_transcribe/media.py` | ffmpeg extract/normalize to 16 kHz mono WAV |
| `src/media_transcribe/clean.py` | transcript cleaning: dedupe + context-echo strip + filler tail (text and segments) |
| `src/media_transcribe/segments.py` | word timings + punctuated text → sentence segments; speaker smoothing |
| `src/media_transcribe/render.py` | output formats (txt/srt/vtt/json/tsv), speaker labels, timestamps |
| `src/media_transcribe/doctor.py` | `--doctor`: core checks + the engine's own checks |
| `src/media_transcribe/engines/base.py` | `Engine` interface (+ optional `features`), `Segment`, `Options`, `Check`, registry |
| `src/media_transcribe/engines/qwen_mlx.py` | the built-in engine: `mlx-qwen3-asr --stdout-only`, or `--output-format json` for timed runs |
| `src/media_transcribe/engines/hf_cache.py` | Hugging Face cache / token helpers (stdlib only) |
| `src/media_transcribe/engines/__init__.py` | registers built-ins + discovers entry-point engines |
| `plugins/claude-code/` | the thin Claude Code adapter (a `transcribe` skill + command) |
| `tests/` | unit tests; use a `FakeEngine`, so no model / Apple Silicon needed |

## Run & test

```bash
pip install -e ".[dev]"                      # or: uv tool install .
pytest                                        # fast; no model needed (FakeEngine)
media-transcribe --doctor                     # environment check
media-transcribe ./media --lang zh --dry-run  # preview a run without transcribing
```

## Invariants — do not break these

1. **The adapter stays thin.** Skill/command only orchestrate the CLI; zero ASR logic there.
2. **Engines return raw text or raw segments.** Cleaning lives in `clean.py` and formatting in `render.py`, never inside an engine.
3. **`Engine.transcribe` / `transcribe_segments` receive a normalized 16 kHz mono WAV.** Don't re-extract in engines.
4. **Output mirrors the input tree** (`<out>/<relpath>.<format>`); a collision pre-flight aborts on duplicate targets (exit 2).
5. **Resumable:** a non-empty target file is skipped unless `--overwrite`; targets are written to `<name>.part` and renamed, so a partial file never counts as done.
6. **Sequential** processing (no `--jobs`).
7. **Exit codes:** `0` ok · `1` some files failed · `2` usage/precondition error (`--doctor`: `2` if any check fails).
8. **Python 3.10+**, but keep modules import-safe on 3.9 (no runtime `X | Y` unions; guard `entry_points(group=…)`).
9. Bad config / missing files / invalid options print a concise error and exit `2` — never a raw traceback.
10. **Optional engine features are opt-in.** An option the engine doesn't list in `features` is rejected (exit 2); `options` reaches `transcribe` only when one was requested, so v0.1 engines keep working.
11. **Transcripts never lose text.** `segments.build_segments` partitions the transcript among the words; keep that property (there's a test for it).

## Add an engine

Implement `engines/base.Engine` (`name`, `is_available`, `transcribe`; optionally
`features` + `transcribe_segments` + `doctor`). Register it as a built-in (one import line
in `engines/__init__.py`) **or** as a third-party package via the
`media_transcribe.engines` entry-point group — see `docs/ENGINE_DEVELOPMENT.md`.

## House rules

- Small, focused modules; match the surrounding style.
- Tests must pass before you declare done; add tests for new behavior.
- If you change a contract (flags, exit codes, output layout), update README **and** tests together.
- This project is reviewed (often via an external review pass) — prefer correctness and clear
  contracts over cleverness.
