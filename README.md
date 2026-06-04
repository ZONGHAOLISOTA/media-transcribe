# media-transcribe

**AI-native, engine-pluggable media-to-text.** Point it at a video/audio file or a
folder and get clean plain-text transcripts. It runs as a standalone CLI — and ships a
thin Claude Code adapter so AI-tool users get a one-command entry point.

> 一个 AI 原生入口友好、引擎可插拔、可独立运行的媒体转文字工具。

## Why two layers

- **`media-transcribe` (the tool)** — a Python CLI that does the real work: ffmpeg audio
  extraction → ASR → transcript cleaning, batched and resumable. Usable from any shell,
  any AI tool, cron, or CI.
- **Plugin adapters (optional)** — thin wrappers (Claude Code today) that recognize
  intent, call the CLI, and summarize. They carry **zero** transcription logic.

This keeps the logic testable and reusable instead of trapped in one vendor's prompt format.

## Install

### 1. The CLI

```bash
uv tool install git+https://github.com/ZONGHAOLISOTA/media-transcribe
# or: pipx install git+https://github.com/ZONGHAOLISOTA/media-transcribe
# or from a clone: pip install -e .
```

Requires `ffmpeg` on PATH (`brew install ffmpeg`).

### 2. An ASR engine

v0.1 ships one fully-working engine, **Qwen MLX** (Apple Silicon). See
[docs/INSTALL_QWEN.md](docs/INSTALL_QWEN.md).

## Quickstart (CLI)

```bash
# a whole folder, Chinese, with a domain-vocabulary file
media-transcribe ./videos --lang zh --context-file vocab.txt

# a single file
media-transcribe lesson01.mp4

# preview without transcribing
media-transcribe ./videos --recursive --dry-run
```

Transcripts mirror the input tree under `<input>/transcripts/` by default. Re-runs skip
files that already have a non-empty transcript; failures are recorded in
`transcripts/.media-transcribe/run.json`, so a re-run retries only what's missing.

To send output elsewhere, pass `--out <dir>` (or set `output_dir` in a config file). An
explicit output path is used **as given** — a relative path resolves against your current
directory, not the input folder.

## Use it from Claude Code (optional)

```
/plugin marketplace add ZONGHAOLISOTA/media-transcribe
/plugin install media-transcribe@media-transcribe
```

Then just ask: *"transcribe the videos in ./lessons"*.

## Engines

| Engine | Status | Notes |
| --- | --- | --- |
| `qwen-mlx` | ✅ v0.1 | Qwen3-ASR via `mlx-qwen3-asr`, Apple Silicon |
| faster-whisper / whisper.cpp / API | 🔜 | interface + docs ready |

Add your own engine **without forking the core** — see
[docs/ENGINE_DEVELOPMENT.md](docs/ENGINE_DEVELOPMENT.md).

## Non-goals (v0.1)

No GUI, no Obsidian/knowledge-base building, no speaker diarization, no subtitle editor,
no multi-platform installer.

## More docs

- **[docs/INSTALL_QWEN.md](docs/INSTALL_QWEN.md)** — install the Qwen MLX engine
- **[docs/ENGINE_DEVELOPMENT.md](docs/ENGINE_DEVELOPMENT.md)** — add your own ASR engine
- **[docs/MAINTAINING.md](docs/MAINTAINING.md)** — set up, test, verify, release (for maintainers)
- **[AGENTS.md](AGENTS.md)** — orientation for AI coding agents (Claude, Codex, …)

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT © 2026 Zonghao Li
