# media-transcribe

**AI-native, engine-pluggable media-to-text.** Point it at a video/audio file or a
folder and get clean transcripts — plain text, or labelled by speaker with timestamps,
or as subtitles. It runs as a standalone CLI — and ships a thin Claude Code adapter so
AI-tool users get a one-command entry point.

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

One fully-working engine ships today, **Qwen MLX** (Apple Silicon). See
[docs/INSTALL_QWEN.md](docs/INSTALL_QWEN.md).

### 3. Check the setup

```bash
media-transcribe --doctor
```

Every line is `[OK]`, `[WARN]` or `[FAIL]`, with the command that fixes it.

## Quickstart (CLI)

```bash
media-transcribe lecture.mp4                          # one speaker
media-transcribe interview.m4a --diarize              # label who speaks
media-transcribe meeting.m4a --diarize --timestamps   # ...with [hh:mm:ss] times
media-transcribe ./videos --context-file vocab.txt    # a folder, with names / jargon
media-transcribe talk.mp4 --format srt                # subtitles
media-transcribe ./videos --recursive --dry-run       # preview without transcribing
```

`media-transcribe --help` lists every option, grouped, with these examples; running it
with no arguments prints the examples.

Transcripts mirror the input tree under `<input>/transcripts/` by default. Re-runs skip
files that already have a non-empty transcript; failures are recorded in
`transcripts/.media-transcribe/run.json`, so a re-run retries only what's missing.

To send output elsewhere, pass `--out <dir>` (or set `output_dir` in a config file). An
explicit output path is used **as given** — a relative path resolves against your current
directory, not the input folder.

## Options

| Option | What it does | Default |
| --- | --- | --- |
| `--lang LANG` | spoken language (`zh`, `en`, `ja`, `ko`, …); Chinese mixed with English → `zh` | `zh` |
| `--model MODEL` | `0.6b`, `1.7b` (more accurate, ~2× slower), `0.6b-8bit`, `1.7b-8bit`, `0.6b-4bit`, `1.7b-4bit`, or a Hugging Face id / local path | `0.6b` |
| `--context "TERMS"` / `--context-file FILE` | names and jargon to bias recognition (merged, de-duplicated) | none |
| `--diarize` / `--no-diarize` | label speakers: `说话人 1：…` for Chinese, `Speaker 1: …` otherwise | off |
| `--num-speakers N` | exact speaker count — only if you are sure; a wrong count hurts | detected |
| `--min-speakers N` / `--max-speakers N` | bounds for detection | engine's (1–8) |
| `--timestamps` | start each line with `[hh:mm:ss]` | off |
| `--format FMT` | `txt`, `srt`, `vtt`, `json` or `tsv` (srt/vtt/tsv are always timed) | `txt` |
| `--out DIR`, `--recursive`, `--overwrite`, `--dry-run` | output folder, subfolders, redo existing, preview | |
| `--doctor` | check ffmpeg, the engine, models and diarization setup | |
| `--config FILE` | YAML defaults for all of the above — see [config.example.yaml](config.example.yaml) | |

The output file takes the format's extension (`ep1.txt`, `ep1.srt`, …). Changing
`--diarize` or `--timestamps` for a file that already has a `.txt` needs `--overwrite`.
Yes/no options also have a `--no-…` form (`--no-diarize`) to switch off a config default.
Timed lines and subtitle cues are split at a comma or similar once a sentence passes
~6 s or ~42 characters' width.

## Speaker labels (`--diarize`)

```text
[00:00:03] 说话人 1：我们先看一下这一版的结果。

[00:00:06] 说话人 2：好的，这次换了新的数据。
```

Speakers are told apart by voice (pyannote), numbered in order of first appearance, and
anonymous — naming them ("doctor", "patient") is up to you. Expect errors at turn
boundaries, on very short replies, and when people talk over each other; leave the
speaker count to detection unless you are certain of it (forcing 2 when a third person
speaks merges that person into someone else).

One-time setup — `media-transcribe --doctor` checks each step:

1. `uv tool install --reinstall --python 3.12 "mlx-qwen3-asr[diarize]"` (adds pyannote +
   PyTorch; use `[diarize,aligner]` for Japanese / Korean; Python 3.12 avoids an old SciPy
   build that recent macOS refuses to load).
2. Accept the terms of
   [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
   on Hugging Face and create a read token.
3. Run once with the token, e.g. `HF_TOKEN=hf_… media-transcribe clip.m4a --diarize`.
   The ~30 MB pipeline is cached; later runs without a token use the cached copy.

## Use it from Claude Code (optional)

```
/plugin marketplace add ZONGHAOLISOTA/media-transcribe
/plugin install media-transcribe@media-transcribe
```

Then just ask: *"transcribe the videos in ./lessons"*.

## Engines

| Engine | Status | Notes |
| --- | --- | --- |
| `qwen-mlx` | ✅ | Qwen3-ASR via `mlx-qwen3-asr`, Apple Silicon; models, timestamps, diarization |
| faster-whisper / whisper.cpp / API | 🔜 | interface + docs ready |

Add your own engine **without forking the core** — see
[docs/ENGINE_DEVELOPMENT.md](docs/ENGINE_DEVELOPMENT.md).

## Non-goals

No GUI, no Obsidian/knowledge-base building, no subtitle editor, no multi-platform
installer.

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
