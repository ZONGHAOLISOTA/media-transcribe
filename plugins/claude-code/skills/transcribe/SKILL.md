---
description: Transcribe video or audio to text. Use when the user wants to turn media (a single file or a folder of videos/audio) into transcripts or notes from recordings. Drives the standalone `media-transcribe` CLI.
---

# Transcribe media to text

You are a **thin adapter** over the `media-transcribe` CLI. Do NOT implement audio
extraction, ASR, or text cleaning yourself — the CLI does all of that. Your job is to
recognize intent, check prerequisites, call the CLI, and summarize.

## Steps

1. **Resolve the target path** the user means — a file or a directory. If it is
   ambiguous, ask before running anything.
2. **Check prerequisites** (actually run these; don't assume):
   - `command -v media-transcribe` — if missing, tell the user to install it:
     `pipx install git+https://github.com/ZONGHAOLISOTA/media-transcribe`
   - `command -v ffmpeg` — if missing: `brew install ffmpeg`.
   - The selected engine: a `--dry-run` warns (without failing) if it is unavailable.
     For the default `qwen-mlx` engine, point to `docs/INSTALL_QWEN.md`.
3. **Call the CLI** — preview with a dry run first, then the real run:
   ```bash
   media-transcribe "<path>" --lang <zh|en|...> [--recursive] [--context-file vocab.txt]
   ```
   - Default language is `zh`; confirm if unclear.
   - Add `--recursive` only if the user wants subfolders included.
   - Pass `--context-file <file>` or `--context "<terms>"` when the user has domain
     vocabulary (improves accuracy on names/jargon).
4. **Summarize**: how many done / skipped / failed, and where the transcripts landed
   (`<input>/transcripts/`, or `--out`). On any failure, point to
   `transcripts/.media-transcribe/run.json`.

## Don't

- Don't re-extract audio or call an ASR model directly — always go through the CLI.
- Don't overwrite existing transcripts unless the user asks (`--overwrite`).
