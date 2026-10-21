---
description: Transcribe video or audio to text. Use when the user wants to turn media (a single file or a folder of videos/audio) into transcripts or notes from recordings — including interviews or meetings with speaker labels, timestamps, or subtitles. Drives the standalone `media-transcribe` CLI.
---

# Transcribe media to text

You are a **thin adapter** over the `media-transcribe` CLI. Do NOT implement audio
extraction, ASR, speaker labelling, or text cleaning yourself — the CLI does all of that.
Your job is to recognize intent, check prerequisites, call the CLI, and summarize.

## Steps

1. **Resolve the target path** the user means — a file or a directory. If it is
   ambiguous, ask before running anything.
2. **Check prerequisites** (actually run these; don't assume):
   - `command -v media-transcribe` — if missing, tell the user to install it:
     `uv tool install git+https://github.com/ZONGHAOLISOTA/media-transcribe`
   - `media-transcribe --doctor` (add `--diarize` if you will use it) — it checks ffmpeg,
     the engine, the model and the speaker-label setup, and prints the fix for each
     `[FAIL]`. Relay those fixes; don't improvise your own.
3. **Pick options** — run `media-transcribe --help` for the current list. Typical choices:
   - one speaker (lecture, talk, video): defaults;
   - interview or meeting: `--diarize` (add `--timestamps` if they want times);
   - `--num-speakers N` **only** when the user states the count — a wrong count merges
     people; otherwise let it be detected;
   - subtitles: `--format srt`; names / jargon: `--context "<terms>"` or `--context-file`;
   - language: default `zh` also covers Chinese mixed with English; `--lang en` for English.
4. **Call the CLI** — preview with a dry run first, then the real run:
   ```bash
   media-transcribe "<path>" [options] --dry-run
   media-transcribe "<path>" [options]
   ```
   Add `--recursive` only if the user wants subfolders included.
5. **Summarize**: how many done / skipped / failed, and where the transcripts landed
   (`<input>/transcripts/`, or `--out`). On any failure, point to
   `transcripts/.media-transcribe/run.json`.

## Don't

- Don't re-extract audio, call an ASR or diarization model directly, or relabel speakers
  yourself — always go through the CLI.
- Don't overwrite existing transcripts unless the user asks (`--overwrite`); note that a
  file already transcribed without `--diarize` is skipped until then.
- Don't ask the user to paste access tokens into the chat; point them to the setup line
  that `--doctor` prints.
