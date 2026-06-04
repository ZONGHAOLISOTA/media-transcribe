---
description: Transcribe a media file or folder to text via the media-transcribe CLI
---

Transcribe the media at: $ARGUMENTS

Follow the `transcribe` skill: resolve the path, verify `media-transcribe` and `ffmpeg`
are installed, run a `--dry-run` first, then the real transcription, and summarize
done / skipped / failed plus the output location. Do not reimplement extraction or ASR —
drive the CLI.
