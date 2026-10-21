---
description: Transcribe a media file or folder to text via the media-transcribe CLI
---

Transcribe the media at: $ARGUMENTS

Follow the `transcribe` skill: resolve the path, run `media-transcribe --doctor`, pick
options from `media-transcribe --help` (e.g. `--diarize` for interviews and meetings),
run a `--dry-run` first, then the real transcription, and summarize done / skipped /
failed plus the output location. Do not reimplement extraction, ASR or speaker
labelling — drive the CLI.
