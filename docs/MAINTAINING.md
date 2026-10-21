# Maintaining media-transcribe

A practical guide for the project **owner/maintainer** — how to set up, test, manually
verify, release, and give feedback. You don't need to know the internals to operate it;
this is the operator's manual. (For internals, see [AGENTS.md](../AGENTS.md).)

## One-time setup

```bash
git clone https://github.com/ZONGHAOLISOTA/media-transcribe
cd media-transcribe

uv tool install .          # installs the `media-transcribe` command on your PATH
brew install ffmpeg        # required for audio extraction (if not already installed)
# ASR engine (Qwen MLX, Apple Silicon): see docs/INSTALL_QWEN.md
```

## Run the tests (fast "did anything break?" check)

```bash
pip install -e ".[dev]" && pytest      # after the first time, just: pytest
```

All tests should pass. They use a **fake engine**, so no model and no Apple Silicon are
needed — this is your quick regression check before trusting any change.

## Verify a *real* transcription (what tests can't fully prove)

Tests stub out the ASR engine. To confirm the real Qwen path works end-to-end:

```bash
media-transcribe "/path/to/a/short/clip.mp4" --lang zh --out /tmp/check
cat /tmp/check/*.txt
```

What "good" looks like:

- `done=1 … failed=0`, and a `.txt` written under `--out`.
- A coherent transcript with **no repeated/garbled tail**.
- Tip: pass `--context-file vocab.txt` (names/jargon, whitespace- or newline-separated) —
  it noticeably improves proper nouns (e.g. a person's name).

Failures, if any, are recorded in `<out>/.media-transcribe/run.json`.

To verify speaker labels (after the one-time setup that `media-transcribe --doctor`
walks you through):

```bash
media-transcribe "/path/to/a/short/conversation.m4a" --diarize --timestamps --out /tmp/check
cat /tmp/check/*.txt
```

"Good" here: every word of the plain transcript is present, turns alternate where the
speakers do, and errors are limited to turn edges and very short replies.

## Test the Claude Code plugin

```bash
claude plugin marketplace add ZONGHAOLISOTA/media-transcribe   # or a local repo path
claude plugin install media-transcribe@media-transcribe
claude plugin list                                             # confirm it's enabled
```

Then, in a Claude Code session, say *"transcribe the videos in &lt;folder&gt;"* or run
`/media-transcribe:transcribe <path>`. Expected: it checks prerequisites, runs the CLI, and
summarizes. Plugins load at **session start**, so install before the session you test in.

## What to check when reviewing changes (your gating role)

- `pytest` passes, and a real smoke run still produces sane text.
- The **adapter is still thin** — no transcription logic crept into the skill/command.
- Contract changes (flags, exit codes, output layout) are reflected in the README and tests.
- A new engine didn't touch the core beyond one registration line.

## Releasing

```bash
git tag -a v0.X.0 -m "media-transcribe v0.X.0 — <summary>"
git push origin main --tags
```

Optional: pin the plugin version in `plugins/claude-code/.claude-plugin/plugin.json` for a
release (it is commit-SHA-versioned during active development).

## Giving feedback

This project is built collaboratively with AI. The most useful feedback is concrete:

- the exact command you ran, the input, what you expected, and what you got;
- paste the transcript snippet or the error text;
- for failures, attach the relevant `run.json` entry.

## Roadmap

- **v0.2.0** — model choice (`--model`), speaker labels (`--diarize`), timestamps,
  subtitle/JSON/TSV output, `--doctor`, grouped `--help` with examples.
- **Next** — a cross-platform fallback engine (e.g. faster-whisper) so users who aren't
  on Apple Silicon work out of the box.
