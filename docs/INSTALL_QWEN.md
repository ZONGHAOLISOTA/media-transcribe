# Installing the Qwen MLX engine

`media-transcribe`'s default engine, `qwen-mlx`, wraps the **`mlx-qwen3-asr`** CLI —
Qwen3-ASR running on Apple Silicon via Apple's MLX framework.

> **Requires an Apple Silicon Mac (M-series).** MLX is Apple-Silicon-only. On other
> platforms use a different engine (planned — see
> [ENGINE_DEVELOPMENT.md](ENGINE_DEVELOPMENT.md)).

## 1. Prerequisites

- macOS on Apple Silicon — check with `uname -m` (should print `arm64`)
- `ffmpeg` on PATH: `brew install ffmpeg`
- A Python tool runner: [`uv`](https://docs.astral.sh/uv/) (recommended) or `pipx`

## 2. Install the engine

```bash
# recommended — isolated tool install
uv tool install mlx-qwen3-asr

# or with pipx
pipx install mlx-qwen3-asr
```

Either way `mlx-qwen3-asr` lands on your PATH (typically `~/.local/bin`). Verify:

```bash
mlx-qwen3-asr --version
mlx-qwen3-asr --doctor      # checks the MLX runtime
```

The first transcription downloads the model (`Qwen/Qwen3-ASR-0.6B` by default) and
caches it; subsequent runs are offline.

## 3. Smoke test through media-transcribe

```bash
media-transcribe path/to/short-clip.m4a --lang zh
cat path/to/transcripts/short-clip.txt
```

## Notes

- **Language:** `media-transcribe` maps common codes to the names the engine expects
  (`zh` → `Chinese`, `en` → `English`); you can also pass a name directly
  (`--lang Chinese`). Default is `zh`.
- **stdout mode:** the engine is invoked with `--stdout-only`, so no intermediate
  transcript files are left behind.
- **Domain vocabulary:** `--context-file` / `--context` terms are forwarded to bias
  recognition toward names and jargon.
