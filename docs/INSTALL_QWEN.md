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

# with speaker labels (--diarize): adds pyannote + PyTorch
uv tool install "mlx-qwen3-asr[diarize]"

# or with pipx
pipx install mlx-qwen3-asr
```

Either way `mlx-qwen3-asr` lands on your PATH (typically `~/.local/bin`). Use **0.4.4 or
newer** — it is several times faster than 0.3.x and its diarization runs the newer
pyannote pipeline on the GPU. Upgrade an existing install with
`uv tool upgrade mlx-qwen3-asr` (or the `[diarize]` line above, which also upgrades).

Then let media-transcribe check everything:

```bash
media-transcribe --doctor
```

The first transcription downloads the model (`Qwen/Qwen3-ASR-0.6B` by default) and
caches it; subsequent runs are offline.

## 3. Speaker labels (optional)

`--diarize` uses the gated pyannote pipeline
[pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1):

1. Install the `[diarize]` extra (above).
2. On Hugging Face, accept that model's terms and create a **read** token.
3. Run once with the token so the pipeline (~30 MB) is downloaded:
   `HF_TOKEN=hf_… media-transcribe short-clip.m4a --diarize`.

Later runs need no token: when none is set and the pipeline is in the Hugging Face cache,
media-transcribe points `mlx-qwen3-asr` at the cached copy (`PYANNOTE_MODEL_ID`). Setting
`PYANNOTE_MODEL_ID` yourself (a model id or a local folder) takes precedence.

## 4. Smoke test through media-transcribe

```bash
media-transcribe path/to/short-clip.m4a --lang zh
cat path/to/transcripts/short-clip.txt
```

## Notes

- **Language:** `media-transcribe` maps common codes to the names the engine expects
  (`zh` → `Chinese`, `en` → `English`); you can also pass a name directly
  (`--lang Chinese`). Default is `zh`; Chinese mixed with English works with `zh`
  (English words stay English).
- **Models:** `--model` takes `0.6b` (default), `1.7b`, the quantized `0.6b-8bit`,
  `1.7b-8bit`, `0.6b-4bit`, `1.7b-4bit` (these need mlx-qwen3-asr ≥ 0.4.3), or any
  Hugging Face id / local path.
- **Japanese / Korean** with `--timestamps`, `--diarize` or `srt`/`vtt`/`tsv` also need
  the aligner's tokenizers: `uv tool install --upgrade "mlx-qwen3-asr[diarize,aligner]"`.
  Chinese and English need nothing extra.
- **Plain runs** invoke the engine with `--stdout-only`, so no intermediate transcript
  files are left behind. **Timed runs** (`--timestamps`, `--diarize`, `srt`/`vtt`/`tsv`)
  request `--output-format json` in a private temp directory, which is removed afterwards.
- **Domain vocabulary:** `--context-file` / `--context` terms are forwarded to bias
  recognition toward names and jargon.
