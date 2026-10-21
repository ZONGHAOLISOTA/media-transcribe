# Adding an engine

`media-transcribe` selects an ASR backend by name (`--engine <name>`). An engine
implements the `Engine` interface and registers itself. There are two ways to add one.

## The interface

```python
from pathlib import Path
from media_transcribe.engines.base import Engine, register


class MyEngine(Engine):
    name = "my-engine"

    def is_available(self) -> tuple[bool, str]:
        # (True, "") if usable now, else (False, "how to fix it")
        ...

    def transcribe(self, wav: Path, *, language: str, context: list[str] | None) -> str:
        # `wav` is a 16 kHz mono PCM WAV produced by the core.
        # Return the RAW transcript text; the core handles cleaning.
        ...
```

That is a complete engine. Everything below is optional.

## Optional features

An engine opts into extra options by listing them in `features`. The CLI rejects an
option the selected engine does not declare (exit `2`), so nothing is silently ignored.

| Feature | Unlocks | Implement |
| --- | --- | --- |
| `"model"` | `--model` | accept `options` in `transcribe` (and honour `options.model`) |
| `"timestamps"` | `--timestamps`, `--format srt/vtt/tsv` | `transcribe_segments` |
| `"diarize"` | `--diarize`, `--num/min/max-speakers` | `transcribe_segments` with `Segment.speaker` set |

```python
from media_transcribe.engines.base import Engine, Options, Segment


class MyEngine(Engine):
    name = "my-engine"
    features = frozenset({"model", "timestamps", "diarize"})

    def is_available(self) -> tuple[bool, str]:
        ...

    def transcribe(self, wav, *, language, context, options: Options | None = None) -> str:
        ...

    def transcribe_segments(self, wav, *, language, context, options: Options) -> list[Segment]:
        # Sentence-sized segments in transcript order, with punctuation, start/end in
        # seconds and (when options.diarize) an engine-local speaker id per segment.
        ...
```

- `options` is passed to `transcribe` **only** when the user requested something, so
  engines without the parameter keep working.
- If your backend returns word timings plus a punctuated transcript (as forced aligners
  do), `media_transcribe.segments.build_segments(text, words)` turns them into
  punctuated, speaker-smoothed sentences — see `engines/qwen_mlx.py`.
- Speaker ids are internal; the core numbers speakers by first appearance and renders
  `说话人 1：` / `Speaker 1: `.

### `--doctor`

Override `doctor(options)` to add environment checks; return a list of
`Check(status, label, detail)` with status `"ok"`, `"warn"` or `"fail"`, and put the fix
command in `detail`. The default reports `is_available()`.

## Option A — built-in (inside this repo)

1. Add `src/media_transcribe/engines/my_engine.py` implementing `Engine`, ending with
   `register(MyEngine())`.
2. Add one line to `_load_builtins()` in `src/media_transcribe/engines/__init__.py`:
   ```python
   from . import my_engine  # noqa: F401
   ```
   Python does not auto-import new modules — this single import is the one required
   core touch.

## Option B — third-party package (no core change)

Ship your engine in its own pip package and declare an entry point:

```toml
# your package's pyproject.toml
[project.entry-points."media_transcribe.engines"]
my-engine = "my_pkg.my_engine:MyEngine"
```

`media-transcribe` discovers it at startup via `importlib.metadata`. Installing your
package is all it takes — **zero** changes to this repo.

## Contract notes

- `transcribe` / `transcribe_segments` receive an already-normalized 16 kHz mono WAV;
  don't re-extract.
- Raise on failure (non-zero exit / empty output) — the batch records it and continues
  with the next file.
- Cleaning (consecutive-duplicate collapse, context-echo strip) is done by the core in
  `clean.py`, and output formats by `render.py`; just return the raw text or segments.
- A broken third-party engine is caught during discovery and skipped — it won't crash
  the tool, it simply won't appear in `--engine` choices.
