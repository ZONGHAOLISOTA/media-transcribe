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

- `transcribe` receives an already-normalized 16 kHz mono WAV; don't re-extract.
- Raise on failure (non-zero exit / empty output) — the batch records it and continues
  with the next file.
- Cleaning (consecutive-duplicate collapse, context-echo strip) is done by the core in
  `clean.py`; just return the raw text.
- A broken third-party engine is caught during discovery and skipped — it won't crash
  the tool, it simply won't appear in `--engine` choices.
