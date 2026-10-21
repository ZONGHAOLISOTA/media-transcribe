"""``media-transcribe --doctor``: check the setup and say how to fix what's missing."""

from __future__ import annotations

import inspect
import shutil
import sys
from typing import Callable

from . import __version__
from .engines.base import Check, Options, get_engine
from .redact import redact
from .render import TIMED_FORMATS, required_features


def collect(engine_name: str, options: Options, *, language: str | None = None,
            fmt: str = "txt") -> list[Check]:
    py = sys.version_info
    checks = [
        Check("ok", "media-transcribe", __version__),
        Check("ok" if py >= (3, 10) else "fail", "python",
              f"{py.major}.{py.minor}.{py.micro}" + ("" if py >= (3, 10) else " — needs 3.10+")),
    ]
    ffmpeg = shutil.which("ffmpeg")
    checks.append(Check("ok", "ffmpeg", ffmpeg) if ffmpeg else
                  Check("fail", "ffmpeg", "not found on PATH — install it: brew install ffmpeg"))
    try:
        engine = get_engine(engine_name)
    except KeyError as e:
        return checks + [Check("fail", "engine", str(e).strip("'\""))]
    timed = options.timestamps or options.diarize or fmt in TIMED_FORMATS  # engine context
    missing = required_features(options, fmt) - set(getattr(engine, "features", ()))
    if missing:
        checks.append(Check("fail", f"engine {engine_name}",
                            f"does not support: {', '.join(sorted(missing))}"))
    try:
        # Pass only the context an engine's doctor() accepts (both are optional).
        try:
            accepted = inspect.signature(engine.doctor).parameters
        except (TypeError, ValueError):
            accepted = {}
        extra = {k: v for k, v in (("language", language), ("timed", timed)) if k in accepted}
        checks += engine.doctor(options, **extra)
    except Exception as e:  # noqa: BLE001 - a broken engine check must not hide the others
        checks.append(Check("fail", f"engine {engine_name}",
                            f"doctor check crashed: {redact(str(e))}"))
    return checks


def run(engine_name: str, options: Options, log: Callable[[str], None] = print, *,
        language: str | None = None, fmt: str = "txt") -> int:
    """Print every check; exit 2 if anything is broken, else 0.

    Every line is redacted on the way out: an engine's check may quote an error
    that carries an access token.
    """
    checks = collect(engine_name, options, language=language, fmt=fmt)
    for c in checks:
        tag = f"[{c.status.upper()}]"
        log(redact(f"{tag:<7} {c.label}: {c.detail}" if c.detail else f"{tag:<7} {c.label}"))
    fails = sum(c.status == "fail" for c in checks)
    warns = sum(c.status == "warn" for c in checks)
    if fails:
        log(f"\n{fails} problem(s), {warns} warning(s) — fix the [FAIL] lines above.")
        return 2
    log(f"\nAll set ({warns} warning(s)).")
    return 0
