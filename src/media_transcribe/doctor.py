"""``media-transcribe --doctor``: check the setup and say how to fix what's missing."""

from __future__ import annotations

import re
import shutil
import sys
from typing import Callable

from . import __version__
from .engines.base import Check, Options, get_engine

_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{6,}")


def collect(engine_name: str, options: Options) -> list[Check]:
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
    try:
        checks += engine.doctor(options)
    except Exception as e:  # noqa: BLE001 - a broken engine check must not hide the others
        reason = _TOKEN_RE.sub("hf_***", str(e))  # never echo an access token
        checks.append(Check("fail", f"engine {engine_name}", f"doctor check crashed: {reason}"))
    return checks


def run(engine_name: str, options: Options, log: Callable[[str], None] = print) -> int:
    """Print every check; exit 2 if anything is broken, else 0."""
    checks = collect(engine_name, options)
    for c in checks:
        tag = f"[{c.status.upper()}]"
        log(f"{tag:<7} {c.label}: {c.detail}" if c.detail else f"{tag:<7} {c.label}")
    fails = sum(c.status == "fail" for c in checks)
    warns = sum(c.status == "warn" for c in checks)
    if fails:
        log(f"\n{fails} problem(s), {warns} warning(s) — fix the [FAIL] lines above.")
        return 2
    log(f"\nAll set ({warns} warning(s)).")
    return 0
