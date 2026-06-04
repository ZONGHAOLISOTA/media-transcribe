"""Qwen3-ASR engine via the ``mlx-qwen3-asr`` CLI (Apple Silicon / MLX).

Runs in stdout mode (``--stdout-only``) and captures stdout as the transcript,
so there is no output directory, no file-name read-back, and no temp artifacts
left behind on failure.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .base import Engine, register

_BINARY = "mlx-qwen3-asr"

# mlx-qwen3-asr expects language *names* (e.g. "Chinese"), not ISO codes.
_LANG_MAP = {
    "zh": "Chinese", "zh-cn": "Chinese", "zh-hans": "Chinese", "cmn": "Chinese",
    "en": "English", "en-us": "English",
    "ja": "Japanese", "ko": "Korean",
}


class QwenMlxEngine(Engine):
    name = "qwen-mlx"

    def is_available(self) -> tuple[bool, str]:
        if shutil.which(_BINARY) is None:
            return (
                False,
                f"{_BINARY} not found on PATH. See docs/INSTALL_QWEN.md "
                "(requires Apple Silicon + MLX).",
            )
        return True, ""

    def transcribe(
        self, wav: Path, *, language: str, context: list[str] | None
    ) -> str:
        lang = _LANG_MAP.get(language.lower(), language)
        cmd = [_BINARY, "--stdout-only", "--no-progress", "--language", lang]
        terms = " ".join(t for t in (context or []) if t).strip()
        if terms:
            cmd += ["--context", terms]
        cmd.append(str(wav))

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"{_BINARY} failed (exit {proc.returncode}): "
                f"{proc.stderr.strip()[:500]}"
            )
        out = proc.stdout.strip()
        if not out:
            raise RuntimeError(f"{_BINARY} produced empty output for {wav.name}")
        return out


register(QwenMlxEngine())
