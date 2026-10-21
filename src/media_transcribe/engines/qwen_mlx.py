"""Qwen3-ASR engine via the ``mlx-qwen3-asr`` CLI (Apple Silicon / MLX).

Plain runs use stdout mode (``--stdout-only``) and capture stdout as the
transcript, so there is no output directory, no file-name read-back, and no
temp artifacts left behind on failure.

Timestamp / diarization runs ask for ``-f json`` in a private temporary
directory. The JSON carries the punctuated transcript plus aligner word timings
(and speaker labels); ``segments.build_segments`` turns those into punctuated
sentences. mlx-qwen3-asr's own ``speaker_segments`` are not used: they drop
zero-length words and carry no punctuation.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

from ..segments import build_segments
from . import hf_cache
from .base import (
    FEATURE_DIARIZE, FEATURE_MODEL, FEATURE_TIMESTAMPS, Check, Engine, Options, Segment,
    register,
)

_BINARY = "mlx-qwen3-asr"
TESTED_VERSION = (0, 4, 4)

DEFAULT_MODEL = "Qwen/Qwen3-ASR-0.6B"
# Short names accepted by --model; anything else is passed through as a
# Hugging Face repo id or a local path. Quantized builds need mlx-qwen3-asr >= 0.4.3.
MODEL_ALIASES = {
    "0.6b": "Qwen/Qwen3-ASR-0.6B",
    "1.7b": "Qwen/Qwen3-ASR-1.7B",
    "0.6b-4bit": "moona3k/mlx-qwen3-asr-0.6b-4bit",
    "0.6b-8bit": "moona3k/mlx-qwen3-asr-0.6b-8bit",
    "1.7b-4bit": "moona3k/mlx-qwen3-asr-1.7b-4bit",
    "1.7b-8bit": "moona3k/mlx-qwen3-asr-1.7b-8bit",
}

# The pyannote pipeline mlx-qwen3-asr loads for --diarize (it changed in 0.4.0).
PYANNOTE_PIPELINE = "pyannote/speaker-diarization-community-1"
_PYANNOTE_LEGACY = "pyannote/speaker-diarization-3.1"
_UPSTREAM_MAX_SPEAKERS = 8  # mlx-qwen3-asr's --max-speakers default

# mlx-qwen3-asr expects language *names* (e.g. "Chinese"), not ISO codes.
_LANG_MAP = {
    "zh": "Chinese", "zh-cn": "Chinese", "zh-hans": "Chinese", "cmn": "Chinese",
    "en": "English", "en-us": "English",
    "ja": "Japanese", "ko": "Korean",
}


def resolve_model(model: str | None) -> str:
    """Alias -> repo id; "~/..." -> an absolute path (mlx-qwen3-asr does not expand it)."""
    if not model:
        return DEFAULT_MODEL
    if model.startswith("~"):
        return str(Path(model).expanduser())
    return MODEL_ALIASES.get(model.lower(), model)


@lru_cache(maxsize=1)
def installed_version() -> tuple[int, ...] | None:
    try:
        proc = subprocess.run([_BINARY, "--version"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", proc.stdout + proc.stderr)
    return tuple(int(x) for x in m.groups()) if m else None


def _pipeline_id(version: tuple[int, ...] | None) -> str:
    if os.environ.get("PYANNOTE_MODEL_ID"):
        return os.environ["PYANNOTE_MODEL_ID"]
    return PYANNOTE_PIPELINE if version and version >= (0, 4, 0) else _PYANNOTE_LEGACY


def _diarization_env() -> dict[str, str] | None:
    """Use the cached pyannote pipeline by path when no token is set.

    The pipeline is gated on Hugging Face: a token is needed to download it
    once. After that, pointing ``PYANNOTE_MODEL_ID`` at the cached copy makes
    the token unnecessary. Older mlx-qwen3-asr (pyannote 3.1) pulls sub-models
    by id, so this only applies from 0.4.0.
    """
    if os.environ.get("PYANNOTE_MODEL_ID") or hf_cache.token():
        return None
    version = installed_version()
    if not version or version < (0, 4, 0):
        return None
    snap = hf_cache.snapshot(PYANNOTE_PIPELINE)
    if snap is None or not (snap / "config.yaml").is_file():
        return None
    return {**os.environ, "PYANNOTE_MODEL_ID": str(snap)}


def _run(cmd: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{_BINARY} failed (exit {proc.returncode}): {proc.stderr.strip()[-500:]}"
        )
    return proc


class QwenMlxEngine(Engine):
    name = "qwen-mlx"
    features = frozenset({FEATURE_MODEL, FEATURE_TIMESTAMPS, FEATURE_DIARIZE})

    def is_available(self) -> tuple[bool, str]:
        if shutil.which(_BINARY) is None:
            return (
                False,
                f"{_BINARY} not found on PATH. See docs/INSTALL_QWEN.md "
                "(requires Apple Silicon + MLX).",
            )
        return True, ""

    def _command(self, language: str, context: list[str] | None,
                 options: Options | None) -> list[str]:
        lang = _LANG_MAP.get(language.lower(), language)
        model = resolve_model(options.model if options else None)
        cmd = [_BINARY, "--no-progress", "--model", model, "--language", lang]
        terms = " ".join(t for t in (context or []) if t).strip()
        if terms:
            cmd += ["--context", terms]
        return cmd

    def transcribe(
        self, wav: Path, *, language: str, context: list[str] | None,
        options: Options | None = None,
    ) -> str:
        proc = _run(self._command(language, context, options) + ["--stdout-only", str(wav)])
        out = proc.stdout.strip()
        if not out:
            raise RuntimeError(f"{_BINARY} produced empty output for {wav.name}")
        return out

    def transcribe_segments(
        self, wav: Path, *, language: str, context: list[str] | None, options: Options
    ) -> list[Segment]:
        with tempfile.TemporaryDirectory(prefix="media-transcribe-qwen-") as td:
            cmd = self._command(language, context, options)
            cmd += ["--timestamps", "--output-format", "json", "--output-dir", td]
            env = None
            if options.diarize:
                cmd.append("--diarize")
                max_speakers = options.max_speakers
                if (max_speakers is None and options.min_speakers is not None
                        and options.min_speakers > _UPSTREAM_MAX_SPEAKERS):
                    max_speakers = options.min_speakers  # else upstream rejects min > 8
                for flag, value in (("--num-speakers", options.num_speakers),
                                    ("--min-speakers", options.min_speakers),
                                    ("--max-speakers", max_speakers)):
                    if value is not None:
                        cmd += [flag, str(value)]
                env = _diarization_env()
            _run(cmd + [str(wav)], env=env)
            result = Path(td) / f"{wav.stem}.json"
            if not result.is_file():
                raise RuntimeError(f"{_BINARY} wrote no JSON result for {wav.name}")
            data = json.loads(result.read_text(encoding="utf-8"))
        text = str(data.get("text") or "").strip()
        if not text:
            raise RuntimeError(f"{_BINARY} produced empty output for {wav.name}")
        words = data.get("segments") or []
        if not words:
            raise RuntimeError(
                f"{_BINARY} returned no word timestamps for {wav.name} (the forced "
                "aligner may not support this language); run without --timestamps/"
                "--diarize or use --format txt/json"
            )
        return build_segments(text, words)

    def doctor(self, options: Options) -> list[Check]:
        path = shutil.which(_BINARY)
        if path is None:
            return [Check("fail", _BINARY, "not found on PATH — install it: "
                          "uv tool install mlx-qwen3-asr (see docs/INSTALL_QWEN.md)")]
        version = installed_version()
        shown = ".".join(map(str, version)) if version else "unknown version"
        if version is None or version < TESTED_VERSION:
            checks = [Check("warn", _BINARY, f"{shown} ({path}); media-transcribe is tested "
                            "with 0.4.4+ — upgrade: uv tool upgrade mlx-qwen3-asr")]
        else:
            checks = [Check("ok", _BINARY, f"{shown} ({path})")]
        checks += _upstream_doctor(options.diarize)
        checks.append(_model_check(options.model))
        checks.append(_diarization_check(options.diarize, version))
        return checks


def _upstream_doctor(diarize: bool) -> list[Check]:
    """Relay ``mlx-qwen3-asr --doctor`` (MLX runtime, diarization extras).

    Its ffmpeg and token lines are skipped: the core checks ffmpeg, and
    ``_diarization_check`` gives a more precise answer about the token.
    """
    try:
        proc = subprocess.run([_BINARY, "--doctor"], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        return [Check("warn", f"{_BINARY} --doctor", f"could not run: {e}")]
    checks: list[Check] = []
    current: Check | None = None  # the check a following "fix:" line belongs to
    for raw in proc.stdout.splitlines():
        line = raw.strip()
        m = re.match(r"\[(OK|WARN|FAIL)\]\s+([^:]+):\s*(.*)", line)
        if not m:
            if line.startswith("fix:") and current is not None:
                current.detail = f"{current.detail} — {line}"
            continue
        status, label, detail = m.group(1).lower(), m.group(2).strip(), m.group(3).strip()
        current = None
        if label.startswith("ffmpeg") or "token" in label:
            continue
        if label.startswith("diarize extras") and status != "ok":
            checks.append(Check("fail" if diarize else "warn", f"{_BINARY}: {label}",
                                "missing — needed only for --diarize: "
                                "uv tool install --upgrade 'mlx-qwen3-asr[diarize]'"))
            continue
        current = Check(status, f"{_BINARY}: {label}", detail)
        checks.append(current)
    return checks


def _model_check(model: str | None) -> Check:
    repo = resolve_model(model)
    if Path(repo).expanduser().exists():
        return Check("ok", "model", f"{repo} (local path)")
    if hf_cache.snapshot(repo):
        return Check("ok", "model", f"{repo} (downloaded)")
    return Check("warn", "model", f"{repo} not downloaded yet — the first run downloads it")


def _diarization_check(requested: bool, version: tuple[int, ...] | None) -> Check:
    """Can --diarize load its pyannote pipeline (token, terms, or a cached copy)?"""
    level = "fail" if requested else "warn"
    label = "diarization model"
    pipeline = _pipeline_id(version)
    if Path(pipeline).expanduser().is_dir():
        return Check("ok", label, f"local pipeline {pipeline}")
    page = f"https://huggingface.co/{pipeline}"
    cached = hf_cache.snapshot(pipeline) is not None and pipeline == PYANNOTE_PIPELINE
    tok = hf_cache.token()
    if tok:
        status, detail = hf_cache.check_access(pipeline, "config.yaml", tok)
        if status == "ok":
            return Check("ok", label, f"{pipeline}: token accepted")
        if status == "denied":
            return Check(level, label, f"{pipeline}: accept the model's terms at {page}")
        if status == "unauthorized":
            hint = (" — remove it (unset the variable, or hf auth logout) to use the "
                    "cached copy") if cached else ""
            return Check("warn" if cached else level, label, f"{pipeline}: {detail}{hint}")
        return Check("warn" if cached else level, label,
                     f"{pipeline}: could not verify the token ({detail})")
    if cached and version and version >= (0, 4, 0):
        return Check("ok", label, f"{pipeline}: downloaded — no token needed")
    return Check(level, label, f"{pipeline} not downloaded — needed only for --diarize: "
                 f"accept the terms at {page}, then run once with HF_TOKEN=<your token>")


register(QwenMlxEngine())
