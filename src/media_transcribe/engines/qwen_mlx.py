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

from ..redact import redact
from ..segments import build_segments
from . import hf_cache
from .base import (
    FEATURE_DIARIZE, FEATURE_MODEL, FEATURE_TIMESTAMPS, Check, Engine, Options, Segment,
    register,
)

_BINARY = "mlx-qwen3-asr"
TESTED_VERSION = (0, 4, 4)

DEFAULT_MODEL = "Qwen/Qwen3-ASR-0.6B"
ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"  # mlx-qwen3-asr's default for --timestamps
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

# Fix commands. `uv tool upgrade` keeps the extras the tool was installed with.
_REINSTALL = "uv tool upgrade --reinstall mlx-qwen3-asr"
_REINSTALL_DIARIZE = 'uv tool install --reinstall --python 3.12 "mlx-qwen3-asr[diarize]"'
_ADD_ALIGNER = 'uv tool install --reinstall --python 3.12 "mlx-qwen3-asr[diarize,aligner]"'
# Japanese / Korean word alignment needs these (the [aligner] extra).
_ALIGNER_TOKENIZERS = {"Japanese": "nagisa", "Korean": "soynlp"}

# mlx-qwen3-asr wants language names ("Chinese") for the codes media-transcribe
# documents; it resolves other common codes itself (en-gb, ja-jp, ...). Every
# zh-* variant is Chinese (upstream knows only zh, zh-cn and zh-tw).
_LANG_MAP = {
    "zh": "Chinese", "cmn": "Chinese", "en": "English", "en-us": "English",
    "ja": "Japanese", "ko": "Korean",
}


def upstream_language(language: str) -> str:
    code = language.strip().lower().replace("_", "-")
    if code.startswith("zh-"):
        return "Chinese"
    return _LANG_MAP.get(code, language)


def resolve_model(model: str | None) -> str:
    """Alias -> repo id; "~/..." -> an absolute path (mlx-qwen3-asr does not expand it)."""
    if not model:
        return DEFAULT_MODEL
    if model.startswith("~"):
        return str(Path(model).expanduser())
    return MODEL_ALIASES.get(model.lower(), model)


def _tail(text: str, limit: int = 300) -> str:
    return redact(text.strip()[-limit:])


@lru_cache(maxsize=1)
def _probe() -> tuple[tuple[int, ...] | None, str]:
    """``(version, error)`` from ``mlx-qwen3-asr --version``; error is "" if it starts."""
    try:
        proc = subprocess.run([_BINARY, "--version"], capture_output=True, text=True,
                              timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        return None, str(e)
    if proc.returncode != 0:
        lines = proc.stderr.strip().splitlines()
        return None, _tail(lines[-1] if lines else f"exit {proc.returncode}")
    m = re.search(rf"{re.escape(_BINARY)}\s+v?(\d+)\.(\d+)\.(\d+)", proc.stdout)
    return (tuple(int(x) for x in m.groups()) if m else None), ""


def installed_version() -> tuple[int, ...] | None:
    return _probe()[0]


def _pipeline_id(version: tuple[int, ...] | None) -> str:
    if os.environ.get("PYANNOTE_MODEL_ID"):
        return os.environ["PYANNOTE_MODEL_ID"]
    return PYANNOTE_PIPELINE if version and version >= (0, 4, 0) else _PYANNOTE_LEGACY


def _cached_pipeline(repo_id: str) -> Path | None:
    """A cached snapshot that can actually load (an interrupted download has no config)."""
    snap = hf_cache.snapshot(repo_id)
    return snap if snap is not None and (snap / "config.yaml").is_file() else None


def _diarization_env() -> dict[str, str] | None:
    """Environment for --diarize runs, or None to inherit ours unchanged.

    The pipeline is gated on Hugging Face: a token is needed to download it
    once. After that, pointing ``PYANNOTE_MODEL_ID`` at the cached copy makes
    the token unnecessary. Older mlx-qwen3-asr (pyannote 3.1) pulls sub-models
    by id, so this only applies from 0.4.0. A "~" in a user-set
    ``PYANNOTE_MODEL_ID`` is expanded, since mlx-qwen3-asr does not.
    """
    user_id = os.environ.get("PYANNOTE_MODEL_ID", "")
    if user_id.startswith("~"):
        return {**os.environ, "PYANNOTE_MODEL_ID": str(Path(user_id).expanduser())}
    if user_id or hf_cache.token():
        return None
    version = installed_version()
    if not version or version < (0, 4, 0):
        return None
    snap = _cached_pipeline(PYANNOTE_PIPELINE)
    return {**os.environ, "PYANNOTE_MODEL_ID": str(snap)} if snap else None


def _run(cmd: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"{_BINARY} failed (exit {proc.returncode}): {_tail(proc.stderr, 500)}")
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
        model = resolve_model(options.model if options else None)
        cmd = [_BINARY, "--no-progress", "--model", model,
               "--language", upstream_language(language)]
        terms = " ".join(t for t in (context or []) if t).strip()
        if terms:
            cmd.append(f"--context={terms}")  # "=" form: terms may start with "-"
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
                        and options.min_speakers >= _UPSTREAM_MAX_SPEAKERS):
                    # "At least N": upstream's default max (8) would cap or reject it.
                    max_speakers = options.min_speakers + _UPSTREAM_MAX_SPEAKERS
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

    def doctor(self, options: Options, *, language: str | None = None,
               timed: bool = False) -> list[Check]:
        path = shutil.which(_BINARY)
        if path is None:
            return [Check("fail", _BINARY, "not found on PATH — install it: "
                          "uv tool install mlx-qwen3-asr (see docs/INSTALL_QWEN.md)")]
        version, error = _probe()
        if error:
            return [Check("fail", _BINARY, f"does not start ({error}) — reinstall: {_REINSTALL}")]
        shown = ".".join(map(str, version)) if version else "unknown version"
        if version is None or version < TESTED_VERSION:
            checks = [Check("warn", _BINARY, f"{shown} ({path}); media-transcribe is tested "
                            "with 0.4.4+ — upgrade: uv tool upgrade mlx-qwen3-asr")]
        else:
            checks = [Check("ok", _BINARY, f"{shown} ({path})")]
        checks += _upstream_doctor(options.diarize)
        python = _tool_python(path)
        if options.diarize:
            checks.append(_import_check(python, "pyannote.audio", "diarization runtime",
                                        _REINSTALL_DIARIZE))
        tokenizer = _ALIGNER_TOKENIZERS.get(upstream_language(language or ""))
        if timed and tokenizer:
            checks.append(_import_check(python, tokenizer, f"{language} word alignment",
                                        _ADD_ALIGNER))
        checks.append(_model_check(options.model))
        checks.append(_model_check(ALIGNER_MODEL, label="timestamp model",
                                   when="the first --timestamps / --diarize / srt run"))
        checks.append(_diarization_check(options.diarize, version))
        return checks


def _tool_python(script: str) -> str | None:
    """The interpreter of the ``mlx-qwen3-asr`` launcher script (its shebang), if any."""
    try:
        with open(script, "rb") as f:
            first = f.readline(512).decode("utf-8", "replace").strip()
    except OSError:
        return None
    if not first.startswith("#!"):
        return None
    parts = first[2:].split()
    if not parts:
        return None
    if Path(parts[0]).name == "env" and len(parts) > 1:
        return shutil.which(parts[1])
    return parts[0] if "python" in Path(parts[0]).name else None


def _import_check(python: str | None, module: str, label: str, fix: str) -> Check:
    """Actually import ``module`` in the engine's environment (installed is not enough)."""
    if python is None:
        return Check("warn", label, f"could not find mlx-qwen3-asr's Python to try `import {module}`")
    try:
        proc = subprocess.run([python, "-c", f"import {module}"], capture_output=True,
                              text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as e:
        return Check("fail", label, f"`import {module}` could not run ({e}) — fix: {fix}")
    if proc.returncode == 0:
        return Check("ok", label, f"`import {module}` works")
    lines = proc.stderr.strip().splitlines()
    reason = _tail(lines[-1], 200) if lines else f"exit {proc.returncode}"
    return Check("fail", label, f"`import {module}` fails: {reason} — fix: {fix}")


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
                                f"missing — needed only for --diarize: {_REINSTALL_DIARIZE}"))
            continue
        current = Check(status, f"{_BINARY}: {label}", redact(detail))
        checks.append(current)
    if proc.returncode != 0 and not checks:
        lines = proc.stderr.strip().splitlines()
        reason = _tail(lines[-1], 200) if lines else f"exit {proc.returncode}"
        return [Check("fail", f"{_BINARY} --doctor", f"crashed: {reason} — reinstall: {_REINSTALL}")]
    for check in checks:  # upstream suggests pip; it is installed as a uv tool here
        if "pip install" in check.detail:
            check.detail = check.detail.split(" — fix:")[0] + f" — fix: {_REINSTALL}"
    return checks


def _model_check(model: str | None, label: str = "model", when: str = "the first run") -> Check:
    """Will ``--model`` resolve: a local model folder, a known alias, or a Hub id?"""
    repo = resolve_model(model)
    folder = Path(repo)
    if folder.exists():
        if (folder / "config.json").is_file():
            return Check("ok", label, f"{repo} (local folder)")
        return Check("fail", label, f"{repo}: not a model folder (no config.json) — check --model")
    if repo.startswith(("/", "./", "../")):
        return Check("fail", label, f"{repo}: path not found — check --model")
    if "/" not in repo:
        names = ", ".join(MODEL_ALIASES)
        return Check("fail", label, f"unknown model {repo!r} — use {names}, a Hugging Face "
                     "id (org/name) or a local folder")
    if hf_cache.snapshot(repo):
        return Check("ok", label, f"{repo} (downloaded)")
    return Check("warn", label, f"{repo} not downloaded yet — {when} downloads it")


def _unset_hint(source: str) -> str:
    return "hf auth logout" if source == hf_cache.LOGIN else f"unset {source}"


def _set_hint(source: str) -> str:
    return "hf auth login" if source in ("", hf_cache.LOGIN) else f"export {source}=<new token>"


def _diarization_check(requested: bool, version: tuple[int, ...] | None) -> Check:
    """Can --diarize load its pyannote pipeline (local folder, cached copy, or token)?"""
    level = "fail" if requested else "warn"
    label = "diarization model"
    pipeline = _pipeline_id(version)
    custom = bool(os.environ.get("PYANNOTE_MODEL_ID"))
    folder = Path(pipeline).expanduser()
    if folder.is_dir():
        if (folder / "config.yaml").is_file():
            return Check("ok", label, f"local pipeline {pipeline}")
        return Check(level, label, f"{pipeline}: no config.yaml — not a pyannote pipeline "
                     "folder (check PYANNOTE_MODEL_ID)")
    if pipeline.startswith(("/", "./", "../", "~")):
        return Check(level, label, f"{pipeline}: folder not found (check PYANNOTE_MODEL_ID)")

    page = f"https://huggingface.co/{pipeline}"
    cached = _cached_pipeline(pipeline) is not None
    tok, source = hf_cache.token_source()
    if tok:
        # A cached copy still loads when the token is refused (huggingface_hub falls
        # back to the cache on 401/403), so those only warn.
        status, detail = hf_cache.check_access(pipeline, "config.yaml", tok)
        if status == "ok":
            return Check("ok", label, f"{pipeline}: token accepted ({source})")
        if status == "denied":
            return Check("warn" if cached else level, label,
                         f"{pipeline}: accept the model's terms at {page}")
        if status == "unauthorized":
            fix = (f"remove it ({_unset_hint(source)}) to use the cached copy" if cached else
                   "create a read token at https://huggingface.co/settings/tokens, then "
                   + _set_hint(source))
            return Check("warn" if cached else level, label,
                         f"{pipeline}: {detail} ({source}) — {fix}")
        return Check("warn" if cached else level, label,
                     f"{pipeline}: could not verify the token at {hf_cache.endpoint()} "
                     f"({detail}) — check the network (or HF_ENDPOINT) and re-run --doctor")
    if cached and (custom or (version and version >= (0, 4, 0))):
        return Check("ok", label, f"{pipeline}: downloaded — no token needed")
    if custom:
        return Check("warn", label, f"{pipeline} not downloaded; if it is gated, accept its "
                     f"terms at {page} and set HF_TOKEN for the first run")
    if cached:
        return Check(level, label, f"{pipeline}: mlx-qwen3-asr < 0.4 still needs a token — "
                     "upgrade: uv tool upgrade mlx-qwen3-asr")
    return Check(level, label, f"{pipeline} not downloaded — needed only for --diarize: "
                 f"accept the terms at {page}, then run once with HF_TOKEN=<your token>")


register(QwenMlxEngine())
