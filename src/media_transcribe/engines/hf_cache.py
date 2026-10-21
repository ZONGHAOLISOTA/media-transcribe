"""Hugging Face cache and token helpers (stdlib only) for engines that use HF models."""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path

# Where a token can come from, in lookup order: the variables mlx-qwen3-asr reads
# for pyannote, the older huggingface_hub name, then the file `hf auth login` writes.
TOKEN_VARS = ("PYANNOTE_AUTH_TOKEN", "HUGGINGFACE_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def _hf_home() -> Path:
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or "~/.cache"
    return Path(base).expanduser() / "huggingface"


LOGIN = "hf auth login"  # token_source() name for the stored-login token file


def token_source() -> tuple[str, str]:
    """``(token, where it came from)`` — a variable name or ``LOGIN``; ``("", "")`` if none."""
    for var in TOKEN_VARS:
        value = os.environ.get(var, "").strip()
        if value:
            return value, var
    path = Path(os.environ.get("HF_TOKEN_PATH") or _hf_home() / "token").expanduser()
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return "", ""
    return (value, LOGIN) if value else ("", "")


def token() -> str:
    return token_source()[0]


def endpoint() -> str:
    """The Hub base URL; a mirror set through ``HF_ENDPOINT`` is honoured like huggingface_hub does."""
    return (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")


def cache_dir() -> Path:
    """The hub cache directory, following huggingface_hub's environment variables."""
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(var):
            return Path(os.environ[var]).expanduser()
    return _hf_home() / "hub"


def snapshot(repo_id: str) -> Path | None:
    """Local snapshot directory of a cached model repo (its ``main`` ref), or None."""
    repo = cache_dir() / ("models--" + repo_id.replace("/", "--"))
    ref = repo / "refs" / "main"
    if ref.is_file():
        snap = repo / "snapshots" / ref.read_text(encoding="utf-8").strip()
        if snap.is_dir():
            return snap
    snaps = sorted((repo / "snapshots").glob("*"), key=lambda p: p.stat().st_mtime) \
        if (repo / "snapshots").is_dir() else []
    return snaps[-1] if snaps else None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Don't follow redirects: the token must not travel to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def check_access(repo_id: str, filename: str, tok: str, timeout: float = 10.0) -> tuple[str, str]:
    """Can ``tok`` download ``filename`` from ``repo_id``?

    Returns ``("ok" | "unauthorized" | "denied" | "error", detail)``; ``denied``
    on a gated repo means the model's terms have not been accepted yet. A
    redirect (to the file's storage) also means access was granted. The detail
    never contains the token.
    """
    if not tok.isprintable() or any(ch.isspace() for ch in tok):
        return "unauthorized", "token contains spaces or control characters"
    url = f"{endpoint()}/{repo_id}/resolve/main/{filename}"
    req = urllib.request.Request(url, method="HEAD", headers={"Authorization": f"Bearer {tok}"})
    try:
        with _OPENER.open(req, timeout=timeout):
            return "ok", ""
    except urllib.error.HTTPError as e:
        if 300 <= e.code < 400:
            return "ok", ""
        if e.code == 401:
            return "unauthorized", "token rejected (invalid or revoked)"
        if e.code == 403:
            return "denied", "terms not accepted for this token's account"
        return "error", f"HTTP {e.code}"
    except urllib.error.URLError as e:
        return "error", str(e.reason)
    except Exception as e:  # noqa: BLE001 - never let a message carry the token
        return "error", type(e).__name__
