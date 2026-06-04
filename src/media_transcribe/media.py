"""Media handling: detect media files and normalize audio to 16 kHz mono WAV."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

VIDEO_EXTS = {
    ".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv",
    ".m4v", ".ts", ".mpg", ".mpeg", ".wmv",
}
AUDIO_EXTS = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".aiff",
}
MEDIA_EXTS = VIDEO_EXTS | AUDIO_EXTS


def is_media(path: Path) -> bool:
    return path.suffix.lower() in MEDIA_EXTS


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it (e.g. `brew install ffmpeg`)."
        )


def extract_wav(src: Path, dst: Path) -> Path:
    """Extract / normalize ``src`` to a 16 kHz mono PCM WAV at ``dst``."""
    ensure_ffmpeg()
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-i", str(src),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {src.name}: {proc.stderr.strip()[:500]}"
        )
    return dst
