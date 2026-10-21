"""Mask access tokens before text reaches the terminal, logs or run.json."""

from __future__ import annotations

import re

_HF_TOKEN = re.compile(r"hf_[A-Za-z0-9]{6,}")


def redact(text: str) -> str:
    return _HF_TOKEN.sub("hf_***", text)
