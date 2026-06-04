"""Optional YAML config loading with a defaults merge.

Precedence (highest first): CLI flags > config file > built-in defaults.
The CLI layer applies the flag overrides; this module supplies the lower two.
"""

from __future__ import annotations

from pathlib import Path

DEFAULTS: dict = {
    "engine": "qwen-mlx",
    "language": "zh",
    "context_file": None,
    "context": "",
    "output_dir": None,   # None -> <input-dir>/transcripts
    "recursive": False,
}


def load_config(path: Path | None) -> dict:
    cfg = dict(DEFAULTS)
    if path is None:
        return cfg
    import yaml  # imported lazily so PyYAML is only needed with --config

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"invalid YAML in {path}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"config {path} must be a mapping")
    cfg.update({k: v for k, v in data.items() if k in DEFAULTS})
    return cfg
