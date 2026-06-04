"""Engine package: registers built-ins and discovers third-party engines.

Importing this package (which happens on the first ``media_transcribe.engines``
reference) populates the registry, because Python does not auto-import new
engine modules.
"""

from __future__ import annotations

from .base import Engine, available_engines, get_engine, register

__all__ = ["Engine", "register", "get_engine", "available_engines"]


def _load_builtins() -> None:
    # Importing the module runs its module-level ``register(...)`` call.
    from . import qwen_mlx  # noqa: F401


def _load_entry_points() -> None:
    """Discover third-party engines declared under ``media_transcribe.engines``."""
    from importlib.metadata import entry_points

    try:
        group = entry_points(group="media_transcribe.engines")
    except TypeError:  # pragma: no cover - very old importlib.metadata
        group = entry_points().get("media_transcribe.engines", [])  # type: ignore[attr-defined]

    for ep in group:
        try:
            obj = ep.load()
            register(obj() if isinstance(obj, type) else obj)
        except Exception:  # noqa: BLE001 - a broken plugin must not crash the tool
            continue


_load_builtins()
_load_entry_points()
