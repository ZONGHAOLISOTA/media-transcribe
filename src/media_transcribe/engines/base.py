"""Engine interface and registry.

Engines are looked up by ``name``. Built-in engines register themselves when
their module is imported (see ``engines/__init__.py``); third-party engines
register via the ``media_transcribe.engines`` entry-point group.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

_REGISTRY: dict[str, "Engine"] = {}


class Engine(ABC):
    """An ASR backend. Subclasses set ``name`` and implement both methods."""

    name: str = ""

    @abstractmethod
    def is_available(self) -> tuple[bool, str]:
        """Return ``(True, "")`` if usable, else ``(False, reason)``.

        The reason should include a fix hint (how to install / configure).
        """

    @abstractmethod
    def transcribe(
        self, wav: Path, *, language: str, context: list[str] | None
    ) -> str:
        """Return raw transcript text for a 16 kHz mono WAV file."""


def register(engine: "Engine") -> None:
    if not engine.name:
        raise ValueError("engine.name must be set")
    _REGISTRY[engine.name] = engine


def get_engine(name: str) -> "Engine":
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown engine {name!r}; available: {', '.join(available_engines()) or '(none)'}"
        ) from None


def available_engines() -> list[str]:
    return sorted(_REGISTRY)
