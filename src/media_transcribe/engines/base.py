"""Engine interface and registry.

Engines are looked up by ``name``. Built-in engines register themselves when
their module is imported (see ``engines/__init__.py``); third-party engines
register via the ``media_transcribe.engines`` entry-point group.

Beyond plain text, an engine may opt into optional features by listing them in
``Engine.features``:

- ``"model"``      — ``transcribe`` / ``transcribe_segments`` honour ``options.model``;
- ``"timestamps"`` — ``transcribe_segments`` returns timed segments;
- ``"diarize"``    — ``transcribe_segments`` labels segments with speakers.

The CLI refuses an option the selected engine does not declare, and only passes
``options`` to ``transcribe`` when one was requested — so v0.1 engines written
without the ``options`` parameter keep working unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..datatypes import (  # noqa: F401 - re-exported as the engine-facing API
    FEATURE_DIARIZE, FEATURE_MODEL, FEATURE_TIMESTAMPS, Check, Options, Segment,
)

_REGISTRY: dict[str, "Engine"] = {}


class Engine(ABC):
    """An ASR backend. Subclasses set ``name`` and implement both abstract methods."""

    name: str = ""
    features: frozenset = frozenset()  # optional features, see module docstring

    @abstractmethod
    def is_available(self) -> tuple[bool, str]:
        """Return ``(True, "")`` if usable, else ``(False, reason)``.

        The reason should include a fix hint (how to install / configure).
        """

    @abstractmethod
    def transcribe(
        self, wav: Path, *, language: str, context: list[str] | None,
        options: Options | None = None,
    ) -> str:
        """Return raw transcript text for a 16 kHz mono WAV file.

        ``options`` is only passed when the user requested a declared feature
        (e.g. ``--model``); engines without features may omit the parameter.
        """

    def transcribe_segments(
        self, wav: Path, *, language: str, context: list[str] | None, options: Options
    ) -> list[Segment]:
        """Return raw timed segments for a 16 kHz mono WAV file.

        Required when ``features`` includes ``"timestamps"`` or ``"diarize"``.
        Segments must be in transcript order and, joined, cover the transcript.
        """
        raise NotImplementedError(f"engine {self.name!r} does not produce timed segments")

    def doctor(self, options: Options, *, language: str | None = None,
               timed: bool = False) -> list[Check]:
        """Environment checks for ``media-transcribe --doctor``. Override to add more.

        ``language`` is the run's ``--lang``; ``timed`` is True when the run needs
        timestamps (``--timestamps``, ``--diarize`` or a timed output format).
        """
        ok, reason = self.is_available()
        return [Check("ok" if ok else "fail", f"engine {self.name}", reason or "available")]


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
