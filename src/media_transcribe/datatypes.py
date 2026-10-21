"""Plain data shared by the core and the engines (no imports from either).

``engines.base`` re-exports these, so engines import them from there.
"""

from __future__ import annotations

from dataclasses import dataclass

FEATURE_MODEL = "model"
FEATURE_TIMESTAMPS = "timestamps"
FEATURE_DIARIZE = "diarize"


@dataclass
class Segment:
    """A timed span of transcript, roughly one sentence, in transcript order."""

    text: str
    start: float          # seconds
    end: float            # seconds
    speaker: str | None = None  # engine-local id (e.g. "SPEAKER_00"); None if not diarized


@dataclass
class Options:
    """Optional per-run engine settings. ``None`` / ``False`` means "not requested"."""

    model: str | None = None
    timestamps: bool = False
    diarize: bool = False
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None

    def requested_features(self) -> set[str]:
        """Engine features these options need (checked against ``Engine.features``)."""
        need: set[str] = set()
        if self.model is not None:
            need.add(FEATURE_MODEL)
        if self.timestamps:
            need.add(FEATURE_TIMESTAMPS)
        if self.diarize:
            need.add(FEATURE_DIARIZE)
        return need


@dataclass
class Check:
    """One ``--doctor`` finding."""

    status: str   # "ok" | "warn" | "fail"
    label: str
    detail: str = ""
