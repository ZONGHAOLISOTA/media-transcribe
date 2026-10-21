"""Render a cleaned transcript into the output formats.

Plain runs give text only; timestamp / diarization runs give ``Segment`` lists.
Speakers are numbered by first appearance and labelled in the run's language
("说话人 1：" for Chinese, "Speaker 1: " otherwise).
"""

from __future__ import annotations

import json

from .datatypes import Options, Segment

FORMATS = ("txt", "srt", "vtt", "json", "tsv")
TIMED_FORMATS = ("srt", "vtt", "tsv")  # need timestamps whatever the flags say
MIN_CUE_SEC = 1.0  # aligner words can be zero-length; players skip such cues


def needs_segments(options: Options, fmt: str) -> bool:
    return options.timestamps or options.diarize or fmt in TIMED_FORMATS


def _is_chinese(language: str) -> bool:
    return language.lower() in ("zh", "zh-cn", "zh-hans", "zh-tw", "cmn", "yue", "chinese",
                                "cantonese")


def speaker_names(segments: list[Segment], language: str) -> dict[str, str]:
    """Map engine speaker ids to display labels, in order of first appearance."""
    word = "说话人" if _is_chinese(language) else "Speaker"
    names: dict[str, str] = {}
    for seg in segments:
        if seg.speaker is not None and seg.speaker not in names:
            names[seg.speaker] = f"{word} {len(names) + 1}"
    return names


def _cjk(ch: str) -> bool:
    return "⺀" <= ch <= "鿿" or "豈" <= ch <= "﫿" or "＀" <= ch <= "￯"


def _join(left: str, right: str) -> str:
    """Concatenate two pieces of transcript, adding a space only between non-CJK text."""
    if not left or not right:
        return left + right
    if _cjk(left[-1]) or _cjk(right[0]):
        return left + right
    return f"{left} {right}"


def _clock(seconds: float, sep: str = ",", millis: bool = True) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h, rest = divmod(ms, 3_600_000)
    m, rest = divmod(rest, 60_000)
    s, ms = divmod(rest, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}" if millis else f"{h:02d}:{m:02d}:{s:02d}"


def render_text(text: str, *, fmt: str, language: str) -> str:
    """Render a plain (untimed) transcript; only ``txt`` and ``json`` apply."""
    if fmt == "json":
        return _json(text.strip(), [], language)
    return text


def render_segments(
    segments: list[Segment], *, fmt: str, language: str, timestamps: bool
) -> str:
    names = speaker_names(segments, language)
    colon = "：" if _is_chinese(language) else ": "

    def label(seg: Segment) -> str:
        return f"{names[seg.speaker]}{colon}" if seg.speaker in names else ""

    if fmt == "txt":
        return _txt(segments, names, colon, timestamps)
    if fmt == "json":
        text = ""
        for seg in segments:
            text = _join(text, seg.text)
        return _json(text, segments, language, names)
    if fmt == "tsv":
        rows = ["start\tend\tspeaker\ttext"]
        rows += [f"{s.start:.3f}\t{s.end:.3f}\t{names.get(s.speaker, '')}\t{s.text}"
                 for s in segments]
        return "\n".join(rows) + "\n"
    if fmt == "srt":
        cues = [f"{i}\n{_clock(start)} --> {_clock(end)}\n{label(s)}{s.text}\n"
                for i, (s, start, end) in enumerate(_cue_times(segments), 1)]
        return "\n".join(cues)
    if fmt == "vtt":
        cues = [f"{_clock(start, '.')} --> {_clock(end, '.')}\n{label(s)}{s.text}\n"
                for s, start, end in _cue_times(segments)]
        return "WEBVTT\n\n" + "\n".join(cues)
    raise ValueError(f"unknown format {fmt!r}")


def _cue_times(segments: list[Segment]) -> list[tuple[Segment, float, float]]:
    """Stretch each cue to at least ``MIN_CUE_SEC``, without running into the next one."""
    out = []
    for k, seg in enumerate(segments):
        end = max(seg.end, seg.start + MIN_CUE_SEC)
        if k + 1 < len(segments) and segments[k + 1].start > seg.start:
            end = min(end, max(seg.end, segments[k + 1].start))
        out.append((seg, seg.start, end))
    return out


def _txt(segments: list[Segment], names: dict[str, str], colon: str, timestamps: bool) -> str:
    if not names:  # no speakers: one line per sentence with times, else running text
        if timestamps:
            return "".join(f"[{_clock(s.start, millis=False)}] {s.text}\n" for s in segments)
        text = ""
        for seg in segments:
            text = _join(text, seg.text)
        return f"{text}\n" if text else ""
    # diarized: one paragraph per speaker turn, blank line between turns
    turns: list[tuple[Segment, str]] = []
    for seg in segments:
        if turns and turns[-1][0].speaker == seg.speaker:
            turns[-1] = (turns[-1][0], _join(turns[-1][1], seg.text))
        else:
            turns.append((seg, seg.text))
    lines = []
    for first, text in turns:
        stamp = f"[{_clock(first.start, millis=False)}] " if timestamps else ""
        who = f"{names[first.speaker]}{colon}" if first.speaker in names else ""
        lines.append(f"{stamp}{who}{text}\n")
    return "\n".join(lines)


def _json(text: str, segments: list[Segment], language: str,
          names: dict[str, str] | None = None) -> str:
    names = names or {}
    data = {
        "language": language,
        "text": text,
        "segments": [
            {"start": round(s.start, 3), "end": round(s.end, 3),
             "speaker": names.get(s.speaker), "text": s.text}
            for s in segments
        ],
    }
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"
