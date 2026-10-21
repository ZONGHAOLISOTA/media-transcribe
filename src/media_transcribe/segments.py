"""Turn word-level timings into punctuated, sentence-sized segments.

Forced aligners return bare words (or CJK characters) with times but no
punctuation, and normalize them ("GPT-4" -> "GPT4", "3.5" -> "35", "9:30" -> "930");
the transcript text has punctuation but no times. ``build_segments`` walks the two
in step and partitions the transcript text among the words, so every character
lands in exactly one segment — nothing is dropped. A word that still cannot be
found keeps the stretch of text up to the next word that can, so its own time and
speaker stay attached to it.

A new segment starts at a sentence end or a speaker change. Diarization turn
boundaries are only accurate to a few hundred milliseconds, so a speaker change
often clips a word or two off the edge of a sentence ("问题不大 / 的。"). Inside a
sentence, a speaker run shorter than ``MIN_RUN_SEC`` is merged into its longer
neighbour. A whole short sentence ("Yeah.") keeps its own label, since short
replies are real turns.
"""

from __future__ import annotations

import unicodedata
from typing import Mapping, Sequence

from .datatypes import Segment

_TERMINATORS = set(".?!。？！…")
_CLOSERS = set("\"'”’」』）)】》〉]")
_SEARCH_WINDOW = 50  # chars to look ahead when a word does not match in place
MIN_RUN_SEC = 0.6    # shorter in-sentence speaker runs are treated as boundary noise


def _fold(ch: str) -> str:
    return "'" if ch in "’‘" else ch.casefold()


def _skippable(ch: str) -> bool:
    """Characters an aligner drops from inside a word ("-" in GPT-4, "." in 3.5)."""
    return not ch.isalnum() and not ch.isspace() and _fold(ch) != "'"


def _match(text: str, pos: int, chars: list[str]) -> int:
    """End index if ``chars`` match ``text`` at ``pos``, else -1.

    Case-insensitive; between two characters of the word, skippable text
    characters are passed over, mirroring the aligner's normalization.
    """
    n = len(text)
    for k, c in enumerate(chars):
        if k:
            while pos < n and _skippable(text[pos]) and _fold(text[pos]) != _fold(c):
                pos += 1
        if pos < n and _fold(text[pos]) == _fold(c):
            pos += 1
        else:
            return -1
    return pos


def _latin(ch: str) -> bool:
    return ch.isascii() and ch.isalnum()


def _find(text: str, pos: int, chars: list[str]) -> tuple[int, int]:
    """(start, end) of ``chars`` in ``text`` at or shortly after ``pos``; (-1, -1) if absent.

    A Latin word only matches at a word start, so a short word ("a") cannot
    latch onto the middle of a longer one ("cat") while searching ahead.
    """
    n = len(text)
    first = _fold(chars[0])
    while pos < n and not text[pos].isalnum() and _fold(text[pos]) != first:
        pos += 1
    for start in range(pos, min(n, pos + _SEARCH_WINDOW)):
        if _fold(text[start]) != first:
            continue
        if _latin(chars[0]) and start > 0 and _latin(text[start - 1]):
            continue
        end = _match(text, start, chars)
        if end >= 0:
            return start, end
    return -1, -1


def locate(text: str, words: Sequence[Mapping]) -> list[tuple[int, int]]:
    """Partition ``text`` into contiguous ``(start, end)`` spans, one per word.

    A matched word's span runs from the end of the previous span through the
    word and its trailing punctuation. A run of unmatched words takes the text
    up to the next matched word (all of it on the run's first word). Text after
    the last word belongs to the last span.
    """
    n = len(text)
    found: list[tuple[int, int] | None] = []
    i = 0
    for w in words:
        chars = [c for c in str(w.get("text", "")) if not c.isspace()]
        start, end = _find(text, i, chars) if chars else (-1, -1)
        if start < 0:
            found.append(None)
            continue
        while (end < n and not text[end].isalnum() and not text[end].isspace()
               and unicodedata.category(text[end]) not in ("Ps", "Pi")):
            end += 1
        found.append((start, end))
        i = end

    spans: list[tuple[int, int]] = []
    prev = 0
    k = 0
    while k < len(found):
        hit = found[k]
        if hit is not None:
            spans.append((prev, hit[1]))
            prev = hit[1]
            k += 1
            continue
        m = k
        while m < len(found) and found[m] is None:
            m += 1
        gap_end = found[m][0] if m < len(found) else n
        spans.append((prev, gap_end))
        spans += [(gap_end, gap_end)] * (m - k - 1)
        prev = gap_end
        k = m
    if spans:
        spans[-1] = (spans[-1][0], n)
    return spans


def _ends_sentence(text: str, start: int, end: int) -> bool:
    p = end - 1
    while p >= start and (text[p] in _CLOSERS or text[p].isspace()):
        p -= 1
    if p < start or text[p] not in _TERMINATORS:
        return False
    # "3.5" / "e.g": a period glued to a digit or lowercase letter is not a sentence
    # end; one glued to a capital ("the end.Next", chunks joined without a space) is.
    nxt = text[end] if end < len(text) else ""
    return not (text[p] == "." and (nxt.isdigit() or nxt.islower()))


def _runs(labels: list, lo: int, hi: int) -> list[tuple[int, int]]:
    """Maximal ``[start, end)`` index ranges of equal labels within ``[lo, hi)``."""
    runs: list[tuple[int, int]] = []
    for k in range(lo, hi):
        if runs and labels[k] == labels[runs[-1][0]]:
            runs[-1] = (runs[-1][0], k + 1)
        else:
            runs.append((k, k + 1))
    return runs


def smooth_speakers(words: Sequence[Mapping], labels: list, ends: list[bool]) -> list:
    """Merge in-sentence speaker runs shorter than ``MIN_RUN_SEC`` into a neighbour.

    The shortest run goes first, so a label flickering on single characters is
    absorbed before it can drag a longer run along with it.
    """
    out = list(labels)

    def duration(run: tuple[int, int]) -> float:
        return float(words[run[1] - 1]["end"]) - float(words[run[0]]["start"])

    lo = 0
    for hi in [k + 1 for k, end in enumerate(ends) if end] + [len(out)]:
        while True:  # each merge removes a run, so this terminates
            runs = _runs(out, lo, hi)
            if len(runs) < 2:
                break
            shortest = min(runs, key=lambda r: (duration(r), r[1] - r[0]))
            if duration(shortest) >= MIN_RUN_SEC:
                break
            idx = runs.index(shortest)
            neighbours = [runs[j] for j in (idx - 1, idx + 1) if 0 <= j < len(runs)]
            target = max(neighbours, key=lambda r: (duration(r), r[1] - r[0]))
            for k in range(*shortest):
                out[k] = out[target[0]]
        lo = max(lo, hi)
    return out


def build_segments(text: str, words: Sequence[Mapping]) -> list[Segment]:
    """Group timed ``words`` (``text``/``start``/``end``/optional ``speaker``) into segments."""
    if not words:
        return [Segment(text.strip(), 0.0, 0.0)] if text.strip() else []
    spans = locate(text, words)
    ends = [_ends_sentence(text, s, e) for s, e in spans]
    speakers = smooth_speakers(words, [w.get("speaker") for w in words], ends)

    segments: list[Segment] = []
    cur: dict | None = None
    for w, (s, e), spk, end_here in zip(words, spans, speakers, ends):
        if cur is not None and spk != cur["speaker"]:
            segments.append(_close(text, cur))
            cur = None
        if cur is None:
            cur = {"speaker": spk, "start": float(w["start"]), "s": s}
        cur["end"], cur["e"] = float(w["end"]), e
        if end_here:
            segments.append(_close(text, cur))
            cur = None
    if cur is not None:
        segments.append(_close(text, cur))
    return [seg for seg in segments if seg.text]


def _close(text: str, cur: dict) -> Segment:
    return Segment(text[cur["s"]:cur["e"]].strip(), cur["start"], cur["end"], cur["speaker"])
