"""Clean ASR transcripts.

Generalized from the original cooking-specific ``clean_transcript.py``: the
domain vocabulary is now passed in as a parameter instead of hardcoded.

Two ASR failure modes are handled:
  1. The model echoes the ``--context`` prompt at the tail, often repeated.
  2. On short clips the model hallucinates trailing filler.

Both manifest as duplicate / context-leaking sentences at the tail. We:
  - collapse runs of consecutive identical sentences, then
  - drop trailing sentences that are context-string echoes or empty.

``clean_segments`` applies the same rules to timed segments (one segment is
roughly one sentence).
"""

from __future__ import annotations

import re
from dataclasses import replace

from .datatypes import Segment

_SENT_END = "。！？\n"
_SPLIT_RE = re.compile(r"(。|！|？|\n)")

# Mandarin hesitation / filler sounds; a tail sentence made only of these is dropped.
_FILLER = set("嗯呃啊哦唔呢嘛吧呀诶欸哎额嗨")


def split_sentences(text: str) -> list[str]:
    """Split CJK text into sentences, keeping the terminal punctuation."""
    parts = _SPLIT_RE.split(text)
    sents: list[str] = []
    i = 0
    while i < len(parts):
        s = parts[i]
        if i + 1 < len(parts) and parts[i + 1] in _SENT_END:
            sents.append(s + parts[i + 1])
            i += 2
        else:
            if s:
                sents.append(s)
            i += 1
    return sents


def _key(sentence: str) -> str:
    return sentence.rstrip(_SENT_END + " ")


def _is_junk_tail(sentence: str, ctx: set[str]) -> bool:
    """True if a tail sentence is empty, a context-vocabulary echo, or pure filler."""
    tail = sentence.strip().rstrip(_SENT_END)
    if not tail:
        return True
    if ctx and len(set(tail.split()) & ctx) >= 3:
        return True
    return all(ch in _FILLER for ch in tail)


def clean(text: str, context_terms: list[str] | None = None) -> str:
    """Return a cleaned transcript.

    ``context_terms`` is the domain vocabulary supplied to the ASR engine; we
    use it to recognise (and strip) context-prompt echoes at the tail. When it
    is empty, only consecutive-duplicate collapsing and empty-tail trimming run.
    """
    sents = split_sentences(text)

    # 1. Collapse consecutive duplicate sentences (ignoring trailing punctuation).
    deduped: list[str] = []
    for s in sents:
        if deduped and _key(deduped[-1]) == _key(s):
            continue
        deduped.append(s)

    # 2. Strip tail sentences that are empty, context-vocabulary echoes, or filler.
    ctx = set(context_terms or [])
    while deduped and _is_junk_tail(deduped[-1], ctx):
        deduped.pop()

    result = "".join(deduped)
    if result and not result.endswith("\n"):
        result += "\n"
    return result


def clean_segments(
    segments: list[Segment], context_terms: list[str] | None = None
) -> list[Segment]:
    """Segment-level ``clean``: merge consecutive duplicates, strip a junk tail."""
    out: list[Segment] = []
    for seg in segments:
        if not seg.text.strip():
            continue
        prev = out[-1] if out else None
        if prev and prev.speaker == seg.speaker and _key(prev.text) == _key(seg.text):
            out[-1] = replace(prev, end=max(prev.end, seg.end))
            continue
        out.append(seg)

    ctx = set(context_terms or [])
    while out and _is_junk_tail(out[-1].text, ctx):
        out.pop()
    return out
