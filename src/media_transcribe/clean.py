"""Clean ASR transcripts.

Generalized from the original cooking-specific ``clean_transcript.py``: the
domain vocabulary is now passed in as a parameter instead of hardcoded.

Two ASR failure modes are handled:
  1. The model echoes the ``--context`` prompt at the tail, often repeated.
  2. On short clips the model hallucinates trailing filler.

Both manifest as duplicate / context-leaking sentences at the tail. We:
  - collapse runs of consecutive identical sentences, then
  - drop trailing sentences that are context-string echoes or empty.

A context echo is the model reciting its prompt: a tail sentence made of nothing
but context terms — at least three distinct ones — in the order they were given.
A real sentence that merely names terms ("We use Kubernetes, Docker and Helm.")
always has some other word and is kept. Sentences are split on CJK enders only,
so a whole English transcript is one "sentence" here.

``clean_segments`` applies the same rules to timed segments, judging the tail a
whole sentence at a time (a sentence may span several segments).
"""

from __future__ import annotations

import re
import string
from dataclasses import replace

from .datatypes import Segment
from .segments import TERMINATORS

_SENT_END = "。！？\n"
_SPLIT_RE = re.compile(r"(。|！|？|\n)")
_TOKEN_STRIP = string.punctuation + "，。！？、；：“”‘’（）《》"

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


def _words(text: str) -> list[str]:
    """Whitespace words with surrounding punctuation stripped; repeats in a row merged."""
    out: list[str] = []
    for raw in text.split():
        word = raw.strip(_TOKEN_STRIP)
        if word and (not out or out[-1] != word):
            out.append(word)
    return out


def _is_context_echo(sentence: str, terms: list[str]) -> bool:
    """Nothing but >= 3 distinct context terms, following the context's order (cyclically)."""
    words = _words(sentence)
    if len(set(words)) < 3:
        return False
    order: dict[str, int] = {}
    for term in terms:
        order.setdefault(term, len(order))
    if any(w not in order for w in words):
        return False
    steps = [order[w] for w in words]
    return all(b == (a + 1) % len(order) for a, b in zip(steps, steps[1:]))


def _is_junk_tail(sentence: str, terms: list[str]) -> bool:
    """True if a tail sentence is empty, a context-prompt echo, or pure filler."""
    tail = sentence.strip().rstrip(_SENT_END)
    if not tail:
        return True
    if terms and _is_context_echo(tail, terms):
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

    # 2. Strip tail sentences that are empty, context-prompt echoes, or filler.
    terms = _words(" ".join(context_terms or []))
    while deduped and _is_junk_tail(deduped[-1], terms):
        deduped.pop()

    result = "".join(deduped)
    if result and not result.endswith("\n"):
        result += "\n"
    return result


def _ends_sentence(text: str) -> bool:
    tail = text.rstrip().rstrip("\"'”’」』）)】》〉]")
    return bool(tail) and tail[-1] in TERMINATORS


def clean_segments(
    segments: list[Segment], context_terms: list[str] | None = None
) -> list[Segment]:
    """Segment-level ``clean``: merge consecutive duplicates, strip a junk tail.

    Like ``clean``, only sentences ending in 。！？ are collapsed, so an English
    "Okay. Okay." survives in both modes.
    """
    out: list[Segment] = []
    for seg in segments:
        if not seg.text.strip():
            continue
        prev = out[-1] if out else None
        if (prev and prev.speaker == seg.speaker and _key(prev.text) == _key(seg.text)
                and seg.text.rstrip()[-1:] in "。！？"):
            out[-1] = replace(prev, end=max(prev.end, seg.end))
            continue
        out.append(seg)

    # Judge the tail a sentence at a time: a sentence split at a clause mark or a
    # speaker change spans several segments.
    terms = _words(" ".join(context_terms or []))
    while out:
        start = len(out) - 1
        while start > 0 and not _ends_sentence(out[start - 1].text):
            start -= 1
        if not _is_junk_tail(" ".join(s.text for s in out[start:]), terms):
            break
        del out[start:]
    return out
