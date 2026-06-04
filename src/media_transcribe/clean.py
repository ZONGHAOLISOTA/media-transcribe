"""Clean ASR transcripts.

Generalized from the original cooking-specific ``clean_transcript.py``: the
domain vocabulary is now passed in as a parameter instead of hardcoded.

Two ASR failure modes are handled:
  1. The model echoes the ``--context`` prompt at the tail, often repeated.
  2. On short clips the model hallucinates trailing filler.

Both manifest as duplicate / context-leaking sentences at the tail. We:
  - collapse runs of consecutive identical sentences, then
  - drop trailing sentences that are context-string echoes or empty.
"""

from __future__ import annotations

import re

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
    while deduped:
        tail = deduped[-1].strip().rstrip(_SENT_END)
        if not tail:
            deduped.pop()
            continue
        if ctx and len(set(tail.split()) & ctx) >= 3:
            deduped.pop()
            continue
        if all(ch in _FILLER for ch in tail):
            deduped.pop()
            continue
        break

    result = "".join(deduped)
    if result and not result.endswith("\n"):
        result += "\n"
    return result
