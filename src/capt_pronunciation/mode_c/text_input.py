"""Text normalization for Mode C.

Punctuation is preserved because commas, semicolons, dashes, and parentheses
often mark appositives, nonrestrictive clauses, and discourse boundaries.
"""

from __future__ import annotations

import re
from typing import List

from .schemas import TextSpan


def normalize_sentence(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", " - ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]])", r"\1", text)
    return text.strip()


def split_sentence_spans(text: str) -> List[TextSpan]:
    spans: List[TextSpan] = []
    start = 0
    for match in re.finditer(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text):
        end = match.start()
        chunk = text[start:end].strip()
        if chunk:
            spans.append(TextSpan(text=chunk, start=start, end=end, label="sentence"))
        start = match.end()
    chunk = text[start:].strip()
    if chunk:
        spans.append(TextSpan(text=chunk, start=start, end=len(text), label="sentence"))
    return spans or [TextSpan(text=text, start=0, end=len(text), label="sentence")]


def word_spans(text: str) -> List[TextSpan]:
    spans: List[TextSpan] = []
    for match in re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+(?:\.[0-9]+)?", text):
        spans.append(TextSpan(text=match.group(0), start=match.start(), end=match.end(), label="word"))
    return spans

