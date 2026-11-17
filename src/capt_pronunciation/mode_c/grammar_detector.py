"""Grammar pattern detection for Mode C."""

from __future__ import annotations

import re
from typing import List, Optional

from .schemas import GrammarPoint, SyntaxToken


STRUCTURE_BOUNDARY_WORDS = {
    "and",
    "or",
    "but",
    "that",
    "which",
    "who",
    "whom",
    "whose",
    "where",
    "when",
    "because",
    "although",
    "though",
    "if",
    "unless",
    "while",
}
GERUND_PHRASE_PREPOSITIONS = {
    "about",
    "after",
    "against",
    "among",
    "around",
    "at",
    "before",
    "behind",
    "beside",
    "between",
    "by",
    "during",
    "for",
    "from",
    "in",
    "into",
    "near",
    "of",
    "on",
    "onto",
    "over",
    "through",
    "to",
    "toward",
    "towards",
    "under",
    "with",
    "without",
}


class GrammarDetector:
    def detect(
        self,
        sentence: str,
        tokens: Optional[List[SyntaxToken]] = None,
    ) -> List[GrammarPoint]:
        points: List[GrammarPoint] = []
        specs = [
            (r"\b(had)\s+been\s+(?:\w+ly\s+)?([A-Za-z]+ed|known|seen|made|given|shown|built|found)\b", "past_perfect_passive", "过去完成时 + 被动语态：表示在过去某个时间之前已经被完成或被接受。"),
            (r"\b(has|have)\s+been\s+(?:\w+ly\s+)?([A-Za-z]+ed|known|seen|made|given|shown|built|found)\b", "present_perfect_passive", "现在完成时 + 被动语态：强调到现在为止已经产生的被动结果。"),
            (r"\b(was|were|is|are|be|been|being)\s+(?:\w+ly\s+)?([A-Za-z]+ed|known|seen|made|given|shown|built|found)\b", "passive_voice", "被动语态：主语是动作的承受者，阅读时要还原“被谁/被什么影响”。"),
            (r"\b(was|were|is|are|am)\s+[A-Za-z]+ing\b", "progressive_aspect", "进行体：强调动作正在发生或处于持续过程。"),
            (r"\b(will|would|can|could|may|might|must|should)\s+have\s+[A-Za-z]+ed\b", "modal_perfect", "情态动词完成式：常表达推测、可能性或对过去情况的判断。"),
            (r"\bto\s+[A-Za-z]+\b", "infinitive_phrase", "不定式结构：常作目的、结果、补语或后置定语。"),
            (r"\bmore\s+\w+\s+than\b|\bas\s+\w+\s+as\b", "comparison", "比较结构：注意比较对象和比较范围。"),
            (r"\b(it)\s+(is|was)\s+[^.?!,;]+?\s+that\b", "cleft_sentence", "强调句/裂句结构：it is/was ... that ... 用来突出中间成分。"),
        ]
        for pattern, point_type, explanation in specs:
            for match in re.finditer(pattern, sentence, flags=re.I):
                span = match.group(0)
                points.append(
                    GrammarPoint(
                        point_id=f"G{len(points)+1:03d}",
                        span=span,
                        type=point_type,
                        explanation_zh=explanation,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.78,
                    )
                )

        points = [
            point
            for point in points
            if (
                point.type != "infinitive_phrase"
                or valid_infinitive(sentence, point.start, point.end, tokens)
            )
            and (
                point.type != "cleft_sentence"
                or valid_cleft(sentence, point.start, point.end, tokens)
            )
        ]

        for match in re.finditer(r"\b[A-Za-z]+ing\b", sentence, flags=re.I):
            if should_skip_ing_point(sentence, match.start(), match.end()):
                continue
            start, end = expand_ing_phrase(sentence, match.start(), match.end())
            points.append(
                GrammarPoint(
                    point_id=f"G{len(points)+1:03d}",
                    span=sentence[start:end].strip(" ,"),
                    type="ing_phrase",
                    explanation_zh="V-ing 结构：可能作动名词、现在分词定语或状语，需要结合位置判断功能。",
                    start=start,
                    end=end,
                    confidence=0.72,
                )
            )

        for match in re.finditer(r"\bsuch as\b[^,;.()]{2,120}", sentence, flags=re.I):
            points.append(
                GrammarPoint(
                    point_id=f"G{len(points)+1:03d}",
                    span=match.group(0).strip(" ,"),
                    type="example_phrase",
                    explanation_zh="举例短语：用来补充说明前面的名词或概念。",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.72,
                )
            )

        for match in re.finditer(
            r"\b(?:after|before|since)\s+\d{1,4}\s*(?:B\.C\.E\.|C\.E\.|BCE|CE|A\.D\.|AD|B\.C\.|BC)?(?=\W|$)",
            sentence,
            flags=re.I,
        ):
            points.append(
                GrammarPoint(
                    point_id=f"G{len(points)+1:03d}",
                    span=match.group(0).strip(" ,"),
                    type="time_prepositional_phrase",
                    explanation_zh="时间介词短语：说明动作发生的时间背景，不是完整从句。",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.78,
                )
            )

        for match in re.finditer(r",\s*[^,]{3,80}?\s*,", sentence):
            text = match.group(0).strip(" ,")
            if text.lower().startswith("such as"):
                continue
            if is_coordinate_list_item(sentence, match.start(), match.end(), text):
                continue
            if text and any(w in text.lower().split() for w in {"which", "who", "although", "because"}):
                continue
            points.append(
                GrammarPoint(
                    point_id=f"G{len(points)+1:03d}",
                    span=text,
                    type="parenthetical_or_appositive",
                    explanation_zh="插入语或同位语：通常补充说明前面的名词或整句话，阅读时可以先跳过。",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.58,
                )
            )

        return dedupe_points(points)


def valid_infinitive(
    sentence: str,
    start: int,
    end: int,
    tokens: Optional[List[SyntaxToken]],
) -> bool:
    candidate = sentence[start:end]
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", candidate)
    if len(words) < 2 or words[0].lower() != "to":
        return False
    following = words[1].lower()
    number_words = {
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "hundred", "thousand",
    }
    if following in number_words:
        return False
    if tokens:
        to_token = next(
            (token for token in tokens if token.start <= start < token.end and token.text.lower() == "to"),
            None,
        )
        if to_token is None:
            to_token = next(
                (token for token in tokens if start <= token.start < end and token.text.lower() == "to"),
                None,
            )
        if to_token is not None:
            following_tokens = [token for token in tokens if token.index > to_token.index]
            next_token = min(following_tokens, key=lambda token: token.index) if following_tokens else None
            if next_token is not None and next_token.pos in {"VERB", "AUX"}:
                return to_token.pos in {"PART", "ADP"} or to_token.tag == "TO"
            if next_token is not None and next_token.tag:
                return False
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z'-]*", following))


def valid_cleft(
    sentence: str,
    start: int,
    end: int,
    tokens: Optional[List[SyntaxToken]],
) -> bool:
    span = sentence[start:end]
    match = re.search(r"\bit\s+(?:is|was)\s+(.+?)\s+that\b", span, flags=re.I)
    if not match:
        return False
    focus_words = re.findall(r"[A-Za-z][A-Za-z'-]*", match.group(1))
    if not focus_words or focus_words[-1].lower() in {"and", "but", "or"}:
        return False
    tail = sentence[start + match.end() :]
    if len(re.findall(r"[A-Za-z][A-Za-z'-]*", tail)) < 2:
        return False
    if tokens:
        that_position = start + match.end() - len("that")
        after = [token for token in tokens if token.start >= that_position]
        if any(
            token.pos in {"VERB", "AUX"}
            or token.tag in {"VBD", "VBP", "VBZ", "MD"}
            for token in after
        ):
            return True
        if any(token.tag for token in after):
            return False
    return bool(re.search(r"\b(?:is|are|was|were|has|have|had|can|could|will|would|\w+(?:ed|s))\b", tail, flags=re.I))


def should_skip_ing_point(sentence: str, start: int, end: int) -> bool:
    prev_word = previous_word(sentence, start)
    next_word = next_word_after(sentence, end)
    before = sentence[max(0, start - 3) : start]
    after = sentence[end : min(len(sentence), end + 3)]

    if re.search(r"(^|[,;:])\s*$", sentence[:start]):
        return False
    if prev_word in {"by", "for", "from", "in", "of", "on", "without", "after", "before", "while", "when"}:
        return False
    if prev_word in {"am", "is", "are", "was", "were", "be", "been", "being"}:
        return True
    if re.match(
        r"\s+(of|to|for|in|on|with|by|from|about|against|among|around|at|before|after|into|onto|over|under|through|towards?|near|beside|behind|between|during)\b",
        sentence[end : end + 16],
        flags=re.I,
    ):
        return False
    if "," in after or ";" in after:
        return False
    if next_word and next_word not in STRUCTURE_BOUNDARY_WORDS and not re.search(r"[,;:]\s*$", before):
        return True
    return False


def expand_ing_phrase(sentence: str, start: int, end: int) -> tuple[int, int]:
    boundary_candidates = [pos for pos in (sentence.find(ch, end) for ch in ",;:.!?") if pos >= 0]
    boundary = min(boundary_candidates) if boundary_candidates else len(sentence)
    text = sentence[start:boundary]
    words = list(re.finditer(r"[A-Za-z][A-Za-z'-]*", text))
    if len(words) > 3 and not any(word.group(0).lower() in GERUND_PHRASE_PREPOSITIONS for word in words[1:4]):
        boundary = start + words[2].end()
    elif len(words) > 6:
        boundary = start + words[5].end()
    return start, boundary


def previous_word(sentence: str, start: int) -> str:
    words = list(re.finditer(r"[A-Za-z][A-Za-z'-]*", sentence[:start]))
    return words[-1].group(0).lower() if words else ""


def next_word_after(sentence: str, end: int) -> str:
    match = re.search(r"[A-Za-z][A-Za-z'-]*", sentence[end:])
    return match.group(0).lower() if match else ""


def dedupe_points(points: List[GrammarPoint]) -> List[GrammarPoint]:
    out: List[GrammarPoint] = []
    seen: set[tuple[str, int, int]] = set()
    for point in sorted(points, key=lambda x: (x.start, x.end)):
        if point.type == "passive_voice" and any(
            existing.type.endswith("_passive") and spans_overlap((point.start, point.end), (existing.start, existing.end))
            for existing in out
        ):
            continue
        key = (point.type, point.start, point.end)
        if key in seen:
            continue
        seen.add(key)
        point.point_id = f"G{len(out)+1:03d}"
        out.append(point)
    return out


def spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and a[1] > b[0]


def is_coordinate_list_item(sentence: str, start: int, end: int, text: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    if not words or len(words) > 3:
        return False
    before = sentence[max(0, start - 16) : start].lower()
    after = sentence[end : min(len(sentence), end + 16)].lower()
    return bool(
        re.search(r"\b(and|or)\s*$", before)
        or re.match(r"\s*(and|or)\b", after)
        or ("," in before and re.match(r"\s*(and|or)\b", after))
    )
