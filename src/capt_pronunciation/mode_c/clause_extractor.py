"""Clause, modifier, and chunk extraction for Mode C."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .schemas import ClauseInfo, MainClause, ModifierInfo, SyntaxToken, TranslationChunk


RELATIVE_MARKERS = {"which", "that", "who", "whom", "whose", "where", "when"}
ADVERBIAL_MARKERS = {
    "because": ("adverbial_clause_reason", "原因状语从句"),
    "although": ("adverbial_clause_concession", "让步状语从句"),
    "though": ("adverbial_clause_concession", "让步状语从句"),
    "if": ("adverbial_clause_condition", "条件状语从句"),
    "unless": ("adverbial_clause_condition", "条件状语从句"),
    "when": ("adverbial_clause_time", "时间状语从句"),
    "while": ("adverbial_clause_time_or_contrast", "时间/对比状语从句"),
    "since": ("adverbial_clause_time_or_reason", "时间/原因状语从句"),
    "before": ("adverbial_clause_time", "时间状语从句"),
    "after": ("adverbial_clause_time", "时间状语从句"),
}
NOUN_CLAUSE_MARKERS = {"what", "whether", "how", "why"}
COMPLEMENT_TAKING_VERBS = {
    "admit", "agree", "argue", "believe", "claim", "conclude", "confirm",
    "discover", "explain", "find", "hear", "hope", "indicate", "know",
    "notice", "predict", "report", "say", "show", "suggest", "tell", "think",
    "said", "thought", "showed", "suggested", "reported", "believed",
    "explained", "found", "discovered", "indicated", "claimed",
}
PREPOSITIONS = {
    "in", "on", "at", "for", "with", "by", "from", "of", "about", "through",
    "without", "within", "over", "under", "between", "among", "during", "before",
    "after", "into", "onto", "against", "around", "as",
}
TEMPORAL_PREPOSITION_MARKERS = {"after", "before", "since"}
CORE_ONLY_MODIFIER_TYPES = {"adjective_modifier", "noun_phrase"}
STRUCTURAL_MODIFIER_TYPES = {
    "adverb_modifier",
    "adverbial_modifier",
    "nominal_adverbial",
    "appositive",
    "prepositional_phrase",
    "time_prepositional_phrase",
}
MODIFIER_PRIORITY = {
    "time_prepositional_phrase": 95,
    "prepositional_phrase": 85,
    "appositive": 80,
    "adverbial_modifier": 75,
    "nominal_adverbial": 70,
    "adverb_modifier": 65,
    "noun_phrase": 25,
    "adjective_modifier": 15,
}


class ClauseExtractor:
    def extract(
        self,
        sentence: str,
        tokens: List[SyntaxToken],
        parser_raw: Dict[str, Any],
    ) -> Dict[str, Any]:
        main_clause = self.extract_main_clause(sentence, tokens)
        clauses = self.extract_clauses(sentence, tokens, main_clause, parser_raw)
        modifiers = self.extract_modifiers(sentence, tokens, parser_raw, clauses)
        chunks = self.build_chunks(sentence, main_clause, clauses)
        return {
            "main_clause": main_clause,
            "clauses": clauses,
            "modifiers": modifiers,
            "chunks": chunks,
        }

    def extract_main_clause(self, sentence: str, tokens: List[SyntaxToken]) -> MainClause:
        root = self._root(tokens)
        if root is None:
            return self._heuristic_main_clause(sentence)
        if not any(t.dep in {"nsubj", "nsubjpass", "csubj"} for t in tokens):
            return self._heuristic_main_clause(sentence)

        subject = self._collect_by_deps(tokens, root.index, {"nsubj", "nsubjpass", "csubj", "expl"})
        complements = self._collect_by_deps(tokens, root.index, {"obj", "dobj", "iobj", "attr", "acomp", "oprd", "ccomp", "xcomp", "prep", "obl"})
        aux = self._collect_by_deps(tokens, root.index, {"aux", "auxpass", "neg"}, include_root=False)
        predicate_parts = aux + [root]
        subject_text = self._span_text(sentence, subject) or self._left_np(sentence, tokens, root.index)
        predicate_text = self._span_text(sentence, predicate_parts) or root.text
        complement_text = self._span_text(sentence, complements)
        full_tokens = sorted(subject + predicate_parts + complements, key=lambda t: t.start)
        full_span = self._span_text(sentence, full_tokens) or f"{subject_text} {predicate_text} {complement_text}".strip()
        coordinated = self._coordinated_predicates(sentence, tokens, root)
        if coordinated:
            predicate_text = "; ".join([predicate_text] + [item["predicate"] for item in coordinated if item["predicate"]])
            complement_pieces = [complement_text] if complement_text else []
            complement_pieces.extend(item["complement"] for item in coordinated if item["complement"])
            complement_text = "; ".join(complement_pieces)
            base_start = min((token.start for token in full_tokens), default=root.start)
            base_end = max((token.end for token in full_tokens), default=root.end)
            coord_end = max(item["end"] for item in coordinated)
            full_span = sentence[base_start : max(base_end, coord_end)].strip(" ,;")
        explanation = "先抓主干：主语、谓语和宾语/补足语构成句子的核心意思。"
        return MainClause(
            subject=subject_text,
            predicate=predicate_text,
            object_or_complement=complement_text,
            full_span=full_span,
            explanation_zh=explanation,
            confidence=0.82 if subject_text and predicate_text else 0.58,
        )

    def extract_clauses(
        self,
        sentence: str,
        tokens: List[SyntaxToken],
        main_clause: MainClause,
        parser_raw: Optional[Dict[str, Any]] = None,
    ) -> List[ClauseInfo]:
        clauses: List[ClauseInfo] = []
        seen: set[Tuple[int, int, str]] = set()

        dep_clause_deps = {"relcl", "advcl", "ccomp", "xcomp", "acl", "csubj"}
        for token in tokens:
            if token.dep in dep_clause_deps:
                span = self._subtree_span(sentence, tokens, token.index)
                marker = self._clause_marker(sentence, tokens, token, span)
                ctype, function = classify_clause(marker, token.dep)
                key = (span[0], span[1], ctype)
                if key in seen:
                    continue
                seen.add(key)
                clauses.append(
                    ClauseInfo(
                        clause_id=f"C{len(clauses)+1:03d}",
                        span=sentence[span[0] : span[1]].strip(" ,;"),
                        type=ctype,
                        function_zh=function,
                        modifies=self._clause_modified_target(sentence, tokens, token, marker, ctype, main_clause),
                        marker=marker,
                        start=span[0],
                        end=span[1],
                        reading_strategy=clause_strategy(function),
                        confidence=0.82,
                    )
                )

        constituency = (parser_raw or {}).get("constituency_spans", [])
        for span_info in constituency if isinstance(constituency, list) else []:
            if not isinstance(span_info, dict) or span_info.get("label") not in {"SBAR", "SBARQ", "SINV", "SQ", "S"}:
                continue
            start = int(span_info.get("start", 0))
            end = int(span_info.get("end", 0))
            if end <= start or end - start >= len(sentence.strip()) - 1:
                continue
            parent_label = str(span_info.get("parent_label", ""))
            marker = self._first_marker(sentence[start:end])
            if span_info.get("label") == "S" and parent_label != "VP" and not marker:
                continue
            if any(span_overlap_ratio((start, end), (item.start, item.end)) >= 0.72 for item in clauses):
                for item in clauses:
                    if span_overlap_ratio((start, end), (item.start, item.end)) >= 0.72:
                        item.confidence = max(item.confidence, 0.90)
                continue
            ctype, function = self._classify_marker_clause(sentence, tokens, start, marker, "ccomp" if parent_label == "VP" else "")
            clauses.append(
                ClauseInfo(
                    clause_id=f"C{len(clauses)+1:03d}",
                    span=sentence[start:end].strip(" ,;"),
                    type=ctype,
                    function_zh=function,
                    modifies=self._marker_modified_target(sentence, tokens, start, marker, ctype, main_clause),
                    marker=marker,
                    start=start,
                    end=end,
                    reading_strategy=clause_strategy(function),
                    confidence=0.78,
                )
            )

        for marker in sorted(RELATIVE_MARKERS | set(ADVERBIAL_MARKERS) | NOUN_CLAUSE_MARKERS, key=len, reverse=True):
            for match in re.finditer(rf"\b{re.escape(marker)}\b", sentence, flags=re.I):
                marker_lower = marker.lower()
                if not self._marker_introduces_clause(sentence, tokens, match.start(), marker_lower):
                    continue
                span = marker_clause_span(sentence, match.start(), marker_lower)
                if not span:
                    continue
                ctype, function = self._classify_marker_clause(
                    sentence,
                    tokens,
                    match.start(),
                    marker_lower,
                    "",
                )
                key = (span[0], span[1], ctype)
                if key in seen:
                    continue
                seen.add(key)
                clauses.append(
                    ClauseInfo(
                        clause_id=f"C{len(clauses)+1:03d}",
                        span=sentence[span[0] : span[1]].strip(" ,;"),
                        type=ctype,
                        function_zh=function,
                        modifies=self._marker_modified_target(sentence, tokens, match.start(), marker_lower, ctype, main_clause),
                        marker=marker_lower,
                        start=span[0],
                        end=span[1],
                        reading_strategy=clause_strategy(function),
                        confidence=0.68,
                    )
                )

        clauses = dedupe_clauses(clauses)
        clauses.sort(key=lambda c: (c.start, c.end))
        for i, clause in enumerate(clauses, start=1):
            clause.clause_id = f"C{i:03d}"
        return clauses

    def extract_modifiers(
        self,
        sentence: str,
        tokens: List[SyntaxToken],
        parser_raw: Dict[str, Any],
        clauses: List[ClauseInfo],
    ) -> List[ModifierInfo]:
        modifiers: List[ModifierInfo] = []
        covered = [(c.start, c.end) for c in clauses]

        dep_map = {
            "amod": ("adjective_modifier", "形容词修饰名词"),
            "advmod": ("adverb_modifier", "副词修饰动词/形容词/全句"),
            "appos": ("appositive", "同位语"),
            "prep": ("prepositional_phrase", "介词短语"),
            "obl": ("adverbial_modifier", "状语成分"),
            "npadvmod": ("nominal_adverbial", "名词性状语"),
        }
        for token in tokens:
            if token.dep not in dep_map:
                continue
            span = self._subtree_span(sentence, tokens, token.index)
            if overlaps_any(span, covered):
                continue
            mtype, function = dep_map[token.dep]
            if mtype == "prepositional_phrase":
                span = trim_prepositional_modifier_span(sentence, span)
            modifiers.append(
                ModifierInfo(
                    modifier_id=f"M{len(modifiers)+1:03d}",
                    span=sentence[span[0] : span[1]].strip(" ,;"),
                    type=mtype,
                    function_zh=function,
                    modifies=self._modified_target(tokens, token),
                    start=span[0],
                    end=span[1],
                    reading_strategy=modifier_strategy(function),
                    confidence=0.78,
                )
            )

        for match in re.finditer(
            r"\b(?:after|before|since)\s+\d{1,4}\s*(?:B\.C\.E\.|C\.E\.|BCE|CE|A\.D\.|AD|B\.C\.|BC)?(?=\W|$)",
            sentence,
            flags=re.I,
        ):
            span = (match.start(), match.end())
            if overlaps_any(span, covered):
                continue
            modifiers.append(
                ModifierInfo(
                    modifier_id=f"M{len(modifiers)+1:03d}",
                    span=sentence[span[0] : span[1]].strip(" ,;"),
                    type="time_prepositional_phrase",
                    function_zh="时间介词短语",
                    modifies="main clause",
                    start=span[0],
                    end=span[1],
                    reading_strategy="把它作为时间背景理解，不要当成完整从句。",
                    confidence=0.78,
                )
            )

        for phrase in parser_raw.get("noun_chunks", []) or []:
            span = (int(phrase.get("start", 0)), int(phrase.get("end", 0)))
            text = str(phrase.get("text", "")).strip()
            if not text or len(text.split()) <= 1 or overlaps_any(span, covered):
                continue
            modifiers.append(
                ModifierInfo(
                    modifier_id=f"M{len(modifiers)+1:03d}",
                    span=text,
                    type="noun_phrase",
                    function_zh="带修饰成分的名词短语",
                    modifies=str(phrase.get("root", "")),
                    start=span[0],
                    end=span[1],
                    reading_strategy="阅读时先抓中心名词，再回看前后的限定和修饰。",
                    confidence=0.76,
                )
            )

        for match in re.finditer(r"\b(" + "|".join(sorted(PREPOSITIONS)) + r")\b[^,;.()]{2,80}", sentence, flags=re.I):
            span = (match.start(), min(match.end(), next_boundary(sentence, match.start())))
            span = trim_prepositional_modifier_span(sentence, span)
            if overlaps_any(span, covered):
                continue
            if overlaps_any(span, [(m.start, m.end) for m in modifiers]):
                continue
            text = sentence[span[0] : span[1]].strip(" ,;")
            if len(text.split()) < 2:
                continue
            if any(normalize_space(m.span).lower() == normalize_space(text).lower() for m in modifiers):
                continue
            modifiers.append(
                ModifierInfo(
                    modifier_id=f"M{len(modifiers)+1:03d}",
                    span=text,
                    type="prepositional_phrase",
                    function_zh="介词短语",
                    modifies="附近的名词、动词或整个分句",
                    start=span[0],
                    end=span[1],
                    reading_strategy="先识别介词短语整体，再判断它修饰前面的名词还是谓语动作。",
                    confidence=0.62,
                )
            )

        modifiers = filter_modifiers(dedupe_modifiers(modifiers))
        modifiers.sort(key=lambda m: (m.start, m.end))
        for i, modifier in enumerate(modifiers, start=1):
            modifier.modifier_id = f"M{i:03d}"
        return modifiers

    def build_chunks(self, sentence: str, main_clause: MainClause, clauses: List[ClauseInfo]) -> List[TranslationChunk]:
        ranges: List[Tuple[int, int, str, str]] = []
        if main_clause.full_span:
            start = sentence.lower().find(main_clause.full_span.lower())
            if start >= 0:
                ranges.append((start, start + len(main_clause.full_span), "main_clause", "主干"))
        for clause in clauses:
            ranges.append((clause.start, clause.end, clause.type, clause.function_zh))
        ranges = sorted(ranges, key=lambda x: (x[0], x[1]))
        chunks: List[TranslationChunk] = []
        used: List[Tuple[int, int]] = []
        for start, end, ctype, role in ranges:
            text = sentence[start:end].strip(" ,;")
            if not text or overlaps_any((start, end), used):
                continue
            used.append((start, end))
            chunks.append(
                TranslationChunk(
                    chunk_id=f"B{len(chunks)+1:03d}",
                    source=text,
                    role=role,
                    translation_zh=f"这一块的作用是：{role}。",
                    start=start,
                    end=end,
                )
            )
        if not chunks:
            chunks.append(
                TranslationChunk(
                    chunk_id="B001",
                    source=sentence,
                    role="whole_sentence",
                    translation_zh="先按完整句子理解，再拆分主干和修饰成分。",
                    start=0,
                    end=len(sentence),
                )
            )
        return chunks

    def _root(self, tokens: List[SyntaxToken]) -> Optional[SyntaxToken]:
        for token in tokens:
            if token.dep == "ROOT":
                return token
        return next((t for t in tokens if t.pos in {"VERB", "AUX"}), None)

    def _collect_by_deps(self, tokens: List[SyntaxToken], head: int, deps: set[str], include_root: bool = True) -> List[SyntaxToken]:
        selected = [t for t in tokens if t.head == head and t.dep in deps]
        expanded: List[SyntaxToken] = []
        for token in selected:
            span = self._subtree_span("", tokens, token.index)
            expanded.extend([t for t in tokens if span[0] <= t.start and t.end <= span[1]])
        return sorted(expanded or selected, key=lambda t: t.start)

    def _span_text(self, sentence: str, tokens: Iterable[SyntaxToken]) -> str:
        toks = sorted(list(tokens), key=lambda t: t.start)
        if not toks:
            return ""
        return sentence[toks[0].start : toks[-1].end].strip()

    def _subtree_span(self, sentence: str, tokens: List[SyntaxToken], root_index: int) -> Tuple[int, int]:
        children = descendants(tokens, root_index)
        group = [tokens[root_index]] + children if 0 <= root_index < len(tokens) else children
        if not group:
            return (0, 0)
        return (min(t.start for t in group), max(t.end for t in group))

    def _coordinated_predicates(self, sentence: str, tokens: List[SyntaxToken], root: SyntaxToken) -> List[Dict[str, Any]]:
        coordinated: List[Dict[str, Any]] = []
        for token in tokens:
            if token.head != root.index or token.dep != "conj" or token.pos not in {"VERB", "AUX"}:
                continue
            span = self._subtree_span(sentence, tokens, token.index)
            comma = sentence.find(",", token.start, span[1])
            end = comma if comma >= 0 else span[1]
            complements = self._collect_by_deps(
                tokens,
                token.index,
                {"obj", "dobj", "iobj", "attr", "acomp", "oprd", "ccomp", "xcomp", "prep", "obl"},
            )
            coordinated.append(
                {
                    "predicate": token.text,
                    "complement": self._span_text(sentence, complements),
                    "start": span[0],
                    "end": end,
                    "span": sentence[span[0] : end].strip(" ,;"),
                }
            )
        return coordinated

    def _modified_target(self, tokens: List[SyntaxToken], token: SyntaxToken) -> str:
        if token.head is not None and 0 <= token.head < len(tokens):
            return tokens[token.head].text
        return ""

    def _clause_modified_target(
        self,
        sentence: str,
        tokens: List[SyntaxToken],
        token: SyntaxToken,
        marker: str,
        clause_type: str,
        main_clause: MainClause,
    ) -> str:
        if self._is_relative_clause(marker, clause_type, token.dep):
            marker_start = self._marker_start_before_token(sentence, token.start, marker)
            target = self._nearest_left_noun_phrase(sentence, tokens, marker_start)
            if target:
                return target
            if token.head is not None and 0 <= token.head < len(tokens):
                target = self._expanded_noun_phrase(sentence, tokens, token.head)
                if target:
                    return target
            target = self._nearest_left_noun_phrase(sentence, tokens, token.start)
            if target:
                return target
        return self._modified_target(tokens, token) or main_clause.full_span or "main clause"

    def _marker_modified_target(
        self,
        sentence: str,
        tokens: List[SyntaxToken],
        marker_start: int,
        marker: str,
        clause_type: str,
        main_clause: MainClause,
    ) -> str:
        if self._is_relative_clause(marker, clause_type, ""):
            target = self._nearest_left_noun_phrase(sentence, tokens, marker_start)
            if target:
                return target
        return main_clause.full_span or "main clause"

    def _clause_marker(
        self,
        sentence: str,
        tokens: List[SyntaxToken],
        token: SyntaxToken,
        span: Tuple[int, int],
    ) -> str:
        if token.dep == "xcomp":
            for candidate in tokens:
                if candidate.head == token.index and candidate.text.lower() == "to":
                    return "to"
            if token.index > 0 and tokens[token.index - 1].text.lower() == "to":
                return "to"
            return ""
        return self._first_marker(sentence[span[0] : span[1]])

    def _marker_introduces_clause(self, sentence: str, tokens: List[SyntaxToken], marker_start: int, marker: str) -> bool:
        span = marker_clause_span(sentence, marker_start, marker)
        if not span:
            return False
        marker_token = self._token_at_char(tokens, marker_start)
        if marker == "that" and marker_token and marker_token.pos in {"DET"}:
            return False
        skip_index = marker_token.index if marker_token else None
        if marker in TEMPORAL_PREPOSITION_MARKERS and marker_token and marker_token.pos == "ADP":
            return self._span_has_finite_verb(tokens, span[0], span[1], skip_index=skip_index)
        return self._span_has_finite_verb(tokens, span[0], span[1], skip_index=skip_index)

    def _classify_marker_clause(
        self,
        sentence: str,
        tokens: List[SyntaxToken],
        marker_start: int,
        marker: str,
        dep: str,
    ) -> Tuple[str, str]:
        if marker == "that" and self._is_complementizer_that(sentence, tokens, marker_start, dep):
            return classify_clause(marker, dep or "ccomp")
        return classify_clause(marker, dep)

    def _is_complementizer_that(
        self,
        sentence: str,
        tokens: List[SyntaxToken],
        marker_start: int,
        dep: str,
    ) -> bool:
        if dep in {"ccomp", "csubj"}:
            return True
        marker_token = self._token_at_char(tokens, marker_start)
        if marker_token is not None:
            head = next((token for token in tokens if token.index == marker_token.head), None)
            if marker_token.dep == "mark" and head is not None and head.dep in {"ccomp", "csubj"}:
                return True
        previous_tokens = [token for token in tokens if token.end <= marker_start]
        previous_token = max(previous_tokens, key=lambda token: token.end) if previous_tokens else None
        if previous_token is not None and self._is_noun_like(previous_token):
            return False
        prefix_words = re.findall(r"[A-Za-z][A-Za-z'-]*", sentence[max(0, marker_start - 100) : marker_start].lower())
        if not prefix_words:
            return False
        lemma_candidates = {
            token.lemma.lower()
            for token in tokens
            if token.end <= marker_start and token.lemma
        }
        return bool(set(prefix_words[-4:]) & COMPLEMENT_TAKING_VERBS or lemma_candidates & COMPLEMENT_TAKING_VERBS)

    def _span_has_finite_verb(
        self,
        tokens: List[SyntaxToken],
        start: int,
        end: int,
        skip_index: Optional[int] = None,
    ) -> bool:
        finite_tags = {"VBD", "VBP", "VBZ", "MD"}
        for token in tokens:
            if token.index == skip_index or token.start < start or token.end > end:
                continue
            if token.pos == "AUX" or token.tag in finite_tags:
                return True
            verb_form = str(token.morph.get("VerbForm", "")).lower()
            if token.pos == "VERB" and verb_form == "fin":
                return True
            if re.search(r"(ed|s)$", token.text, flags=re.I) and token.text.lower() not in {"this", "his", "as"}:
                return True
        return False

    def _token_at_char(self, tokens: List[SyntaxToken], char_index: int) -> Optional[SyntaxToken]:
        for token in tokens:
            if token.start <= char_index < token.end:
                return token
        return None

    def _marker_start_before_token(self, sentence: str, token_start: int, marker: str) -> int:
        marker = (marker or "").lower()
        if not marker:
            return token_start
        window_start = max(0, token_start - 80)
        window = sentence[window_start : token_start + len(marker)]
        matches = list(re.finditer(rf"\b{re.escape(marker)}\b", window, flags=re.I))
        if not matches:
            return token_start
        return window_start + matches[-1].start()

    def _is_relative_clause(self, marker: str, clause_type: str, dep: str) -> bool:
        marker = (marker or "").lower()
        if marker == "that":
            return "relative_clause" in clause_type or dep in {"relcl", "acl"}
        if marker in ADVERBIAL_MARKERS and "relative_clause" not in clause_type and dep not in {"relcl", "acl"}:
            return False
        return marker in RELATIVE_MARKERS or "relative_clause" in clause_type or dep in {"relcl", "acl"}

    def _nearest_left_noun_phrase(self, sentence: str, tokens: List[SyntaxToken], marker_start: int) -> str:
        for token in reversed(tokens):
            if token.end > marker_start:
                continue
            if not self._is_noun_like(token):
                continue
            phrase = self._expanded_noun_phrase(sentence, tokens, token.index)
            if phrase:
                return phrase
        return self._nearest_left_text_phrase(sentence, marker_start)

    def _expanded_noun_phrase(self, sentence: str, tokens: List[SyntaxToken], head_index: int) -> str:
        if not 0 <= head_index < len(tokens):
            return ""
        head = tokens[head_index]
        if not self._is_noun_like(head):
            return ""
        allowed_left_deps = {"det", "amod", "compound", "poss", "nummod", "quantmod", "predet"}
        group = [head]
        group.extend(
            token
            for token in tokens
            if token.head == head.index and token.start <= head.start and token.dep in allowed_left_deps
        )
        group = sorted(group, key=lambda t: t.start)
        included = {token.index for token in group}
        while group:
            left_index = group[0].index - 1
            if left_index < 0 or left_index in included:
                break
            left = tokens[left_index]
            gap = sentence[left.end : group[0].start]
            if re.search(r"[,;:.!?()]", gap):
                break
            if not self._looks_like_np_left_modifier(left):
                break
            group.insert(0, left)
            included.add(left.index)
        return sentence[group[0].start : group[-1].end].strip(" ,;")

    def _nearest_left_text_phrase(self, sentence: str, marker_start: int) -> str:
        prefix = sentence[:marker_start].rstrip(" ,;")
        matches = list(re.finditer(r"[A-Za-z][A-Za-z'-]*", prefix))
        if not matches:
            return ""
        stop = {
            "although",
            "because",
            "if",
            "unless",
            "when",
            "while",
            "since",
            "before",
            "after",
            "that",
            "which",
            "who",
            "whom",
            "whose",
            "where",
        }
        selected = [matches[-1]]
        for match in reversed(matches[:-1]):
            gap = prefix[match.end() : selected[0].start()]
            word = match.group(0).lower()
            if "," in gap or ";" in gap or "." in gap or word in stop:
                break
            if len(selected) >= 4:
                break
            selected.insert(0, match)
        return prefix[selected[0].start() : selected[-1].end()].strip(" ,;")

    def _is_noun_like(self, token: SyntaxToken) -> bool:
        if token.pos in {"NOUN", "PROPN", "PRON"}:
            return True
        return token.tag in {"NN", "NNS", "NNP", "NNPS", "PRP"}

    def _looks_like_np_left_modifier(self, token: SyntaxToken) -> bool:
        lower = token.text.lower()
        if lower in {"a", "an", "the", "this", "that", "these", "those", "many", "much", "some", "any", "each", "every", "no", "new"}:
            return True
        if token.pos in {"DET", "ADJ", "NUM"}:
            return True
        if token.dep in {"det", "amod", "compound", "poss", "nummod", "quantmod", "predet"}:
            return True
        if token.pos in {"ADP", "AUX", "VERB", "CCONJ", "SCONJ", "PUNCT", "PART"}:
            return False
        if lower in set(ADVERBIAL_MARKERS) | RELATIVE_MARKERS | NOUN_CLAUSE_MARKERS | PREPOSITIONS:
            return False
        return not re.search(r"(ed|ing)$", lower)

    def _first_marker(self, text: str) -> str:
        match = re.search(r"\b(which|that|who|whom|whose|where|when|because|although|if|while|since|unless|whether|what|how|why)\b", text, flags=re.I)
        return match.group(1).lower() if match else ""

    def _left_np(self, sentence: str, tokens: List[SyntaxToken], root_index: int) -> str:
        left = [t for t in tokens[:root_index] if t.pos not in {"ADP", "PUNCT"}]
        if not left:
            return ""
        return sentence[left[0].start : left[-1].end].strip(" ,;")

    def _heuristic_main_clause(self, sentence: str) -> MainClause:
        parts = [p.strip() for p in re.split(r",|;", sentence) if p.strip()]
        main = parts[-1] if parts and starts_with_subordinator(parts[0]) else max(parts, key=len) if parts else sentence
        main_core = re.split(r"\b(because|although|though|if|unless|when|while|since|before|after)\b", main, maxsplit=1, flags=re.I)[0].strip()
        if len(main_core.split()) >= 3:
            main = main_core
        words = main.split()
        subject = " ".join(words[:2]) if len(words) >= 2 else (words[0] if words else "")
        predicate = words[2] if len(words) >= 3 else ""
        complement = " ".join(words[3:]) if len(words) >= 4 else ""
        return MainClause(
            subject=subject,
            predicate=predicate,
            object_or_complement=complement,
            full_span=main,
            explanation_zh="模板规则推定的主干：先理解这一部分，再回看前后的从句和修饰。",
            confidence=0.48,
        )


def classify_clause(marker: str, dep: str) -> Tuple[str, str]:
    marker = marker.lower()
    if dep == "xcomp" or marker == "to":
        return "nonfinite_complement_clause", "非谓语补足结构"
    if marker in ADVERBIAL_MARKERS:
        return ADVERBIAL_MARKERS[marker]
    if dep in {"ccomp", "csubj"}:
        return "nominal_clause", "nominal complement clause"
    if marker in RELATIVE_MARKERS or dep in {"relcl", "acl"}:
        if marker == "which":
            return "nonrestrictive_or_restrictive_relative_clause", "定语从句"
        return "relative_clause", "定语从句"
    if marker in NOUN_CLAUSE_MARKERS or dep in {"ccomp", "csubj"}:
        return "nominal_clause", "名词性从句"
    if dep == "xcomp":
        return "nonfinite_complement_clause", "非谓语补足结构"
    return "clause", "从句/分句"


def clause_strategy(function: str) -> str:
    if "定语" in function:
        return "先抓它前面的名词，再把这个从句作为补充说明接回去。"
    if "状语" in function:
        return "可以先放到一边，读完主句后再理解它提供的背景、原因、条件或时间。"
    if "名词性" in function:
        return "把整个从句当作一个名词成分，判断它在句中作主语、宾语还是表语。"
    return "先判断它和主句的逻辑关系，再整合意思。"


def modifier_strategy(function: str) -> str:
    if "介词" in function:
        return "把介词短语作为一个整体阅读，再判断它修饰前面的名词还是动作。"
    if "名词短语" in function:
        return "先找到中心名词，再理解限定词和修饰语。"
    return "阅读时可先跳过修饰成分，抓住主干后再补回细节。"


def marker_clause_span(sentence: str, marker_start: int, marker: str = "") -> Tuple[int, int]:
    if marker_start == 0 and marker in ADVERBIAL_MARKERS:
        end = leading_adverbial_end(sentence)
        return (0, end)
    start = marker_start if marker else previous_boundary(sentence, marker_start)
    end = next_boundary(sentence, marker_start)
    return (start, end)


def leading_adverbial_end(sentence: str) -> int:
    comma_positions = [m.start() for m in re.finditer(",", sentence)]
    if not comma_positions:
        return len(sentence)
    relative_or_sub = RELATIVE_MARKERS | set(ADVERBIAL_MARKERS) | NOUN_CLAUSE_MARKERS
    aux_like = {"is", "are", "was", "were", "be", "been", "being", "has", "have", "had", "will", "would", "can", "could", "may", "might", "must", "should"}
    for comma in comma_positions:
        tail = sentence[comma + 1 :].strip()
        match = re.match(r"([A-Za-z]+)", tail)
        first = match.group(1).lower() if match else ""
        if first and first not in relative_or_sub and first not in aux_like and len(tail.split()) >= 3:
            return comma
    return comma_positions[-1]


def previous_boundary(sentence: str, index: int) -> int:
    candidates = [sentence.rfind(ch, 0, index) for ch in ",;:.!?()"]
    pos = max(candidates)
    return pos + 1 if pos >= 0 else 0


def next_boundary(sentence: str, index: int) -> int:
    positions = [sentence.find(ch, index + 1) for ch in ",;:.!?()"]
    positions = [p for p in positions if p >= 0]
    return min(positions) if positions else len(sentence)


def descendants(tokens: List[SyntaxToken], root_index: int) -> List[SyntaxToken]:
    out: List[SyntaxToken] = []
    pending = [root_index]
    seen = {root_index}
    while pending:
        head = pending.pop()
        for token in tokens:
            if token.index in seen:
                continue
            if token.head == head:
                out.append(token)
                pending.append(token.index)
                seen.add(token.index)
    return out


def overlaps_any(span: Tuple[int, int], spans: List[Tuple[int, int]]) -> bool:
    for start, end in spans:
        if span[0] < end and span[1] > start:
            return True
    return False


def span_overlap_ratio(left: Tuple[int, int], right: Tuple[int, int]) -> float:
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    if intersection <= 0:
        return 0.0
    return intersection / max(1, min(left[1] - left[0], right[1] - right[0]))


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def starts_with_subordinator(text: str) -> bool:
    first = text.strip().split(" ", 1)[0].lower() if text.strip() else ""
    return first in set(ADVERBIAL_MARKERS) | RELATIVE_MARKERS | NOUN_CLAUSE_MARKERS


def dedupe_modifiers(modifiers: List[ModifierInfo]) -> List[ModifierInfo]:
    grouped: Dict[str, ModifierInfo] = {}
    for modifier in modifiers:
        key = normalize_space(modifier.span).lower()
        current = grouped.get(key)
        if current is None or modifier_priority(modifier) > modifier_priority(current):
            grouped[key] = modifier
    return list(grouped.values())


def trim_prepositional_modifier_span(sentence: str, span: Tuple[int, int]) -> Tuple[int, int]:
    text = sentence[span[0] : span[1]]
    stripped = text.strip(" ,;")
    if not stripped:
        return span
    leading_offset = text.find(stripped)
    start = span[0] + max(leading_offset, 0)
    words = list(re.finditer(r"[A-Za-z][A-Za-z'.-]*", stripped))
    if len(words) < 3:
        return (start, start + len(stripped))
    first = words[0].group(0).lower()
    if first not in PREPOSITIONS:
        return (start, start + len(stripped))
    for word in words[2:]:
        lower = word.group(0).lower()
        if lower in PREPOSITIONS:
            return (start, start + word.start())
    return (start, start + len(stripped))


def filter_modifiers(modifiers: List[ModifierInfo]) -> List[ModifierInfo]:
    candidates = [modifier for modifier in modifiers if is_high_value_modifier(modifier)]
    candidates.sort(key=lambda m: (-modifier_priority(m), -(m.end - m.start), m.start))

    selected: List[ModifierInfo] = []
    for modifier in candidates:
        if any(is_redundant_modifier(modifier, kept) for kept in selected):
            continue
        selected.append(modifier)

    selected.sort(key=lambda m: (m.start, m.end))
    return selected


def is_high_value_modifier(modifier: ModifierInfo) -> bool:
    text = normalize_space(modifier.span)
    if not text or not re.search(r"[A-Za-z0-9]", text):
        return False
    if text.lower().startswith("as "):
        # "such as ..." is handled as a grammar/example phrase, not as a generic modifier.
        return False
    if modifier.type in CORE_ONLY_MODIFIER_TYPES:
        return False
    if modifier.type not in STRUCTURAL_MODIFIER_TYPES:
        return len(text.split()) >= 2
    if modifier.type in {"prepositional_phrase", "time_prepositional_phrase"}:
        return len(text.split()) >= 2
    if modifier.type == "adverb_modifier":
        return len(text) > 1
    return True


def is_redundant_modifier(candidate: ModifierInfo, kept: ModifierInfo) -> bool:
    same_span = candidate.start == kept.start and candidate.end == kept.end
    if same_span:
        return modifier_priority(candidate) <= modifier_priority(kept)
    candidate_inside_kept = kept.start <= candidate.start and candidate.end <= kept.end
    kept_inside_candidate = candidate.start <= kept.start and kept.end <= candidate.end
    if candidate_inside_kept and modifier_priority(candidate) <= modifier_priority(kept):
        return True
    if kept_inside_candidate and modifier_priority(candidate) < modifier_priority(kept):
        return True
    return False


def modifier_priority(modifier: ModifierInfo) -> int:
    return MODIFIER_PRIORITY.get(modifier.type, 40)


def dedupe_clauses(clauses: List[ClauseInfo]) -> List[ClauseInfo]:
    grouped: Dict[Tuple[int, str, str], ClauseInfo] = {}
    for clause in clauses:
        key = (clause.start, clause.marker.lower(), clause.type)
        current = grouped.get(key)
        if current is None or (clause.end - clause.start) > (current.end - current.start):
            grouped[key] = clause

    out: List[ClauseInfo] = []
    seen: set[Tuple[str, str, int, int]] = set()
    for clause in sorted(grouped.values(), key=lambda c: (c.start, -(c.end - c.start))):
        key = (normalize_space(clause.span).lower(), clause.type, clause.start, clause.end)
        if key in seen:
            continue
        seen.add(key)
        out.append(clause)
    return out
