"""End-to-end Mode C long-sentence analyzer."""

from __future__ import annotations

import uuid
from typing import Optional

from ..contracts import CaptSettings
from .clause_extractor import ClauseExtractor
from .explainer import LongSentenceExplainer
from .grammar_detector import GrammarDetector
from .role_mapper import RoleMapper
from .schemas import LongSentenceEvidence, LongSentenceReport
from .syntax_parser import SyntaxParser
from .text_input import normalize_sentence


class LongSentenceAnalyzer:
    def __init__(self, settings: CaptSettings):
        self.settings = settings
        self.syntax_parser = SyntaxParser(settings)
        self.clause_extractor = ClauseExtractor()
        self.grammar_detector = GrammarDetector()
        self.role_mapper = RoleMapper()
        self.explainer = LongSentenceExplainer(settings)

    def analyze(
        self,
        sentence: str,
        level: str = "TOEFL",
        detail: str = "full",
        analysis_id: Optional[str] = None,
    ) -> LongSentenceReport:
        normalized = normalize_sentence(sentence)
        if not normalized:
            raise ValueError("sentence is required for Mode C long-sentence analysis")
        analysis_id = analysis_id or uuid.uuid4().hex[:12]

        parsed = self.syntax_parser.parse(normalized)
        extracted = self.clause_extractor.extract(normalized, parsed["tokens"], parsed)
        grammar_points = self.grammar_detector.detect(normalized, parsed["tokens"])
        semantic_roles = self.role_mapper.map_roles(
            normalized,
            extracted["main_clause"],
            extracted["clauses"],
            parsed["tokens"],
        )
        confidence = estimate_confidence(parsed["backend"], extracted["main_clause"].confidence, extracted["clauses"], grammar_points)
        evidence = LongSentenceEvidence(
            sentence=sentence,
            normalized_sentence=normalized,
            level=level,
            detail=detail,
            parser_backend=parsed["backend"],
            tokens=parsed["tokens"],
            sentence_spans=parsed["sentence_spans"],
            main_clause=extracted["main_clause"],
            clauses=extracted["clauses"],
            modifiers=extracted["modifiers"],
            grammar_points=grammar_points,
            semantic_roles=semantic_roles,
            chunks=extracted["chunks"],
            parse_trees=parsed["parse_trees"],
            confidence=confidence,
            warnings=parsed["warnings"],
            raw={
                "noun_chunks": parsed.get("noun_chunks", []),
                "constituency_spans": parsed.get("constituency_spans", []),
            },
        )
        explanation = self.explainer.explain(evidence)
        return LongSentenceReport(analysis_id=analysis_id, evidence=evidence, explanation=explanation)


def estimate_confidence(backend: str, main_confidence: float, clauses, grammar_points) -> float:
    base = 0.38 if "heuristic" in backend else 0.58
    if "spacy" in backend:
        base += 0.18
    if "stanza" in backend:
        base += 0.10
    if clauses:
        base += 0.04
    base = (base + main_confidence) / 2.0
    return round(max(0.0, min(0.95, base)), 3)
