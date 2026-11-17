"""Schemas for Mode C long-sentence analysis."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from ..contracts import BaseModel, Field


class TextSpan(BaseModel):
    text: str
    start: int = 0
    end: int = 0
    label: str = ""


class SyntaxToken(BaseModel):
    index: int
    text: str
    lemma: str = ""
    pos: str = ""
    tag: str = ""
    dep: str = ""
    head: Optional[int] = None
    start: int = 0
    end: int = 0
    morph: Dict[str, Any] = Field(default_factory=dict)


class MainClause(BaseModel):
    subject: str = ""
    predicate: str = ""
    object_or_complement: str = ""
    full_span: str = ""
    explanation_zh: str = ""
    confidence: float = 0.0


class ClauseInfo(BaseModel):
    clause_id: str
    span: str
    type: str
    function_zh: str
    modifies: str = ""
    marker: str = ""
    start: int = 0
    end: int = 0
    reading_strategy: str = ""
    confidence: float = 0.0


class ModifierInfo(BaseModel):
    modifier_id: str
    span: str
    type: str
    function_zh: str
    modifies: str = ""
    start: int = 0
    end: int = 0
    reading_strategy: str = ""
    confidence: float = 0.0


class GrammarPoint(BaseModel):
    point_id: str
    span: str
    type: str
    explanation_zh: str
    start: int = 0
    end: int = 0
    confidence: float = 0.0


class SemanticRole(BaseModel):
    predicate: str
    arg0: str = ""
    arg1: str = ""
    argm: Dict[str, str] = Field(default_factory=dict)
    explanation_zh: str = ""
    confidence: float = 0.0


class TranslationChunk(BaseModel):
    chunk_id: str
    source: str
    role: str
    translation_zh: str = ""
    start: int = 0
    end: int = 0


class LongSentenceEvidence(BaseModel):
    sentence: str
    normalized_sentence: str
    level: str = "TOEFL"
    detail: str = "full"
    parser_backend: str = "heuristic"
    tokens: List[SyntaxToken] = Field(default_factory=list)
    sentence_spans: List[TextSpan] = Field(default_factory=list)
    main_clause: MainClause = Field(default_factory=MainClause)
    clauses: List[ClauseInfo] = Field(default_factory=list)
    modifiers: List[ModifierInfo] = Field(default_factory=list)
    grammar_points: List[GrammarPoint] = Field(default_factory=list)
    semantic_roles: List[SemanticRole] = Field(default_factory=list)
    chunks: List[TranslationChunk] = Field(default_factory=list)
    parse_trees: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    warnings: List[str] = Field(default_factory=list)
    raw: Dict[str, Any] = Field(default_factory=dict)


class LongSentenceExplanation(BaseModel):
    summary_zh: str = ""
    main_clause_steps: List[str] = Field(default_factory=list)
    modifier_explanations: List[str] = Field(default_factory=list)
    clause_explanations: List[str] = Field(default_factory=list)
    grammar_explanations: List[str] = Field(default_factory=list)
    step_by_step_explanation: List[str] = Field(default_factory=list)
    chunk_translation: List[TranslationChunk] = Field(default_factory=list)
    full_translation: str = ""
    reading_strategy: str = ""
    generator: str = "template"
    llm_raw: Optional[str] = None


class LongSentenceReport(BaseModel):
    analysis_id: str
    mode: Literal["long_sentence"] = "long_sentence"
    evidence: LongSentenceEvidence
    explanation: LongSentenceExplanation

