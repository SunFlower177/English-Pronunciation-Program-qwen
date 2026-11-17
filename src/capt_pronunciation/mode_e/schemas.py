"""Schemas for Mode E writing training."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from ..contracts import BaseModel, Field
from ..learning.schemas import LearningIssue, LearningResult, PracticeItem


class PromptAnalysis(BaseModel):
    exam_type: str = "general"
    task_type: str = "opinion"
    topic_keywords: List[str] = Field(default_factory=list)
    required_parts: List[str] = Field(default_factory=list)
    expected_min_words: int = 250


class EssayStats(BaseModel):
    word_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    avg_sentence_length: float = 0.0
    lexical_diversity: float = 0.0


class StructureIssue(BaseModel):
    issue_id: str
    type: str
    paragraph: Optional[int] = None
    evidence: str = ""
    suggestion_hint: str = ""
    severity: str = "medium"
    confidence: float = 0.7


class LocalWritingError(BaseModel):
    error_id: str
    span: str
    type: str
    correction: str = ""
    explanation_hint: str = ""
    sentence_index: Optional[int] = None
    severity: str = "medium"
    confidence: float = 0.7


class LexicalIssue(BaseModel):
    issue_id: str
    span: str
    type: str
    suggestion: str = ""
    explanation_hint: str = ""
    severity: str = "low"
    confidence: float = 0.65


class RubricScores(BaseModel):
    task_response: float = 0.0
    coherence_cohesion: float = 0.0
    lexical_resource: float = 0.0
    grammar_accuracy: float = 0.0
    overall: float = 0.0
    confidence: float = 0.7


class RevisionComparison(BaseModel):
    has_previous: bool = False
    previous_word_count: int = 0
    current_word_count: int = 0
    previous_error_count: int = 0
    current_error_count: int = 0
    improvement_summary: str = ""
    improved_dimensions: List[str] = Field(default_factory=list)
    remaining_priorities: List[str] = Field(default_factory=list)


class WritingEvidence(BaseModel):
    prompt: str
    essay: str
    previous_essay: str = ""
    prompt_analysis: PromptAnalysis
    stats: EssayStats
    thesis: str = ""
    topic_sentences: List[str] = Field(default_factory=list)
    structure_issues: List[StructureIssue] = Field(default_factory=list)
    local_errors: List[LocalWritingError] = Field(default_factory=list)
    lexical_issues: List[LexicalIssue] = Field(default_factory=list)
    rubric_scores: RubricScores = Field(default_factory=RubricScores)
    revision_comparison: RevisionComparison = Field(default_factory=RevisionComparison)
    revision_tasks: List[str] = Field(default_factory=list)
    raw: Dict[str, Any] = Field(default_factory=dict)


class WritingFeedback(BaseModel):
    summary_zh: str = ""
    rubric_feedback: List[str] = Field(default_factory=list)
    global_feedback: List[str] = Field(default_factory=list)
    local_feedback: List[str] = Field(default_factory=list)
    lexical_feedback: List[str] = Field(default_factory=list)
    revision_tasks: List[str] = Field(default_factory=list)
    generator: str = "template"
    llm_raw: Optional[str] = None


class WritingReport(BaseModel):
    analysis_id: str
    mode: Literal["writing"] = "writing"
    evidence: WritingEvidence
    feedback: WritingFeedback
    learning_issues: List[LearningIssue] = Field(default_factory=list)
    practice_items: List[PracticeItem] = Field(default_factory=list)
    learning: Optional[LearningResult] = None
