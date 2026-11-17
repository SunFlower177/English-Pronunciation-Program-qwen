"""Schemas for Mode D listening training."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from ..contracts import AsrResult, AudioInfo, BaseModel, Field
from ..learning.schemas import LearningIssue, LearningResult, PracticeItem


class DictationAlignmentItem(BaseModel):
    item_id: str
    ref_index: Optional[int] = None
    ref_word: str = ""
    student_word: str = ""
    operation: Literal["equal", "deletion", "insertion", "substitution"] = "equal"
    error_type: str = ""
    severity: str = "info"
    confidence: float = 1.0
    start: Optional[float] = None
    end: Optional[float] = None
    explanation_hint: str = ""


class NoteCoverageItem(BaseModel):
    point_id: str
    reference_point: str
    covered: bool = False
    coverage_score: float = 0.0
    matched_note: str = ""


class ListeningScores(BaseModel):
    dictation_accuracy: float = 0.0
    note_coverage: float = 0.0
    overall: float = 0.0


class ListeningEvidence(BaseModel):
    reference_transcript: str
    student_transcript: str = ""
    student_notes: str = ""
    asr_text: str = ""
    transcript_source: Literal["user", "asr"] = "user"
    alignment: List[DictationAlignmentItem] = Field(default_factory=list)
    note_coverage: List[NoteCoverageItem] = Field(default_factory=list)
    missing_key_points: List[str] = Field(default_factory=list)
    scores: ListeningScores = Field(default_factory=ListeningScores)
    raw: Dict[str, Any] = Field(default_factory=dict)


class ListeningFeedback(BaseModel):
    summary_zh: str = ""
    dictation_feedback: List[str] = Field(default_factory=list)
    note_feedback: List[str] = Field(default_factory=list)
    training_plan: List[str] = Field(default_factory=list)
    generator: str = "template"
    llm_raw: Optional[str] = None


class ListeningReport(BaseModel):
    analysis_id: str
    mode: Literal["listening"] = "listening"
    audio: Optional[AudioInfo] = None
    asr: Optional[AsrResult] = None
    evidence: ListeningEvidence
    feedback: ListeningFeedback
    learning_issues: List[LearningIssue] = Field(default_factory=list)
    practice_items: List[PracticeItem] = Field(default_factory=list)
    learning: Optional[LearningResult] = None
