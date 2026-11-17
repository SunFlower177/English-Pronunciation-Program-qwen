"""Unified cross-mode learning schemas.

The pronunciation, listening, reading, and writing modules keep their own
domain-specific evidence. This file defines the small common layer used for
linking modules and recommending follow-up practice.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from ..contracts import BaseModel, Field


LearningMode = Literal["A", "B", "C", "D", "E", "reading", "imitation", "long_sentence", "listening", "writing"]


class LearningIssue(BaseModel):
    issue_id: str
    user_id: str = "local_user"
    source_mode: LearningMode
    source_task: str = ""
    issue_type: str
    skill_tags: List[str] = Field(default_factory=list)
    span_text: str = ""
    reference_text: str = ""
    user_output: str = ""
    audio_start: Optional[float] = None
    audio_end: Optional[float] = None
    severity: str = "medium"
    confidence: float = 0.5
    evidence: Dict[str, Any] = Field(default_factory=dict)
    recommended_followups: List[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    recommendation_id: str
    target_mode: LearningMode
    title: str
    rationale: str
    priority: float = 0.0
    source_issue_ids: List[str] = Field(default_factory=list)
    skill_tags: List[str] = Field(default_factory=list)


class PracticeItem(BaseModel):
    practice_id: str
    target_mode: LearningMode
    practice_type: str
    title: str
    instruction: str
    source_issue_ids: List[str] = Field(default_factory=list)
    skill_tags: List[str] = Field(default_factory=list)
    payload: Dict[str, Any] = Field(default_factory=dict)
    success_criteria: Dict[str, Any] = Field(default_factory=dict)


class MasteryRecord(BaseModel):
    skill_tag: str
    score: float = 0.65
    attempts: int = 0
    recent_errors: int = 0
    last_seen: str = ""


class LearningResult(BaseModel):
    user_id: str = "local_user"
    issues: List[LearningIssue] = Field(default_factory=list)
    profile_updates: Dict[str, MasteryRecord] = Field(default_factory=dict)
    recommendations: List[Recommendation] = Field(default_factory=list)
    practice_items: List[PracticeItem] = Field(default_factory=list)
