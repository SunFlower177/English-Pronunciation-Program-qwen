"""Cross-mode orchestration for evidence storage and follow-up practice."""

from __future__ import annotations

from typing import Iterable, List

from ..contracts import CaptSettings
from .evidence_hub import EvidenceHub
from .practice_generator import PracticeGenerator
from .recommendation_engine import RecommendationEngine
from .schemas import LearningIssue, LearningResult


class LearningOrchestrator:
    def __init__(self, settings: CaptSettings):
        self.hub = EvidenceHub(settings)
        self.recommender = RecommendationEngine()
        self.practice_generator = PracticeGenerator()

    def process(
        self,
        user_id: str,
        source_mode: str,
        source_task: str,
        issues: Iterable[LearningIssue],
    ) -> LearningResult:
        normalized: List[LearningIssue] = []
        for issue in issues:
            issue.user_id = user_id
            if not issue.source_mode:
                issue.source_mode = source_mode  # type: ignore[assignment]
            if not issue.source_task:
                issue.source_task = source_task
            normalized.append(issue)
        saved = self.hub.save_issues(normalized)
        profile_updates = self.hub.update_profile(user_id, saved)
        recommendations = self.recommender.recommend(user_id, saved)
        practice_items = self.practice_generator.create(
            recommendations,
            {issue.issue_id: issue for issue in saved},
        )
        return LearningResult(
            user_id=user_id,
            issues=saved,
            profile_updates=profile_updates,
            recommendations=recommendations,
            practice_items=practice_items,
        )
