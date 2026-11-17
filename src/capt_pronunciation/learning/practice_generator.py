"""Generate concrete practice payloads from cross-mode recommendations."""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List

from .schemas import LearningIssue, PracticeItem, Recommendation


class PracticeGenerator:
    def create(
        self,
        recommendations: Iterable[Recommendation],
        issues_by_id: Dict[str, LearningIssue],
    ) -> List[PracticeItem]:
        items: List[PracticeItem] = []
        for rec in recommendations:
            source_issues = [issues_by_id[i] for i in rec.source_issue_ids if i in issues_by_id]
            issue = source_issues[0] if source_issues else None
            items.append(self._create_one(rec, issue))
        return items

    def _create_one(self, rec: Recommendation, issue: LearningIssue | None) -> PracticeItem:
        span = issue.span_text if issue else ""
        reference = issue.reference_text if issue else ""
        payload = {
            "focus_span": span,
            "reference_text": reference,
            "user_output": issue.user_output if issue else "",
            "audio_start": issue.audio_start if issue else None,
            "audio_end": issue.audio_end if issue else None,
            "evidence": issue.evidence if issue else {},
        }
        instruction = rec.rationale
        practice_type = "review"
        criteria = {"complete": True}

        if rec.target_mode == "B":
            practice_type = "shadowing_snippet"
            instruction = f"先听参考片段，再模仿朗读：{span or reference[:80]}。重点关注连读、弱读和节奏。"
            criteria = {"mode": "B", "expected": "learner gap/rhythm closer to reference audio"}
        elif rec.target_mode == "A":
            practice_type = "read_aloud_drill"
            instruction = f"朗读目标句，先慢速再自然速度：{reference or span}。"
            criteria = {"mode": "A", "repeat": 3}
        elif rec.target_mode == "C":
            practice_type = "syntax_breakdown"
            instruction = f"把这个片段放入长难句模式，先找主干，再看修饰或从句：{span or reference[:120]}。"
            criteria = {"mode": "C", "identify_main_clause": True}
        elif rec.target_mode == "D":
            practice_type = "listening_discrimination"
            instruction = f"复听或听辨这个片段，写下你听到的词：{span or reference[:80]}。"
            criteria = {"mode": "D", "dictation_accuracy_min": 0.85}
        elif rec.target_mode == "E":
            practice_type = "writing_revision"
            instruction = f"根据反馈修改这个表达或段落：{span or reference[:120]}。修改后再次提交比较。"
            criteria = {"mode": "E", "revision_required": True}

        return PracticeItem(
            practice_id=stable_id("PRACTICE", rec.recommendation_id, span),
            target_mode=rec.target_mode,
            practice_type=practice_type,
            title=rec.title,
            instruction=instruction,
            source_issue_ids=rec.source_issue_ids,
            skill_tags=rec.skill_tags,
            payload=payload,
            success_criteria=criteria,
        )


def stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"
