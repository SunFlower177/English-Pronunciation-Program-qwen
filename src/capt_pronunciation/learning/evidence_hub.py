"""Lightweight local storage for cross-mode learning evidence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ..contracts import CaptSettings, model_to_dict
from .schemas import LearningIssue, MasteryRecord


class EvidenceHub:
    def __init__(self, settings: CaptSettings):
        self.root = Path(settings.work_dir) / "learning_hub"
        self.root.mkdir(parents=True, exist_ok=True)
        self.issues_path = self.root / "learning_issues.jsonl"
        self.profile_path = self.root / "learner_profile.json"

    def save_issues(self, issues: Iterable[LearningIssue]) -> List[LearningIssue]:
        saved = list(issues)
        if not saved:
            return []
        with self.issues_path.open("a", encoding="utf-8") as f:
            for issue in saved:
                data = model_to_dict(issue)
                data["saved_at"] = now_iso()
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        return saved

    def load_profile(self, user_id: str) -> Dict[str, MasteryRecord]:
        if not self.profile_path.exists():
            return {}
        try:
            data = json.loads(self.profile_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        raw = data.get(user_id, {})
        profile: Dict[str, MasteryRecord] = {}
        for tag, item in raw.items():
            try:
                profile[tag] = MasteryRecord(**item)
            except Exception:
                continue
        return profile

    def save_profile(self, user_id: str, profile: Dict[str, MasteryRecord]) -> None:
        data: Dict[str, Any] = {}
        if self.profile_path.exists():
            try:
                data = json.loads(self.profile_path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        data[user_id] = {tag: model_to_dict(record) for tag, record in profile.items()}
        self.profile_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def update_profile(self, user_id: str, issues: Iterable[LearningIssue]) -> Dict[str, MasteryRecord]:
        profile = self.load_profile(user_id)
        updated: Dict[str, MasteryRecord] = {}
        for issue in issues:
            for tag in issue.skill_tags:
                record = profile.get(tag) or MasteryRecord(skill_tag=tag)
                record.attempts += 1
                record.recent_errors += 1
                record.last_seen = now_iso()
                penalty = severity_penalty(issue.severity) * max(0.35, min(1.0, issue.confidence))
                record.score = round(max(0.0, min(1.0, record.score - penalty)), 3)
                profile[tag] = record
                updated[tag] = record
        self.save_profile(user_id, profile)
        return updated


def severity_penalty(severity: str) -> float:
    if severity == "high":
        return 0.10
    if severity == "medium":
        return 0.06
    if severity == "low":
        return 0.03
    return 0.015


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
