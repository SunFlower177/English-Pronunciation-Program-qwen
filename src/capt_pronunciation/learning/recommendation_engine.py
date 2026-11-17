"""Rule-based cross-mode recommendation engine."""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Tuple

from .schemas import LearningIssue, Recommendation


class RecommendationEngine:
    """Generate a small number of high-value cross-mode follow-ups.

    The first version is deliberately rule-based for transparency. The rules
    operate on normalized skill tags, so every mode can participate without
    importing each other directly.
    """

    def recommend(self, user_id: str, recent_issues: Iterable[LearningIssue]) -> List[Recommendation]:
        candidates: Dict[Tuple[str, str], Recommendation] = {}
        for issue in recent_issues:
            for rec in self._recommend_for_issue(issue):
                key = (rec.target_mode, rec.title)
                if key in candidates:
                    existing = candidates[key]
                    existing.priority = max(existing.priority, rec.priority)
                    existing.source_issue_ids = sorted(set(existing.source_issue_ids + rec.source_issue_ids))
                    existing.skill_tags = sorted(set(existing.skill_tags + rec.skill_tags))
                else:
                    candidates[key] = rec
        ordered = sorted(candidates.values(), key=lambda x: x.priority, reverse=True)
        return ordered[:3]

    def _recommend_for_issue(self, issue: LearningIssue) -> List[Recommendation]:
        tags = set(issue.skill_tags)
        recs: List[Recommendation] = []
        base_priority = severity_value(issue.severity) + issue.confidence

        if "listening.function_words" in tags or "grammar.auxiliary_verb" in tags:
            recs.append(make_rec(issue, "B", "跟读弱读功能词", "把漏听的功能词放回真实语流中模仿。", base_priority + 0.50, ["speaking.weak_form"]))
            recs.append(make_rec(issue, "C", "分析相关语法结构", "先看懂助动词链或从句结构，再回到听力复听。", base_priority + 0.35, ["grammar.structure"]))

        if any(tag.startswith("pronunciation.") for tag in tags) or any(tag.startswith("phoneme.") for tag in tags):
            recs.append(make_rec(issue, "D", "做最小对立听辨", "把发音问题反向转成听辨练习，确认耳朵能分出来。", base_priority + 0.45, ["listening.minimal_pair"]))
            recs.append(make_rec(issue, "A", "朗读目标音素句", "用文本朗读模式检查目标音素是否稳定。", base_priority + 0.25, ["speaking.phoneme"]))

        if "speaking.linking" in tags or "listening.linking" in tags:
            recs.append(make_rec(issue, "B", "跟读连读片段", "参考录音能直接提供自然连读目标。", base_priority + 0.50, ["speaking.linking"]))
            recs.append(make_rec(issue, "D", "连读听写复听", "先听出词边界，再做跟读。", base_priority + 0.30, ["listening.linking"]))

        if "reading.relative_clause" in tags or "grammar.relative_clause" in tags or "listening.clause_chunking" in tags:
            recs.append(make_rec(issue, "C", "长难句分块分析", "用主干和从句分块处理复杂结构。", base_priority + 0.55, ["reading.syntax"]))
            recs.append(make_rec(issue, "E", "用同类结构造句", "把读懂的结构迁移到写作。", base_priority + 0.25, ["writing.complex_sentence"]))

        if any(tag.startswith("writing.") for tag in tags):
            recs.append(make_rec(issue, "E", "修改当前作文段落", "先处理最影响分数的写作问题。", base_priority + 0.45, ["writing.revision"]))
            if "writing.grammar" in tags or "grammar.sentence_boundary" in tags:
                recs.append(make_rec(issue, "C", "拆解问题句主干", "先看出句子骨架，再修改语法。", base_priority + 0.35, ["reading.syntax"]))

        if not recs:
            recs.append(make_rec(issue, issue.source_mode, "复习当前问题", "先在原模块完成一次针对性复练。", base_priority, issue.skill_tags))
        return recs


def make_rec(
    issue: LearningIssue,
    target_mode: str,
    title: str,
    rationale: str,
    priority: float,
    extra_tags: List[str],
) -> Recommendation:
    rec_id = stable_id("REC", issue.issue_id, target_mode, title)
    return Recommendation(
        recommendation_id=rec_id,
        target_mode=target_mode,  # type: ignore[arg-type]
        title=title,
        rationale=rationale,
        priority=round(priority, 3),
        source_issue_ids=[issue.issue_id],
        skill_tags=sorted(set(issue.skill_tags + extra_tags)),
    )


def stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def severity_value(severity: str) -> float:
    return {"high": 1.0, "medium": 0.7, "low": 0.4, "info": 0.2}.get(severity, 0.5)
