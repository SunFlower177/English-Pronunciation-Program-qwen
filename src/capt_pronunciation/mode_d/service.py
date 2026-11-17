"""Mode D listening coach: dictation, notes, and drills."""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..contracts import AlignmentResult, AsrResult, AudioInfo, CaptSettings, model_to_dict
from ..learning.qwen import qwen_json
from ..learning.schemas import LearningIssue, PracticeItem
from ..reference_alignment import FUNCTION_WORDS, normalize_word
from .schemas import (
    DictationAlignmentItem,
    ListeningEvidence,
    ListeningFeedback,
    ListeningReport,
    ListeningScores,
    NoteCoverageItem,
)


class ListeningAnalyzer:
    def __init__(self, settings: CaptSettings):
        self.settings = settings

    def analyze(
        self,
        reference_transcript: str,
        student_transcript: str = "",
        student_notes: str = "",
        audio: Optional[AudioInfo] = None,
        asr: Optional[AsrResult] = None,
        alignment: Optional[AlignmentResult] = None,
        analysis_id: Optional[str] = None,
    ) -> ListeningReport:
        analysis_id = analysis_id or uuid.uuid4().hex[:12]
        reference_transcript = clean_text(reference_transcript or (asr.text if asr else ""))
        if not reference_transcript:
            raise ValueError("Mode D requires a reference transcript or recoverable ASR transcript")
        student_transcript = clean_text(student_transcript)
        student_notes = clean_text(student_notes)
        transcript_source = "user" if reference_transcript else "asr"

        time_map = ref_time_map(alignment)
        dictation_alignment = align_dictation(reference_transcript, student_transcript, time_map)
        note_coverage = analyze_notes(reference_transcript, student_notes)
        scores = compute_listening_scores(dictation_alignment, note_coverage)
        evidence = ListeningEvidence(
            reference_transcript=reference_transcript,
            student_transcript=student_transcript,
            student_notes=student_notes,
            asr_text=asr.text if asr else "",
            transcript_source=transcript_source,  # type: ignore[arg-type]
            alignment=dictation_alignment,
            note_coverage=note_coverage,
            missing_key_points=[x.reference_point for x in note_coverage if not x.covered],
            scores=scores,
            raw={"alignment_source": alignment.source if alignment else "text_only"},
        )
        learning_issues = build_learning_issues(evidence)
        practice_items = build_listening_practice(evidence, learning_issues)
        feedback = self.explain(evidence, learning_issues, practice_items)
        return ListeningReport(
            analysis_id=analysis_id,
            audio=audio,
            asr=asr,
            evidence=evidence,
            feedback=feedback,
            learning_issues=learning_issues,
            practice_items=practice_items,
        )

    def explain(
        self,
        evidence: ListeningEvidence,
        learning_issues: List[LearningIssue],
        practice_items: List[PracticeItem],
    ) -> ListeningFeedback:
        template = template_feedback(evidence, learning_issues, practice_items)
        if not self.settings.use_qwen_text:
            return template
        payload = {
            "evidence": model_to_dict(evidence),
            "learning_issues": [model_to_dict(x) for x in learning_issues[:30]],
            "practice_items": [model_to_dict(x) for x in practice_items[:10]],
            "instructions": [
                "Only explain listed evidence and learning_issues.",
                "Do not invent transcript content or errors.",
                "Use Chinese as the main feedback language.",
                "Return JSON with summary_zh, dictation_feedback, note_feedback, training_plan.",
            ],
        }
        system = (
            "You are an evidence-grounded TOEFL/IELTS listening coach. "
            "You must only use the supplied JSON evidence. Return valid JSON only."
        )
        user = (
            "请根据 evidence JSON 生成中文听力反馈。不要添加 evidence 中不存在的错误。"
            "输出 JSON: {summary_zh, dictation_feedback, note_feedback, training_plan}.\n\n"
            f"evidence_json={json.dumps(payload, ensure_ascii=False)}"
        )
        parsed, raw = qwen_json(self.settings, system, user, max_tokens=self.settings.qwen_summary_max_tokens + 700)
        if not parsed:
            template.llm_raw = raw
            template.generator = "template+qwen_raw_unparsed" if raw else template.generator
            return template
        return ListeningFeedback(
            summary_zh=str(parsed.get("summary_zh") or template.summary_zh),
            dictation_feedback=list_of_strings(parsed.get("dictation_feedback")) or template.dictation_feedback,
            note_feedback=list_of_strings(parsed.get("note_feedback")) or template.note_feedback,
            training_plan=list_of_strings(parsed.get("training_plan")) or template.training_plan,
            generator="qwen_text_evidence_grounded",
            llm_raw=raw,
        )


def clean_text(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def words_with_raw(text: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for match in re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+(?:\.[0-9]+)?", text):
        raw = match.group(0)
        norm = normalize_word(raw)
        if norm:
            out.append((raw, norm))
    return out


def ref_time_map(alignment: Optional[AlignmentResult]) -> Dict[int, Tuple[Optional[float], Optional[float]]]:
    out: Dict[int, Tuple[Optional[float], Optional[float]]] = {}
    if alignment is None:
        return out
    for item in alignment.word_alignments:
        if item.ref_index is not None:
            out[item.ref_index] = (item.start, item.end)
    return out


def align_dictation(
    reference_transcript: str,
    student_transcript: str,
    time_map: Dict[int, Tuple[Optional[float], Optional[float]]],
) -> List[DictationAlignmentItem]:
    ref = words_with_raw(reference_transcript)
    hyp = words_with_raw(student_transcript)
    n, m = len(ref), len(hyp)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    back: List[List[Tuple[int, int, str]]] = [[(0, 0, "")] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = (i - 1, 0, "deletion")
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = (0, j - 1, "insertion")
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1][1] == hyp[j - 1][1]:
                sub_cost = 0.0
                op = "equal"
            else:
                ratio = difflib.SequenceMatcher(None, ref[i - 1][1], hyp[j - 1][1]).ratio()
                sub_cost = 0.45 if ratio >= 0.82 else 0.85 if ratio >= 0.68 else 1.0
                op = "substitution"
            candidates = [
                (dp[i - 1][j - 1] + sub_cost, i - 1, j - 1, op),
                (dp[i - 1][j] + 1.0, i - 1, j, "deletion"),
                (dp[i][j - 1] + 1.0, i, j - 1, "insertion"),
            ]
            best = min(candidates, key=lambda x: x[0])
            dp[i][j] = best[0]
            back[i][j] = (best[1], best[2], best[3])

    pairs: List[Tuple[Optional[int], Optional[int], str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        pi, pj, op = back[i][j]
        if op in {"equal", "substitution"}:
            pairs.append((i - 1, j - 1, op))
        elif op == "deletion":
            pairs.append((i - 1, None, op))
        elif op == "insertion":
            pairs.append((None, j - 1, op))
        i, j = pi, pj
    pairs.reverse()

    items: List[DictationAlignmentItem] = []
    for idx, (ri, hj, op) in enumerate(pairs, start=1):
        ref_raw = ref[ri][0] if ri is not None else ""
        hyp_raw = hyp[hj][0] if hj is not None else ""
        start, end = time_map.get(ri, (None, None)) if ri is not None else (None, None)
        err_type, severity, hint = classify_dictation_error(ref[ri][1] if ri is not None else "", hyp[hj][1] if hj is not None else "", op)
        items.append(
            DictationAlignmentItem(
                item_id=f"D{idx:04d}",
                ref_index=ri,
                ref_word=ref_raw,
                student_word=hyp_raw,
                operation=op,  # type: ignore[arg-type]
                error_type=err_type,
                severity=severity,
                confidence=confidence_for_alignment(op, ref[ri][1] if ri is not None else "", hyp[hj][1] if hj is not None else ""),
                start=start,
                end=end,
                explanation_hint=hint,
            )
        )
    return items


def classify_dictation_error(ref_word: str, hyp_word: str, op: str) -> Tuple[str, str, str]:
    if op == "equal":
        return "", "info", ""
    if op == "deletion":
        if ref_word in FUNCTION_WORDS:
            return "function_word_omission", "medium", "功能词在自然语流里常被弱读，容易漏听。"
        return "content_word_omission", "high", "关键词漏听会直接影响主旨或细节理解。"
    if op == "insertion":
        return "extra_word", "low", "听写中多写了原文没有的词，可能是把连读或噪音误分成单词。"
    ratio = difflib.SequenceMatcher(None, ref_word, hyp_word).ratio()
    if ratio >= 0.82:
        return "spelling_or_word_form", "low", "词形或拼写接近，但书写形式不准确。"
    if is_morphology_pair(ref_word, hyp_word):
        return "morphology_confusion", "medium", "单复数、时态或派生词尾没有听清。"
    if ref_word in FUNCTION_WORDS or hyp_word in FUNCTION_WORDS:
        return "function_word_confusion", "medium", "功能词或弱读形式发生错听。"
    if may_be_sound_alike(ref_word, hyp_word):
        return "sound_alike_confusion", "medium", "两个词形接近，可能是音近词错听。"
    return "content_word_misheard", "high", "内容词错听，通常会影响细节理解。"


def confidence_for_alignment(op: str, ref_word: str, hyp_word: str) -> float:
    if op == "equal":
        return 1.0
    if op in {"deletion", "insertion"}:
        return 0.85 if ref_word not in FUNCTION_WORDS else 0.70
    ratio = difflib.SequenceMatcher(None, ref_word, hyp_word).ratio()
    return round(max(0.55, min(0.95, 1.0 - ratio * 0.35)), 2)


def is_morphology_pair(a: str, b: str) -> bool:
    if not a or not b:
        return False
    stems = {
        a.rstrip("s").rstrip("ed").rstrip("ing"),
        b.rstrip("s").rstrip("ed").rstrip("ing"),
    }
    return len(stems) == 1 or a.rstrip("s") == b.rstrip("s")


def may_be_sound_alike(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return a[0] == b[0] or a[-1:] == b[-1:] or difflib.SequenceMatcher(None, a, b).ratio() > 0.55


def analyze_notes(transcript: str, notes: str) -> List[NoteCoverageItem]:
    points = extract_key_points(transcript)
    note_lines = split_note_lines(notes)
    out: List[NoteCoverageItem] = []
    for idx, point in enumerate(points, start=1):
        best_line = ""
        best = 0.0
        for line in note_lines or [notes]:
            score = text_overlap(point, line)
            if score > best:
                best = score
                best_line = line
        out.append(
            NoteCoverageItem(
                point_id=f"P{idx}",
                reference_point=point,
                covered=best >= 0.32,
                coverage_score=round(best, 3),
                matched_note=best_line,
            )
        )
    return out


def compute_listening_scores(
    alignment: List[DictationAlignmentItem],
    note_coverage: List[NoteCoverageItem],
) -> ListeningScores:
    ref_items = [x for x in alignment if x.operation != "insertion"]
    if ref_items:
        penalty = 0.0
        for item in alignment:
            if item.operation == "equal":
                continue
            penalty += {"high": 1.0, "medium": 0.65, "low": 0.35}.get(item.severity, 0.4) * item.confidence
        dictation = max(0.0, 100.0 * (1.0 - penalty / max(1, len(ref_items))))
    else:
        dictation = 0.0
    note_score = 0.0
    if note_coverage:
        note_score = 100.0 * sum(x.coverage_score for x in note_coverage) / len(note_coverage)
    parts = [dictation, note_score]
    return ListeningScores(
        dictation_accuracy=round(dictation, 1),
        note_coverage=round(note_score, 1),
        overall=round(sum(parts) / max(1, len(parts)), 1),
    )


def build_learning_issues(evidence: ListeningEvidence) -> List[LearningIssue]:
    issues: List[LearningIssue] = []
    for item in evidence.alignment:
        if item.operation == "equal":
            continue
        tags = tags_for_dictation_item(item)
        issues.append(
            LearningIssue(
                issue_id=f"LD_{item.item_id}",
                source_mode="D",
                source_task="listening_dictation",
                issue_type=item.error_type,
                skill_tags=tags,
                span_text=item.ref_word or item.student_word,
                reference_text=item.ref_word,
                user_output=item.student_word,
                audio_start=item.start,
                audio_end=item.end,
                severity=item.severity,
                confidence=item.confidence,
                evidence=model_to_dict(item),
            )
        )
    for point in evidence.note_coverage:
        if point.covered:
            continue
        issues.append(
            LearningIssue(
                issue_id=f"LN_{point.point_id}",
                source_mode="D",
                source_task="note_analysis",
                issue_type="missing_key_point",
                skill_tags=["listening.note_coverage", "listening.main_idea"],
                span_text=point.reference_point,
                reference_text=point.reference_point,
                user_output=point.matched_note,
                severity="medium",
                confidence=round(1.0 - point.coverage_score, 2),
                evidence=model_to_dict(point),
            )
        )
    return issues[:60]


def tags_for_dictation_item(item: DictationAlignmentItem) -> List[str]:
    tags = ["listening.dictation"]
    if "function_word" in item.error_type:
        tags.extend(["listening.function_words", "speaking.weak_form"])
    if "content_word" in item.error_type:
        tags.extend(["listening.content_words", "vocab.recognition"])
    if "morphology" in item.error_type or "word_form" in item.error_type:
        tags.extend(["grammar.morphology", "writing.word_form"])
    if "sound_alike" in item.error_type:
        tags.extend(["listening.minimal_pair", "pronunciation.contrast"])
    return sorted(set(tags))


def build_listening_practice(evidence: ListeningEvidence, issues: List[LearningIssue]) -> List[PracticeItem]:
    items: List[PracticeItem] = []
    for issue in issues[:8]:
        practice_type = "replay_dictation"
        instruction = f"回放并重写这个片段：{issue.span_text}"
        if "speaking.weak_form" in issue.skill_tags:
            practice_type = "weak_form_loop"
            instruction = f"循环听 3 次，再跟读弱读/连读片段：{issue.span_text}"
        elif "listening.minimal_pair" in issue.skill_tags:
            practice_type = "minimal_pair_discrimination"
            instruction = f"比较这个词和相近读音词，写下你听到的目标词：{issue.span_text}"
        elif issue.issue_type == "missing_key_point":
            practice_type = "note_reconstruction"
            instruction = "只听一次，重新写下这个关键点，再和原文对照。"
        items.append(
            PracticeItem(
                practice_id=stable_id("DPR", issue.issue_id, practice_type),
                target_mode="D",
                practice_type=practice_type,
                title=instruction[:48],
                instruction=instruction,
                source_issue_ids=[issue.issue_id],
                skill_tags=issue.skill_tags,
                payload={
                    "reference_transcript": evidence.reference_transcript,
                    "focus_span": issue.span_text,
                    "audio_start": issue.audio_start,
                    "audio_end": issue.audio_end,
                },
                success_criteria={"retry_until": "dictation or note coverage improves"},
            )
        )
    return items


def template_feedback(
    evidence: ListeningEvidence,
    learning_issues: List[LearningIssue],
    practice_items: List[PracticeItem],
) -> ListeningFeedback:
    counts: Dict[str, int] = {}
    for issue in learning_issues:
        counts[issue.issue_type] = counts.get(issue.issue_type, 0) + 1
    summary = (
        f"本次听力整体分数 {evidence.scores.overall:.1f}。"
        f"听写准确率 {evidence.scores.dictation_accuracy:.1f}，笔记覆盖 {evidence.scores.note_coverage:.1f}。"
    )
    if counts:
        top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
        summary += "主要问题：" + "、".join(f"{k} {v} 个" for k, v in top) + "。"

    dictation_feedback = []
    for item in evidence.alignment:
        if item.operation != "equal":
            dictation_feedback.append(
                f"{item.ref_word or item.student_word}: {item.explanation_hint} "
                f"参考={item.ref_word or '[无]'}，你的答案={item.student_word or '[漏写]'}。"
            )
    if not dictation_feedback:
        dictation_feedback.append("听写部分没有发现明显逐词错误。")

    note_feedback = [
        f"{p.point_id}: {'已覆盖' if p.covered else '缺失'} - {p.reference_point}"
        for p in evidence.note_coverage
    ] or ["没有输入笔记；建议听完后用关键词记录主旨、原因、转折和例子。"]

    training_plan = [p.instruction for p in practice_items[:5]]
    if not training_plan:
        training_plan = ["复听原文，先抓主旨，再做逐句听写。"]
    return ListeningFeedback(
        summary_zh=summary,
        dictation_feedback=dictation_feedback[:20],
        note_feedback=note_feedback[:10],
        training_plan=training_plan,
        generator="template",
    )


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def split_note_lines(text: str) -> List[str]:
    return [x.strip("-* \t") for x in re.split(r"[\n;]+", text) if x.strip("-* \t")]


def extract_key_points(transcript: str) -> List[str]:
    sentences = split_sentences(transcript)
    points: List[str] = []
    for sentence in sentences[:8]:
        words = [w for _, w in words_with_raw(sentence)]
        content = [w for w in words if w not in FUNCTION_WORDS and len(w) > 3]
        if len(content) >= 2 or len(sentence.split()) >= 6:
            points.append(sentence)
    return points or sentences[:5]


def text_overlap(a: str, b: str) -> float:
    aw = {w for _, w in words_with_raw(a) if w not in FUNCTION_WORDS}
    bw = {w for _, w in words_with_raw(b) if w not in FUNCTION_WORDS}
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / max(1, len(aw | bw))


def list_of_strings(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(x) for x in value if str(x).strip()]


def stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"
