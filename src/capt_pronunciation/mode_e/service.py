"""Mode E writing coach: rubric scoring, errors, and revision loop."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..contracts import CaptSettings, model_to_dict
from ..learning.qwen import qwen_json
from ..learning.schemas import LearningIssue, PracticeItem
from ..reference_alignment import FUNCTION_WORDS, normalize_word
from .schemas import (
    EssayStats,
    LexicalIssue,
    LocalWritingError,
    PromptAnalysis,
    RevisionComparison,
    RubricScores,
    StructureIssue,
    WritingEvidence,
    WritingFeedback,
    WritingReport,
)


class WritingAnalyzer:
    def __init__(self, settings: CaptSettings):
        self.settings = settings

    def analyze(
        self,
        prompt: str,
        essay: str,
        exam_type: str = "IELTS_Task_2",
        previous_essay: str = "",
        analysis_id: Optional[str] = None,
    ) -> WritingReport:
        analysis_id = analysis_id or uuid.uuid4().hex[:12]
        prompt = clean_text(prompt)
        essay = normalize_paragraphs(essay)
        previous_essay = normalize_paragraphs(previous_essay)
        if not prompt:
            raise ValueError("Mode E requires a writing prompt")
        if not essay:
            raise ValueError("Mode E requires an essay")

        prompt_analysis = analyze_prompt(prompt, exam_type)
        stats = essay_stats(essay)
        thesis = detect_thesis(essay)
        topic_sentences = detect_topic_sentences(essay)
        structure_issues = analyze_structure(prompt_analysis, essay, thesis, topic_sentences, stats)
        local_errors = detect_local_errors(essay)
        lexical_issues = detect_lexical_issues(essay)
        scores = score_rubric(prompt_analysis, stats, thesis, structure_issues, local_errors, lexical_issues)
        revision_tasks = build_revision_tasks(structure_issues, local_errors, lexical_issues, scores, prompt_analysis.exam_type)
        revision = compare_revision(previous_essay, essay, local_errors, lexical_issues, structure_issues)
        evidence = WritingEvidence(
            prompt=prompt,
            essay=essay,
            previous_essay=previous_essay,
            prompt_analysis=prompt_analysis,
            stats=stats,
            thesis=thesis,
            topic_sentences=topic_sentences,
            structure_issues=structure_issues,
            local_errors=local_errors,
            lexical_issues=lexical_issues,
            rubric_scores=scores,
            revision_comparison=revision,
            revision_tasks=revision_tasks,
            raw={"paragraphs": split_paragraphs(essay), "sentences": split_sentences(essay)},
        )
        learning_issues = build_learning_issues(evidence)
        practice_items = build_writing_practice(evidence, learning_issues)
        feedback = self.explain(evidence, learning_issues, practice_items)
        return WritingReport(
            analysis_id=analysis_id,
            evidence=evidence,
            feedback=feedback,
            learning_issues=learning_issues,
            practice_items=practice_items,
        )

    def explain(
        self,
        evidence: WritingEvidence,
        learning_issues: List[LearningIssue],
        practice_items: List[PracticeItem],
    ) -> WritingFeedback:
        template = template_feedback(evidence, learning_issues, practice_items)
        if not self.settings.use_qwen_text:
            return template
        payload = {
            "evidence": model_to_dict(evidence),
            "learning_issues": [model_to_dict(x) for x in learning_issues[:40]],
            "practice_items": [model_to_dict(x) for x in practice_items[:10]],
            "instructions": [
                "Use exam_type to choose the rubric: IELTS_Task_2 uses IELTS 0-9 bands; TOEFL tasks use TOEFL-style 0-5 diagnostic scoring.",
                "Candidate errors in evidence are detector proposals. Confirm correct ones, reject false positives, and add obvious high-confidence errors from the essay.",
                "Do not rewrite the full essay.",
                "Return JSON with rubric_scores, summary_zh, rubric_feedback, global_feedback, local_feedback, lexical_feedback, revision_tasks, confirmed_errors, rejected_candidate_errors, added_errors.",
            ],
        }
        system = (
            "You are an evidence-grounded IELTS/TOEFL writing coach. "
            "Use the essay and supplied evidence. Apply the correct exam rubric, be conservative, and return valid JSON only."
        )
        user = (
            "请根据 evidence JSON 对作文进行评分和反馈。\n"
            "如果 exam_type 是 IELTS_Task_2，请按 IELTS Task 2 四项标准估分："
            "task_response, coherence_cohesion, lexical_resource, grammar_accuracy，范围 0-9，可用 0.5 分档，overall 为四项平均后的 IELTS 估计分。\n"
            "如果 exam_type 是 TOEFL，请按 TOEFL 写作 0-5 评分逻辑估计 overall，并给出 relevance/task, development/organization, lexical, grammar 四个诊断子项。\n"
            "请检查 evidence.local_errors 是否有误判；例如 many + adjective + plural noun 不是 plural error。"
            "允许补充作文中明显存在但规则漏掉的高置信语法错误。不要整篇代写。\n"
            "输出 JSON: {rubric_scores:{task_response, coherence_cohesion, lexical_resource, grammar_accuracy, overall, confidence}, "
            "summary_zh, rubric_feedback, global_feedback, local_feedback, lexical_feedback, revision_tasks, "
            "confirmed_errors, rejected_candidate_errors, added_errors}.\n\n"
            f"evidence_json={json.dumps(payload, ensure_ascii=False)}"
        )
        parsed, raw = qwen_json(self.settings, system, user, max_tokens=self.settings.qwen_summary_max_tokens + 900)
        if not parsed:
            template.llm_raw = raw
            template.generator = "template+qwen_raw_unparsed" if raw else template.generator
            return template
        apply_qwen_rubric_scores(evidence, parsed)
        template = template_feedback(evidence, learning_issues, practice_items)
        return WritingFeedback(
            summary_zh=str(parsed.get("summary_zh") or template.summary_zh),
            rubric_feedback=list_of_strings(parsed.get("rubric_feedback")) or template.rubric_feedback,
            global_feedback=list_of_strings(parsed.get("global_feedback")) or template.global_feedback,
            local_feedback=list_of_strings(parsed.get("local_feedback")) or template.local_feedback,
            lexical_feedback=list_of_strings(parsed.get("lexical_feedback")) or template.lexical_feedback,
            revision_tasks=list_of_strings(parsed.get("revision_tasks")) or template.revision_tasks,
            generator="qwen_text_evidence_grounded",
            llm_raw=raw,
        )


def clean_text(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", text).strip()


def normalize_paragraphs(text: str) -> str:
    text = (text or "").strip().replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not paragraphs:
        paragraphs = [clean_text(text)] if text.strip() else []
    return "\n\n".join(paragraphs)


def analyze_prompt(prompt: str, exam_type: str) -> PromptAnalysis:
    lower = prompt.lower()
    task_type = "opinion"
    required_parts = ["state a clear position", "develop reasons", "support with examples"]
    if "advantage" in lower and "disadvantage" in lower:
        task_type = "advantages_disadvantages"
        required_parts = ["discuss advantages", "discuss disadvantages", "give a conclusion"]
    elif "discuss both" in lower:
        task_type = "discuss_both_views"
        required_parts = ["discuss view 1", "discuss view 2", "give your own opinion"]
    elif "problem" in lower and "solution" in lower:
        task_type = "problem_solution"
        required_parts = ["identify problems", "propose solutions", "support with examples"]
    elif "academic discussion" in lower or "professor" in lower:
        task_type = "toefl_academic_discussion"
        required_parts = ["respond to the question", "connect to the discussion", "give a developed reason"]
    keywords = extract_keywords(prompt, limit=10)
    min_words = 100 if "toefl" in exam_type.lower() and "discussion" in task_type else 250
    if "task 1" in exam_type.lower():
        min_words = 150
    return PromptAnalysis(
        exam_type=exam_type or "general",
        task_type=task_type,
        topic_keywords=keywords,
        required_parts=required_parts,
        expected_min_words=min_words,
    )


def essay_stats(essay: str) -> EssayStats:
    words = tokenize_words(essay)
    sentences = split_sentences(essay)
    paragraphs = split_paragraphs(essay)
    lemmas = [normalize_word(w) for w in words if normalize_word(w)]
    content = [w for w in lemmas if w not in FUNCTION_WORDS]
    lexical_diversity = len(set(content)) / max(1, len(content))
    return EssayStats(
        word_count=len(words),
        sentence_count=len(sentences),
        paragraph_count=len(paragraphs),
        avg_sentence_length=round(len(words) / max(1, len(sentences)), 1),
        lexical_diversity=round(lexical_diversity, 3),
    )


def detect_thesis(essay: str) -> str:
    paragraphs = split_paragraphs(essay)
    intro = paragraphs[0] if paragraphs else essay
    sentences = split_sentences(intro)
    stance_markers = re.compile(r"\b(I believe|I think|in my opinion|I agree|I disagree|this essay|I would argue)\b", re.I)
    for sentence in sentences:
        if stance_markers.search(sentence):
            return sentence
    return sentences[-1] if sentences else ""


def detect_topic_sentences(essay: str) -> List[str]:
    out: List[str] = []
    for p in split_paragraphs(essay):
        sentences = split_sentences(p)
        if sentences:
            out.append(sentences[0])
    return out


def analyze_structure(
    prompt: PromptAnalysis,
    essay: str,
    thesis: str,
    topic_sentences: List[str],
    stats: EssayStats,
) -> List[StructureIssue]:
    issues: List[StructureIssue] = []
    if stats.word_count < prompt.expected_min_words:
        issues.append(
            StructureIssue(
                issue_id="WS001",
                type="under_length",
                evidence=f"文章约 {stats.word_count} 词，目标任务通常至少需要 {prompt.expected_min_words} 词。",
                suggestion_hint="补充理由、例子或反方讨论，而不是重复同一句观点。",
                severity="high" if stats.word_count < prompt.expected_min_words * 0.7 else "medium",
                confidence=0.95,
            )
        )
    if not thesis:
        issues.append(
            StructureIssue(
                issue_id="WS002",
                type="missing_clear_position",
                evidence="开头没有检测到清楚立场句。",
                suggestion_hint="在引言末尾用一句话明确你的观点。",
                severity="high",
                confidence=0.85,
            )
        )
    coverage = prompt_coverage(prompt.topic_keywords, essay)
    if coverage < 0.45:
        issues.append(
            StructureIssue(
                issue_id="WS003",
                type="weak_task_coverage",
                evidence=f"题目关键词覆盖率较低：{coverage:.2f}。",
                suggestion_hint="检查题目中的每个关键词是否都被回应。",
                severity="high",
                confidence=0.78,
            )
        )
    if stats.paragraph_count < 3 and stats.word_count >= 120:
        issues.append(
            StructureIssue(
                issue_id="WS004",
                type="weak_paragraphing",
                evidence=f"文章只有 {stats.paragraph_count} 段，段落层次不够清晰。",
                suggestion_hint="至少使用引言、主体段、结论，主体段每段一个中心。",
                severity="medium",
                confidence=0.80,
            )
        )
    repeated = repeated_topic_sentences(topic_sentences)
    if repeated:
        issues.append(
            StructureIssue(
                issue_id="WS005",
                type="repeated_argument",
                evidence=f"这些主题句语义接近：{'; '.join(repeated[:2])}",
                suggestion_hint="让不同主体段承担不同论点，避免换词重复。",
                severity="medium",
                confidence=0.70,
            )
        )
    if not has_example(essay) and stats.word_count >= 120:
        issues.append(
            StructureIssue(
                issue_id="WS006",
                type="underdeveloped_argument",
                evidence="没有明显例子或具体情境支撑论点。",
                suggestion_hint="每个主体段至少加入一个具体例子、数据、场景或因果解释。",
                severity="medium",
                confidence=0.75,
            )
        )
    return issues


def detect_local_errors(essay: str) -> List[LocalWritingError]:
    errors: List[LocalWritingError] = []
    sentences = split_sentences(essay)
    idx = 1
    for s_idx, sentence in enumerate(sentences, start=1):
        for match in re.finditer(r"\bmany\s+([A-Za-z]+)(?:\s+([A-Za-z]+))?", sentence, re.I):
            first = match.group(1)
            second = match.group(2) or ""
            if is_many_adjective_plural_noun(first, second):
                continue
            if not is_plural_like(first):
                span = f"many {first}"
                idx = add_local_error(
                    errors,
                    idx,
                    span,
                    "plural_error",
                    f"many {pluralize_regular(first)}",
                    "many 后面通常接可数名词复数；如果 many 后接形容词，后面还需要有复数名词。",
                    s_idx,
                    "medium",
                    0.86,
                )

        for pattern, correction, hint, kind, severity, confidence in grammar_patterns():
            for match in re.finditer(pattern, sentence, re.I):
                replacement = correction(match) if callable(correction) else correction
                idx = add_local_error(errors, idx, match.group(0), kind, replacement, hint, s_idx, severity, confidence)

        for match in re.finditer(r"\b(people|students|children|researchers|teachers)\s+(is|was)\b", sentence, re.I):
            plural, verb = match.groups()
            correction = f"{plural} {'are' if verb.lower() == 'is' else 'were'}"
            idx = add_local_error(errors, idx, match.group(0), "subject_verb_agreement", correction, "复数主语需要复数谓语。", s_idx, "high", 0.86)
        for match in re.finditer(r"\b(learning|education|research|information|knowledge)\s+are\b", sentence, re.I):
            idx = add_local_error(errors, idx, match.group(0), "subject_verb_agreement", f"{match.group(1)} is", "抽象名词或动名词短语整体作单数。", s_idx, "high", 0.84)
        for match in re.finditer(r"\b((?:online|classroom|traditional classroom|normal classroom)\s+learning)\s+are\b", sentence, re.I):
            idx = add_local_error(errors, idx, match.group(0), "subject_verb_agreement", f"{match.group(1)} is", "动名词短语作主语时，谓语通常用单数。", s_idx, "high", 0.86)

        for match in re.finditer(
            r"\b((?:online|classroom|traditional classroom|normal classroom)\s+learning|learning|education|research|information|knowledge)\s+((?:still|also|usually|often|sometimes|always)\s+)?(have|need|do)\b",
            sentence,
            re.I,
        ):
            subject, adverb, verb = match.groups()
            correction_verb = {"have": "has", "need": "needs", "do": "does"}[verb.lower()]
            correction = f"{subject} {adverb or ''}{correction_verb}".strip()
            idx = add_local_error(errors, idx, match.group(0), "subject_verb_agreement", correction, "动名词或抽象名词短语作主语时，谓语通常用第三人称单数。", s_idx, "high", 0.88)

        idx = detect_parser_agreement_errors(sentence, s_idx, errors, idx)
        if len(tokenize_words(sentence)) > 38 and re.search(r"\b(and|but|because|so)\b", sentence, re.I):
            idx = add_local_error(errors, idx, sentence[:120], "run_on_sentence", "split or subordinate the sentence", "句子过长且连接关系混杂，容易形成 run-on sentence。", s_idx, "medium", 0.72)
        if looks_like_fragment(sentence):
            idx = add_local_error(errors, idx, sentence, "sentence_fragment", "add a main subject and finite verb", "这个句子可能缺少完整主干。", s_idx, "high", 0.68)
        for match in re.finditer(r"\b(although|because|while)\b[^.!?]*\.\s*$", sentence, re.I):
            if "," not in sentence and len(tokenize_words(sentence)) < 12:
                idx = add_local_error(errors, idx, sentence, "subordinate_clause_fragment", "connect it to a main clause", "从属连词引导的从句不能单独作为完整句。", s_idx, "high", 0.72)
    return errors[:60]


def grammar_patterns() -> List[Tuple[str, Any, str, str, str, float]]:
    return [
        (
            r"\bI\s+am\s+((?:partly|fully|strongly|totally)\s+)?agree\b",
            lambda m: f"I {m.group(1) or ''}agree".strip(),
            "agree 在这里是动词，不需要 be 动词。",
            "verb_form_error",
            "high",
            0.92,
        ),
        (
            r"\bis\s+more\s+freedom\s+and\s+convenient\b",
            "offers more freedom and convenience / is more flexible and convenient",
            "freedom 是名词，不能和 convenient 直接并列作形容词补语。",
            "parallel_structure_error",
            "medium",
            0.84,
        ),
        (
            r"\bon\s+internet\b",
            "on the internet",
            "internet 通常需要冠词 the。",
            "article_error",
            "medium",
            0.86,
        ),
        (
            r"\bsome\s+knowledges\b",
            "some knowledge",
            "knowledge 通常是不可数名词，不用复数形式。",
            "uncountable_noun_error",
            "high",
            0.90,
        ),
        (
            r"\binformations\b",
            "information",
            "information 通常是不可数名词，不用复数形式。",
            "uncountable_noun_error",
            "high",
            0.92,
        ),
        (
            r"\bmore\s+([A-Za-z]+er)\b",
            lambda m: m.group(1),
            "已经带 -er 的比较级前面通常不再加 more。",
            "comparative_error",
            "high",
            0.90,
        ),
        (
            r"\bcommunicate\s+with\s+teacher\b",
            "communicate with teachers / communicate with the teacher",
            "teacher 作可数名词时需要冠词或复数形式。",
            "article_number_error",
            "medium",
            0.82,
        ),
        (
            r"\bfeel\s+boring\b",
            "feel bored",
            "描述人的感受通常用 bored，boring 描述事物令人无聊。",
            "adjective_choice_error",
            "high",
            0.90,
        ),
        (
            r"\b(?:do|does|did|don't|doesn't|didn't|not)\b[^.!?]{0,40}\bnothing\b",
            lambda m: re.sub(r"\bnothing\b", "anything", m.group(0), flags=re.I),
            "否定句中通常用 anything，避免双重否定。",
            "double_negative",
            "high",
            0.88,
        ),
        (
            r"\bcannot\s+always\s+do\s+good\b",
            "cannot always do well",
            "修饰动作 do 应使用副词 well，而不是形容词 good。",
            "adverb_adjective_error",
            "medium",
            0.84,
        ),
        (
            r"\bmore\s+chance\b",
            "more chances",
            "chance 作可数名词时，more 后通常用复数。",
            "plural_error",
            "medium",
            0.86,
        ),
        (
            r"\bteacher\s*['’]\s*s\s+careful\s+watching\b",
            "teachers' careful supervision / guidance from teachers",
            "careful watching 搭配不自然，学术写作中可用 supervision 或 guidance。",
            "collocation_error",
            "medium",
            0.78,
        ),
    ]


def add_local_error(
    errors: List[LocalWritingError],
    idx: int,
    span: str,
    kind: str,
    correction: str,
    hint: str,
    sentence_index: int,
    severity: str,
    confidence: float,
) -> int:
    span = re.sub(r"\s+", " ", span).strip()
    if not span:
        return idx
    lower = span.lower()
    for existing in errors:
        if existing.sentence_index == sentence_index and existing.type == kind:
            existing_lower = existing.span.lower()
            if lower in existing_lower or existing_lower in lower:
                return idx
    errors.append(make_error(idx, span, kind, correction, hint, sentence_index, severity, confidence))
    return idx + 1


def is_many_adjective_plural_noun(first: str, second: str) -> bool:
    return bool(second) and not is_plural_like(first) and is_plural_like(second)


def is_plural_like(word: str) -> bool:
    lower = word.lower()
    return lower in {"people", "children", "men", "women"} or lower.endswith("s")


def pluralize_regular(word: str) -> str:
    lower = word.lower()
    if lower.endswith("y") and len(lower) > 1 and lower[-2] not in "aeiou":
        return word[:-1] + "ies"
    if lower.endswith(("s", "x", "ch", "sh")):
        return word + "es"
    return word + "s"


_SPACY_NLP: Any = None
_SPACY_LOAD_FAILED = False


def detect_parser_agreement_errors(
    sentence: str,
    sentence_index: int,
    errors: List[LocalWritingError],
    idx: int,
) -> int:
    nlp = get_spacy_nlp()
    if nlp is None:
        return idx
    try:
        doc = nlp(sentence)
    except Exception:
        return idx
    for token in doc:
        if token.pos_ not in {"VERB", "AUX"}:
            continue
        subjects = [child for child in token.children if child.dep_ in {"nsubj", "nsubjpass"}]
        if not subjects:
            continue
        if has_modal_aux(token):
            continue
        subject = subjects[0]
        number = subject_number(subject)
        if number == "singular" and token.text.lower() in {"are", "were", "have", "do", "need"}:
            corrected = singular_verb_form(token.text)
            if agreement_error_already_exists(errors, sentence_index, subject_subtree(subject), token.text):
                continue
            idx = add_local_error(
                errors,
                idx,
                f"{subject_subtree(subject)} {token.text}",
                "subject_verb_agreement",
                f"{subject_subtree(subject)} {corrected}",
                "根据句法主干，单数主语需要对应的单数谓语形式。",
                sentence_index,
                "high",
                0.76,
            )
        elif number == "plural" and token.text.lower() in {"is", "was", "has", "does"}:
            corrected = plural_verb_form(token.text)
            if agreement_error_already_exists(errors, sentence_index, subject_subtree(subject), token.text):
                continue
            idx = add_local_error(
                errors,
                idx,
                f"{subject_subtree(subject)} {token.text}",
                "subject_verb_agreement",
                f"{subject_subtree(subject)} {corrected}",
                "根据句法主干，复数主语需要对应的复数谓语形式。",
                sentence_index,
                "high",
                0.76,
            )
    return idx


def has_modal_aux(token: Any) -> bool:
    modals = {"can", "cannot", "can't", "could", "should", "would", "will", "may", "might", "must"}
    try:
        return any(child.dep_ == "aux" and child.text.lower() in modals for child in token.children)
    except Exception:
        return False


def agreement_error_already_exists(errors: List[LocalWritingError], sentence_index: int, subject: str, verb: str) -> bool:
    subject_words = [w for w in tokenize_words(subject.lower()) if w not in FUNCTION_WORDS]
    verb_lower = verb.lower()
    for error in errors:
        if error.sentence_index != sentence_index or error.type != "subject_verb_agreement":
            continue
        span = error.span.lower()
        if verb_lower in span and all(w in span for w in subject_words):
            return True
    return False


def get_spacy_nlp():
    global _SPACY_NLP, _SPACY_LOAD_FAILED
    if _SPACY_NLP is not None:
        return _SPACY_NLP
    if _SPACY_LOAD_FAILED:
        return None
    try:
        import spacy  # type: ignore

        model = os.environ.get("CAPT_MODEC_SPACY_MODEL", "en_core_web_sm")
        _SPACY_NLP = spacy.load(model, disable=["ner"])
        return _SPACY_NLP
    except Exception:
        _SPACY_LOAD_FAILED = True
        return None


def subject_number(token: Any) -> str:
    lower = token.text.lower()
    subtree = subject_subtree(token).lower()
    if lower in {"people", "children", "men", "women", "students", "teachers", "researchers"}:
        return "plural"
    if lower in {"i", "you", "we", "they"}:
        return "plural"
    if lower in {"he", "she", "it", "this", "that"}:
        return "singular"
    if lower in {"learning", "education", "research", "information", "knowledge", "university"}:
        return "singular"
    if subtree.endswith(" learning"):
        return "singular"
    if getattr(token, "tag_", "") in {"NNS", "NNPS"}:
        return "plural"
    if getattr(token, "tag_", "") in {"NN", "NNP", "VBG"}:
        return "singular"
    return "unknown"


def subject_subtree(token: Any) -> str:
    try:
        children = list(token.subtree)
        start = min(x.i for x in children)
        end = max(x.i for x in children)
        return token.doc[start : end + 1].text
    except Exception:
        return str(getattr(token, "text", ""))


def singular_verb_form(verb: str) -> str:
    lower = verb.lower()
    mapping = {"are": "is", "were": "was", "have": "has", "do": "does", "need": "needs"}
    return mapping.get(lower, third_person_singular(verb))


def plural_verb_form(verb: str) -> str:
    lower = verb.lower()
    mapping = {"is": "are", "was": "were", "has": "have", "does": "do"}
    return mapping.get(lower, verb)


def third_person_singular(verb: str) -> str:
    lower = verb.lower()
    if lower.endswith("y") and len(lower) > 1 and lower[-2] not in "aeiou":
        return verb[:-1] + "ies"
    if lower.endswith(("s", "x", "ch", "sh", "o")):
        return verb + "es"
    return verb + "s"


def make_error(
    idx: int,
    span: str,
    kind: str,
    correction: str,
    hint: str,
    sentence_index: int,
    severity: str,
    confidence: float,
) -> LocalWritingError:
    return LocalWritingError(
        error_id=f"WE{idx:04d}",
        span=span,
        type=kind,
        correction=correction,
        explanation_hint=hint,
        sentence_index=sentence_index,
        severity=severity,
        confidence=confidence,
    )


def detect_lexical_issues(essay: str) -> List[LexicalIssue]:
    issues: List[LexicalIssue] = []
    idx = 1
    patterns = [
        (r"\bmake a research\b", "collocation_error", "conduct research / do research", "research 通常和 conduct/do 搭配，不说 make a research。"),
        (r"\bdo a mistake\b", "collocation_error", "make a mistake", "mistake 常和 make 搭配。"),
        (r"\bbig influence\b", "collocation_style", "significant influence / major impact", "学术写作中 influence/impact 前常用 significant/major。"),
        (r"\bvery good\b", "vague_wording", "beneficial / effective / valuable", "very good 太泛，可以换成更具体的评价词。"),
    ]
    for pattern, kind, suggestion, hint in patterns:
        for match in re.finditer(pattern, essay, re.I):
            issues.append(
                LexicalIssue(
                    issue_id=f"WL{idx:04d}",
                    span=match.group(0),
                    type=kind,
                    suggestion=suggestion,
                    explanation_hint=hint,
                    severity="medium" if kind == "collocation_error" else "low",
                    confidence=0.86,
                )
            )
            idx += 1
    repeated = repeated_words(essay)
    for word, count in repeated[:6]:
        issues.append(
            LexicalIssue(
                issue_id=f"WL{idx:04d}",
                span=word,
                type="word_repetition",
                suggestion=f"用更具体的表达替换部分 {word}",
                explanation_hint=f"{word} 出现 {count} 次，可能显得重复。",
                severity="low",
                confidence=0.65,
            )
        )
        idx += 1
    return issues[:40]


def score_rubric(
    prompt: PromptAnalysis,
    stats: EssayStats,
    thesis: str,
    structure_issues: List[StructureIssue],
    local_errors: List[LocalWritingError],
    lexical_issues: List[LexicalIssue],
) -> RubricScores:
    if is_toefl_exam(prompt.exam_type):
        return score_toefl_rubric(prompt, stats, thesis, structure_issues, local_errors, lexical_issues)
    return score_ielts_rubric(prompt, stats, thesis, structure_issues, local_errors, lexical_issues)


def score_ielts_rubric(
    prompt: PromptAnalysis,
    stats: EssayStats,
    thesis: str,
    structure_issues: List[StructureIssue],
    local_errors: List[LocalWritingError],
    lexical_issues: List[LexicalIssue],
) -> RubricScores:
    coverage = 1.0
    for issue in structure_issues:
        if issue.type == "weak_task_coverage":
            coverage = 0.55
    length_factor = min(1.0, stats.word_count / max(1, prompt.expected_min_words))
    high_errors = [x for x in local_errors if x.severity == "high"]
    medium_errors = [x for x in local_errors if x.severity == "medium"]
    grammar_density = weighted_error_density(local_errors, stats.word_count)

    task = 5.0 + 1.0 * coverage + 0.5 * length_factor + (0.5 if thesis else -0.7)
    if has_example_text(stats, structure_issues):
        task += 0.3
    if stats.paragraph_count >= 4:
        task += 0.2
    task -= sum(0.5 for x in structure_issues if x.severity == "high")

    coherence = 5.5 + min(1.0, max(0, stats.paragraph_count - 2) * 0.25)
    coherence -= sum(0.6 for x in structure_issues if x.type in {"weak_paragraphing", "repeated_argument", "underdeveloped_argument"})
    if has_transition_words_from_text(stats, structure_issues):
        coherence += 0.5
    if any(x.type == "run_on_sentence" for x in local_errors):
        coherence -= 0.3

    lexical = 5.0 + min(1.2, stats.lexical_diversity * 1.8)
    lexical -= sum(0.35 if x.severity == "medium" else 0.18 for x in lexical_issues)

    grammar = 6.7 - min(3.0, grammar_density * 0.38)
    if stats.avg_sentence_length >= 14:
        grammar += 0.3
    if any(x.type in {"run_on_sentence", "sentence_fragment"} for x in local_errors):
        grammar -= 0.5

    caps = ielts_score_caps(prompt, stats, thesis, structure_issues, local_errors, lexical_issues)
    task = min(task, caps["task_response"])
    coherence = min(coherence, caps["coherence_cohesion"])
    lexical = min(lexical, caps["lexical_resource"])
    grammar = min(grammar, caps["grammar_accuracy"])

    task = round_ielts_band(task)
    coherence = round_ielts_band(coherence)
    lexical = round_ielts_band(lexical)
    grammar = round_ielts_band(grammar)
    overall = round_ielts_band((task + coherence + lexical + grammar) / 4.0)
    confidence = 0.62 + min(0.18, stats.word_count / 1200.0)
    return RubricScores(
        task_response=task,
        coherence_cohesion=coherence,
        lexical_resource=lexical,
        grammar_accuracy=grammar,
        overall=overall,
        confidence=round(confidence, 2),
    )


def score_toefl_rubric(
    prompt: PromptAnalysis,
    stats: EssayStats,
    thesis: str,
    structure_issues: List[StructureIssue],
    local_errors: List[LocalWritingError],
    lexical_issues: List[LexicalIssue],
) -> RubricScores:
    coverage = 1.0
    if any(x.type == "weak_task_coverage" for x in structure_issues):
        coverage = 0.55
    length_factor = min(1.0, stats.word_count / max(1, prompt.expected_min_words))
    grammar_density = weighted_error_density(local_errors, stats.word_count)
    relevance = 2.7 + 0.9 * coverage + 0.4 * length_factor + (0.3 if thesis else -0.5)
    development = 2.6 + min(0.7, max(0, stats.paragraph_count - 1) * 0.18)
    if has_transition_words_from_text(stats, structure_issues):
        development += 0.3
    development -= sum(0.35 for x in structure_issues if x.severity in {"medium", "high"})
    lexical = 2.7 + min(0.8, stats.lexical_diversity * 1.2)
    lexical -= sum(0.18 if x.severity == "medium" else 0.10 for x in lexical_issues)
    grammar = 3.4 - min(1.8, grammar_density * 0.22)
    if any(x.type in {"run_on_sentence", "sentence_fragment"} for x in local_errors):
        grammar -= 0.25
    relevance = round_to_half(clamp_range(relevance, 0.0, 5.0))
    development = round_to_half(clamp_range(development, 0.0, 5.0))
    lexical = round_to_half(clamp_range(lexical, 0.0, 5.0))
    grammar = round_to_half(clamp_range(grammar, 0.0, 5.0))
    overall = round_to_half((relevance + development + lexical + grammar) / 4.0)
    return RubricScores(
        task_response=relevance,
        coherence_cohesion=development,
        lexical_resource=lexical,
        grammar_accuracy=grammar,
        overall=overall,
        confidence=round(0.62 + min(0.18, stats.word_count / 1200.0), 2),
    )


def ielts_score_caps(
    prompt: PromptAnalysis,
    stats: EssayStats,
    thesis: str,
    structure_issues: List[StructureIssue],
    local_errors: List[LocalWritingError],
    lexical_issues: List[LexicalIssue],
) -> Dict[str, float]:
    caps = {
        "task_response": 7.5,
        "coherence_cohesion": 8.0,
        "lexical_resource": 8.0,
        "grammar_accuracy": 8.0,
    }
    if stats.word_count < prompt.expected_min_words:
        caps["task_response"] = min(caps["task_response"], 6.0 if stats.word_count >= prompt.expected_min_words * 0.7 else 5.0)
    if not thesis:
        caps["task_response"] = min(caps["task_response"], 5.5)
    if any(x.type == "weak_task_coverage" for x in structure_issues):
        caps["task_response"] = min(caps["task_response"], 5.5)
    if not has_example_text(stats, structure_issues):
        caps["task_response"] = min(caps["task_response"], 6.5)
    if any(x.type in {"weak_paragraphing", "repeated_argument"} for x in structure_issues):
        caps["coherence_cohesion"] = min(caps["coherence_cohesion"], 6.5)
    repetition_count = sum(1 for x in lexical_issues if x.type == "word_repetition")
    medium_lex = sum(1 for x in lexical_issues if x.severity == "medium")
    if repetition_count >= 5:
        caps["lexical_resource"] = min(caps["lexical_resource"], 6.5)
    if medium_lex >= 2:
        caps["lexical_resource"] = min(caps["lexical_resource"], 6.0)
    high_errors = sum(1 for x in local_errors if x.severity == "high")
    grammar_density = weighted_error_density(local_errors, stats.word_count)
    if grammar_density >= 4.0 or high_errors >= 6:
        caps["grammar_accuracy"] = min(caps["grammar_accuracy"], 5.5)
    elif grammar_density >= 2.3 or high_errors >= 3:
        caps["grammar_accuracy"] = min(caps["grammar_accuracy"], 6.0)
    elif grammar_density >= 1.2 or high_errors >= 1:
        caps["grammar_accuracy"] = min(caps["grammar_accuracy"], 6.5)
    return caps


def weighted_error_density(errors: List[LocalWritingError], word_count: int) -> float:
    weights = {"high": 1.25, "medium": 0.8, "low": 0.45}
    total = sum(weights.get(x.severity, 0.7) for x in errors)
    return total / max(1, word_count) * 100.0


def has_example_text(stats: EssayStats, structure_issues: List[StructureIssue]) -> bool:
    return stats.word_count < 120 or not any(x.type == "underdeveloped_argument" for x in structure_issues)


def is_toefl_exam(exam_type: str) -> bool:
    return "toefl" in (exam_type or "").lower()


def max_score_for_exam(exam_type: str) -> float:
    return 5.0 if is_toefl_exam(exam_type) else 9.0


def rubric_scale_label(exam_type: str) -> str:
    return "TOEFL 0-5" if is_toefl_exam(exam_type) else "IELTS 0-9"


def round_ielts_band(value: float) -> float:
    return round_to_half(clamp_range(value, 0.0, 9.0))


def round_to_half(value: float) -> float:
    return round(value * 2.0) / 2.0


def clamp_range(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def build_revision_tasks(
    structure_issues: List[StructureIssue],
    local_errors: List[LocalWritingError],
    lexical_issues: List[LexicalIssue],
    scores: RubricScores,
    exam_type: str = "IELTS_Task_2",
) -> List[str]:
    tasks: List[str] = []
    for issue in structure_issues[:3]:
        tasks.append(issue.suggestion_hint)
    for error in local_errors[:5]:
        tasks.append(f"修改 '{error.span}'：{error.explanation_hint} 建议：{error.correction}")
    for issue in lexical_issues[:3]:
        tasks.append(f"替换或调整 '{issue.span}'：{issue.suggestion}")
    weakest = min(
        [
            ("Task Response", scores.task_response),
            ("Coherence & Cohesion", scores.coherence_cohesion),
            ("Lexical Resource", scores.lexical_resource),
            ("Grammar Accuracy", scores.grammar_accuracy),
        ],
        key=lambda x: x[1],
    )
    tasks.insert(0, f"优先提升 {weakest[0]}，当前估计为 {weakest[1]:.1f}/{max_score_for_exam(exam_type):.0f}（{rubric_scale_label(exam_type)}）。")
    return unique(tasks)[:8]


def compare_revision(
    previous_essay: str,
    essay: str,
    local_errors: List[LocalWritingError],
    lexical_issues: List[LexicalIssue],
    structure_issues: List[StructureIssue],
) -> RevisionComparison:
    if not previous_essay:
        return RevisionComparison(
            has_previous=False,
            current_word_count=len(tokenize_words(essay)),
            current_error_count=len(local_errors) + len(lexical_issues) + len(structure_issues),
            improvement_summary="没有提供上一版作文；提交上一版后可以比较修改是否真正改善。",
        )
    prev_errors = detect_local_errors(previous_essay)
    prev_lex = detect_lexical_issues(previous_essay)
    prev_struct = analyze_structure(analyze_prompt("", "general"), previous_essay, detect_thesis(previous_essay), detect_topic_sentences(previous_essay), essay_stats(previous_essay))
    prev_count = len(prev_errors) + len(prev_lex) + len(prev_struct)
    cur_count = len(local_errors) + len(lexical_issues) + len(structure_issues)
    improved: List[str] = []
    if len(tokenize_words(essay)) > len(tokenize_words(previous_essay)):
        improved.append("development")
    if cur_count < prev_count:
        improved.append("error_reduction")
    if paragraph_count(essay) >= paragraph_count(previous_essay):
        improved.append("organization")
    summary = "本版比上一版更完整。" if improved else "本版相对上一版的可见改善有限。"
    return RevisionComparison(
        has_previous=True,
        previous_word_count=len(tokenize_words(previous_essay)),
        current_word_count=len(tokenize_words(essay)),
        previous_error_count=prev_count,
        current_error_count=cur_count,
        improvement_summary=summary,
        improved_dimensions=improved,
        remaining_priorities=["处理最高严重度错误", "补足论证例子", "减少重复词"][:3],
    )


def build_learning_issues(evidence: WritingEvidence) -> List[LearningIssue]:
    issues: List[LearningIssue] = []
    for item in evidence.structure_issues:
        issues.append(
            LearningIssue(
                issue_id=f"EW_{item.issue_id}",
                source_mode="E",
                source_task="writing_structure",
                issue_type=item.type,
                skill_tags=tags_for_structure(item),
                span_text=item.evidence,
                reference_text=evidence.prompt,
                user_output=item.evidence,
                severity=item.severity,
                confidence=item.confidence,
                evidence=model_to_dict(item),
            )
        )
    for item in evidence.local_errors:
        issues.append(
            LearningIssue(
                issue_id=f"EW_{item.error_id}",
                source_mode="E",
                source_task="writing_grammar",
                issue_type=item.type,
                skill_tags=tags_for_local_error(item),
                span_text=item.span,
                reference_text=item.correction,
                user_output=item.span,
                severity=item.severity,
                confidence=item.confidence,
                evidence=model_to_dict(item),
            )
        )
    for item in evidence.lexical_issues:
        issues.append(
            LearningIssue(
                issue_id=f"EW_{item.issue_id}",
                source_mode="E",
                source_task="writing_lexis",
                issue_type=item.type,
                skill_tags=["writing.lexical_resource", "vocab.collocation" if "collocation" in item.type else "writing.style"],
                span_text=item.span,
                reference_text=item.suggestion,
                user_output=item.span,
                severity=item.severity,
                confidence=item.confidence,
                evidence=model_to_dict(item),
            )
        )
    return issues[:80]


def tags_for_structure(item: StructureIssue) -> List[str]:
    if item.type == "weak_task_coverage":
        return ["writing.task_response"]
    if item.type in {"weak_paragraphing", "repeated_argument", "underdeveloped_argument"}:
        return ["writing.coherence", "writing.argument_development"]
    return ["writing.task_response"]


def tags_for_local_error(item: LocalWritingError) -> List[str]:
    tags = ["writing.grammar"]
    if "agreement" in item.type:
        tags.extend(["grammar.subject_verb_agreement", "reading.syntax"])
    if "plural" in item.type:
        tags.extend(["grammar.morphology", "listening.word_endings"])
    if "uncountable" in item.type:
        tags.extend(["grammar.noun_countability", "writing.lexical_resource"])
    if "article" in item.type:
        tags.extend(["grammar.articles", "writing.accuracy"])
    if "comparative" in item.type:
        tags.extend(["grammar.comparatives", "writing.accuracy"])
    if "double_negative" in item.type:
        tags.extend(["grammar.negation", "reading.syntax"])
    if "adjective" in item.type or "adverb" in item.type:
        tags.extend(["grammar.word_form", "vocab.collocation"])
    if "verb_form" in item.type:
        tags.extend(["grammar.verb_form", "writing.accuracy"])
    if "parallel" in item.type:
        tags.extend(["grammar.parallelism", "writing.coherence"])
    if "collocation" in item.type:
        tags.extend(["vocab.collocation", "writing.lexical_resource"])
    if "fragment" in item.type or "run_on" in item.type:
        tags.extend(["grammar.sentence_boundary", "reading.main_clause"])
    return sorted(set(tags))


def build_writing_practice(evidence: WritingEvidence, issues: List[LearningIssue]) -> List[PracticeItem]:
    items: List[PracticeItem] = []
    for issue in issues[:10]:
        practice_type = "sentence_revision"
        instruction = f"修改这个表达：{issue.span_text}。参考目标：{issue.reference_text}"
        target_mode = "E"
        if "reading.syntax" in issue.skill_tags or "reading.main_clause" in issue.skill_tags:
            target_mode = "C"
            practice_type = "sentence_structure_analysis"
            instruction = f"先用长难句模式拆解这个问题句，再回到作文修改：{issue.span_text}"
        items.append(
            PracticeItem(
                practice_id=stable_id("EPR", issue.issue_id, practice_type),
                target_mode=target_mode,  # type: ignore[arg-type]
                practice_type=practice_type,
                title=instruction[:48],
                instruction=instruction,
                source_issue_ids=[issue.issue_id],
                skill_tags=issue.skill_tags,
                payload={"prompt": evidence.prompt, "focus_span": issue.span_text, "suggested_target": issue.reference_text},
                success_criteria={"revision": "submit a revised sentence or paragraph"},
            )
        )
    return items


def template_feedback(
    evidence: WritingEvidence,
    learning_issues: List[LearningIssue],
    practice_items: List[PracticeItem],
) -> WritingFeedback:
    s = evidence.rubric_scores
    scale = max_score_for_exam(evidence.prompt_analysis.exam_type)
    scale_label = rubric_scale_label(evidence.prompt_analysis.exam_type)
    if is_toefl_exam(evidence.prompt_analysis.exam_type):
        dimension_names = ("任务相关", "展开与组织", "词汇表达", "语法控制")
    else:
        dimension_names = ("任务回应", "连贯", "词汇", "语法")
    summary = (
        f"当前写作估计总分 {s.overall:.1f}/{scale:.0f}（{scale_label}）。"
        f"{dimension_names[0]} {s.task_response:.1f}，{dimension_names[1]} {s.coherence_cohesion:.1f}，"
        f"{dimension_names[2]} {s.lexical_resource:.1f}，{dimension_names[3]} {s.grammar_accuracy:.1f}。"
    )
    rubric_feedback = [
        f"{dimension_names[0]}: {s.task_response:.1f}/{scale:.0f}。题目类型：{evidence.prompt_analysis.task_type}。",
        f"{dimension_names[1]}: {s.coherence_cohesion:.1f}/{scale:.0f}。段落数 {evidence.stats.paragraph_count}，主题句 {len(evidence.topic_sentences)} 个。",
        f"{dimension_names[2]}: {s.lexical_resource:.1f}/{scale:.0f}。词汇多样性估计 {evidence.stats.lexical_diversity:.2f}。",
        f"{dimension_names[3]}: {s.grammar_accuracy:.1f}/{scale:.0f}。检测到 {len(evidence.local_errors)} 个局部语法/句法问题。",
    ]
    global_feedback = [f"{x.type}: {x.evidence} 建议：{x.suggestion_hint}" for x in evidence.structure_issues] or ["整体结构没有发现高置信度问题。"]
    local_feedback = [f"{x.span} -> {x.correction}: {x.explanation_hint}" for x in evidence.local_errors] or ["局部语法错误较少，继续提高句式准确性。"]
    lexical_feedback = [f"{x.span}: {x.explanation_hint} 建议：{x.suggestion}" for x in evidence.lexical_issues] or ["没有发现明显高频搭配或重复问题。"]
    revision_tasks = evidence.revision_tasks or [p.instruction for p in practice_items[:5]]
    if evidence.revision_comparison.has_previous:
        revision_tasks.append(evidence.revision_comparison.improvement_summary)
    return WritingFeedback(
        summary_zh=summary,
        rubric_feedback=rubric_feedback,
        global_feedback=global_feedback[:12],
        local_feedback=local_feedback[:20],
        lexical_feedback=lexical_feedback[:12],
        revision_tasks=revision_tasks[:10],
        generator="template",
    )


def apply_qwen_rubric_scores(evidence: WritingEvidence, parsed: Dict[str, Any]) -> None:
    raw_scores = parsed.get("rubric_scores") or parsed.get("scores")
    if not isinstance(raw_scores, dict):
        return
    scale = max_score_for_exam(evidence.prompt_analysis.exam_type)
    aliases = {
        "task_response": ["task_response", "task", "relevance", "task_achievement"],
        "coherence_cohesion": ["coherence_cohesion", "coherence", "organization", "development_organization"],
        "lexical_resource": ["lexical_resource", "lexical", "vocabulary", "language_use_lexis"],
        "grammar_accuracy": ["grammar_accuracy", "grammatical_range_accuracy", "grammar", "language_control"],
        "overall": ["overall", "estimated_score", "score"],
        "confidence": ["confidence"],
    }
    values: Dict[str, float] = {}
    for target, names in aliases.items():
        for name in names:
            if name in raw_scores:
                try:
                    values[target] = normalize_qwen_score(float(raw_scores[name]), scale)
                    break
                except (TypeError, ValueError):
                    continue
    required = ["task_response", "coherence_cohesion", "lexical_resource", "grammar_accuracy"]
    if not any(k in values for k in required):
        return
    current = evidence.rubric_scores
    task = values.get("task_response", current.task_response)
    coherence = values.get("coherence_cohesion", current.coherence_cohesion)
    lexical = values.get("lexical_resource", current.lexical_resource)
    grammar = values.get("grammar_accuracy", current.grammar_accuracy)
    overall = values.get("overall", (task + coherence + lexical + grammar) / 4.0)
    if is_toefl_exam(evidence.prompt_analysis.exam_type):
        task = round_to_half(clamp_range(task, 0.0, 5.0))
        coherence = round_to_half(clamp_range(coherence, 0.0, 5.0))
        lexical = round_to_half(clamp_range(lexical, 0.0, 5.0))
        grammar = round_to_half(clamp_range(grammar, 0.0, 5.0))
        overall = round_to_half(clamp_range(overall, 0.0, 5.0))
    else:
        task = round_ielts_band(task)
        coherence = round_ielts_band(coherence)
        lexical = round_ielts_band(lexical)
        grammar = round_ielts_band(grammar)
        overall = round_ielts_band(overall)
    confidence = clamp_range(values.get("confidence", current.confidence), 0.0, 1.0)
    evidence.raw["rule_rubric_scores"] = model_to_dict(current)
    evidence.raw["qwen_rubric_scores"] = raw_scores
    evidence.rubric_scores = RubricScores(
        task_response=task,
        coherence_cohesion=coherence,
        lexical_resource=lexical,
        grammar_accuracy=grammar,
        overall=overall,
        confidence=round(confidence, 2),
    )


def normalize_qwen_score(value: float, scale: float) -> float:
    if scale == 5.0 and 5.0 < value <= 10.0:
        return value / 2.0
    if scale == 9.0 and value > 9.0:
        return min(9.0, value)
    return value


def split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " ").strip())
    return [p.strip() for p in parts if p.strip()]


def tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+(?:\.[0-9]+)?", text)


def extract_keywords(text: str, limit: int = 10) -> List[str]:
    words = [normalize_word(w) for w in tokenize_words(text)]
    words = [w for w in words if w and w not in FUNCTION_WORDS and len(w) > 3]
    out: List[str] = []
    for word in words:
        if word not in out:
            out.append(word)
        if len(out) >= limit:
            break
    return out


def prompt_coverage(keywords: List[str], essay: str) -> float:
    if not keywords:
        return 1.0
    essay_words = {normalize_word(w) for w in tokenize_words(essay)}
    return len([k for k in keywords if k in essay_words]) / max(1, len(keywords))


def repeated_topic_sentences(topic_sentences: List[str]) -> List[str]:
    repeated: List[str] = []
    for i, a in enumerate(topic_sentences):
        for b in topic_sentences[i + 1 :]:
            if text_overlap(a, b) > 0.55:
                repeated.append(f"{a} / {b}")
    return repeated


def has_example(text: str) -> bool:
    return bool(re.search(r"\b(for example|for instance|such as|in my case|a study|research shows)\b", text, re.I))


def has_transition_words_from_text(stats: EssayStats, structure_issues: List[StructureIssue]) -> bool:
    return stats.paragraph_count >= 3 and not any(x.type == "weak_paragraphing" for x in structure_issues)


def looks_like_fragment(sentence: str) -> bool:
    words = [normalize_word(w) for w in tokenize_words(sentence)]
    if len(words) < 3:
        return False
    finite = {"is", "are", "was", "were", "be", "been", "am", "do", "does", "did", "have", "has", "had", "will", "would", "can", "could", "should", "may", "might", "must"}
    has_verbish = any(w in finite or w.endswith("ed") or w.endswith("s") for w in words[1:])
    starts_sub = words[0] in {"although", "because", "while", "when", "if"}
    return starts_sub and not re.search(r",\s*\w+", sentence) or not has_verbish and len(words) > 6


def repeated_words(essay: str) -> List[Tuple[str, int]]:
    words = [normalize_word(w) for w in tokenize_words(essay)]
    words = [w for w in words if w and w not in FUNCTION_WORDS and len(w) > 4]
    counts: Dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    threshold = 4 if len(words) > 180 else 3
    return sorted([(w, c) for w, c in counts.items() if c >= threshold], key=lambda x: (-x[1], x[0]))


def text_overlap(a: str, b: str) -> float:
    aw = {normalize_word(w) for w in tokenize_words(a) if normalize_word(w) not in FUNCTION_WORDS}
    bw = {normalize_word(w) for w in tokenize_words(b) if normalize_word(w) not in FUNCTION_WORDS}
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / len(aw | bw)


def paragraph_count(text: str) -> int:
    return len(split_paragraphs(text))


def clamp_score(value: float) -> float:
    return max(0.0, min(10.0, value))


def unique(values: List[str]) -> List[str]:
    out: List[str] = []
    for value in values:
        value = value.strip()
        if value and value not in out:
            out.append(value)
    return out


def list_of_strings(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(x) for x in value if str(x).strip()]


def stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"
