"""Evidence-grounded feedback generation.

Functional module 5/7.

This module deliberately separates *evidence* from *wording*. Qwen2-Audio can
be used to turn structured evidence into learner-friendly explanations, but it
is instructed not to add new errors that are absent from the evidence JSON. A
fully deterministic template fallback is included for reproducibility and tests.
"""
from __future__ import annotations

import html
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .contracts import (
    AlignmentResult,
    AsrResult,
    CaptSettings,
    EvidenceBundle,
    EvidenceIssue,
    FeedbackItem,
    FeedbackReport,
    Severity,
    model_to_dict,
)
from .deterministic_scoring import load_score_config, score_evidence

logger = logging.getLogger(__name__)


class FeedbackGenerationError(RuntimeError):
    pass


class EvidenceGroundedFeedbackGenerator:
    def __init__(self, settings: CaptSettings):
        self.settings = settings
        self._qwen_processor = None
        self._qwen_model = None

    def generate(
        self,
        reference_text: str,
        audio_path: str | Path,
        asr: AsrResult,
        alignment: AlignmentResult,
        evidence: EvidenceBundle,
    ) -> FeedbackReport:
        template_report = self._template_feedback(reference_text, asr, alignment, evidence)
        if self.settings.use_qwen_text:
            self._attach_qwen_rubric_scores(template_report, reference_text, asr, alignment, evidence)
            try:
                llm_raw = self._qwen_text_feedback(reference_text, asr, alignment, evidence)
                parsed = self._parse_llm_json(llm_raw)
                if parsed:
                    return self._merge_llm_with_template(parsed, template_report, llm_raw, generator="qwen_text_evidence_grounded")
                template_report.llm_raw = llm_raw
                template_report.generator = "template+qwen_text_raw_unparsed"
                return template_report
            except Exception as exc:
                logger.warning("Qwen text feedback failed; trying next fallback: %s", exc)
                template_report.llm_raw = f"Qwen text failed: {exc}"

        if not self.settings.use_qwen2_audio:
            return template_report
        try:
            llm_raw = self._qwen2_audio_feedback(reference_text, audio_path, asr, alignment, evidence)
            parsed = self._parse_llm_json(llm_raw)
            if parsed:
                return self._merge_llm_with_template(parsed, template_report, llm_raw, generator="qwen2_audio_evidence_grounded")
            template_report.llm_raw = llm_raw
            template_report.generator = "template+qwen2_audio_raw_unparsed"
            return template_report
        except Exception as exc:
            logger.warning("Qwen2-Audio feedback failed; using template fallback: %s", exc)
            template_report.llm_raw = f"Qwen2-Audio failed: {exc}"
            template_report.generator = "template_after_qwen2_audio_failure"
            return template_report

    def _template_feedback(self, reference_text: str, asr: AsrResult, alignment: AlignmentResult, evidence: EvidenceBundle) -> FeedbackReport:
        all_issues = evidence.all_issues
        scores = compute_scores(evidence, self.settings, alignment)
        items: List[FeedbackItem] = []
        for issue in all_issues[:40]:
            items.append(
                FeedbackItem(
                    issue_id=issue.issue_id,
                    display_text=format_issue_display(issue),
                    practice=practice_examples(issue),
                    explanation_zh=issue.suggestion_hint or "建议放慢速度，重新听标准发音后再练习。",
                    explanation_en=simple_english_explanation(issue),
                )
            )
        if not all_issues:
            summary_zh = "这次朗读没有检测到明显的词级、音素级或韵律问题。建议继续保持，并尝试更自然的语速和语调。"
            summary_en = "No major word-, phoneme-, or prosody-level issues were detected. Keep practicing natural speed and intonation."
        else:
            by_type = count_by_type(all_issues)
            summary_zh = (
                f"本次共检测到 {len(all_issues)} 个可解释证据点。"
                f"其中词级 {by_type.get('word', 0)} 个，音素级 {by_type.get('phoneme', 0)} 个，"
                f"停顿/连读 {by_type.get('pause_linking', 0)} 个，重音/语调 {by_type.get('prosody', 0)} 个。"
                "建议先处理高严重度和高置信度问题，再练自然度问题。"
            )
            summary_en = (
                f"Detected {len(all_issues)} evidence-grounded issues. "
                "Prioritize high-severity and high-confidence issues before naturalness suggestions."
            )
        return FeedbackReport(
            summary_zh=summary_zh,
            summary_en=summary_en,
            overall_scores=scores,
            annotated_text_html=build_annotated_html(alignment, evidence),
            items=items,
            generator="template",
        )

    def _qwen_text_feedback(self, reference_text: str, asr: AsrResult, alignment: AlignmentResult, evidence: EvidenceBundle) -> str:
        if not self.settings.qwen_api_key:
            raise FeedbackGenerationError("QWEN_API_KEY is required when CAPT_USE_QWEN_TEXT=1")

        evidence_json = build_evidence_prompt_json(reference_text, asr, alignment, evidence)
        if self.settings.qwen_feedback_mode == "batch":
            return self._qwen_text_feedback_batch(evidence_json)

        return self._qwen_text_feedback_per_issue(evidence_json)

    def _qwen_text_feedback_batch(self, evidence_json: Dict[str, Any]) -> str:
        system = (
            "You are an evidence-grounded English pronunciation feedback generator. "
            "You must not invent errors. You may only explain evidence items whose issue_id appears in the JSON. "
            "Return valid JSON only."
        )
        user = (
            "请基于 evidence_json 生成中文为主、英文辅助的发音反馈。"
            "必须输出 JSON: {summary_zh, summary_en, items:[{issue_id, display_text, explanation_zh, explanation_en, practice}]}。"
            "不要新增 evidence_json 中不存在的错误；低置信度问题要用'可能'或'建议'表达。\n\n"
            f"evidence_json={json.dumps(evidence_json, ensure_ascii=False)}"
        )
        return self._qwen_chat_completion(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=self.settings.qwen_summary_max_tokens + self.settings.qwen_issue_max_tokens,
        )

    def _qwen_text_feedback_per_issue(self, evidence_json: Dict[str, Any]) -> str:
        """Ask Qwen to rewrite each issue, then ask for one global summary.

        Detection, scoring, and ordering still come from deterministic evidence.
        Qwen only rewrites learner-facing wording for existing issue IDs.
        """

        issues = list(evidence_json.get("issues", []))[: max(0, self.settings.qwen_max_feedback_issues)]
        items: List[Dict[str, Any]] = []
        raw_calls: List[Dict[str, str]] = []
        system = (
            "你是证据约束的英语发音反馈改写器。你只能解释输入 JSON 中给出的一个 pronunciation issue，"
            "不能新增错误，不能改变 issue_id、type、expected、observed、confidence。"
            "如果证据不足，请使用'可能'、'建议'等保守表达。只输出 JSON。"
        )
        for issue in issues:
            payload = {
                "reference_text": evidence_json.get("reference_text", ""),
                "asr_text": evidence_json.get("asr_text", ""),
                "scores": evidence_json.get("scores", {}),
                "issue": issue,
                "local_context": local_context_for_issue(evidence_json.get("reference_text", ""), issue),
                "output_schema": {
                    "issue_id": "same as issue.issue_id",
                    "display_text": "problem word or phrase plus a concise evidence label",
                    "explanation_zh": "2-3 learner-friendly Chinese sentences grounded in the evidence",
                    "explanation_en": "one short English sentence",
                    "practice": ["2-4 concrete practice steps"],
                },
            }
            user = (
                "请只针对 payload.issue 生成反馈。"
                "display_text 要保留问题单词/短语；explanation_zh 要说明出现了什么错误、为什么会影响听感；"
                "practice 要给出具体练习方式，例如单音素、慢读、词组、句中回放。"
                "不要输出 markdown，不要解释 JSON 以外的内容。\n\n"
                f"payload={json.dumps(payload, ensure_ascii=False)}"
            )
            try:
                raw = self._qwen_chat_completion(
                    [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    max_tokens=self.settings.qwen_issue_max_tokens,
                )
                raw_calls.append({"issue_id": str(issue.get("issue_id", "")), "raw": raw[:2000]})
                parsed = self._parse_llm_json(raw) or {}
                item = sanitize_qwen_item(parsed, str(issue.get("issue_id", "")))
                if item:
                    items.append(item)
            except Exception as exc:
                logger.warning("Qwen issue feedback failed for %s: %s", issue.get("issue_id"), exc)
                raw_calls.append({"issue_id": str(issue.get("issue_id", "")), "error": str(exc)[:1000]})

        summary = self._qwen_summary_from_evidence(evidence_json, items, raw_calls)
        result = {
            "summary_zh": summary.get("summary_zh", ""),
            "summary_en": summary.get("summary_en", ""),
            "items": items,
            "raw_calls": raw_calls,
        }
        return json.dumps(result, ensure_ascii=False)

    def _qwen_summary_from_evidence(
        self,
        evidence_json: Dict[str, Any],
        items: List[Dict[str, Any]],
        raw_calls: List[Dict[str, str]],
    ) -> Dict[str, str]:
        system = (
            "你是证据约束的英语发音综合评价生成器。只能基于输入 evidence_json 和已生成的 items 总结，"
            "不能新增没有证据的错误。只输出 JSON。"
        )
        summary_payload = {
            "reference_text": evidence_json.get("reference_text", ""),
            "asr_text": evidence_json.get("asr_text", ""),
            "scores": evidence_json.get("scores", {}),
            "prosody": evidence_json.get("prosody", {}),
            "issue_counts": issue_counts_from_prompt_json(evidence_json),
            "top_issues": evidence_json.get("issues", [])[:10],
            "generated_items": items[:10],
            "output_schema": {
                "summary_zh": "2-4 Chinese sentences with priority suggestions",
                "summary_en": "1-2 short English sentences",
            },
        }
        user = (
            "请在完整 evidence 的基础上输出综合评价。"
            "需要说明主要问题类别、优先练习顺序，以及哪些方面表现较好。"
            "不要输出 markdown，不要输出 JSON 以外内容。\n\n"
            f"payload={json.dumps(summary_payload, ensure_ascii=False)}"
        )
        try:
            raw = self._qwen_chat_completion(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=self.settings.qwen_summary_max_tokens,
            )
            raw_calls.append({"issue_id": "__summary__", "raw": raw[:2000]})
            parsed = self._parse_llm_json(raw) or {}
            return {
                "summary_zh": str(parsed.get("summary_zh") or ""),
                "summary_en": str(parsed.get("summary_en") or ""),
            }
        except Exception as exc:
            logger.warning("Qwen summary feedback failed: %s", exc)
            raw_calls.append({"issue_id": "__summary__", "error": str(exc)[:1000]})
            return {"summary_zh": "", "summary_en": ""}

    def _qwen_chat_completion(self, messages: List[Dict[str, str]], max_tokens: int | None = None) -> str:
        import urllib.error
        import urllib.request

        payload: Dict[str, Any] = {
            "model": self.settings.qwen_model,
            "messages": messages,
            "temperature": self.settings.qwen_temperature,
            "response_format": {"type": "json_object"},
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        url = self.settings.qwen_base_url.rstrip("/") + "/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.settings.qwen_api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise FeedbackGenerationError(f"Qwen API HTTP {exc.code}: {exc.read().decode('utf-8', errors='ignore')[:1000]}") from exc
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    def _qwen2_audio_feedback(
        self,
        reference_text: str,
        audio_path: str | Path,
        asr: AsrResult,
        alignment: AlignmentResult,
        evidence: EvidenceBundle,
    ) -> str:
        try:
            import librosa  # type: ignore
            import torch  # type: ignore
            from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration  # type: ignore
        except Exception as exc:
            raise FeedbackGenerationError("transformers, torch and librosa are required for Qwen2-Audio feedback") from exc

        if self._qwen_processor is None or self._qwen_model is None:
            self._qwen_processor = AutoProcessor.from_pretrained(self.settings.qwen2_audio_model)
            self._qwen_model = Qwen2AudioForConditionalGeneration.from_pretrained(
                self.settings.qwen2_audio_model,
                device_map="auto",
            )
        processor = self._qwen_processor
        model = self._qwen_model
        evidence_json = build_evidence_prompt_json(reference_text, asr, alignment, evidence)
        prompt = (
            "你是英语发音训练系统的反馈生成器。你必须严格基于下面的 evidence_json 给反馈，"
            "不能新增 evidence_json 中不存在的错误。请输出 JSON，不要输出 Markdown。"
            "JSON schema: {summary_zh:string, summary_en:string, items:[{issue_id:string, display_text:string, "
            "explanation_zh:string, explanation_en:string, practice:[string]}]}。\n\n"
            f"evidence_json:\n{json.dumps(evidence_json, ensure_ascii=False)}"
        )
        conversation = [
            {"role": "system", "content": "You generate evidence-grounded pronunciation feedback. Do not hallucinate."},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": "local_audio.wav"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audio, _ = librosa.load(str(audio_path), sr=processor.feature_extractor.sampling_rate, mono=True)
        inputs = processor(text=text, audios=[audio], return_tensors="pt", padding=True)
        # Move tensors to the first model device where possible. device_map=auto
        # can shard modules; input_ids on cuda is enough for common setups, while
        # CPU fallback remains valid.
        if torch.cuda.is_available() and self.settings.device == "cuda":
            inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
        generated = model.generate(**inputs, max_new_tokens=self.settings.qwen2_max_new_tokens)
        input_len = inputs["input_ids"].shape[1]
        generated = generated[:, input_len:]
        return processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def _parse_llm_json(self, raw: str) -> Optional[Dict[str, Any]]:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    def _merge_llm_with_template(self, parsed: Dict[str, Any], template: FeedbackReport, llm_raw: str, generator: str = "qwen_evidence_grounded") -> FeedbackReport:
        template_by_id = {item.issue_id: item for item in template.items}
        rewritten_by_id: Dict[str, FeedbackItem] = {}
        for item in parsed.get("items", []) or []:
            issue_id = str(item.get("issue_id", ""))
            base = template_by_id.get(issue_id)
            if base is None:
                # Enforce evidence grounding: unknown issue IDs are discarded.
                continue
            rewritten_by_id[issue_id] = FeedbackItem(
                issue_id=issue_id,
                display_text=str(item.get("display_text") or base.display_text),
                explanation_zh=str(item.get("explanation_zh") or base.explanation_zh),
                explanation_en=str(item.get("explanation_en") or base.explanation_en),
                practice=[str(x) for x in (item.get("practice") or base.practice)[:5]],
            )
        items = [rewritten_by_id.get(item.issue_id, item) for item in template.items]
        return FeedbackReport(
            summary_zh=str(parsed.get("summary_zh") or template.summary_zh),
            summary_en=str(parsed.get("summary_en") or template.summary_en),
            overall_scores=template.overall_scores,
            qwen_rubric_scores=template.qwen_rubric_scores,
            qwen_rubric_rationale=template.qwen_rubric_rationale,
            qwen_rubric_raw=template.qwen_rubric_raw,
            annotated_text_html=template.annotated_text_html,
            items=items,
            llm_raw=llm_raw,
            generator=generator,
        )

    def _attach_qwen_rubric_scores(
        self,
        report: FeedbackReport,
        reference_text: str,
        asr: AsrResult,
        alignment: AlignmentResult,
        evidence: EvidenceBundle,
    ) -> None:
        if not self.settings.qwen_api_key:
            return
        try:
            raw = self._qwen_text_rubric_scores(reference_text, asr, alignment, evidence)
            parsed = self._parse_llm_json(raw) or {}
            scores, rationale = sanitize_qwen_rubric_scores(parsed)
            if scores:
                report.qwen_rubric_scores = scores
                report.qwen_rubric_rationale = rationale
                report.qwen_rubric_raw = raw[:4000]
        except Exception as exc:
            logger.warning("Qwen rubric scoring failed; keeping ratio scores only: %s", exc)

    def _qwen_text_rubric_scores(
        self,
        reference_text: str,
        asr: AsrResult,
        alignment: AlignmentResult,
        evidence: EvidenceBundle,
    ) -> str:
        evidence_json = build_evidence_prompt_json(reference_text, asr, alignment, evidence)
        payload = {
            "task": "speech_ocean762_rubric_scoring",
            "reference_text": reference_text,
            "asr_text": asr.text,
            "ratio_scores_0_100": evidence_json.get("scores", {}),
            "prosody": evidence_json.get("prosody", {}),
            "issues": evidence_json.get("issues", []),
            "phoneme_hypotheses": evidence_json.get("phoneme_hypotheses", []),
            "rubric": SPEECHOCEAN762_RUBRIC_TEXT,
            "output_schema": {
                "accuracy": "0-10 number",
                "phoneme": "0-10 number",
                "fluency": "0-10 number",
                "prosody": "0-10 number",
                "rationale": {
                    "accuracy": "short evidence-grounded reason",
                    "phoneme": "short evidence-grounded reason",
                    "fluency": "short evidence-grounded reason",
                    "prosody": "short evidence-grounded reason",
                },
            },
        }
        system = (
            "You are a strict SpeechOcean762 pronunciation rubric scorer. "
            "Use only the provided evidence. Do not invent errors. "
            "Return valid JSON only. Scores must be 0-10 numbers; do not include markdown."
        )
        user = (
            "Score this pronunciation attempt using the textual SpeechOcean762 rubric and the evidence payload. "
            "The deterministic ratio scores are provided for reference, but your rubric scores should follow the rubric bands. "
            "Do not score issues that are not present in the evidence.\n\n"
            f"payload={json.dumps(payload, ensure_ascii=False)}"
        )
        return self._qwen_chat_completion(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=self.settings.qwen_summary_max_tokens,
        )


# ---------- Template/report helpers ----------


SPEECHOCEAN762_RUBRIC_TEXT = """
SpeechOcean762 manual scoring rubric, summarized as text:

Phoneme-level accuracy:
- 2: the phone is pronounced correctly.
- 1: the phone is pronounced with a heavy accent.
- 0: the pronunciation is incorrect or missed.

Word-level accuracy:
- 10: the pronunciation of the whole word is correct.
- 7-9: most phones in the word are correct, but the word has a heavy accent.
- 4-6: no more than 30% of phones in the word are wrongly pronounced.
- 2-3: more than 30% of phones are wrong, or the word is mispronounced as another word.
- 0-1: the whole pronunciation is hard to distinguish, or the word is missed.

Word-level stress:
- 10: stress position is correct, or the word is monosyllabic.
- 5: stress position is incorrect.

Sentence-level accuracy:
- 9-10: excellent overall sentence pronunciation without obvious mispronunciation.
- 7-8: good overall sentence pronunciation with a few mispronunciations.
- 5-6: many mispronunciations, but the sentence is still understandable.
- 3-4: awkward pronunciation with many serious mispronunciations.
- 0-2: the whole sentence is hard to understand, or there is no voice.

Sentence-level fluency:
- 8-10: coherent speech without noticeable pauses, repetition, or stammering.
- 6-7: generally coherent speech with a few pauses, repetitions, or stammering.
- 4-5: incoherent speech with many pauses, repetitions, or stammering.
- 0-3: the speaker cannot read the sentence as a whole, or there is no voice.

Sentence-level prosody:
- 9-10: correct intonation, stable speaking speed and rhythm.
- 7-8: nearly correct intonation at a stable speaking speed.
- 3-6: unstable speech speed, or inappropriate intonation.
- 0-2: too much stammering for prosodic scoring, or there is no voice.
""".strip()


def compute_scores(
    evidence: EvidenceBundle,
    settings: CaptSettings | None = None,
    alignment: AlignmentResult | None = None,
) -> Dict[str, float]:
    """Return continuous deterministic scores on a 0--100 display scale.

    All values come from fixed evidence transforms and YAML weights. Qwen is
    not involved. The legacy ratio helpers remain below for reading old
    reports, but new reports use continuous_v2.
    """

    config_path = settings.deterministic_score_config if settings is not None else None
    config = load_score_config(config_path)
    mode = "modeB" if isinstance(evidence.continuous_features.get("mode_b"), dict) else "modeA"
    return score_evidence(evidence, alignment, mode=mode, config=config).scores


def reference_word_count(evidence: EvidenceBundle, alignment: AlignmentResult | None = None) -> int:
    if alignment is not None and alignment.reference_tokens:
        return len(alignment.reference_tokens)
    transcript = evidence.textpa.transcript or ""
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", transcript)
    if words:
        return len(words)
    refs = [issue.ref_index for issue in evidence.all_issues if issue.ref_index is not None]
    return max(refs) + 1 if refs else 0


def ratio_accuracy_score(evidence: EvidenceBundle, ref_words: int) -> float:
    reference_problem_words = {
        int(issue.ref_index)
        for issue in evidence.word_issues + evidence.phoneme_issues
        if issue.ref_index is not None
    }
    extra_units = sum(1 for issue in evidence.word_issues + evidence.phoneme_issues if issue.ref_index is None)
    return ratio_percent(ref_words, len(reference_problem_words) + extra_units)


def ratio_phoneme_score(evidence: EvidenceBundle, ref_words: int, accuracy_score: float) -> float:
    if evidence.phoneme_hypotheses:
        analyzed_count = len(
            {
                (h.word, round(float(h.start or 0.0), 2), round(float(h.end or 0.0), 2))
                for h in evidence.phoneme_hypotheses
            }
        )
        problem_units = issue_unit_count(evidence.phoneme_issues)
        analyzed_score = ratio_percent(analyzed_count, problem_units)
        if ref_words <= 0:
            return analyzed_score
        coverage = max(0.0, min(1.0, analyzed_count / max(1, ref_words)))
        # Phone recognition only covers a subset of the sentence, especially
        # because mismatched ASR/reference words are skipped. Treat the
        # unanalysed part as falling back to Accuracy instead of assuming 100.
        return round(accuracy_score * (1.0 - coverage) + analyzed_score * coverage, 1)
    elif evidence.phoneme_issues:
        denominator = ref_words if ref_words > 0 else len(evidence.phoneme_issues)
    else:
        return accuracy_score if evidence.word_issues else 100.0

    problem_units = issue_unit_count(evidence.phoneme_issues)
    return ratio_percent(denominator, problem_units)


def ratio_fluency_score(
    evidence: EvidenceBundle,
    alignment: AlignmentResult | None,
    ref_words: int,
) -> float:
    boundary_units = evaluable_boundary_count(alignment, ref_words)
    rhythm_units = fluency_rhythm_unit_count(evidence)
    denominator = boundary_units + rhythm_units
    problems = set()
    for issue in evidence.pause_issues + evidence.linking_issues:
        problems.add(boundary_issue_key(issue))
    for issue in evidence.rhythm_issues:
        problems.add(rhythm_issue_key(issue))
    return ratio_percent(max(denominator, len(problems)), len(problems))


def ratio_prosody_score(
    evidence: EvidenceBundle,
    alignment: AlignmentResult | None,
    ref_words: int,
) -> float:
    stress_units = evaluable_stress_unit_count(alignment)
    sentence_units = evaluable_sentence_count(alignment, ref_words)
    rhythm_units = prosody_rhythm_unit_count(evidence)
    denominator = stress_units + sentence_units + sentence_units + rhythm_units
    problems = set()
    for issue in evidence.stress_issues:
        problems.add(("stress", issue_word_key(issue)))
    for issue in evidence.sentence_stress_issues:
        problems.add(("sentence_stress", sentence_issue_key(issue)))
    for issue in evidence.intonation_issues:
        problems.add(("intonation", sentence_issue_key(issue)))
    for issue in evidence.rhythm_issues:
        problems.add(("rhythm", rhythm_issue_key(issue)))
    return ratio_percent(max(denominator, len(problems)), len(problems))


def ratio_percent(total_units: int, problem_units: int) -> float:
    total_units = max(0, int(total_units))
    problem_units = max(0, int(problem_units))
    if total_units <= 0:
        return 0.0 if problem_units else 100.0
    correct_ratio = 1.0 - min(problem_units, total_units) / total_units
    return round(max(0.0, min(1.0, correct_ratio)) * 100.0, 1)


def issue_unit_count(issues: List[EvidenceIssue]) -> int:
    return len({issue_word_key(issue) for issue in issues})


def issue_word_key(issue: EvidenceIssue) -> Any:
    if issue.ref_index is not None:
        return ("ref", int(issue.ref_index))
    if issue.word:
        return ("word", str(issue.word).lower(), round(float(issue.start or 0.0), 2), round(float(issue.end or 0.0), 2))
    return ("issue", issue.issue_id)


def boundary_issue_key(issue: EvidenceIssue) -> Any:
    if issue.ref_index is not None:
        return ("boundary_after", int(issue.ref_index))
    if issue.word:
        return ("boundary_text", str(issue.word).lower())
    return ("boundary_issue", issue.issue_id)


def sentence_issue_key(issue: EvidenceIssue) -> Any:
    sentence_index = issue.evidence.get("sentence_index") if isinstance(issue.evidence, dict) else None
    if sentence_index is not None:
        return int(sentence_index)
    if issue.ref_index is not None:
        return int(issue.ref_index)
    return issue.issue_id


def rhythm_issue_key(issue: EvidenceIssue) -> Any:
    evidence = issue.evidence if isinstance(issue.evidence, dict) else {}
    for key in ("sentence_index", "paired_word_count"):
        if evidence.get(key) is not None:
            return (key, evidence[key])
    return issue.word or issue.issue_id


def evaluable_boundary_count(alignment: AlignmentResult | None, ref_words: int) -> int:
    if alignment is not None and alignment.reference_tokens:
        return max(0, len(alignment.reference_tokens) - 1)
    return max(0, ref_words - 1)


def evaluable_sentence_count(alignment: AlignmentResult | None, ref_words: int) -> int:
    if alignment is not None and alignment.reference_tokens:
        return max(1, len({int(tok.sentence_index) for tok in alignment.reference_tokens}))
    return 1 if ref_words > 0 else 0


def evaluable_stress_unit_count(alignment: AlignmentResult | None) -> int:
    if alignment is None:
        return 0
    return sum(
        1
        for tok in alignment.reference_tokens
        if tok.is_content_word and len(tok.stress_pattern) >= 2 and 1 in tok.stress_pattern
    )


def fluency_rhythm_unit_count(evidence: EvidenceBundle) -> int:
    metrics = [
        evidence.prosody.pause_ratio,
        evidence.prosody.mean_length_run_words,
        evidence.prosody.speech_rate_wpm,
    ]
    return sum(1 for value in metrics if value is not None)


def prosody_rhythm_unit_count(evidence: EvidenceBundle) -> int:
    metrics = [
        evidence.prosody.npvi_word_duration,
        evidence.prosody.word_duration_cv,
        evidence.prosody.speech_rate_wpm,
    ]
    return sum(1 for value in metrics if value is not None)


def count_by_type(issues: List[EvidenceIssue]) -> Dict[str, int]:
    out = {"word": 0, "phoneme": 0, "pause_linking": 0, "prosody": 0}
    for issue in issues:
        t = issue.type.value
        if t in {"word_omission", "word_insertion", "word_substitution"}:
            out["word"] += 1
        elif t.startswith("phoneme_"):
            out["phoneme"] += 1
        elif "pause" in t or "linking" in t or t == "rhythm":
            out["pause_linking"] += 1
        elif t in {"word_stress", "sentence_stress", "intonation"}:
            out["prosody"] += 1
        else:
            out["prosody"] += 1
    return out


def format_issue_display(issue: EvidenceIssue) -> str:
    word = issue.word or ""
    if issue.expected and issue.observed:
        return f"{word}: expected {issue.expected}, observed {issue.observed} ({issue.type.value}, confidence={issue.confidence:.2f})"
    return f"{word}: {issue.type.value} (confidence={issue.confidence:.2f})"


def simple_english_explanation(issue: EvidenceIssue) -> str:
    if issue.expected and issue.observed:
        return f"For '{issue.word}', the expected target is {issue.expected}, but the detected evidence suggests {issue.observed}."
    return f"Review the pronunciation around '{issue.word}'."


def sanitize_qwen_item(parsed: Dict[str, Any], expected_issue_id: str) -> Optional[Dict[str, Any]]:
    item = parsed.get("item") if isinstance(parsed.get("item"), dict) else parsed
    issue_id = str(item.get("issue_id") or expected_issue_id)
    if issue_id != expected_issue_id:
        return None
    practice = item.get("practice") or []
    if isinstance(practice, str):
        practice = [practice]
    return {
        "issue_id": issue_id,
        "display_text": str(item.get("display_text") or ""),
        "explanation_zh": str(item.get("explanation_zh") or ""),
        "explanation_en": str(item.get("explanation_en") or ""),
        "practice": [str(x) for x in practice[:5]],
    }


def sanitize_qwen_rubric_scores(parsed: Dict[str, Any]) -> tuple[Dict[str, float], Dict[str, str]]:
    raw_scores = parsed.get("rubric_scores") if isinstance(parsed.get("rubric_scores"), dict) else None
    if raw_scores is None:
        raw_scores = parsed.get("scores") if isinstance(parsed.get("scores"), dict) else parsed
    scores: Dict[str, float] = {}
    for key in ("accuracy", "phoneme", "fluency", "prosody"):
        value = raw_scores.get(key) if isinstance(raw_scores, dict) else None
        if value is None:
            continue
        try:
            score_10 = float(value)
        except Exception:
            continue
        if score_10 > 10.0:
            score_10 = score_10 / 10.0
        scores[key] = round(max(0.0, min(10.0, score_10)) * 10.0, 1)
    if len(scores) == 4:
        scores["overall"] = round(sum(scores.values()) / 4.0, 1)
    rationale_raw = parsed.get("rationale") if isinstance(parsed.get("rationale"), dict) else {}
    rationale = {key: str(rationale_raw.get(key) or "") for key in ("accuracy", "phoneme", "fluency", "prosody")}
    return scores, rationale


def local_context_for_issue(reference_text: str, issue: Dict[str, Any], window: int = 4) -> str:
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", reference_text)
    ref_index = issue.get("ref_index")
    if isinstance(ref_index, int) and words:
        start = max(0, ref_index - window)
        end = min(len(words), ref_index + window + 1)
        return " ".join(words[start:end])
    word = str(issue.get("word") or "").strip()
    if word:
        pattern = re.compile(rf"(.{{0,80}}{re.escape(word)}.{{0,80}})", re.I)
        match = pattern.search(reference_text)
        if match:
            return match.group(1).strip()
    return reference_text[:200]


def issue_counts_from_prompt_json(evidence_json: Dict[str, Any]) -> Dict[str, int]:
    counts = {"word": 0, "phoneme": 0, "pause_linking": 0, "prosody": 0}
    for issue in evidence_json.get("issues", []) or []:
        issue_type = str(issue.get("type") or "")
        if issue_type in {"word_omission", "word_insertion", "word_substitution"}:
            counts["word"] += 1
        elif issue_type.startswith("phoneme_"):
            counts["phoneme"] += 1
        elif "pause" in issue_type or "linking" in issue_type or issue_type == "rhythm":
            counts["pause_linking"] += 1
        elif "stress" in issue_type or "intonation" in issue_type:
            counts["prosody"] += 1
    return counts


def practice_examples(issue: EvidenceIssue) -> List[str]:
    word = issue.word or ""
    if issue.type.value.startswith("phoneme") and issue.expected:
        return [f"/{issue.expected}/", word, f"slow: {word}", f"normal: {word}"]
    if issue.type.value == "linking_missing" and word:
        return [word, word.replace(" ", "‿")]
    if issue.type.value.startswith("pause"):
        return ["read once slowly with punctuation pauses", "read again at natural speed"]
    if word:
        return [word, f"slow: {word}", f"sentence context: ... {word} ..."]
    return []


def build_annotated_html(alignment: AlignmentResult, evidence: EvidenceBundle) -> str:
    by_ref: Dict[int, List[EvidenceIssue]] = {}
    for issue in evidence.all_issues:
        if issue.ref_index is not None:
            by_ref.setdefault(issue.ref_index, []).append(issue)
    parts: List[str] = []
    for tok in alignment.reference_tokens:
        issues = by_ref.get(tok.index, [])
        classes = ["token"]
        if issues:
            severities = {i.severity.value for i in issues}
            classes.append("issue")
            if "high" in severities:
                classes.append("high")
            elif "medium" in severities:
                classes.append("medium")
            else:
                classes.append("low")
        title = "\n".join(format_issue_display(i) for i in issues)
        span = (
            f'<span class="{" ".join(classes)}" data-index="{tok.index}" '
            f'data-issues="{html.escape(json.dumps([i.issue_id for i in issues], ensure_ascii=False))}" '
            f'title="{html.escape(title)}">{html.escape(tok.raw)}</span>'
        )
        parts.append(html.escape(tok.before_punct) + span + html.escape(tok.after_punct))
    # Insertions are not tied to reference words; add a compact note.
    insertions = [i for i in evidence.word_issues if i.ref_index is None]
    if insertions:
        ins = "; ".join(html.escape(format_issue_display(i)) for i in insertions)
        parts.append(f'<div class="insertion-note">Extra words detected: {ins}</div>')
    return " ".join(parts)


def build_evidence_prompt_json(reference_text: str, asr: AsrResult, alignment: AlignmentResult, evidence: EvidenceBundle) -> Dict[str, Any]:
    issues = sorted(evidence.all_issues, key=lambda x: (severity_rank(x.severity), -x.confidence), reverse=True)[:30]
    return {
        "reference_text": reference_text,
        "asr_text": asr.text,
        "scores": compute_scores(evidence, alignment=alignment),
        "prosody": model_to_dict(evidence.prosody),
        "textpa_cues": model_to_dict(evidence.textpa),
        "issues": [model_to_dict(i) for i in issues],
        "phoneme_hypotheses": [model_to_dict(h) for h in evidence.phoneme_hypotheses[:20]],
        "rules": [
            "Only discuss issues listed in issues[].",
            "Do not invent additional pronunciation errors.",
            "Use Chinese explanations with short English labels.",
            "Prioritize high severity and high confidence items.",
        ],
    }


def severity_rank(sev: Severity) -> int:
    return {Severity.INFO: 0, Severity.LOW: 1, Severity.MEDIUM: 2, Severity.HIGH: 3}.get(sev, 0)
