"""Template and Qwen explainers for Mode C."""

from __future__ import annotations

import json
import logging
import re
import socket
from typing import Any, Dict, List, Optional

from ..contracts import CaptSettings, model_to_dict
from .schemas import LongSentenceEvidence, LongSentenceExplanation, TranslationChunk

logger = logging.getLogger(__name__)


class LongSentenceExplainer:
    def __init__(self, settings: CaptSettings):
        self.settings = settings

    def explain(self, evidence: LongSentenceEvidence) -> LongSentenceExplanation:
        template = self.template_explanation(evidence)
        if not self.settings.use_qwen_text:
            return template
        if not self.settings.qwen_api_key:
            template.llm_raw = "Qwen disabled: QWEN_API_KEY is not set."
            return template
        try:
            raw = self._qwen_explanation(evidence)
            parsed = parse_json_object(raw)
            if not parsed:
                template.llm_raw = raw
                template.generator = "template+qwen_raw_unparsed"
                return template
            return merge_qwen_explanation(parsed, template, raw)
        except Exception as exc:
            logger.warning("Mode C Qwen explanation failed; using template fallback: %s", exc)
            template.llm_raw = f"Qwen failed: {exc}"
            template.generator = "template_after_qwen_failure"
            return template

    def template_explanation(self, evidence: LongSentenceEvidence) -> LongSentenceExplanation:
        main = evidence.main_clause
        main_steps = [
            f"主语：{main.subject or '未稳定识别'}",
            f"谓语：{main.predicate or '未稳定识别'}",
            f"宾语/补足语：{main.object_or_complement or '未稳定识别'}",
            f"主干意思：先理解为“{main.full_span or evidence.normalized_sentence}”。",
        ]
        clause_explanations = [
            f"{c.span}：{c.function_zh}，修饰 {c.modifies or '主句'}。{c.reading_strategy}"
            for c in evidence.clauses
        ] or ["没有检测到明显从句；可以直接按主干和修饰成分理解。"]
        modifier_explanations = [
            f"{m.span}：{m.function_zh}，修饰 {m.modifies or '附近成分'}。{m.reading_strategy}"
            for m in evidence.modifiers[:12]
        ] or ["没有检测到复杂修饰成分；重点放在主干和从句关系上。"]
        grammar_explanations = [
            f"{g.span}：{g.explanation_zh}"
            for g in evidence.grammar_points
        ] or ["没有检测到需要优先说明的复杂语法点。"]
        step_by_step = [
            "第一步：先抓主干，不要被句首状语、插入语或长修饰打断。",
            f"第二步：把主干读成一个完整意思：{main.full_span or evidence.normalized_sentence}",
            "第三步：回看从句，判断它们是补充名词、说明原因/让步/条件，还是充当名词成分。",
            "第四步：把介词短语、分词短语和同位语接回它们修饰的对象。",
            "第五步：按“背景/修饰 + 主干 + 原因/结果”的顺序整合整句。",
        ]
        chunks = [
            TranslationChunk(
                chunk_id=c.chunk_id,
                source=c.source,
                role=c.role,
                translation_zh=c.translation_zh or f"这一块承担“{c.role}”的作用，模板模式下先按结构理解。",
                start=c.start,
                end=c.end,
            )
            for c in evidence.chunks
        ]
        full_translation = (
            "模板模式不会自由翻译整句；建议先按以下顺序理解："
            f"主干“{main.full_span or evidence.normalized_sentence}”，"
            "再补充从句、修饰成分和语法点。开启 Qwen 后可生成自然整句翻译。"
        )
        reading_strategy = build_reading_strategy(evidence)
        summary = (
            f"这句话的核心是先抓主干：{main.full_span or evidence.normalized_sentence}。"
            f"系统检测到 {len(evidence.clauses)} 个从句、{len(evidence.modifiers)} 个修饰成分、"
            f"{len(evidence.grammar_points)} 个语法点。建议先读主干，再逐层补回从句和修饰。"
        )
        return LongSentenceExplanation(
            summary_zh=summary,
            main_clause_steps=main_steps,
            modifier_explanations=modifier_explanations,
            clause_explanations=clause_explanations,
            grammar_explanations=grammar_explanations,
            step_by_step_explanation=step_by_step,
            chunk_translation=chunks,
            full_translation=full_translation,
            reading_strategy=reading_strategy,
            generator="template",
        )

    def _qwen_explanation(self, evidence: LongSentenceEvidence) -> str:
        payload = build_qwen_payload(evidence)
        system = (
            "你是托福/学术英语长难句分析老师。你只能根据 evidence_json 解释句子结构，"
            "不能新增 evidence_json 中没有的从句、语法点或修饰关系。"
            "如果证据置信度低，要使用“可能”。只输出 JSON，不要 Markdown。"
        )
        user = (
            "请按照以下栏目输出：主干分析、修饰成分、从句结构、语法点、分块翻译、整句自然翻译、阅读策略。"
            "输出 JSON schema: {summary_zh, main_clause_steps, modifier_explanations, clause_explanations, "
            "grammar_explanations, step_by_step_explanation, chunk_translation:[{chunk_id,source,role,translation_zh}], "
            "full_translation, reading_strategy}。"
            "modifier_explanations、clause_explanations、grammar_explanations 必须是字符串数组，不要输出对象数组；"
            "每一项使用“原文片段: 中文解释”的格式。"
            "chunk_translation 必须只使用 evidence_json.chunks 中已有 chunk_id/source，不要新增不存在的片段。\n\n"
            f"evidence_json={json.dumps(payload, ensure_ascii=False)}"
        )
        return self._qwen_chat_completion(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=getattr(self.settings, "mode_c_qwen_max_tokens", 1600),
        )

    def _qwen_chat_completion(self, messages: List[Dict[str, str]], max_tokens: int = 1600) -> str:
        import urllib.error
        import urllib.request

        request_payload: Dict[str, Any] = {
            "model": self.settings.qwen_model,
            "messages": messages,
            "temperature": self.settings.qwen_temperature,
            "response_format": {"type": "json_object"},
            "max_tokens": max_tokens,
        }
        url = self.settings.qwen_base_url.rstrip("/") + "/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(request_payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.settings.qwen_api_key}"},
            method="POST",
        )
        timeout_s = getattr(self.settings, "mode_c_qwen_timeout_s", 240)
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (TimeoutError, socket.timeout) as exc:
            raise RuntimeError(f"Qwen API read timed out after {timeout_s}s") from exc
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Qwen API HTTP {exc.code}: {exc.read().decode('utf-8', errors='ignore')[:1000]}") from exc
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")


def build_qwen_payload(evidence: LongSentenceEvidence) -> Dict[str, Any]:
    data = {
        "sentence": evidence.sentence,
        "normalized_sentence": evidence.normalized_sentence,
        "level": evidence.level,
        "detail": evidence.detail,
        "parser_backend": evidence.parser_backend,
        "main_clause": model_to_dict(evidence.main_clause),
        "clauses": [model_to_dict(item) for item in evidence.clauses],
        "modifiers": [model_to_dict(item) for item in evidence.modifiers[:12]],
        "grammar_points": [model_to_dict(item) for item in evidence.grammar_points],
        "semantic_roles": [model_to_dict(item) for item in evidence.semantic_roles],
        "chunks": [model_to_dict(item) for item in evidence.chunks],
        "confidence": evidence.confidence,
        "warnings": evidence.warnings,
    }
    data["instructions"] = [
        "Only explain structures present in this evidence JSON.",
        "Use Chinese as the main teaching language.",
        "Follow the six-part display: main clause, modifiers, clauses, grammar, chunk translation, full translation, reading strategy.",
        "If parser_backend contains heuristic, express uncertain structures conservatively.",
    ]
    return data


def merge_qwen_explanation(parsed: Dict[str, Any], template: LongSentenceExplanation, raw: str) -> LongSentenceExplanation:
    valid_chunks = {c.chunk_id: c for c in template.chunk_translation}
    qwen_chunks: List[TranslationChunk] = []
    for item in parsed.get("chunk_translation", []) or []:
        chunk_id = str(item.get("chunk_id") or "")
        base = valid_chunks.get(chunk_id)
        if base is None:
            continue
        qwen_chunks.append(
            TranslationChunk(
                chunk_id=chunk_id,
                source=base.source,
                role=str(item.get("role") or base.role),
                translation_zh=str(item.get("translation_zh") or base.translation_zh),
                start=base.start,
                end=base.end,
            )
        )
    if not qwen_chunks:
        qwen_chunks = template.chunk_translation
    return LongSentenceExplanation(
        summary_zh=str(parsed.get("summary_zh") or template.summary_zh),
        main_clause_steps=list_of_strings(parsed.get("main_clause_steps")) or template.main_clause_steps,
        modifier_explanations=list_of_explanation_strings(parsed.get("modifier_explanations"), "modifier") or template.modifier_explanations,
        clause_explanations=list_of_explanation_strings(parsed.get("clause_explanations"), "clause") or template.clause_explanations,
        grammar_explanations=list_of_explanation_strings(parsed.get("grammar_explanations"), "grammar") or template.grammar_explanations,
        step_by_step_explanation=list_of_strings(parsed.get("step_by_step_explanation")) or template.step_by_step_explanation,
        chunk_translation=qwen_chunks,
        full_translation=str(parsed.get("full_translation") or template.full_translation),
        reading_strategy=str(parsed.get("reading_strategy") or template.reading_strategy),
        generator="qwen_text_evidence_grounded",
        llm_raw=raw,
    )


def parse_json_object(raw: str) -> Optional[Dict[str, Any]]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None


def list_of_strings(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(x) for x in value if str(x).strip()]


def list_of_explanation_strings(value: Any, kind: str) -> List[str]:
    if not isinstance(value, list):
        return []
    result: List[str] = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = format_explanation_item(item, kind)
        else:
            text = str(item).strip()
        if text:
            result.append(text)
    return result


def format_explanation_item(item: Dict[str, Any], kind: str) -> str:
    span = first_nonempty(item, ["span", "source", "text", "phrase", "clause", "point"])
    if kind == "grammar":
        explanation = first_nonempty(item, ["explanation_zh", "explanation", "function_zh", "function", "analysis"])
        return join_span_explanation(span, [explanation])

    function = first_nonempty(item, ["function_zh", "function", "type_zh", "type", "explanation_zh", "explanation"])
    modifies = first_nonempty(item, ["modifies", "target", "head"])
    strategy = first_nonempty(item, ["reading_strategy", "strategy", "advice"])
    details: List[str] = []
    if function:
        details.append(function)
    if modifies and modifies not in function:
        details.append(f"修饰 {modifies}")
    if strategy:
        details.append(strategy)
    return join_span_explanation(span, details)


def first_nonempty(item: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def join_span_explanation(span: str, parts: List[str]) -> str:
    clean_parts = [part.strip().rstrip("。") for part in parts if part and part.strip()]
    explanation = "。".join(clean_parts)
    if explanation:
        explanation += "。"
    if span and explanation:
        return f"{span}: {explanation}"
    return span or explanation


def build_reading_strategy(evidence: LongSentenceEvidence) -> str:
    main = evidence.main_clause.full_span or evidence.normalized_sentence
    clause_bits = []
    for clause in evidence.clauses[:5]:
        clause_bits.append(f"{clause.function_zh}“{clause.span}”")
    if clause_bits:
        return (
            f"先读主干“{main}”。然后依次补回："
            + "；".join(clause_bits)
            + "。最后再看介词短语、分词短语和插入语提供的细节。"
        )
    return f"先读主干“{main}”，再回看名词短语和介词短语中的限定信息。"
