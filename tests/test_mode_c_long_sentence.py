import json

from capt_pronunciation.contracts import CaptSettings
from capt_pronunciation.mode_c import LongSentenceAnalyzer
from capt_pronunciation.mode_c.clause_extractor import ClauseExtractor
from capt_pronunciation.mode_c.grammar_detector import GrammarDetector
from capt_pronunciation.mode_c.schemas import SyntaxToken


SENTENCE = (
    "Although the theory, which had been widely accepted for decades, was challenged by new evidence, "
    "many researchers continued to rely on it because it explained observations that alternative models could not."
)


def _analyzer(settings):
    analyzer = LongSentenceAnalyzer(settings)
    analyzer.syntax_parser._spacy_failed = True
    analyzer.syntax_parser._stanza_failed = True
    return analyzer


def test_mode_c_template_outputs_full_sections(tmp_path):
    settings = CaptSettings(work_dir=tmp_path, upload_dir=tmp_path, use_qwen_text=False)
    report = _analyzer(settings).analyze(SENTENCE, analysis_id="mode-c-unit")

    assert report.mode == "long_sentence"
    assert report.evidence.normalized_sentence.startswith("Although the theory")
    assert report.evidence.main_clause.full_span
    assert any("although" in c.marker for c in report.evidence.clauses)
    assert any("which" in c.marker for c in report.evidence.clauses)
    assert any(c.marker == "which" and c.modifies == "the theory" for c in report.evidence.clauses)
    assert any(c.marker == "that" and c.modifies == "observations" for c in report.evidence.clauses)
    assert any(g.type == "past_perfect_passive" for g in report.evidence.grammar_points)
    assert report.explanation.main_clause_steps
    assert report.explanation.clause_explanations
    assert report.explanation.grammar_explanations
    assert report.explanation.chunk_translation
    assert report.explanation.reading_strategy
    assert report.explanation.generator == "template"


def test_mode_c_qwen_explanation_merges_evidence_chunks(tmp_path, monkeypatch):
    settings = CaptSettings(
        work_dir=tmp_path,
        upload_dir=tmp_path,
        use_qwen_text=True,
        qwen_api_key="test-key",
    )
    analyzer = _analyzer(settings)

    def fake_qwen(messages, max_tokens=1600):
        payload = json.loads(messages[-1]["content"].split("evidence_json=", 1)[1])
        first_chunk = payload["chunks"][0]
        return json.dumps(
            {
                "summary_zh": "Qwen 综合：先抓主干，再处理让步和原因从句。",
                "main_clause_steps": ["Qwen 主干步骤"],
                "modifier_explanations": ["Qwen 修饰说明"],
                "clause_explanations": ["Qwen 从句说明"],
                "grammar_explanations": ["Qwen 语法说明"],
                "step_by_step_explanation": ["Qwen 第一步", "Qwen 第二步"],
                "chunk_translation": [
                    {
                        "chunk_id": first_chunk["chunk_id"],
                        "source": first_chunk["source"],
                        "role": first_chunk["role"],
                        "translation_zh": "Qwen 分块翻译",
                    }
                ],
                "full_translation": "Qwen 整句翻译。",
                "reading_strategy": "Qwen 阅读策略。",
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(analyzer.explainer, "_qwen_chat_completion", fake_qwen)
    report = analyzer.analyze(SENTENCE, analysis_id="mode-c-qwen-unit")

    assert report.explanation.generator == "qwen_text_evidence_grounded"
    assert report.explanation.summary_zh.startswith("Qwen 综合")
    assert report.explanation.full_translation == "Qwen 整句翻译。"
    assert report.explanation.chunk_translation[0].translation_zh == "Qwen 分块翻译"
def test_mode_c_evidence_handles_coordinated_predicates_and_nested_relatives():
    sentence = (
        "Iron technology certainly helped farmers and so accelerated the spread of agricultural communities "
        "in West Africa after 400 B.C.E., such as those of the upper Niger River, where farming settlements "
        "began to form clusters of villages that specialized in rice, cotton, or dried fish, and eventually "
        "developed into large market towns."
    )
    words = [
        ("Iron", "PROPN", "compound", 1, "NNP"),
        ("technology", "NOUN", "nsubj", 3, "NN"),
        ("certainly", "ADV", "advmod", 3, "RB"),
        ("helped", "VERB", "ROOT", 3, "VBD"),
        ("farmers", "NOUN", "dobj", 3, "NNS"),
        ("and", "CCONJ", "cc", 7, "CC"),
        ("so", "ADV", "advmod", 7, "RB"),
        ("accelerated", "VERB", "conj", 3, "VBD"),
        ("the", "DET", "det", 9, "DT"),
        ("spread", "NOUN", "dobj", 7, "NN"),
        ("of", "ADP", "prep", 9, "IN"),
        ("agricultural", "ADJ", "amod", 12, "JJ"),
        ("communities", "NOUN", "pobj", 10, "NNS"),
        ("in", "ADP", "prep", 12, "IN"),
        ("West", "PROPN", "compound", 15, "NNP"),
        ("Africa", "PROPN", "pobj", 13, "NNP"),
        ("after", "ADP", "prep", 7, "IN"),
        ("400", "NUM", "nummod", 18, "CD"),
        ("B.C.E.", "PROPN", "pobj", 16, "NNP"),
        ("such", "ADJ", "amod", 22, "JJ"),
        ("as", "ADP", "prep", 7, "IN"),
        ("those", "PRON", "pobj", 20, "DT"),
        ("of", "ADP", "prep", 21, "IN"),
        ("the", "DET", "det", 26, "DT"),
        ("upper", "ADJ", "amod", 26, "JJ"),
        ("Niger", "PROPN", "compound", 26, "NNP"),
        ("River", "PROPN", "pobj", 22, "NNP"),
        ("where", "ADV", "advmod", 30, "WRB"),
        ("farming", "NOUN", "compound", 29, "NN"),
        ("settlements", "NOUN", "nsubj", 30, "NNS"),
        ("began", "VERB", "relcl", 26, "VBD"),
        ("to", "PART", "aux", 32, "TO"),
        ("form", "VERB", "xcomp", 30, "VB"),
        ("clusters", "NOUN", "dobj", 32, "NNS"),
        ("of", "ADP", "prep", 33, "IN"),
        ("villages", "NOUN", "pobj", 34, "NNS"),
        ("that", "PRON", "nsubj", 37, "WDT"),
        ("specialized", "VERB", "relcl", 33, "VBD"),
        ("in", "ADP", "prep", 37, "IN"),
        ("rice", "NOUN", "pobj", 38, "NN"),
        ("cotton", "NOUN", "conj", 39, "NN"),
        ("or", "CCONJ", "cc", 42, "CC"),
        ("dried", "ADJ", "amod", 42, "JJ"),
        ("fish", "NOUN", "conj", 39, "NN"),
        ("and", "CCONJ", "cc", 45, "CC"),
        ("eventually", "ADV", "advmod", 45, "RB"),
        ("developed", "VERB", "conj", 37, "VBD"),
        ("into", "ADP", "prep", 45, "IN"),
        ("large", "ADJ", "amod", 50, "JJ"),
        ("market", "NOUN", "compound", 50, "NN"),
        ("towns", "NOUN", "pobj", 47, "NNS"),
    ]

    cursor = 0
    tokens = []
    for index, (text, pos, dep, head, tag) in enumerate(words):
        start = sentence.index(text, cursor)
        end = start + len(text)
        cursor = end
        tokens.append(SyntaxToken(index=index, text=text, pos=pos, dep=dep, head=head, tag=tag, start=start, end=end))

    evidence = ClauseExtractor().extract(sentence, tokens, {"noun_chunks": []})
    clauses = evidence["clauses"]
    grammar_points = GrammarDetector().detect(sentence)

    assert "accelerated the spread" in evidence["main_clause"].full_span
    assert not any(clause.span.lower().startswith("after 400") for clause in clauses)
    assert any(clause.marker == "where" and clause.modifies == "the upper Niger River" for clause in clauses)
    assert any(clause.marker == "that" and clause.modifies == "villages" for clause in clauses)
    assert any(clause.span.startswith("to form") and clause.type == "nonfinite_complement_clause" for clause in clauses)
    modifier_spans = {mod.span for mod in evidence["modifiers"]}
    assert {"certainly", "so", "of agricultural communities", "in West Africa", "after 400 B.C.E."}.issubset(modifier_spans)
    assert "Iron technology" not in modifier_spans
    assert "the spread" not in modifier_spans
    assert "agricultural communities" not in modifier_spans
    assert "West Africa" not in modifier_spans
    assert "400 B.C.E." not in modifier_spans
    assert "such" not in modifier_spans
    assert "upper" not in modifier_spans
    assert any(point.type == "example_phrase" for point in grammar_points)
    assert not any(point.type == "ing_phrase" and point.span.lower() == "farming" for point in grammar_points)
    assert not any(point.type == "parenthetical_or_appositive" and point.span == "cotton" for point in grammar_points)


def test_mode_c_grammar_filters_nominal_ing_but_keeps_structural_ing():
    sentence = "Where farming settlements expanded, using iron tools helped farmers working in villages."
    points = GrammarDetector().detect(sentence)
    ing_spans = {point.span.lower() for point in points if point.type == "ing_phrase"}

    assert "farming" not in ing_spans
    assert "using iron tools" in ing_spans
    assert "working in villages" in ing_spans


def test_mode_c_distinguishes_complementizer_that_from_relative_that(tmp_path):
    settings = CaptSettings(work_dir=tmp_path, upload_dir=tmp_path, use_qwen_text=False)
    report = _analyzer(settings).analyze(
        "The researcher said that the results were reliable.",
        analysis_id="mode-c-that",
    )
    that_clauses = [clause for clause in report.evidence.clauses if clause.marker == "that"]

    assert that_clauses
    assert all("relative" not in clause.type for clause in that_clauses)
    assert any(clause.type == "nominal_clause" for clause in that_clauses)


def test_mode_c_filters_false_infinitive_and_false_cleft(tmp_path):
    settings = CaptSettings(work_dir=tmp_path, upload_dir=tmp_path, use_qwen_text=False)
    analyzer = _analyzer(settings)
    to_report = analyzer.analyze(
        "She spoke to two students after class.",
        analysis_id="mode-c-to-two",
    )
    cleft_report = analyzer.analyze(
        "It was late and that caused a delay.",
        analysis_id="mode-c-false-cleft",
    )

    assert not any(point.type == "infinitive_phrase" for point in to_report.evidence.grammar_points)
    assert not any(point.type == "cleft_sentence" for point in cleft_report.evidence.grammar_points)


def test_mode_c_constituency_span_adds_or_confirms_clause():
    sentence = "The study suggests that the method works."
    words = [
        ("The", "DET", "det", 1, "DT"),
        ("study", "NOUN", "nsubj", 2, "NN"),
        ("suggests", "VERB", "ROOT", 2, "VBZ"),
        ("that", "SCONJ", "mark", 6, "IN"),
        ("the", "DET", "det", 5, "DT"),
        ("method", "NOUN", "nsubj", 6, "NN"),
        ("works", "VERB", "ccomp", 2, "VBZ"),
    ]
    tokens = []
    cursor = 0
    for index, (text, pos, dep, head, tag) in enumerate(words):
        start = sentence.index(text, cursor)
        end = start + len(text)
        cursor = end
        tokens.append(
            SyntaxToken(
                index=index,
                text=text,
                lemma=text.lower(),
                pos=pos,
                dep=dep,
                head=head,
                tag=tag,
                start=start,
                end=end,
            )
        )
    start = sentence.index("that")
    extracted = ClauseExtractor().extract(
        sentence,
        tokens,
        {
            "noun_chunks": [],
            "constituency_spans": [
                {
                    "label": "SBAR",
                    "parent_label": "VP",
                    "start": start,
                    "end": len(sentence) - 1,
                }
            ],
        },
    )
    that_clause = next(clause for clause in extracted["clauses"] if clause.marker == "that")

    assert that_clause.type == "nominal_clause"
    assert that_clause.confidence >= 0.82
