import json
from pathlib import Path

from capt_pronunciation.contracts import (
    AlignmentResult,
    AsrResult,
    CaptSettings,
    EvidenceBundle,
    EvidenceIssue,
    IssueType,
    ReferenceToken,
    Severity,
)
from capt_pronunciation.feedback import EvidenceGroundedFeedbackGenerator


def _alignment(words):
    return AlignmentResult(
        reference_tokens=[ReferenceToken(index=i, raw=word, normalized=word.lower()) for i, word in enumerate(words)],
        word_alignments=[],
    )


def test_qwen_per_issue_feedback_preserves_template_items(monkeypatch):
    settings = CaptSettings(
        use_qwen_text=True,
        qwen_api_key="test-key",
        qwen_feedback_mode="per_issue",
        qwen_model="qwen3.7-plus",
    )
    generator = EvidenceGroundedFeedbackGenerator(settings)
    alignment = _alignment(["pick", "it", "up"])
    evidence = EvidenceBundle(
        word_issues=[
            EvidenceIssue(
                issue_id="I0001",
                type=IssueType.WORD_SUBSTITUTION,
                severity=Severity.HIGH,
                ref_index=0,
                word="pick",
                expected="pick",
                observed="big",
                confidence=0.9,
            )
        ],
        pause_issues=[
            EvidenceIssue(
                issue_id="I0002",
                type=IssueType.PAUSE_UNEXPECTED,
                severity=Severity.MEDIUM,
                ref_index=1,
                word="pick it",
                confidence=0.8,
                suggestion_hint="fallback pause explanation",
            )
        ],
    )

    def fake_chat(messages, max_tokens=None):
        payload = json.loads(messages[-1]["content"].split("payload=", 1)[1])
        if payload.get("task") == "speech_ocean762_rubric_scoring":
            return json.dumps(
                {
                    "accuracy": 7.0,
                    "phoneme": 6.5,
                    "fluency": 8.0,
                    "prosody": 7.5,
                    "rationale": {
                        "accuracy": "one word substitution",
                        "phoneme": "limited phone evidence",
                        "fluency": "one pause issue",
                        "prosody": "no prosody issue",
                    },
                }
            )
        issue = payload.get("issue")
        if issue:
            if issue["issue_id"] == "I0002":
                raise RuntimeError("simulated item failure")
            return json.dumps(
                {
                    "issue_id": "I0001",
                    "display_text": "pick",
                    "explanation_zh": "Qwen generated issue explanation",
                    "explanation_en": "Qwen generated issue explanation.",
                    "practice": ["pick", "slow: pick", "pick it up"],
                }
            )
        return json.dumps({"summary_zh": "Qwen generated summary", "summary_en": "Qwen summary."})

    monkeypatch.setattr(generator, "_qwen_chat_completion", fake_chat)

    report = generator.generate("pick it up", Path("dummy.wav"), AsrResult(text="big it up"), alignment, evidence)

    assert report.generator == "qwen_text_evidence_grounded"
    assert report.summary_zh == "Qwen generated summary"
    assert len(report.items) == 2
    assert report.items[0].issue_id == "I0001"
    assert report.items[0].explanation_zh == "Qwen generated issue explanation"
    assert report.items[1].issue_id == "I0002"
    assert report.items[1].explanation_zh == "fallback pause explanation"
    assert report.qwen_rubric_scores == {
        "accuracy": 70.0,
        "phoneme": 65.0,
        "fluency": 80.0,
        "prosody": 75.0,
        "overall": 72.5,
    }
    assert report.qwen_rubric_rationale["accuracy"] == "one word substitution"
