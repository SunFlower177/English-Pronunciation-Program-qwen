import json

from capt_pronunciation.contracts import (
    AnalysisReport,
    AlignmentResult,
    AsrResult,
    AudioInfo,
    BoundaryFeature,
    CaptSettings,
    EvidenceBundle,
    EvidenceIssue,
    FeedbackReport,
    IssueType,
    ReferenceProfile,
    ReferenceToken,
    SentenceProsodyFeature,
    Severity,
    WordAcousticFeature,
)
from capt_pronunciation.feedback import EvidenceGroundedFeedbackGenerator, compute_scores
from capt_pronunciation.imitation_analysis import ImitationAnalyzer
from capt_pronunciation.runtime import CaptPipeline
from capt_pronunciation.ssl_comparison import cosine_soft_dtw


def _audio() -> AudioInfo:
    return AudioInfo(path="dummy.wav", sample_rate=16000, duration_s=1.0)


def _profile(ref_gap=0.02, learner=False) -> ReferenceProfile:
    prefix = "learner" if learner else "reference"
    return ReferenceProfile(
        reference_text="pick it up",
        audio=_audio(),
        prosody={},
        word_features=[
            WordAcousticFeature(ref_index=0, word="pick", start=0.0, end=0.3, duration=0.3, prominence=0.1),
            WordAcousticFeature(ref_index=1, word="it", start=0.3 + ref_gap, end=0.55 + ref_gap, duration=0.25, prominence=0.0),
            WordAcousticFeature(ref_index=2, word="up", start=0.6 + ref_gap, end=0.8 + ref_gap, duration=0.2, prominence=0.2),
        ],
        boundary_features=[
            BoundaryFeature(
                left_index=0,
                right_index=1,
                left_word="pick",
                right_word="it",
                start=0.3,
                end=0.3 + ref_gap,
                gap_s=ref_gap,
                linking_opportunity={"type": "c_v_linking", "description": "consonant-vowel linking k+i"},
                energy_continuity=0.9,
                has_linking=not learner,
            )
        ],
        sentence_features=[
            SentenceProsodyFeature(
                sentence_index=0,
                start=0.0,
                end=0.8,
                final_slope_st_per_s=-2.0 if not learner else 2.0,
                pitch_range_st=4.0,
                f0_contour_st=[0.0, 0.2, -0.1, -0.8] if not learner else [0.0, -0.1, 0.3, 1.1],
                prominent_words=[0] if not learner else [2],
            )
        ],
        raw={"kind": prefix},
    )


def test_imitation_pause_and_linking_comparison():
    settings = CaptSettings()
    analyzer = ImitationAnalyzer(settings)
    analyzer.base._issue_counter = 0
    reference = _profile(ref_gap=0.02)
    learner = _profile(ref_gap=0.32, learner=True)

    pause_issues = analyzer._compare_pauses(reference, learner)
    linking_issues = analyzer._compare_linking(reference, learner)

    assert pause_issues
    assert pause_issues[0].type.value == "pause_unexpected"
    assert pause_issues[0].evidence["reference_gap_s"] == 0.02
    assert pause_issues[0].evidence["learner_gap_s"] == 0.32
    assert linking_issues
    assert linking_issues[0].type.value == "linking_missing"


def test_imitation_intonation_and_sentence_stress_comparison():
    settings = CaptSettings()
    analyzer = ImitationAnalyzer(settings)
    analyzer.base._issue_counter = 0
    reference = _profile(ref_gap=0.02)
    learner = _profile(ref_gap=0.02, learner=True)

    intonation = analyzer._compare_intonation(reference, learner)
    sentence_stress = analyzer._compare_sentence_stress(reference, learner)

    assert intonation
    assert intonation[0].type.value == "intonation"
    assert sentence_stress
    assert sentence_stress[0].type.value == "sentence_stress"


def test_analysis_report_accepts_imitation_mode():
    report = AnalysisReport(
        analysis_id="unit",
        mode="imitation",
        reference_text="pick it up",
        reference_text_source="user",
        reference_audio=_audio(),
        reference_profile=_profile(ref_gap=0.02),
        audio=_audio(),
        asr=AsrResult(text="pick it up"),
        alignment=AlignmentResult(reference_tokens=[], word_alignments=[]),
        evidence=EvidenceBundle(),
        feedback=FeedbackReport(summary_zh="ok"),
    )

    assert report.mode == "imitation"
    assert report.reference_profile is not None


def test_imitation_mode_uses_rubric_scores_and_keeps_evidence(tmp_path):
    settings = CaptSettings(work_dir=tmp_path, upload_dir=tmp_path)
    reference_text = "pick it up"
    tokens = [
        ReferenceToken(index=0, raw="pick", normalized="pick"),
        ReferenceToken(index=1, raw="it", normalized="it"),
        ReferenceToken(index=2, raw="up", normalized="up"),
    ]
    alignment = AlignmentResult(reference_tokens=tokens, word_alignments=[])
    evidence = EvidenceBundle(
        pause_issues=[
            EvidenceIssue(
                issue_id="I0001",
                type=IssueType.PAUSE_UNEXPECTED,
                severity=Severity.HIGH,
                ref_index=1,
                word="pick it",
                confidence=1.0,
            )
        ]
    )

    class FakeAudioProcessor:
        def prepare_audio(self, path, analysis_id):
            return _audio()

    class FakeTranscriber:
        def transcribe(self, path):
            return AsrResult(text=reference_text)

    class FakeReferenceProcessor:
        def tokenize(self, text):
            return tokens

    class FakeAligner:
        reference_processor = FakeReferenceProcessor()

        def align(self, text, asr, audio_path, analysis_id):
            return alignment

    class FakeReferenceProfiler:
        def build(self, *args, **kwargs):
            return _profile(ref_gap=0.02)

    class FakeImitationAnalyzer:
        def analyze(self, *args, **kwargs):
            return evidence

    class FakeFeedback:
        def generate(self, *args, **kwargs):
            return FeedbackReport(
                summary_zh="ok",
                overall_scores={
                    "accuracy": 1.0,
                    "phoneme": 1.0,
                    "fluency": 1.0,
                    "prosody": 1.0,
                    "overall": 1.0,
                },
            )

    pipeline = object.__new__(CaptPipeline)
    pipeline.settings = settings
    pipeline.audio_processor = FakeAudioProcessor()
    pipeline.transcriber = FakeTranscriber()
    pipeline.aligner = FakeAligner()
    pipeline.reference_profiler = FakeReferenceProfiler()
    pipeline.imitation_analyzer = FakeImitationAnalyzer()
    pipeline.feedback = FakeFeedback()

    report = pipeline.analyze_imitation("reference.wav", "learner.wav", reference_text, analysis_id="unit")

    assert report.mode == "imitation"
    assert report.evidence.pause_issues == evidence.pause_issues
    assert report.feedback.overall_scores == compute_scores(evidence, settings, alignment)
    assert report.feedback.overall_scores["overall"] != 1.0


def test_imitation_mode_can_use_qwen_per_issue_feedback(tmp_path, monkeypatch):
    settings = CaptSettings(
        work_dir=tmp_path,
        upload_dir=tmp_path,
        use_qwen_text=True,
        qwen_api_key="test-key",
        qwen_model="qwen3.7-plus",
        qwen_feedback_mode="per_issue",
    )
    reference_text = "pick it up"
    tokens = [
        ReferenceToken(index=0, raw="pick", normalized="pick"),
        ReferenceToken(index=1, raw="it", normalized="it"),
        ReferenceToken(index=2, raw="up", normalized="up"),
    ]
    alignment = AlignmentResult(reference_tokens=tokens, word_alignments=[])
    evidence = EvidenceBundle(
        linking_issues=[
            EvidenceIssue(
                issue_id="I0001",
                type=IssueType.LINKING_MISSING,
                severity=Severity.MEDIUM,
                ref_index=1,
                word="pick it",
                expected="smooth linking",
                observed="clear gap",
                confidence=0.82,
            )
        ]
    )

    class FakeAudioProcessor:
        def prepare_audio(self, path, analysis_id):
            return _audio()

    class FakeTranscriber:
        def transcribe(self, path):
            return AsrResult(text=reference_text)

    class FakeReferenceProcessor:
        def tokenize(self, text):
            return tokens

    class FakeAligner:
        reference_processor = FakeReferenceProcessor()

        def align(self, text, asr, audio_path, analysis_id):
            return alignment

    class FakeReferenceProfiler:
        def build(self, *args, **kwargs):
            return _profile(ref_gap=0.02)

    class FakeImitationAnalyzer:
        def analyze(self, *args, **kwargs):
            return evidence

    feedback = EvidenceGroundedFeedbackGenerator(settings)

    def fake_chat(messages, max_tokens=None):
        payload = json.loads(messages[-1]["content"].split("payload=", 1)[1])
        if payload.get("task") == "speech_ocean762_rubric_scoring":
            return json.dumps(
                {
                    "accuracy": 8.0,
                    "phoneme": 8.0,
                    "fluency": 7.0,
                    "prosody": 7.5,
                    "rationale": {
                        "accuracy": "shared text is mostly matched",
                        "phoneme": "no phone issue in evidence",
                        "fluency": "linking issue affects smoothness",
                        "prosody": "minor naturalness concern",
                    },
                },
                ensure_ascii=False,
            )
        issue = payload.get("issue")
        if issue:
            return json.dumps(
                {
                    "issue_id": issue["issue_id"],
                    "display_text": "pick it",
                    "explanation_zh": "Qwen 解释：参考录音这里更连贯，学习者录音中间有明显间隔。",
                    "explanation_en": "The learner leaves a gap where the reference is linked.",
                    "practice": ["pick it", "pick_it", "slow: pick it", "normal: pick it up"],
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "summary_zh": "Qwen 综合评价：录音与录音对比显示，优先练习连读自然度。",
                "summary_en": "Practice smoother linking first.",
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(feedback, "_qwen_chat_completion", fake_chat)

    pipeline = object.__new__(CaptPipeline)
    pipeline.settings = settings
    pipeline.audio_processor = FakeAudioProcessor()
    pipeline.transcriber = FakeTranscriber()
    pipeline.aligner = FakeAligner()
    pipeline.reference_profiler = FakeReferenceProfiler()
    pipeline.imitation_analyzer = FakeImitationAnalyzer()
    pipeline.feedback = feedback

    report = pipeline.analyze_imitation("reference.wav", "learner.wav", reference_text, analysis_id="unit-qwen")

    assert report.mode == "imitation"
    assert report.evidence.linking_issues == evidence.linking_issues
    assert report.feedback.generator == "qwen_text_evidence_grounded"
    assert report.feedback.items[0].explanation_zh.startswith("Qwen 解释")
    assert report.feedback.summary_zh.startswith("Qwen 综合评价")
    assert report.feedback.overall_scores == compute_scores(evidence, settings, alignment)
    assert report.feedback.qwen_rubric_scores["overall"] == 76.2


def test_ssl_soft_dtw_prefers_matching_frame_sequences():
    import numpy as np

    reference = np.eye(4, dtype=float)
    matching = np.eye(4, dtype=float)
    mismatching = -np.eye(4, dtype=float)

    matching_distance = cosine_soft_dtw(reference, matching)
    mismatching_distance = cosine_soft_dtw(reference, mismatching)

    assert matching_distance is not None
    assert mismatching_distance is not None
    assert matching_distance < mismatching_distance
