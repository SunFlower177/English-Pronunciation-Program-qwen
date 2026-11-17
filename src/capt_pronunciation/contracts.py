"""Shared contracts for the evidence-grounded CAPT system.

This file intentionally combines configuration and data schemas so the project
stays within the requested 6--7 functional modules. All other modules should
import data models from here instead of defining ad-hoc dictionaries.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

try:
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover - pydantic is expected in runtime env
    BaseModel = object  # type: ignore

    def Field(default=None, **_: Any):  # type: ignore
        return default


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class IssueType(str, Enum):
    WORD_OMISSION = "word_omission"
    WORD_INSERTION = "word_insertion"
    WORD_SUBSTITUTION = "word_substitution"
    PHONEME_SUBSTITUTION = "phoneme_substitution"
    PHONEME_DELETION = "phoneme_deletion"
    PHONEME_INSERTION = "phoneme_insertion"
    PAUSE_MISSING = "pause_missing"
    PAUSE_UNEXPECTED = "pause_unexpected"
    LINKING_MISSING = "linking_missing"
    STRESS = "word_stress"
    SENTENCE_STRESS = "sentence_stress"
    INTONATION = "intonation"
    RHYTHM = "rhythm"
    FLUENCY = "fluency"
    TEXTPA_BASELINE = "textpa_baseline"


@dataclass
class CaptSettings:
    """Runtime settings loaded from environment variables.

    The defaults are conservative: they run with CPU and fall back to heuristic
    behavior when heavy optional models are not installed. For actual paper
    experiments, use CUDA, install WhisperX/MFA/Transformers, and set
    CAPT_USE_* flags to true.
    """

    work_dir: Path = field(default_factory=lambda: Path(os.getenv("CAPT_WORK_DIR", "./runs")).resolve())
    upload_dir: Path = field(default_factory=lambda: Path(os.getenv("CAPT_UPLOAD_DIR", "./uploads")).resolve())
    device: str = os.getenv("CAPT_DEVICE", "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu")
    compute_type: str = os.getenv("CAPT_COMPUTE_TYPE", "float16")

    asr_backend: Literal["whisperx", "faster_whisper", "transformers_whisper"] = os.getenv("CAPT_ASR_BACKEND", "whisperx")  # type: ignore
    whisperx_model: str = os.getenv("CAPT_WHISPERX_MODEL", "small.en")
    transformers_whisper_model: str = os.getenv("CAPT_TRANSFORMERS_WHISPER_MODEL", "openai/whisper-tiny.en")
    whisperx_batch_size: int = int(os.getenv("CAPT_WHISPERX_BATCH_SIZE", "8"))
    use_whisperx_alignment: bool = os.getenv("CAPT_USE_WHISPERX_ALIGNMENT", "1") == "1"

    use_mfa: bool = os.getenv("CAPT_USE_MFA", "0") == "1"
    mfa_command: str = os.getenv("CAPT_MFA_COMMAND", "mfa")
    mfa_dictionary: str = os.getenv("CAPT_MFA_DICTIONARY", "english_us_arpa")
    mfa_acoustic_model: str = os.getenv("CAPT_MFA_ACOUSTIC", "english_us_arpa")
    mfa_g2p_model: str = os.getenv("CAPT_MFA_G2P", "english_us_mfa")
    mfa_num_jobs: int = int(os.getenv("CAPT_MFA_JOBS", "2"))
    use_ctc_alignment: bool = os.getenv("CAPT_USE_CTC_ALIGNMENT", "0") == "1"

    phoneme_model: str = os.getenv("CAPT_PHONEME_MODEL", "facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    use_phoneme_model: bool = os.getenv("CAPT_USE_PHONEME_MODEL", "1") == "1"
    phoneme_require_equal_word: bool = os.getenv("CAPT_PHONEME_REQUIRE_EQUAL_WORD", "0") == "1"
    phoneme_skip_function_words: bool = os.getenv("CAPT_PHONEME_SKIP_FUNCTION_WORDS", "0") == "1"
    phoneme_min_word_duration_s: float = float(os.getenv("CAPT_PHONEME_MIN_WORD_DURATION", "0.28"))
    phoneme_min_confidence: float = float(os.getenv("CAPT_PHONEME_MIN_CONFIDENCE", "0.75"))
    phoneme_max_issues_per_word: int = int(os.getenv("CAPT_PHONEME_MAX_ISSUES_PER_WORD", "3"))
    phoneme_max_edit_ratio: float = float(os.getenv("CAPT_PHONEME_MAX_EDIT_RATIO", "0.45"))
    phoneme_max_word_penalty: float = float(os.getenv("CAPT_PHONEME_MAX_WORD_PENALTY", "75"))
    prosody_require_equal_words: bool = os.getenv("CAPT_PROSODY_REQUIRE_EQUAL_WORDS", "1") == "1"
    prosody_prefer_forced_alignment: bool = os.getenv("CAPT_PROSODY_PREFER_FORCED_ALIGNMENT", "1") == "1"

    use_qwen2_audio: bool = os.getenv("CAPT_USE_QWEN2_AUDIO", "0") == "1"
    qwen2_audio_model: str = os.getenv("CAPT_QWEN2_AUDIO_MODEL", "Qwen/Qwen2-Audio-7B-Instruct")
    qwen2_max_new_tokens: int = int(os.getenv("CAPT_QWEN2_MAX_NEW_TOKENS", "900"))

    target_accent: Literal["american", "british"] = os.getenv("CAPT_TARGET_ACCENT", "american")  # type: ignore
    language: str = os.getenv("CAPT_LANGUAGE", "en")
    sample_rate: int = int(os.getenv("CAPT_SAMPLE_RATE", "16000"))

    # Heuristic thresholds. They should be tuned on Speechocean762/L2-ARCTIC or
    # your self-collected validation data before paper submission.
    long_pause_threshold_s: float = float(os.getenv("CAPT_LONG_PAUSE_THRESHOLD", "0.55"))
    missing_pause_threshold_s: float = float(os.getenv("CAPT_MISSING_PAUSE_THRESHOLD", "0.12"))
    linking_gap_threshold_s: float = float(os.getenv("CAPT_LINKING_GAP_THRESHOLD", "0.18"))
    min_word_confidence: float = float(os.getenv("CAPT_MIN_WORD_CONFIDENCE", "0.50"))
    phrase_pause_threshold_s: float = float(os.getenv("CAPT_PHRASE_PAUSE_THRESHOLD", "0.25"))
    statement_rise_threshold_st_per_s: float = float(os.getenv("CAPT_STATEMENT_RISE_ST_S", "2.5"))
    question_fall_threshold_st_per_s: float = float(os.getenv("CAPT_QUESTION_FALL_ST_S", "-3.0"))
    monotone_pitch_range_st: float = float(os.getenv("CAPT_MONOTONE_RANGE_ST", "3.0"))
    stress_prominence_margin: float = float(os.getenv("CAPT_STRESS_PROM_MARGIN", "0.25"))
    imitation_reference_short_pause_s: float = float(os.getenv("CAPT_IMITATION_REF_SHORT_PAUSE", "0.08"))
    imitation_reference_long_pause_s: float = float(os.getenv("CAPT_IMITATION_REF_LONG_PAUSE", "0.30"))
    imitation_learner_break_s: float = float(os.getenv("CAPT_IMITATION_LEARNER_BREAK", "0.20"))
    imitation_learner_missing_pause_s: float = float(os.getenv("CAPT_IMITATION_LEARNER_MISSING_PAUSE", "0.08"))
    imitation_linking_ref_gap_s: float = float(os.getenv("CAPT_IMITATION_LINK_REF_GAP", "0.08"))
    imitation_linking_learner_gap_s: float = float(os.getenv("CAPT_IMITATION_LINK_LEARNER_GAP", "0.18"))
    imitation_f0_dtw_threshold: float = float(os.getenv("CAPT_IMITATION_F0_DTW_THRESHOLD", "1.8"))
    imitation_final_slope_diff_threshold: float = float(os.getenv("CAPT_IMITATION_FINAL_SLOPE_DIFF", "2.5"))
    imitation_duration_diff_threshold: float = float(os.getenv("CAPT_IMITATION_DURATION_DIFF", "0.70"))
    imitation_use_ssl_embeddings: bool = os.getenv("CAPT_IMITATION_USE_SSL", "0") == "1"
    imitation_ssl_model: str = os.getenv("CAPT_IMITATION_SSL_MODEL", "microsoft/wavlm-base-plus")
    imitation_ssl_layer: int = int(os.getenv("CAPT_IMITATION_SSL_LAYER", "9"))
    imitation_ssl_max_frames: int = int(os.getenv("CAPT_IMITATION_SSL_MAX_FRAMES", "240"))
    deterministic_score_config: Optional[str] = os.getenv("CAPT_DETERMINISTIC_SCORE_CONFIG")
    use_qwen_text: bool = os.getenv("CAPT_USE_QWEN_TEXT", "0") == "1"
    qwen_backend: str = os.getenv("QWEN_BACKEND", "openai_compatible")
    qwen_api_key: Optional[str] = os.getenv("QWEN_API_KEY")
    qwen_base_url: str = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    qwen_model: str = os.getenv("QWEN_MODEL", "qwen3.7-plus")
    qwen_feedback_mode: Literal["per_issue", "batch"] = os.getenv("CAPT_QWEN_FEEDBACK_MODE", "per_issue")  # type: ignore
    qwen_max_feedback_issues: int = int(os.getenv("CAPT_QWEN_MAX_FEEDBACK_ISSUES", "30"))
    qwen_temperature: float = float(os.getenv("QWEN_TEMPERATURE", "0.2"))
    qwen_issue_max_tokens: int = int(os.getenv("CAPT_QWEN_ISSUE_MAX_TOKENS", "450"))
    qwen_summary_max_tokens: int = int(os.getenv("CAPT_QWEN_SUMMARY_MAX_TOKENS", "650"))
    mode_c_spacy_model: str = os.getenv("CAPT_MODEC_SPACY_MODEL", "en_core_web_trf")
    mode_c_qwen_max_tokens: int = int(os.getenv("CAPT_MODEC_QWEN_MAX_TOKENS", "1600"))
    mode_c_qwen_timeout_s: int = int(os.getenv("CAPT_MODEC_QWEN_TIMEOUT_SECONDS", "240"))

    @classmethod
    def from_env(cls) -> "CaptSettings":
        settings = cls()
        settings.work_dir.mkdir(parents=True, exist_ok=True)
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        return settings


class AudioInfo(BaseModel):
    path: str
    original_path: Optional[str] = None
    sample_rate: int
    duration_s: float
    channels: int = 1
    vad_segments: List[Dict[str, float]] = Field(default_factory=list)
    rms: Optional[float] = None


class AsrWord(BaseModel):
    word: str
    start: Optional[float] = None
    end: Optional[float] = None
    score: Optional[float] = None
    source: str = "asr"


class AsrSegment(BaseModel):
    text: str
    start: Optional[float] = None
    end: Optional[float] = None
    words: List[AsrWord] = Field(default_factory=list)


class AsrResult(BaseModel):
    text: str
    language: str = "en"
    segments: List[AsrSegment] = Field(default_factory=list)
    words: List[AsrWord] = Field(default_factory=list)
    raw: Dict[str, Any] = Field(default_factory=dict)


class ReferenceToken(BaseModel):
    index: int
    raw: str
    normalized: str
    before_punct: str = ""
    after_punct: str = ""
    sentence_index: int = 0
    canonical_arpabet: List[str] = Field(default_factory=list)
    canonical_ipa: List[str] = Field(default_factory=list)
    stress_pattern: List[int] = Field(default_factory=list)
    syllable_count: int = 0
    pos_hint: str = "unknown"
    is_content_word: bool = False
    boundary_after: str = "none"  # none | weak | phrase | clause | sentence


class PhoneInterval(BaseModel):
    phone: str
    start: Optional[float] = None
    end: Optional[float] = None
    source: str = "estimated"
    score: Optional[float] = None


class WordAlignment(BaseModel):
    ref_index: Optional[int] = None
    ref_word: Optional[str] = None
    hyp_word: Optional[str] = None
    start: Optional[float] = None
    end: Optional[float] = None
    operation: Literal["equal", "substitute", "delete", "insert"]
    confidence: Optional[float] = None
    phones: List[PhoneInterval] = Field(default_factory=list)


class AlignmentResult(BaseModel):
    reference_tokens: List[ReferenceToken]
    word_alignments: List[WordAlignment]
    source: str = "hybrid"
    raw: Dict[str, Any] = Field(default_factory=dict)


class PhonemeHypothesis(BaseModel):
    ref_index: Optional[int] = None
    word: str
    start: Optional[float] = None
    end: Optional[float] = None
    predicted_phones: List[str] = Field(default_factory=list)
    canonical_phones: List[str] = Field(default_factory=list)
    source: str = "wav2vec2_phoneme"
    confidence: Optional[float] = None
    ctc_gop: Optional[float] = None


class EvidenceIssue(BaseModel):
    issue_id: str
    type: IssueType
    severity: Severity
    ref_index: Optional[int] = None
    word: Optional[str] = None
    observed: Optional[str] = None
    expected: Optional[str] = None
    start: Optional[float] = None
    end: Optional[float] = None
    confidence: float = 0.5
    evidence: Dict[str, Any] = Field(default_factory=dict)
    suggestion_hint: Optional[str] = None


class TextPACues(BaseModel):
    transcript: str = ""
    phonemes_cmu: str = ""
    phonemes_ipa: str = ""
    canonical_ipa: str = ""
    ipa_match_score: Optional[float] = None
    cmu_match_score: Optional[float] = None


class ProsodyFeatures(BaseModel):
    speech_rate_wpm: Optional[float] = None
    articulation_rate_wpm: Optional[float] = None
    pause_count: int = 0
    total_pause_s: float = 0.0
    pause_ratio: Optional[float] = None
    mean_length_run_words: Optional[float] = None
    npvi_word_duration: Optional[float] = None
    word_duration_cv: Optional[float] = None
    pitch_mean_hz: Optional[float] = None
    pitch_range_hz: Optional[float] = None
    pitch_range_st: Optional[float] = None
    final_slope_st_per_s: Optional[float] = None
    intensity_mean_db: Optional[float] = None
    raw: Dict[str, Any] = Field(default_factory=dict)


class WordAcousticFeature(BaseModel):
    ref_index: int
    word: str
    start: float
    end: float
    duration: float
    intensity_mean_db: Optional[float] = None
    f0_mean_hz: Optional[float] = None
    f0_peak_hz: Optional[float] = None
    f0_range_st: Optional[float] = None
    prominence: Optional[float] = None
    prominent_syllable: Optional[int] = None
    stress_contrast: Optional[float] = None
    syllables: List[Dict[str, Any]] = Field(default_factory=list)


class BoundaryFeature(BaseModel):
    left_index: int
    right_index: int
    left_word: str
    right_word: str
    start: Optional[float] = None
    end: Optional[float] = None
    gap_s: float
    boundary_after: str = "none"
    linking_opportunity: Optional[Dict[str, Any]] = None
    energy_continuity: Optional[float] = None
    voicing_continuity: Optional[float] = None
    has_linking: bool = False


class SentenceProsodyFeature(BaseModel):
    sentence_index: int
    start: float
    end: float
    final_slope_st_per_s: Optional[float] = None
    pitch_range_st: Optional[float] = None
    f0_contour_st: List[float] = Field(default_factory=list)
    f0_times: List[float] = Field(default_factory=list)
    prominent_words: List[int] = Field(default_factory=list)


class ReferenceProfile(BaseModel):
    reference_text: str
    reference_text_source: Literal["user", "reference_asr"] = "user"
    asr_text: str = ""
    audio: Optional[AudioInfo] = None
    alignment: Optional[AlignmentResult] = None
    prosody: ProsodyFeatures = Field(default_factory=ProsodyFeatures)
    word_features: List[WordAcousticFeature] = Field(default_factory=list)
    boundary_features: List[BoundaryFeature] = Field(default_factory=list)
    sentence_features: List[SentenceProsodyFeature] = Field(default_factory=list)
    raw: Dict[str, Any] = Field(default_factory=dict)


class EvidenceBundle(BaseModel):
    word_issues: List[EvidenceIssue] = Field(default_factory=list)
    phoneme_issues: List[EvidenceIssue] = Field(default_factory=list)
    pause_issues: List[EvidenceIssue] = Field(default_factory=list)
    linking_issues: List[EvidenceIssue] = Field(default_factory=list)
    stress_issues: List[EvidenceIssue] = Field(default_factory=list)
    sentence_stress_issues: List[EvidenceIssue] = Field(default_factory=list)
    intonation_issues: List[EvidenceIssue] = Field(default_factory=list)
    rhythm_issues: List[EvidenceIssue] = Field(default_factory=list)
    phoneme_hypotheses: List[PhonemeHypothesis] = Field(default_factory=list)
    prosody: ProsodyFeatures = Field(default_factory=ProsodyFeatures)
    textpa: TextPACues = Field(default_factory=TextPACues)
    # Continuous, auditable measurements consumed by deterministic_scoring.
    # Missing measurements stay missing; they are never silently treated as correct.
    continuous_features: Dict[str, Any] = Field(default_factory=dict)
    coverage: Dict[str, float] = Field(default_factory=dict)
    scoring_evidence_version: str = "continuous_evidence_v2"

    @property
    def all_issues(self) -> List[EvidenceIssue]:
        return (
            self.word_issues
            + self.phoneme_issues
            + self.pause_issues
            + self.linking_issues
            + self.stress_issues
            + self.sentence_stress_issues
            + self.intonation_issues
            + self.rhythm_issues
        )


class FeedbackItem(BaseModel):
    issue_id: str
    display_text: str
    practice: List[str] = Field(default_factory=list)
    explanation_zh: str = ""
    explanation_en: str = ""


class FeedbackReport(BaseModel):
    summary_zh: str
    summary_en: str = ""
    overall_scores: Dict[str, float] = Field(default_factory=dict)
    qwen_rubric_scores: Dict[str, float] = Field(default_factory=dict)
    qwen_rubric_rationale: Dict[str, str] = Field(default_factory=dict)
    qwen_rubric_raw: Optional[str] = None
    annotated_text_html: str = ""
    items: List[FeedbackItem] = Field(default_factory=list)
    llm_raw: Optional[str] = None
    generator: str = "template"


class AnalysisReport(BaseModel):
    analysis_id: str
    mode: Literal["reading", "imitation", "free_speaking"] = "reading"
    reference_text: Optional[str] = None
    reference_text_source: Optional[Literal["user", "reference_asr"]] = None
    reference_audio: Optional[AudioInfo] = None
    reference_profile: Optional[ReferenceProfile] = None
    audio: AudioInfo
    asr: AsrResult
    alignment: AlignmentResult
    evidence: EvidenceBundle
    feedback: FeedbackReport


def model_to_dict(model: Any) -> Dict[str, Any]:
    """Pydantic-v1/v2 compatible dump helper."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model)
