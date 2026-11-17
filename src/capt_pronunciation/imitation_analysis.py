"""Reference-audio imitation diagnostics.

This module keeps the public report format identical to scripted reading mode,
but the evidence targets come from a reference recording instead of text-only
rules wherever that is more informative.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .audio_io import AudioProcessor, Waveform
from .comparative_features import (
    by_boundary,
    by_ref_index,
    by_sentence,
    contour_correlation,
    contour_dtw_distance,
    median_duration,
    normalized_duration,
    tempo_ratio,
    top_overlap,
)
from .contracts import (
    AlignmentResult,
    AsrResult,
    AudioInfo,
    CaptSettings,
    EvidenceBundle,
    EvidenceIssue,
    IssueType,
    PhonemeHypothesis,
    ReferenceProfile,
    Severity,
)
from .diagnostics import (
    EvidenceAnalyzer,
    phone_edit_alignment,
    phone_hint,
    phone_issue_priority,
    should_analyze_phoneme_word,
    simplify_phone,
)
from .deterministic_scoring import global_sequence_similarity
from .reference_profile import ReferenceProfileBuilder
from .ssl_comparison import SslEmbeddingComparator


class ImitationAnalyzer:
    """Compares learner audio against a reference-audio profile."""

    def __init__(self, settings: CaptSettings):
        self.settings = settings
        self.audio = AudioProcessor(settings)
        self.base = EvidenceAnalyzer(settings)
        self.profile_builder = ReferenceProfileBuilder(settings)
        self.ssl_comparator = SslEmbeddingComparator(settings)

    def analyze(
        self,
        *,
        reference_profile: ReferenceProfile,
        learner_audio: AudioInfo,
        learner_asr: AsrResult,
        learner_alignment: AlignmentResult,
        reference_text: str,
    ) -> EvidenceBundle:
        self.base._issue_counter = 0
        learner_profile = self.profile_builder.build(
            learner_audio,
            learner_alignment,
            reference_text,
            reference_text_source=reference_profile.reference_text_source,
            asr_text=learner_asr.text,
        )
        bundle = EvidenceBundle(prosody=learner_profile.prosody)

        # Word accuracy remains learner ASR vs shared reference transcript.
        bundle.word_issues = self.base.detect_word_issues(learner_alignment)

        learner_wave = self.audio.load_wave(learner_audio.path, sr=learner_audio.sample_rate)
        reference_wave: Optional[Waveform] = None
        if reference_profile.audio is not None and reference_profile.alignment is not None:
            reference_wave = self.audio.load_wave(reference_profile.audio.path, sr=reference_profile.audio.sample_rate)

        hypotheses, phone_issues, phone_features = self._detect_phoneme_issues(
            reference_profile=reference_profile,
            reference_wave=reference_wave,
            learner_wave=learner_wave,
            learner_alignment=learner_alignment,
        )
        bundle.phoneme_hypotheses = hypotheses
        bundle.phoneme_issues = phone_issues
        bundle.textpa = self.base.build_textpa_cues(learner_alignment, hypotheses)

        bundle.pause_issues = self._compare_pauses(reference_profile, learner_profile)
        bundle.linking_issues = self._compare_linking(reference_profile, learner_profile)
        bundle.stress_issues = self._compare_word_stress(reference_profile, learner_profile, learner_alignment)
        bundle.sentence_stress_issues = self._compare_sentence_stress(reference_profile, learner_profile)
        bundle.intonation_issues = self._compare_intonation(reference_profile, learner_profile)
        bundle.rhythm_issues = self._compare_rhythm(reference_profile, learner_profile)

        ssl_features = (
            self.ssl_comparator.compare(reference_wave, learner_wave)
            if reference_wave is not None
            else {"ssl_soft_dtw": None, "ssl_frame_coverage": 0.0}
        )
        comparison = self._continuous_reference_features(
            reference_profile,
            learner_profile,
            learner_alignment,
        )
        comparison.update(phone_features)
        comparison.update(ssl_features)
        bundle.continuous_features["mode_b"] = comparison
        bundle.coverage = {
            "phoneme": float(phone_features.get("phone_coverage") or 0.0),
            "reference_phoneme": float(phone_features.get("reference_phone_coverage") or 0.0),
            "boundary": float(comparison.get("boundary_coverage") or 0.0),
            "duration": float(comparison.get("duration_coverage") or 0.0),
            "sentence_prosody": float(comparison.get("sentence_coverage") or 0.0),
            "ssl": float(ssl_features.get("ssl_frame_coverage") or 0.0),
        }

        bundle.prosody.raw["reference_audio_comparison"] = {
            "reference_text_source": reference_profile.reference_text_source,
            "reference_alignment_source": reference_profile.alignment.source if reference_profile.alignment else None,
            "learner_alignment_source": learner_alignment.source,
            "reference_word_count": len(reference_profile.word_features),
            "learner_word_count": len(learner_profile.word_features),
        }
        return bundle

    def _detect_phoneme_issues(
        self,
        *,
        reference_profile: ReferenceProfile,
        reference_wave: Optional[Waveform],
        learner_wave: Waveform,
        learner_alignment: AlignmentResult,
    ) -> Tuple[List[PhonemeHypothesis], List[EvidenceIssue], Dict[str, Optional[float]]]:
        learner_hyp = self._recognize_phonemes_by_ref(learner_alignment, learner_wave)
        reference_hyp: Dict[int, PhonemeHypothesis] = {}
        if reference_wave is not None and reference_profile.alignment is not None:
            reference_hyp = self._recognize_phonemes_by_ref(reference_profile.alignment, reference_wave)

        issues: List[EvidenceIssue] = []
        learner_canonical_values: List[float] = []
        learner_reference_values: List[float] = []
        for ref_index, hyp in learner_hyp.items():
            canonical = [simplify_phone(p) for p in hyp.canonical_phones if simplify_phone(p)]
            predicted = [simplify_phone(p) for p in hyp.predicted_phones if simplify_phone(p)]
            if not canonical or not predicted:
                continue
            edits = [(op, exp, obs) for op, exp, obs in phone_edit_alignment(canonical, predicted) if op != "equal"]
            edit_ratio = len(edits) / max(1, len(canonical))
            ref_h = reference_hyp.get(ref_index)
            ref_pred = [simplify_phone(p) for p in (ref_h.predicted_phones if ref_h else []) if simplify_phone(p)]
            learner_canonical_sim = global_sequence_similarity(canonical, predicted)
            reference_canonical_sim = global_sequence_similarity(canonical, ref_pred) if ref_pred else None
            learner_reference_sim = global_sequence_similarity(ref_pred, predicted) if ref_pred else None
            learner_canonical_values.append(float(learner_canonical_sim))
            if learner_reference_sim is not None:
                learner_reference_values.append(float(learner_reference_sim))

            if hyp.confidence is None or hyp.confidence < self.settings.phoneme_min_confidence:
                continue

            # If the reference recording itself realizes a reduced/variant form
            # and the learner follows it closely, treat this as imitation rather
            # than a canonical phoneme error.
            if (
                reference_canonical_sim is not None
                and learner_reference_sim is not None
                and reference_canonical_sim < 0.82
                and learner_reference_sim >= 0.86
            ):
                continue

            edits = sorted(edits, key=lambda item: phone_issue_priority(*item), reverse=True)
            for op, exp, obs in edits[: max(0, self.settings.phoneme_max_issues_per_word)]:
                issue_type = {
                    "substitute": IssueType.PHONEME_SUBSTITUTION,
                    "delete": IssueType.PHONEME_DELETION,
                    "insert": IssueType.PHONEME_INSERTION,
                }[op]
                issues.append(
                    self.base.issue(
                        issue_type,
                        Severity.HIGH if edit_ratio > self.settings.phoneme_max_edit_ratio else Severity.MEDIUM,
                        ref_index,
                        hyp.word,
                        observed=obs or "[missing]",
                        expected=exp or "[extra]",
                        start=hyp.start,
                        end=hyp.end,
                        confidence=min(0.9, max(0.45, hyp.confidence or 0.55)),
                        evidence={
                            "comparison_mode": "reference_audio",
                            "canonical_phones": canonical,
                            "learner_predicted_phones": predicted,
                            "reference_predicted_phones": ref_pred,
                            "learner_canonical_similarity": _round(learner_canonical_sim),
                            "reference_canonical_similarity": _round(reference_canonical_sim),
                            "learner_reference_similarity": _round(learner_reference_sim),
                            "phone_operation": op,
                            "recognizer_confidence": hyp.confidence,
                            "ctc_gop": hyp.ctc_gop,
                            "edit_ratio": round(edit_ratio, 3),
                        },
                        hint=phone_hint(exp, obs, hyp.word),
                    )
                )
        ref_count = len(learner_alignment.reference_tokens)
        features: Dict[str, Optional[float]] = {
            "phone_reference_similarity": _mean(learner_reference_values),
            "phone_canonical_similarity": _mean(learner_canonical_values),
            "phone_coverage": len(learner_hyp) / max(1, ref_count),
            "reference_phone_coverage": len(reference_hyp) / max(1, ref_count),
        }
        return list(learner_hyp.values()), issues, features

    def _recognize_phonemes_by_ref(self, alignment: AlignmentResult, waveform: Waveform) -> Dict[int, PhonemeHypothesis]:
        out: Dict[int, PhonemeHypothesis] = {}
        if not self.base.phoneme_recognizer.is_available():
            return out
        token_by_index = {t.index: t for t in alignment.reference_tokens}
        for ali in alignment.word_alignments:
            if ali.ref_index is None:
                continue
            token = token_by_index.get(ali.ref_index)
            if token is None or not should_analyze_phoneme_word(ali, token, self.settings):
                continue
            segment = self.audio.crop(waveform, ali.start, ali.end, pad_s=0.02)
            try:
                fallback = [simplify_phone(p) for p in token.canonical_ipa if simplify_phone(p)]
                canonical_model = self.base.phoneme_recognizer.canonical_phones(token.normalized, fallback)
                predicted, conf, ctc_gop = self.base.phoneme_recognizer.recognize(segment, canonical_model)
            except Exception:
                continue
            canonical = [simplify_phone(p) for p in canonical_model if simplify_phone(p)]
            out[int(ali.ref_index)] = PhonemeHypothesis(
                ref_index=int(ali.ref_index),
                word=token.normalized,
                start=ali.start,
                end=ali.end,
                predicted_phones=[simplify_phone(p) for p in predicted if simplify_phone(p)],
                canonical_phones=canonical,
                confidence=conf,
                ctc_gop=ctc_gop,
            )
        return out

    def _continuous_reference_features(
        self,
        reference: ReferenceProfile,
        learner: ReferenceProfile,
        learner_alignment: AlignmentResult,
    ) -> Dict[str, Optional[float]]:
        """Compute Mode B distances before any issue threshold is applied."""
        ref_boundaries = by_boundary(reference.boundary_features)
        learner_boundaries = by_boundary(learner.boundary_features)
        ratio = tempo_ratio(reference.word_features, learner.word_features)
        boundary_keys = sorted(set(ref_boundaries) & set(learner_boundaries))
        pause_errors = [
            abs(
                float(learner_boundaries[key].gap_s)
                - float(ref_boundaries[key].gap_s) * ratio
            )
            for key in boundary_keys
        ]

        ref_words = by_ref_index(reference.word_features)
        learner_words = by_ref_index(learner.word_features)
        paired_words = sorted(set(ref_words) & set(learner_words))
        ref_median = median_duration(reference.word_features)
        learner_median = median_duration(learner.word_features)
        duration_errors: List[float] = []
        stress_matches: List[float] = []
        for index in paired_words:
            ref_word = ref_words[index]
            learner_word = learner_words[index]
            ref_duration = normalized_duration(ref_word.duration, ref_median)
            learner_duration = normalized_duration(learner_word.duration, learner_median)
            if ref_duration is not None and learner_duration is not None:
                duration_errors.append(abs(float(learner_duration) - float(ref_duration)))
            if (
                ref_word.prominent_syllable is not None
                and learner_word.prominent_syllable is not None
            ):
                stress_matches.append(
                    1.0
                    if int(ref_word.prominent_syllable) == int(learner_word.prominent_syllable)
                    else 0.0
                )

        ref_sentences = by_sentence(reference.sentence_features)
        learner_sentences = by_sentence(learner.sentence_features)
        sentence_keys = sorted(set(ref_sentences) & set(learner_sentences))
        f0_dtw_values: List[float] = []
        f0_correlations: List[float] = []
        stress_overlaps: List[float] = []
        for index in sentence_keys:
            ref_sentence = ref_sentences[index]
            learner_sentence = learner_sentences[index]
            dtw = contour_dtw_distance(
                ref_sentence.f0_contour_st,
                learner_sentence.f0_contour_st,
            )
            corr = contour_correlation(
                ref_sentence.f0_contour_st,
                learner_sentence.f0_contour_st,
            )
            if dtw is not None:
                f0_dtw_values.append(float(dtw))
            if corr is not None:
                f0_correlations.append(float(corr))
            if ref_sentence.prominent_words and learner_sentence.prominent_words:
                stress_overlaps.append(
                    top_overlap(
                        ref_sentence.prominent_words,
                        learner_sentence.prominent_words,
                    )
                )

        return {
            "tempo_log_ratio": math.log(max(ratio, 1e-8)),
            "pause_mae_s": _mean(pause_errors),
            "duration_mae": _mean(duration_errors),
            "f0_dtw": _mean(f0_dtw_values),
            "f0_correlation": _mean(f0_correlations),
            "sentence_stress_overlap": _mean(stress_overlaps),
            "word_stress_similarity": _mean(stress_matches),
            "boundary_coverage": len(boundary_keys) / max(1, len(ref_boundaries)),
            "duration_coverage": len(paired_words) / max(1, len(ref_words)),
            "sentence_coverage": len(sentence_keys) / max(1, len(ref_sentences)),
            "reference_word_count": float(len(learner_alignment.reference_tokens)),
        }

    def _compare_pauses(self, reference: ReferenceProfile, learner: ReferenceProfile) -> List[EvidenceIssue]:
        issues: List[EvidenceIssue] = []
        ref_boundaries = by_boundary(reference.boundary_features)
        learner_boundaries = by_boundary(learner.boundary_features)
        ratio = tempo_ratio(reference.word_features, learner.word_features)
        for key, ref_b in ref_boundaries.items():
            learner_b = learner_boundaries.get(key)
            if learner_b is None:
                continue
            ref_gap = float(ref_b.gap_s)
            learner_gap = float(learner_b.gap_s)
            expected = ref_gap * ratio
            evidence = {
                "comparison_mode": "reference_audio",
                "reference_gap_s": round(ref_gap, 3),
                "learner_gap_s": round(learner_gap, 3),
                "tempo_ratio": round(ratio, 3),
                "expected_learner_gap_s": round(expected, 3),
                "boundary_after": ref_b.boundary_after,
            }
            if ref_gap <= self.settings.imitation_reference_short_pause_s and learner_gap >= self.settings.imitation_learner_break_s:
                issues.append(
                    self.base.issue(
                        IssueType.PAUSE_UNEXPECTED,
                        Severity.MEDIUM if learner_gap >= 0.45 else Severity.LOW,
                        ref_b.left_index,
                        f"{ref_b.left_word} {ref_b.right_word}",
                        observed=f"learner pause={learner_gap:.2f}s",
                        expected=f"reference gap={ref_gap:.2f}s",
                        start=learner_b.start,
                        end=learner_b.end,
                        confidence=min(0.92, 0.45 + learner_gap),
                        evidence=evidence,
                        hint="参考录音这里基本没有停顿，你的朗读在这里断开较明显。",
                    )
                )
            elif ref_gap >= self.settings.imitation_reference_long_pause_s and learner_gap <= self.settings.imitation_learner_missing_pause_s:
                issues.append(
                    self.base.issue(
                        IssueType.PAUSE_MISSING,
                        Severity.MEDIUM,
                        ref_b.left_index,
                        ref_b.left_word,
                        observed=f"learner pause={learner_gap:.2f}s",
                        expected=f"reference pause={ref_gap:.2f}s",
                        start=learner_b.start,
                        end=learner_b.end,
                        confidence=0.70,
                        evidence=evidence,
                        hint="参考录音这里有明显停顿，你的朗读略过了这个边界。",
                    )
                )
            elif ref_gap > 0.12 and learner_gap > expected + 0.35:
                issues.append(
                    self.base.issue(
                        IssueType.RHYTHM,
                        Severity.LOW,
                        ref_b.left_index,
                        f"{ref_b.left_word} {ref_b.right_word}",
                        observed=f"learner pause={learner_gap:.2f}s",
                        expected=f"tempo-adjusted reference pause about {expected:.2f}s",
                        start=learner_b.start,
                        end=learner_b.end,
                        confidence=min(0.75, 0.35 + learner_gap - expected),
                        evidence=evidence,
                        hint="这里两段录音都有停顿，但你的停顿相对参考录音偏长。",
                    )
                )
        return issues

    def _compare_linking(self, reference: ReferenceProfile, learner: ReferenceProfile) -> List[EvidenceIssue]:
        issues: List[EvidenceIssue] = []
        learner_boundaries = by_boundary(learner.boundary_features)
        for ref_b in reference.boundary_features:
            if not ref_b.has_linking or not ref_b.linking_opportunity:
                continue
            learner_b = learner_boundaries.get((ref_b.left_index, ref_b.right_index))
            if learner_b is None:
                continue
            if float(learner_b.gap_s) <= self.settings.imitation_linking_learner_gap_s:
                continue
            issues.append(
                self.base.issue(
                    IssueType.LINKING_MISSING,
                    Severity.LOW,
                    ref_b.left_index,
                    f"{ref_b.left_word} {ref_b.right_word}",
                    observed=f"learner gap={learner_b.gap_s:.2f}s",
                    expected=f"reference links this boundary ({ref_b.gap_s:.2f}s gap)",
                    start=learner_b.start,
                    end=learner_b.end,
                    confidence=min(0.82, 0.40 + float(learner_b.gap_s) * 1.4),
                    evidence={
                        "comparison_mode": "reference_audio",
                        "reference_has_linking": True,
                        "reference_gap_s": ref_b.gap_s,
                        "learner_gap_s": learner_b.gap_s,
                        "linking_opportunity": ref_b.linking_opportunity,
                        "reference_energy_continuity": ref_b.energy_continuity,
                        "learner_energy_continuity": learner_b.energy_continuity,
                    },
                    hint="参考录音这里是连起来读的，你的朗读更像两个分开的词。",
                )
            )
        return issues

    def _compare_word_stress(
        self,
        reference: ReferenceProfile,
        learner: ReferenceProfile,
        learner_alignment: AlignmentResult,
    ) -> List[EvidenceIssue]:
        issues: List[EvidenceIssue] = []
        ref_words = by_ref_index(reference.word_features)
        learner_words = by_ref_index(learner.word_features)
        tokens = {t.index: t for t in learner_alignment.reference_tokens}
        for ref_index, learner_w in learner_words.items():
            token = tokens.get(ref_index)
            if token is None or not token.is_content_word or len(token.stress_pattern) < 2 or 1 not in token.stress_pattern:
                continue
            ref_w = ref_words.get(ref_index)
            canonical_primary = token.stress_pattern.index(1) + 1
            learner_primary = learner_w.prominent_syllable
            reference_primary = ref_w.prominent_syllable if ref_w else None
            if learner_primary is None:
                continue
            evidence = {
                "comparison_mode": "reference_audio",
                "canonical_primary_syllable": canonical_primary,
                "reference_prominent_syllable": reference_primary,
                "learner_prominent_syllable": learner_primary,
                "reference_stress_contrast": ref_w.stress_contrast if ref_w else None,
                "learner_stress_contrast": learner_w.stress_contrast,
                "reference_syllables": ref_w.syllables if ref_w else [],
                "learner_syllables": learner_w.syllables,
            }
            if learner_primary != canonical_primary and learner_primary != reference_primary:
                issues.append(
                    self.base.issue(
                        IssueType.STRESS,
                        Severity.MEDIUM,
                        ref_index,
                        learner_w.word,
                        observed=f"syllable {learner_primary} more prominent",
                        expected=f"primary stress on syllable {canonical_primary}",
                        start=learner_w.start,
                        end=learner_w.end,
                        confidence=0.64,
                        evidence=evidence,
                        hint="这个词的重音位置和词典/参考录音都不一致。",
                    )
                )
            elif ref_w and ref_w.stress_contrast is not None and learner_w.stress_contrast is not None:
                if ref_w.stress_contrast - learner_w.stress_contrast > 0.45:
                    issues.append(
                        self.base.issue(
                            IssueType.STRESS,
                            Severity.LOW,
                            ref_index,
                            learner_w.word,
                            observed="learner stress contrast weaker than reference",
                            expected="clearer reference-like stress contrast",
                            start=learner_w.start,
                            end=learner_w.end,
                            confidence=0.50,
                            evidence=evidence,
                            hint="参考录音的重读音节更突出，你的重音对比偏弱。",
                        )
                    )
        return issues

    def _compare_sentence_stress(self, reference: ReferenceProfile, learner: ReferenceProfile) -> List[EvidenceIssue]:
        issues: List[EvidenceIssue] = []
        ref_sentences = by_sentence(reference.sentence_features)
        learner_sentences = by_sentence(learner.sentence_features)
        for sentence_index, ref_s in ref_sentences.items():
            learner_s = learner_sentences.get(sentence_index)
            if learner_s is None or len(ref_s.prominent_words) < 1 or len(learner_s.prominent_words) < 1:
                continue
            overlap = top_overlap(ref_s.prominent_words, learner_s.prominent_words)
            if overlap >= 0.35:
                continue
            issues.append(
                self.base.issue(
                    IssueType.SENTENCE_STRESS,
                    Severity.LOW,
                    ref_s.prominent_words[0] if ref_s.prominent_words else None,
                    "sentence stress",
                    observed=f"prominent words differ from reference (overlap={overlap:.2f})",
                    expected="reference-like prominent content words",
                    start=learner_s.start,
                    end=learner_s.end,
                    confidence=0.48,
                    evidence={
                        "comparison_mode": "reference_audio",
                        "sentence_index": sentence_index,
                        "reference_prominent_words": ref_s.prominent_words,
                        "learner_prominent_words": learner_s.prominent_words,
                        "overlap": round(overlap, 3),
                    },
                    hint="这一句中你突出强调的词和参考录音不太一致。",
                )
            )
        return issues

    def _compare_intonation(self, reference: ReferenceProfile, learner: ReferenceProfile) -> List[EvidenceIssue]:
        issues: List[EvidenceIssue] = []
        learner_sentences = by_sentence(learner.sentence_features)
        for ref_s in reference.sentence_features:
            learner_s = learner_sentences.get(ref_s.sentence_index)
            if learner_s is None:
                continue
            slope_issue = False
            slope_diff: Optional[float] = None
            if ref_s.final_slope_st_per_s is not None and learner_s.final_slope_st_per_s is not None:
                slope_diff = float(learner_s.final_slope_st_per_s) - float(ref_s.final_slope_st_per_s)
                slope_issue = abs(slope_diff) >= self.settings.imitation_final_slope_diff_threshold
                if ref_s.final_slope_st_per_s * learner_s.final_slope_st_per_s < 0 and abs(slope_diff) >= 1.4:
                    slope_issue = True
            dtw = contour_dtw_distance(ref_s.f0_contour_st, learner_s.f0_contour_st)
            corr = contour_correlation(ref_s.f0_contour_st, learner_s.f0_contour_st)
            dtw_issue = dtw is not None and dtw >= self.settings.imitation_f0_dtw_threshold
            if not slope_issue and not dtw_issue:
                continue
            observed = []
            expected = []
            if slope_diff is not None:
                observed.append(f"final slope diff={slope_diff:.2f} st/s")
                expected.append(f"reference final slope={ref_s.final_slope_st_per_s:.2f} st/s")
            if dtw is not None:
                observed.append(f"F0 DTW={dtw:.2f}")
                expected.append("closer normalized F0 contour")
            issues.append(
                self.base.issue(
                    IssueType.INTONATION,
                    Severity.LOW,
                    None,
                    "sentence intonation",
                    observed=", ".join(observed),
                    expected=", ".join(expected),
                    start=learner_s.start,
                    end=learner_s.end,
                    confidence=0.55 if slope_issue else 0.48,
                    evidence={
                        "comparison_mode": "reference_audio",
                        "sentence_index": ref_s.sentence_index,
                        "reference_final_slope_st_per_s": ref_s.final_slope_st_per_s,
                        "learner_final_slope_st_per_s": learner_s.final_slope_st_per_s,
                        "final_slope_diff_st_per_s": _round(slope_diff),
                        "f0_dtw_distance": _round(dtw),
                        "f0_contour_correlation": _round(corr),
                        "reference_pitch_range_st": ref_s.pitch_range_st,
                        "learner_pitch_range_st": learner_s.pitch_range_st,
                    },
                    hint="这一句的音高走势和参考录音差异较明显。",
                )
            )
        return issues

    def _compare_rhythm(self, reference: ReferenceProfile, learner: ReferenceProfile) -> List[EvidenceIssue]:
        issues: List[EvidenceIssue] = []
        ref_words = by_ref_index(reference.word_features)
        learner_words = by_ref_index(learner.word_features)
        ref_median = median_duration(reference.word_features)
        learner_median = median_duration(learner.word_features)
        diffs: List[Dict[str, float | int | str]] = []
        for ref_index, learner_w in learner_words.items():
            ref_w = ref_words.get(ref_index)
            if ref_w is None:
                continue
            ref_norm = normalized_duration(ref_w.duration, ref_median)
            learner_norm = normalized_duration(learner_w.duration, learner_median)
            if ref_norm is None or learner_norm is None:
                continue
            diff = learner_norm - ref_norm
            if abs(diff) >= self.settings.imitation_duration_diff_threshold:
                diffs.append({"ref_index": ref_index, "word": learner_w.word, "duration_diff": round(diff, 3)})
        paired = len(set(ref_words) & set(learner_words))
        if paired >= 6 and len(diffs) >= max(3, int(math.ceil(paired * 0.25))):
            issues.append(
                self.base.issue(
                    IssueType.RHYTHM,
                    Severity.LOW,
                    None,
                    "duration pattern",
                    observed=f"{len(diffs)} words differ strongly in relative duration",
                    expected="reference-like relative word duration pattern",
                    start=None,
                    end=None,
                    confidence=min(0.72, 0.35 + len(diffs) / max(1, paired)),
                    evidence={
                        "comparison_mode": "reference_audio",
                        "paired_word_count": paired,
                        "large_duration_differences": diffs[:20],
                        "reference_median_word_duration": _round(ref_median),
                        "learner_median_word_duration": _round(learner_median),
                    },
                    hint="整体节奏和参考录音有差异，一些词的相对时长偏长或偏短。",
                )
            )
        return issues


def _round(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        if not math.isfinite(float(value)):
            return None
        return round(float(value), 4)
    except Exception:
        return None


def _mean(values: List[float]) -> Optional[float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None
