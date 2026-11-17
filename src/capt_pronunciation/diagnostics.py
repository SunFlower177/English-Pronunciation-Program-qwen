"""Evidence extraction: word, phoneme, pause, linking, stress, intonation.

Functional module 4/7.

The design rule is: do not ask an LLM to guess pronunciation mistakes. This
module produces structured, auditable evidence. The feedback module may only
paraphrase or prioritize these evidence items.
"""
from __future__ import annotations

import difflib
import logging
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .audio_io import AudioProcessor, Waveform
from .contracts import (
    AlignmentResult,
    AudioInfo,
    CaptSettings,
    EvidenceBundle,
    EvidenceIssue,
    IssueType,
    PhonemeHypothesis,
    ProsodyFeatures,
    TextPACues,
    ReferenceToken,
    Severity,
    WordAlignment,
)
from .deterministic_scoring import global_sequence_similarity
from .reference_alignment import is_consonant_phone, is_vowel_phone, normalize_word

logger = logging.getLogger(__name__)


COMMON_HINTS = {
    ("θ", "s"): "把舌尖轻轻放在上下齿之间再送气，避免把 /θ/ 读成 /s/。",
    ("θ", "t"): "不要把 /θ/ 读成 /t/；/θ/ 是摩擦音，不是爆破音。",
    ("ð", "z"): "读 /ð/ 时舌尖应在上下齿之间，并保持声带振动。",
    ("ð", "d"): "不要把 /ð/ 读成 /d/；/ð/ 需要持续摩擦。",
    ("v", "w"): "读 /v/ 时上齿轻触下唇；/w/ 是双唇圆唇音，口型不同。",
    ("w", "v"): "读 /w/ 时双唇圆起，不要让上齿碰下唇。",
    ("r", "l"): "注意 /r/ 舌尖不要顶住上齿龈，和 /l/ 的舌位不同。",
    ("l", "r"): "注意 /l/ 舌尖需要接触上齿龈，和 /r/ 区分开。",
    ("ɪ", "i"): "短元音 /ɪ/ 比 /iː/ 更短、更放松，不要拉得太长。",
    ("i", "ɪ"): "长元音 /iː/ 需要更紧、更长，不要读成短 /ɪ/。",
    ("æ", "ɛ"): "读 /æ/ 时嘴巴张得更开，舌位比 /ɛ/ 更低。",
    ("ɛ", "æ"): "读 /ɛ/ 时嘴巴不要张得过大，避免接近 /æ/。",
}


class PhonemeRecognitionError(RuntimeError):
    pass


class Wav2Vec2PhonemeRecognizer:
    """Generic Hugging Face CTC phoneme recognizer.

    It supports Wav2Vec2Phoneme checkpoints and common wav2vec2 phoneme CTC
    checkpoints through AutoProcessor + AutoModelForCTC.
    """

    def __init__(self, settings: CaptSettings):
        self.settings = settings
        self.processor = None
        self.model = None
        self._disabled_reason: Optional[str] = None

    def is_available(self) -> bool:
        if not self.settings.use_phoneme_model:
            self._disabled_reason = "CAPT_USE_PHONEME_MODEL=0"
            return False
        try:
            self._ensure_loaded()
            return True
        except Exception as exc:
            self._disabled_reason = str(exc)
            logger.warning("Phoneme recognizer unavailable: %s", exc)
            return False

    def recognize(
        self,
        waveform: Waveform,
        canonical_phones: Optional[Sequence[str]] = None,
    ) -> Tuple[List[str], Optional[float], Optional[float]]:
        if waveform.samples.size == 0:
            return [], None, None
        self._ensure_loaded()
        import torch  # type: ignore

        assert self.processor is not None and self.model is not None
        inputs = self.processor(
            waveform.samples,
            sampling_rate=waveform.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        device = self.settings.device if self.settings.device == "cuda" and torch.cuda.is_available() else "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            pred_ids = torch.argmax(logits, dim=-1)
            token_ids = pred_ids[0]
            token_probs = torch.max(probs[0], dim=-1).values
            blank_id = int(
                getattr(getattr(self.model, "config", None), "pad_token_id", None)
                or getattr(getattr(self.processor, "tokenizer", None), "pad_token_id", 0)
                or 0
            )
            previous = torch.cat([torch.tensor([blank_id], device=token_ids.device), token_ids[:-1]])
            emission_mask = (token_ids != blank_id) & (token_ids != previous)
            if torch.any(emission_mask):
                confidence = float(torch.mean(token_probs[emission_mask]).detach().cpu().item())
            else:
                confidence = None
            ctc_gop = self._ctc_target_quality(logits[0], canonical_phones or [], blank_id)
        decoded = self.processor.batch_decode(pred_ids)[0]
        phones = split_phone_string(decoded)
        return phones, confidence, ctc_gop

    def canonical_phones(self, word: str, fallback: Optional[Sequence[str]] = None) -> List[str]:
        """Return canonical phones in the recognizer tokenizer's own inventory."""
        self._ensure_loaded()
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            return [simplify_phone(p) for p in (fallback or []) if simplify_phone(p)]
        try:
            phonemize = getattr(tokenizer, "phonemize", None)
            if callable(phonemize):
                decoded = phonemize(str(word))
            else:
                encoded = tokenizer(str(word), add_special_tokens=False, do_phonemize=True)
                decoded = tokenizer.decode(getattr(encoded, "input_ids", []), skip_special_tokens=True)
            phones = [
                phone.strip()
                for phone in re.sub(r"\\s+", " ", str(decoded).strip()).split(" ")
                if phone.strip() and phone.strip() not in {"|", "<pad>", "<s>", "</s>"}
            ]
            unk_id = getattr(tokenizer, "unk_token_id", None)
            valid: List[str] = []
            for phone in phones:
                token_id = tokenizer.convert_tokens_to_ids(phone)
                if token_id is None or (unk_id is not None and int(token_id) == int(unk_id)):
                    continue
                valid.append(phone)
            if valid:
                return valid
        except Exception as exc:
            logger.debug("Cannot generate model-inventory phones for %s: %s", word, exc)
        return [str(p).strip() for p in (fallback or []) if str(p).strip()]

    def _ctc_target_quality(
        self,
        logits: Any,
        canonical_phones: Sequence[str],
        blank_id: int,
    ) -> Optional[float]:
        """Return competitor-normalized forced-alignment GOP in ``[0, 1]``."""
        if not canonical_phones:
            return None
        try:
            import torch  # type: ignore
            import torchaudio  # type: ignore

            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer is None:
                return None
            unk_id = getattr(tokenizer, "unk_token_id", None)
            target_ids: List[int] = []
            for phone in canonical_phones:
                token_id = tokenizer.convert_tokens_to_ids(str(phone))
                if token_id is None:
                    return None
                token_id = int(token_id)
                if token_id == blank_id or token_id < 0 or token_id >= int(logits.shape[-1]):
                    return None
                if unk_id is not None and token_id == int(unk_id):
                    return None
                target_ids.append(token_id)
            if not target_ids or len(target_ids) > int(logits.shape[0]):
                return None
            log_probs = torch.log_softmax(logits.detach().float(), dim=-1).cpu()
            targets = torch.tensor(target_ids, dtype=torch.long).unsqueeze(0)
            paths, _ = torchaudio.functional.forced_align(
                log_probs.unsqueeze(0), targets, blank=blank_id
            )
            path = paths[0]
            emitting = path != blank_id
            if not torch.any(emitting):
                return None
            frame_indices = torch.arange(path.shape[0], dtype=torch.long)[emitting]
            aligned_ids = path[emitting].long()
            target_logp = log_probs[frame_indices, aligned_ids]
            competitors = log_probs[frame_indices].clone()
            competitors[:, blank_id] = -torch.inf
            competitors[torch.arange(len(frame_indices)), aligned_ids] = -torch.inf
            margin = target_logp - torch.max(competitors, dim=-1).values
            quality = torch.sigmoid(margin).mean()
            return max(0.0, min(1.0, float(quality.item())))
        except Exception:
            return None

    def _ensure_loaded(self) -> None:
        if self.processor is not None and self.model is not None:
            return
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCTC, AutoProcessor  # type: ignore

            logger.info("Loading phoneme model: %s", self.settings.phoneme_model)
            self.processor = AutoProcessor.from_pretrained(self.settings.phoneme_model)
            self.model = AutoModelForCTC.from_pretrained(self.settings.phoneme_model)
            device = self.settings.device if self.settings.device == "cuda" and torch.cuda.is_available() else "cpu"
            self.model.to(device)
            self.model.eval()
        except Exception as exc:
            self.processor = None
            self.model = None
            raise PhonemeRecognitionError(
                f"Cannot load phoneme model {self.settings.phoneme_model}. "
                "Install transformers/torch/espeak-ng as needed, or set CAPT_USE_PHONEME_MODEL=0."
            ) from exc


class EvidenceAnalyzer:
    def __init__(self, settings: CaptSettings):
        self.settings = settings
        self.audio = AudioProcessor(settings)
        self.phoneme_recognizer = Wav2Vec2PhonemeRecognizer(settings)
        self._issue_counter = 0

    def analyze(self, audio_info: AudioInfo, alignment: AlignmentResult) -> EvidenceBundle:
        self._issue_counter = 0
        waveform = self.audio.load_wave(audio_info.path, sr=audio_info.sample_rate)
        acoustic = self.audio.summarize_acoustics(waveform, audio_info.vad_segments, len(alignment.reference_tokens))
        prosody = self.compute_prosody_features(acoustic, alignment, audio_info.duration_s)
        bundle = EvidenceBundle(prosody=prosody)

        # Segmental evidence.
        bundle.word_issues = self.detect_word_issues(alignment)
        hypotheses, phone_issues = self.detect_phoneme_issues(waveform, alignment)
        bundle.phoneme_hypotheses = hypotheses
        bundle.phoneme_issues = phone_issues
        bundle.textpa = self.build_textpa_cues(alignment, hypotheses)

        # Suprasegmental / innovation evidence.
        tracks = prosody.raw.get("tracks", {})
        bundle.pause_issues = self.detect_pause_issues(alignment)
        bundle.linking_issues = self.detect_linking_issues(alignment)
        bundle.stress_issues = self.detect_stress_issues(alignment, tracks)
        bundle.sentence_stress_issues = self.detect_sentence_stress_issues(alignment, tracks)
        bundle.intonation_issues = self.detect_intonation_issues(alignment, tracks)
        bundle.rhythm_issues = self.detect_rhythm_issues(alignment, prosody)
        ref_count = len(alignment.reference_tokens)
        timed_count = len(
            {
                int(item.ref_index)
                for item in alignment.word_alignments
                if item.ref_index is not None
                and item.operation != "delete"
                and item.start is not None
                and item.end is not None
            }
        )
        phone_count = len(
            {
                item.ref_index if item.ref_index is not None else (item.word, index)
                for index, item in enumerate(hypotheses)
            }
        )
        bundle.coverage = {
            "word_timing": round(timed_count / max(1, ref_count), 4),
            "phoneme": round(phone_count / max(1, ref_count), 4),
            "pitch": 1.0 if prosody.pitch_range_st is not None else 0.0,
            "intensity": 1.0 if prosody.intensity_mean_db is not None else 0.0,
        }
        bundle.continuous_features["mode_a"] = {
            "reference_word_count": ref_count,
            "timed_word_count": timed_count,
            "phoneme_word_count": phone_count,
        }
        return bundle

    def compute_prosody_features(self, acoustic: Dict[str, Any], alignment: AlignmentResult, duration_s: float) -> ProsodyFeatures:
        tracks = acoustic.get("tracks", {}) if isinstance(acoustic, dict) else {}
        pitch = tracks.get("pitch", {}) if isinstance(tracks, dict) else {}
        f0 = [float(x) for x in (pitch.get("f0", []) or []) if x and float(x) > 0 and math.isfinite(float(x))]
        pitch_mean = float(np.mean(f0)) if f0 else None
        pitch_range_hz = float(np.percentile(f0, 95) - np.percentile(f0, 5)) if len(f0) >= 5 else None
        pitch_range_st = None
        if f0 and pitch_mean and pitch_mean > 0:
            st = [12.0 * math.log2(v / pitch_mean) for v in f0 if v > 0]
            if len(st) >= 5:
                pitch_range_st = float(np.percentile(st, 95) - np.percentile(st, 5))

        timed_words = [
            a
            for a in alignment.word_alignments
            if a.operation != "delete"
            and a.hyp_word
            and a.start is not None
            and a.end is not None
        ]
        timed_words.sort(key=lambda a: float(a.start or 0.0))
        gaps = [max(0.0, float(b.start) - float(a.end)) for a, b in zip(timed_words, timed_words[1:]) if a.end is not None and b.start is not None]
        pauses = [g for g in gaps if g >= 0.18]
        long_pauses = [g for g in pauses if g >= self.settings.long_pause_threshold_s]
        word_durations = [max(0.01, float(w.end) - float(w.start)) for w in timed_words if w.start is not None and w.end is not None]
        npvi = normalized_pairwise_variability(word_durations) if len(word_durations) >= 3 else None
        dur_cv = float(np.std(word_durations) / (np.mean(word_durations) + 1e-8)) if len(word_durations) >= 2 else None
        speech_window_s = None
        if timed_words:
            speech_window_s = max(0.01, float(timed_words[-1].end) - float(timed_words[0].start))
        realized_word_count = len(timed_words)
        articulation_s = (
            max(0.01, float(speech_window_s) - sum(pauses))
            if speech_window_s is not None
            else None
        )
        speech_rate = (
            realized_word_count / speech_window_s * 60.0
            if speech_window_s is not None and realized_word_count
            else _as_float(acoustic.get("speech_rate_wpm"))
        )
        articulation_rate = (
            realized_word_count / articulation_s * 60.0
            if articulation_s is not None and realized_word_count
            else _as_float(acoustic.get("articulation_rate_wpm"))
        )
        pause_ratio = float(sum(pauses) / speech_window_s) if speech_window_s else None
        long_pause_ratio = float(sum(long_pauses) / speech_window_s) if speech_window_s else None
        pause_rate_per_min = float(len(pauses) / speech_window_s * 60.0) if speech_window_s else None
        mean_pause_s = float(np.mean(pauses)) if pauses else 0.0
        filled_pause_words = {"uh", "um", "erm", "er", "hmm", "mm"}
        filled_pause_count = sum(
            1 for word in timed_words if normalize_word(str(word.hyp_word or "")) in filled_pause_words
        )
        filled_pause_rate = (
            filled_pause_count / max(1, realized_word_count) * 100.0
            if realized_word_count
            else None
        )
        run_lengths: List[int] = []
        cur = 1 if timed_words else 0
        for g in gaps:
            if g >= 0.18:
                if cur:
                    run_lengths.append(cur)
                cur = 1
            else:
                cur += 1
        if cur:
            run_lengths.append(cur)
        mlr = float(np.mean(run_lengths)) if run_lengths else None

        final_slope = None
        if timed_words:
            last_end = float(timed_words[-1].end)
            start = max(0.0, last_end - 0.8)
            hz_slope = pitch_slope(pitch.get("times", []) or [], pitch.get("f0", []) or [], start, last_end)
            if hz_slope is not None and pitch_mean and pitch_mean > 0:
                # Convert approximate Hz/s to semitone/s around speaker median.
                final_slope = 12.0 / math.log(2) * hz_slope / pitch_mean

        return ProsodyFeatures(
            speech_rate_wpm=round(speech_rate, 3) if speech_rate is not None else None,
            articulation_rate_wpm=round(articulation_rate, 3) if articulation_rate is not None else None,
            pause_count=len(pauses),
            total_pause_s=round(float(sum(pauses)), 3),
            pause_ratio=round(pause_ratio, 3) if pause_ratio is not None else None,
            mean_length_run_words=round(mlr, 3) if mlr is not None else None,
            npvi_word_duration=round(npvi, 3) if npvi is not None else None,
            word_duration_cv=round(dur_cv, 3) if dur_cv is not None else None,
            pitch_mean_hz=round(pitch_mean, 2) if pitch_mean is not None else None,
            pitch_range_hz=round(pitch_range_hz, 2) if pitch_range_hz is not None else None,
            pitch_range_st=round(pitch_range_st, 2) if pitch_range_st is not None else None,
            final_slope_st_per_s=round(final_slope, 3) if final_slope is not None else None,
            intensity_mean_db=_as_float(acoustic.get("intensity_mean_db")),
            raw={
                "tracks": tracks,
                "word_durations": word_durations,
                "word_gaps": gaps,
                "pause_gaps": pauses,
                "long_pause_gaps": long_pauses,
                "speech_window_s": speech_window_s,
                "articulation_s": articulation_s,
                "realized_word_count": realized_word_count,
                "pause_rate_per_min": pause_rate_per_min,
                "mean_pause_s": mean_pause_s,
                "long_pause_ratio": long_pause_ratio,
                "filled_pause_count": filled_pause_count,
                "filled_pause_rate_per_100_words": filled_pause_rate,
            },
        )

    def build_textpa_cues(self, alignment: AlignmentResult, hypotheses: List[PhonemeHypothesis]) -> TextPACues:
        # This extends TextPA's transcript/CMU/IPA textual cues with scores but keeps them auditable.
        transcript_words = [a.hyp_word or a.ref_word or "" for a in alignment.word_alignments if a.operation != "delete" and (a.hyp_word or a.ref_word)]
        transcript = " ".join(transcript_words)
        token_by_index = {t.index: t for t in alignment.reference_tokens}
        seq = [a for a in alignment.word_alignments if a.ref_index is not None and a.operation != "insert"]
        seq.sort(key=lambda a: a.ref_index if a.ref_index is not None else 10**9)
        cmu_parts: List[str] = []
        canonical_ipa: List[str] = []
        for i, ali in enumerate(seq):
            tok = token_by_index.get(ali.ref_index if ali.ref_index is not None else -1)
            if tok is None:
                continue
            cmu_parts.extend([re.sub(r"[0-2]", "", ph).upper() for ph in tok.canonical_arpabet])
            if i + 1 < len(seq) and ali.end is not None and seq[i + 1].start is not None:
                gap = float(seq[i + 1].start) - float(ali.end)
                if gap >= 0.08:
                    cmu_parts.append(f"({gap:.2f}s pause)")
        # Canonical and recognized phones must cover exactly the same words.
        # Never fall back to canonical phone intervals, which would be circular.
        hyp_phones: List[str] = []
        ordered_hypotheses = sorted(
            hypotheses,
            key=lambda item: (
                item.ref_index if item.ref_index is not None else 10**9,
                float(item.start or 0.0),
            ),
        )
        for hypothesis in ordered_hypotheses:
            canonical_ipa.extend(
                [simplify_phone(phone) for phone in hypothesis.canonical_phones if simplify_phone(phone)]
            )
            hyp_phones.extend(
                [simplify_phone(phone) for phone in hypothesis.predicted_phones if simplify_phone(phone)]
            )
        ipa_match = global_sequence_similarity(canonical_ipa, hyp_phones) if canonical_ipa or hyp_phones else None
        return TextPACues(
            transcript=transcript,
            phonemes_cmu=" ".join(cmu_parts),
            phonemes_ipa=" ".join(hyp_phones),
            canonical_ipa=" ".join(canonical_ipa),
            ipa_match_score=round(ipa_match, 4) if ipa_match is not None else None,
            cmu_match_score=None,
        )

    def detect_word_issues(self, alignment: AlignmentResult) -> List[EvidenceIssue]:
        issues: List[EvidenceIssue] = []
        for ali in alignment.word_alignments:
            if ali.operation == "equal":
                continue
            if ali.operation == "delete":
                issues.append(
                    self.issue(
                        IssueType.WORD_OMISSION,
                        Severity.HIGH,
                        ali.ref_index,
                        ali.ref_word,
                        observed="[missing]",
                        expected=ali.ref_word,
                        start=ali.start,
                        end=ali.end,
                        confidence=0.85,
                        evidence={"operation": ali.operation},
                        hint=f"这个词可能被漏读了：{ali.ref_word}。",
                    )
                )
            elif ali.operation == "insert":
                issues.append(
                    self.issue(
                        IssueType.WORD_INSERTION,
                        Severity.MEDIUM,
                        None,
                        ali.hyp_word,
                        observed=ali.hyp_word,
                        expected="[no word in reference]",
                        start=ali.start,
                        end=ali.end,
                        confidence=0.75,
                        evidence={"operation": ali.operation},
                        hint=f"录音中可能多读了一个词：{ali.hyp_word}。",
                    )
                )
            elif ali.operation == "substitute":
                ratio = difflib.SequenceMatcher(None, normalize_word(ali.ref_word or ""), normalize_word(ali.hyp_word or "")).ratio()
                issues.append(
                    self.issue(
                        IssueType.WORD_SUBSTITUTION,
                        Severity.HIGH if ratio < 0.5 else Severity.MEDIUM,
                        ali.ref_index,
                        ali.ref_word,
                        observed=ali.hyp_word,
                        expected=ali.ref_word,
                        start=ali.start,
                        end=ali.end,
                        confidence=max(0.55, 1.0 - ratio),
                        evidence={"operation": ali.operation, "string_similarity": round(ratio, 3)},
                        hint=f"目标词是 {ali.ref_word}，系统听到的更接近 {ali.hyp_word}。",
                    )
                )
        return issues

    def detect_phoneme_issues(self, waveform: Waveform, alignment: AlignmentResult) -> Tuple[List[PhonemeHypothesis], List[EvidenceIssue]]:
        hypotheses: List[PhonemeHypothesis] = []
        issues: List[EvidenceIssue] = []
        if not self.phoneme_recognizer.is_available():
            return hypotheses, issues
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
                canonical_model = self.phoneme_recognizer.canonical_phones(token.normalized, fallback)
                predicted, conf, ctc_gop = self.phoneme_recognizer.recognize(segment, canonical_model)
            except Exception as exc:
                logger.warning("Phoneme recognition failed for word=%s: %s", ali.ref_word, exc)
                continue
            canonical = [simplify_phone(p) for p in canonical_model if simplify_phone(p)]
            predicted_simple = [simplify_phone(p) for p in predicted if simplify_phone(p)]
            hypotheses.append(
                PhonemeHypothesis(
                    ref_index=int(ali.ref_index),
                    word=token.normalized,
                    start=ali.start,
                    end=ali.end,
                    predicted_phones=predicted_simple,
                    canonical_phones=canonical,
                    confidence=conf,
                    ctc_gop=ctc_gop,
                )
            )
            if conf is None or conf < self.settings.phoneme_min_confidence:
                continue
            edits = [(op, exp, obs) for op, exp, obs in phone_edit_alignment(canonical, predicted_simple) if op != "equal"]
            edit_ratio = len(edits) / max(1, len(canonical))
            edits = sorted(edits, key=lambda item: phone_issue_priority(*item), reverse=True)
            for op, exp, obs in edits[: max(0, self.settings.phoneme_max_issues_per_word)]:
                issue_type = {
                    "substitute": IssueType.PHONEME_SUBSTITUTION,
                    "delete": IssueType.PHONEME_DELETION,
                    "insert": IssueType.PHONEME_INSERTION,
                }[op]
                hint = phone_hint(exp, obs, token.normalized)
                issues.append(
                    self.issue(
                        issue_type,
                        Severity.HIGH if edit_ratio > self.settings.phoneme_max_edit_ratio or ali.operation != "equal" else Severity.MEDIUM,
                        ali.ref_index,
                        token.normalized,
                        observed=obs or "[missing]",
                        expected=exp or "[extra]",
                        start=ali.start,
                        end=ali.end,
                        confidence=min(0.9, max(0.45, conf or 0.55)),
                        evidence={
                            "canonical_phones": canonical,
                            "predicted_phones": predicted_simple,
                            "phone_operation": op,
                            "recognizer_confidence": conf,
                            "ctc_gop": ctc_gop,
                            "edit_ratio": round(edit_ratio, 3),
                            "timing_source": word_timing_source(ali),
                        },
                        hint=hint,
                    )
                )
        return hypotheses, issues

    def detect_pause_issues(self, alignment: AlignmentResult) -> List[EvidenceIssue]:
        """Detect missing/unexpected pauses using word gaps + reference boundary type.

        Innovation over TextPA: pauses are not only embedded as "(0.21s pause)" text;
        each pause is interpreted against punctuation and lightweight phrase/clause
        boundary evidence.
        """
        issues: List[EvidenceIssue] = []
        token_by_index = {t.index: t for t in alignment.reference_tokens}
        seq = [a for a in alignment.word_alignments if a.ref_index is not None and a.operation != "insert"]
        seq.sort(key=lambda a: a.ref_index if a.ref_index is not None else 10**9)
        for cur, nxt in zip(seq, seq[1:]):
            if cur.end is None or nxt.start is None or cur.ref_index is None or nxt.ref_index is None:
                continue
            gap = float(nxt.start) - float(cur.end)
            if gap < 0:
                continue
            token = token_by_index.get(cur.ref_index)
            next_token = token_by_index.get(nxt.ref_index)
            if token is None or next_token is None:
                continue
            boundary = token.boundary_after or "none"
            expected_min, expected_max = expected_pause_range(boundary)
            evidence = {
                "gap_s": round(gap, 3),
                "boundary_after": boundary,
                "punctuation": token.after_punct,
                "previous_word": token.normalized,
                "next_word": next_token.normalized,
                "expected_range_s": [expected_min, expected_max],
            }
            if boundary in {"phrase", "clause", "sentence"} and gap < expected_min:
                issues.append(
                    self.issue(
                        IssueType.PAUSE_MISSING,
                        Severity.LOW if boundary == "phrase" else Severity.MEDIUM,
                        cur.ref_index,
                        cur.ref_word,
                        observed=f"pause={gap:.2f}s",
                        expected=f"{boundary} boundary pause >= {expected_min:.2f}s",
                        start=cur.end,
                        end=nxt.start,
                        confidence=0.58 if boundary == "phrase" else 0.70,
                        evidence=evidence,
                        hint="这里是短语/分句边界，朗读时可以稍微停一下，帮助听者分组理解。",
                    )
                )
            elif boundary == "none" and gap > expected_max:
                issues.append(
                    self.issue(
                        IssueType.PAUSE_UNEXPECTED,
                        Severity.HIGH if gap > 0.85 else Severity.MEDIUM,
                        cur.ref_index,
                        f"{token.normalized} {next_token.normalized}",
                        observed=f"pause={gap:.2f}s",
                        expected=f"within-phrase gap <= {expected_max:.2f}s",
                        start=cur.end,
                        end=nxt.start,
                        confidence=min(0.92, 0.45 + gap),
                        evidence=evidence,
                        hint="这里像是短语内部停顿，容易让一句话听起来断裂；建议把这两个词连成一个节奏组。",
                    )
                )
            elif boundary in {"phrase", "clause"} and gap > expected_max + 0.55:
                issues.append(
                    self.issue(
                        IssueType.PAUSE_UNEXPECTED,
                        Severity.LOW,
                        cur.ref_index,
                        f"{token.normalized} {next_token.normalized}",
                        observed=f"pause={gap:.2f}s",
                        expected=f"{boundary} boundary pause roughly <= {expected_max:.2f}s",
                        start=cur.end,
                        end=nxt.start,
                        confidence=0.55,
                        evidence=evidence,
                        hint="这里可以停顿，但当前停顿偏长，可能让节奏显得犹豫。",
                    )
                )
        return issues

    def detect_linking_issues(self, alignment: AlignmentResult) -> List[EvidenceIssue]:
        """Detect missed connected-speech opportunities.

        This module deliberately reports naturalness suggestions, not absolute
        correctness errors, because linking is style- and rate-dependent.
        """
        issues: List[EvidenceIssue] = []
        token_by_index = {t.index: t for t in alignment.reference_tokens}
        seq = [a for a in alignment.word_alignments if is_reliable_timed_word(a, token_by_index.get(a.ref_index), self.settings)]
        seq.sort(key=lambda a: a.ref_index if a.ref_index is not None else 10**9)
        for cur, nxt in zip(seq, seq[1:]):
            if cur.ref_index is None or nxt.ref_index is None or cur.end is None or nxt.start is None:
                continue
            cur_tok = token_by_index.get(cur.ref_index)
            nxt_tok = token_by_index.get(nxt.ref_index)
            if cur_tok is None or nxt_tok is None:
                continue
            if cur_tok.boundary_after in {"phrase", "clause", "sentence"} or any(p in cur_tok.after_punct for p in ".!?;:,—"):
                continue
            opp = linking_opportunity(cur_tok, nxt_tok)
            if opp is None:
                continue
            gap = float(nxt.start) - float(cur.end)
            if gap < 0:
                continue
            threshold = self.settings.linking_gap_threshold_s
            if opp["type"] in {"weak_form_chunk", "c_v_linking"}:
                threshold = min(threshold, 0.16)
            if gap > threshold:
                issues.append(
                    self.issue(
                        IssueType.LINKING_MISSING,
                        Severity.LOW,
                        cur.ref_index,
                        f"{cur_tok.normalized} {nxt_tok.normalized}",
                        observed=f"gap={gap:.2f}s",
                        expected=opp["description"],
                        start=cur.end,
                        end=nxt.start,
                        confidence=min(0.82, 0.40 + gap * 1.4),
                        evidence={
                            **opp,
                            "previous_word": cur_tok.normalized,
                            "next_word": nxt_tok.normalized,
                            "gap_s": round(gap, 3),
                            "threshold_s": threshold,
                        },
                        hint=f"{cur_tok.normalized} 和 {nxt_tok.normalized} 之间可以更自然地连起来读；这里更像两个孤立单词。",
                    )
                )
        return issues

    def detect_stress_issues(self, alignment: AlignmentResult, tracks: Dict[str, Any]) -> List[EvidenceIssue]:
        """Detect lexical stress errors in multi-syllable words.

        Canonical stress comes from CMUdict digits. Observed prominence is
        estimated from duration, intensity and F0 in vowel nuclei or estimated
        syllable windows.
        """
        issues: List[EvidenceIssue] = []
        intensity = tracks.get("intensity", {}) if isinstance(tracks, dict) else {}
        pitch = tracks.get("pitch", {}) if isinstance(tracks, dict) else {}
        itimes = intensity.get("times", []) or []
        dbs = intensity.get("db", []) or []
        ptimes = pitch.get("times", []) or []
        f0s = pitch.get("f0", []) or []
        if not itimes and not ptimes:
            return issues
        token_by_index = {t.index: t for t in alignment.reference_tokens}
        for ali in alignment.word_alignments:
            token = token_by_index.get(ali.ref_index)
            if not is_reliable_timed_word(ali, token, self.settings, require_content=True):
                continue
            if token is None or len(token.stress_pattern) < 2 or 1 not in token.stress_pattern:
                continue
            windows = syllable_windows_from_alignment(ali, token.stress_pattern)
            if len(windows) != len(token.stress_pattern):
                windows = equal_syllable_windows(float(ali.start), float(ali.end), len(token.stress_pattern))
            prominences: List[Dict[str, Any]] = []
            for start, end in windows:
                dur = max(0.01, end - start)
                energy = track_mean(itimes, dbs, start, end)
                f0_mean = track_mean(ptimes, [x if x and x > 0 else float('nan') for x in f0s], start, end)
                prominences.append({"start": start, "end": end, "duration": dur, "intensity_db": energy, "f0_hz": f0_mean})
            dur_vals = [p["duration"] for p in prominences]
            int_vals = [p["intensity_db"] for p in prominences]
            f0_vals = [p["f0_hz"] for p in prominences]
            dur_z = zscores(dur_vals)
            int_z = zscores([v if v is not None else np.nan for v in int_vals])
            f0_z = zscores([v if v is not None else np.nan for v in f0_vals])
            scores = []
            for i in range(len(prominences)):
                vals = [dur_z[i], int_z[i], f0_z[i]]
                vals = [v for v in vals if math.isfinite(v)]
                scores.append(float(np.mean(vals)) if vals else 0.0)
            primary_idx = token.stress_pattern.index(1)
            observed_idx = int(np.argmax(scores)) if scores else primary_idx
            margin = scores[primary_idx] - max([s for i, s in enumerate(scores) if i != primary_idx] or [-10.0])
            for i, p in enumerate(prominences):
                p["prominence_score"] = round(scores[i], 3)
                p["expected_stress"] = token.stress_pattern[i]
            if observed_idx != primary_idx and margin < self.settings.stress_prominence_margin:
                issues.append(
                    self.issue(
                        IssueType.STRESS,
                        Severity.MEDIUM,
                        ali.ref_index,
                        token.normalized,
                        observed=f"syllable {observed_idx + 1} more prominent",
                        expected=f"primary stress on syllable {primary_idx + 1}",
                        start=ali.start,
                        end=ali.end,
                        confidence=0.62,
                        evidence={
                            "stress_pattern": token.stress_pattern,
                            "expected_primary_syllable": primary_idx + 1,
                            "observed_prominent_syllable": observed_idx + 1,
                            "prominence_margin": round(margin, 3),
                            "syllables": prominences,
                        },
                        hint="这个词的主重音位置可能不自然。练习时把目标重读音节读得更长、更响亮，并弱化其他音节。",
                    )
                )
            elif margin < 0.05:
                issues.append(
                    self.issue(
                        IssueType.STRESS,
                        Severity.LOW,
                        ali.ref_index,
                        token.normalized,
                        observed="primary stress contrast weak",
                        expected=f"clear stress on syllable {primary_idx + 1}",
                        start=ali.start,
                        end=ali.end,
                        confidence=0.45,
                        evidence={
                            "stress_pattern": token.stress_pattern,
                            "expected_primary_syllable": primary_idx + 1,
                            "prominence_margin": round(margin, 3),
                            "syllables": prominences,
                        },
                        hint="这个词的主重音不够突出，听起来每个音节强度比较平均。",
                    )
                )
        return issues

    def detect_intonation_issues(self, alignment: AlignmentResult, tracks: Dict[str, Any]) -> List[EvidenceIssue]:
        issues: List[EvidenceIssue] = []
        pitch = tracks.get("pitch", {}) if isinstance(tracks, dict) else {}
        times = pitch.get("times", []) or []
        f0 = pitch.get("f0", []) or []
        voiced = [float(x) for x in f0 if x and float(x) > 0]
        if not times or not f0 or len(voiced) < 6:
            return issues
        median_f0 = float(np.median(voiced))
        token_by_index = {t.index: t for t in alignment.reference_tokens}
        groups: Dict[int, List[WordAlignment]] = {}
        for ali in alignment.word_alignments:
            tok = token_by_index.get(ali.ref_index)
            if not is_reliable_timed_word(ali, tok, self.settings):
                continue
            groups.setdefault(tok.sentence_index, []).append(ali)
        for sentence_idx, words in groups.items():
            words = [w for w in words if w.start is not None and w.end is not None]
            if len(words) < 2:
                continue
            words.sort(key=lambda w: w.start or 0.0)
            sent_start = float(words[0].start or 0.0)
            sent_end = float(words[-1].end or 0.0)
            last = words[-1]
            tok = token_by_index.get(last.ref_index) if last.ref_index is not None else None
            if tok is None:
                continue
            end_punct = tok.after_punct[-1] if tok.after_punct and tok.after_punct[-1] in ".!?" else "."
            start = max(sent_start, sent_end - 0.8)
            slope_hz = pitch_slope(times, f0, start, sent_end)
            if slope_hz is not None and median_f0 > 0:
                slope_st = 12.0 / math.log(2) * slope_hz / median_f0
                if end_punct in ".!" and slope_st > self.settings.statement_rise_threshold_st_per_s:
                    issues.append(
                        self.issue(
                            IssueType.INTONATION,
                            Severity.LOW,
                            last.ref_index,
                            last.ref_word,
                            observed=f"final pitch rises {slope_st:.2f} st/s",
                            expected="falling or stable statement ending",
                            start=start,
                            end=sent_end,
                            confidence=0.55,
                            evidence={"sentence_index": sentence_idx, "final_punctuation": end_punct, "final_slope_st_per_s": round(slope_st, 3)},
                            hint="陈述句结尾通常可以略微下降或保持稳定；当前句尾上扬较明显，可能听起来像疑问句。",
                        )
                    )
                elif end_punct == "?" and slope_st < self.settings.question_fall_threshold_st_per_s:
                    issues.append(
                        self.issue(
                            IssueType.INTONATION,
                            Severity.LOW,
                            last.ref_index,
                            last.ref_word,
                            observed=f"final pitch falls {slope_st:.2f} st/s",
                            expected="rising or clearly focused question ending",
                            start=start,
                            end=sent_end,
                            confidence=0.50,
                            evidence={"sentence_index": sentence_idx, "final_punctuation": end_punct, "final_slope_st_per_s": round(slope_st, 3)},
                            hint="疑问句结尾可以根据语境适当上扬，或者至少突出疑问信息焦点。",
                        )
                    )
            segment_f0 = [float(v) for t, v in zip(times, f0) if sent_start <= float(t) <= sent_end and v and float(v) > 0]
            if len(segment_f0) >= 6 and median_f0 > 0:
                st = [12.0 * math.log2(v / median_f0) for v in segment_f0]
                range_st = float(np.percentile(st, 95) - np.percentile(st, 5))
                if range_st < self.settings.monotone_pitch_range_st and sent_end - sent_start > 2.0:
                    issues.append(
                        self.issue(
                            IssueType.INTONATION,
                            Severity.LOW,
                            words[0].ref_index,
                            "sentence",
                            observed=f"pitch range={range_st:.2f} st",
                            expected=f"pitch range > {self.settings.monotone_pitch_range_st:.1f} st for expressive reading",
                            start=sent_start,
                            end=sent_end,
                            confidence=0.48,
                            evidence={"sentence_index": sentence_idx, "pitch_range_st": round(range_st, 3)},
                            hint="这句话整体音高变化较小，听起来可能偏平。可以在关键词上稍微提高音高和音量。",
                        )
                    )
        return issues

    def detect_sentence_stress_issues(self, alignment: AlignmentResult, tracks: Dict[str, Any]) -> List[EvidenceIssue]:
        issues: List[EvidenceIssue] = []
        intensity = tracks.get("intensity", {}) if isinstance(tracks, dict) else {}
        itimes = intensity.get("times", []) or []
        dbs = intensity.get("db", []) or []
        if not itimes or not dbs:
            return issues
        token_by_index = {t.index: t for t in alignment.reference_tokens}
        content_vals: List[float] = []
        function_vals: List[float] = []
        examples: List[Dict[str, Any]] = []
        for ali in alignment.word_alignments:
            tok = token_by_index.get(ali.ref_index)
            if not is_reliable_timed_word(ali, tok, self.settings):
                continue
            e = track_mean(itimes, dbs, float(ali.start), float(ali.end))
            if e is None:
                continue
            if tok.is_content_word:
                content_vals.append(e)
            else:
                function_vals.append(e)
            if len(examples) < 20:
                examples.append({"word": tok.normalized, "content": tok.is_content_word, "intensity_db": round(e, 2)})
        if len(content_vals) >= 3 and len(function_vals) >= 2:
            diff = float(np.mean(content_vals) - np.mean(function_vals))
            if diff < 1.0:
                issues.append(
                    self.issue(
                        IssueType.SENTENCE_STRESS,
                        Severity.LOW,
                        None,
                        "sentence stress",
                        observed=f"content-function contrast={diff:.2f}dB",
                        expected="content words should usually be more prominent than function words",
                        start=None,
                        end=None,
                        confidence=0.46,
                        evidence={"content_mean_db": round(float(np.mean(content_vals)), 2), "function_mean_db": round(float(np.mean(function_vals)), 2), "examples": examples},
                        hint="整体重音可能过于平均。英语朗读通常突出名词、动词、形容词等内容词，弱化冠词、介词、助动词等功能词。",
                    )
                )
        return issues

    def detect_rhythm_issues(self, alignment: AlignmentResult, prosody: ProsodyFeatures) -> List[EvidenceIssue]:
        issues: List[EvidenceIssue] = []
        timed_words = [a for a in alignment.word_alignments if a.ref_index is not None and a.operation != "delete" and a.start is not None and a.end is not None]
        if len(timed_words) < 4:
            return issues
        pause_ratio = prosody.pause_ratio
        mlr = prosody.mean_length_run_words
        speech_rate = prosody.speech_rate_wpm
        if pause_ratio is not None and pause_ratio > 0.30:
            issues.append(
                self.issue(
                    IssueType.RHYTHM,
                    Severity.MEDIUM,
                    None,
                    "overall rhythm",
                    observed=f"pause ratio={pause_ratio:.2f}",
                    expected="lower pause ratio for smooth read-aloud speech",
                    start=None,
                    end=None,
                    confidence=min(0.75, 0.35 + pause_ratio),
                    evidence={"pause_ratio": pause_ratio, "total_pause_s": prosody.total_pause_s, "pause_count": prosody.pause_count},
                    hint="整体停顿比例偏高，节奏容易显得断续。建议先按短语分组，再连续读完每个短语。",
                )
            )
        if mlr is not None and mlr < 3.0 and len(timed_words) >= 8:
            issues.append(
                self.issue(
                    IssueType.RHYTHM,
                    Severity.LOW,
                    None,
                    "mean length of run",
                    observed=f"{mlr:.1f} words per run",
                    expected="longer phrase-level runs",
                    start=None,
                    end=None,
                    confidence=0.50,
                    evidence={"mean_length_run_words": mlr},
                    hint="两次停顿之间说出的词数偏少，可能每几个词就断一次。可以把句子切成意义块练习。",
                )
            )
        if speech_rate is not None and speech_rate < 85 and len(timed_words) >= 5:
            issues.append(
                self.issue(
                    IssueType.RHYTHM,
                    Severity.LOW,
                    None,
                    "speech rate",
                    observed=f"speech rate={speech_rate:.1f} wpm",
                    expected="moderate read-aloud rate",
                    start=None,
                    end=None,
                    confidence=0.42,
                    evidence={"speech_rate_wpm": speech_rate, "articulation_rate_wpm": prosody.articulation_rate_wpm},
                    hint="整体语速偏慢。可以先保证准确，再逐步把短语内部读得更连续。",
                )
            )
        return issues

    def issue(
        self,
        issue_type: IssueType,
        severity: Severity,
        ref_index: Optional[int],
        word: Optional[str],
        observed: Optional[str],
        expected: Optional[str],
        start: Optional[float],
        end: Optional[float],
        confidence: float,
        evidence: Dict[str, Any],
        hint: str,
    ) -> EvidenceIssue:
        self._issue_counter += 1
        return EvidenceIssue(
            issue_id=f"I{self._issue_counter:04d}",
            type=issue_type,
            severity=severity,
            ref_index=ref_index,
            word=word,
            observed=observed,
            expected=expected,
            start=start,
            end=end,
            confidence=round(float(confidence), 3),
            evidence=evidence,
            suggestion_hint=hint,
        )


# ---------- Utility functions ----------


def is_equal_reference_word(ali: WordAlignment, token: Optional[ReferenceToken]) -> bool:
    if token is None or ali.operation != "equal" or not ali.hyp_word:
        return False
    return normalize_word(ali.hyp_word) == token.normalized


def word_timing_source(ali: WordAlignment) -> str:
    sources = {p.source for p in ali.phones if p.source}
    if "mfa" in sources:
        return "mfa"
    if sources:
        return "+".join(sorted(sources))
    return "asr"


def is_reliable_timed_word(
    ali: WordAlignment,
    token: Optional[ReferenceToken],
    settings: CaptSettings,
    *,
    require_content: bool = False,
    min_duration_s: float = 0.05,
) -> bool:
    if token is None or ali.ref_index is None or ali.start is None or ali.end is None:
        return False
    if float(ali.end) <= float(ali.start):
        return False
    if float(ali.end) - float(ali.start) < min_duration_s:
        return False
    if settings.prosody_require_equal_words and not is_equal_reference_word(ali, token):
        return False
    if require_content and not token.is_content_word:
        return False
    if ali.confidence is not None and ali.confidence < settings.min_word_confidence:
        return False
    return True


def should_analyze_phoneme_word(ali: WordAlignment, token: ReferenceToken, settings: CaptSettings) -> bool:
    if not token.canonical_ipa or ali.start is None or ali.end is None:
        return False
    if settings.phoneme_require_equal_word and not is_equal_reference_word(ali, token):
        return False
    if settings.phoneme_skip_function_words and not token.is_content_word:
        return False
    if ali.confidence is not None and ali.confidence < settings.min_word_confidence:
        return False
    duration = float(ali.end) - float(ali.start)
    if duration < settings.phoneme_min_word_duration_s:
        return False
    canonical = [simplify_phone(p) for p in token.canonical_ipa if simplify_phone(p)]
    return len(canonical) >= 2


def phone_issue_priority(op: str, expected: Optional[str], observed: Optional[str]) -> float:
    score = 0.0
    if expected and observed and (expected, observed) in COMMON_HINTS:
        score += 3.0
    if op == "substitute":
        score += 2.0
    elif op == "delete":
        score += 1.5
    else:
        score += 1.0
    if expected and is_consonant_phone(expected):
        score += 1.0
    if observed and is_consonant_phone(observed):
        score += 0.5
    if expected in {"ə", "ʌ"} or observed in {"ə", "ʌ"}:
        score -= 0.25
    return score


def split_phone_string(decoded: str) -> List[str]:
    decoded = decoded.strip()
    decoded = decoded.replace("|", " ").replace("<pad>", " ").replace("<s>", " ").replace("</s>", " ")
    decoded = re.sub(r"\s+", " ", decoded)
    phones = [simplify_phone(p) for p in decoded.split(" ") if simplify_phone(p)]
    # Remove repeated CTC-ish duplicates if decoding did not collapse them.
    out: List[str] = []
    for p in phones:
        if not out or out[-1] != p:
            out.append(p)
    return out


def simplify_phone(phone: str) -> str:
    p = str(phone).strip().lower()
    p = re.sub(r"[ˈˌː:0-2]", "", p)
    p = p.replace("ɡ", "g")
    p = p.replace("ɹ", "r")
    p = p.replace("ɚ", "ɝ")
    p = p.replace("əɹ", "ɝ")
    p = p.replace("t͡ʃ", "tʃ").replace("d͡ʒ", "dʒ")
    p = re.sub(r"[^a-zɑæʌɔɛɝɪʊəθðŋʃʒʧʤɒɜɐøœɯɨʉɾʔ]+", "", p)
    return p


def phone_edit_alignment(ref: Sequence[str], hyp: Sequence[str]) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """Return op, expected, observed alignment for phones."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[(0, 0, "")] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = (i - 1, 0, "delete")
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = (0, j - 1, "insert")
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            same = ref[i - 1] == hyp[j - 1]
            sub_cost = 0 if same else phone_substitution_cost(ref[i - 1], hyp[j - 1])
            candidates = [
                (dp[i - 1][j - 1] + sub_cost, i - 1, j - 1, "equal" if same else "substitute"),
                (dp[i - 1][j] + 1, i - 1, j, "delete"),
                (dp[i][j - 1] + 1, i, j - 1, "insert"),
            ]
            best = min(candidates, key=lambda x: x[0])
            dp[i][j] = best[0]
            back[i][j] = (best[1], best[2], best[3])
    ops: List[Tuple[str, Optional[str], Optional[str]]] = []
    i, j = n, m
    while i > 0 or j > 0:
        pi, pj, op = back[i][j]
        if op in {"equal", "substitute"}:
            ops.append((op, ref[i - 1], hyp[j - 1]))
        elif op == "delete":
            ops.append((op, ref[i - 1], None))
        elif op == "insert":
            ops.append((op, None, hyp[j - 1]))
        i, j = pi, pj
    ops.reverse()
    return ops


def phone_substitution_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    # Similar vowels/consonants are cheaper, encouraging substitutions over
    # delete+insert for understandable learner mistakes.
    if (is_vowel_phone(a) and is_vowel_phone(b)) or (is_consonant_phone(a) and is_consonant_phone(b)):
        return 0.75
    return 1.0


def phone_hint(expected: Optional[str], observed: Optional[str], word: str) -> str:
    if expected and observed and (expected, observed) in COMMON_HINTS:
        return COMMON_HINTS[(expected, observed)]
    if expected and observed:
        return f"在 {word} 中，目标音素 /{expected}/ 听起来更接近 /{observed}/，建议单独练习该音素后再放回单词。"
    if expected and not observed:
        if is_consonant_phone(expected):
            return f"{word} 中的 /{expected}/ 可能被弱化或省略了，注意词尾/词中的辅音要交代清楚。"
        return f"{word} 中的元音 /{expected}/ 可能不够清楚，建议放慢速度练习。"
    if observed and not expected:
        return f"{word} 附近可能多出了 /{observed}/，注意不要在辅音连缀中插入额外元音。"
    return "建议放慢速度，单独练习该词的标准音素序列。"


def track_mean(times: Sequence[float], values: Sequence[float], start: float, end: float) -> Optional[float]:
    vals = [float(v) for t, v in zip(times, values) if start <= float(t) <= end and math.isfinite(float(v))]
    if not vals:
        return None
    return float(np.mean(vals))


def pitch_slope(times: Sequence[float], f0: Sequence[float], start: float, end: float) -> Optional[float]:
    xs: List[float] = []
    ys: List[float] = []
    for t, v in zip(times, f0):
        t = float(t)
        v = float(v)
        if start <= t <= end and v > 0 and math.isfinite(v):
            xs.append(t)
            ys.append(v)
    if len(xs) < 4 or max(xs) - min(xs) < 0.1:
        return None
    # Robust-ish slope via first/last tercile means rather than full linear fit.
    k = max(1, len(xs) // 3)
    y1 = float(np.mean(ys[:k]))
    y2 = float(np.mean(ys[-k:]))
    x1 = float(np.mean(xs[:k]))
    x2 = float(np.mean(xs[-k:]))
    if x2 <= x1:
        return None
    return (y2 - y1) / (x2 - x1)


def expected_pause_range(boundary: str) -> Tuple[float, float]:
    if boundary == "sentence":
        return 0.30, 1.60
    if boundary == "clause":
        return 0.20, 1.10
    if boundary == "phrase":
        return 0.12, 0.75
    if boundary == "weak":
        return 0.05, 0.35
    return 0.00, 0.22


def linking_opportunity(cur_tok: ReferenceToken, nxt_tok: ReferenceToken) -> Optional[Dict[str, Any]]:
    weak_pairs = {
        ("want", "to"): "want to can reduce/link toward /wɑːnə/ in natural speech",
        ("going", "to"): "going to often reduces toward /gənə/ in natural speech",
        ("have", "to"): "have to often links as /hæftə/",
        ("kind", "of"): "kind of often links/reduces toward /kaɪndəv/",
        ("pick", "it"): "pick it is a strong C-V linking opportunity",
    }
    pair = (cur_tok.normalized, nxt_tok.normalized)
    if pair in weak_pairs:
        return {"type": "weak_form_chunk", "description": weak_pairs[pair]}
    if not cur_tok.canonical_ipa or not nxt_tok.canonical_ipa:
        return None
    last_phone = simplify_phone(cur_tok.canonical_ipa[-1])
    first_phone = simplify_phone(nxt_tok.canonical_ipa[0])
    if is_consonant_phone(last_phone) and is_vowel_phone(first_phone):
        return {
            "type": "c_v_linking",
            "description": f"consonant-vowel linking {last_phone}+{first_phone}",
            "previous_final_phone": last_phone,
            "next_initial_phone": first_phone,
        }
    if is_vowel_phone(last_phone) and is_vowel_phone(first_phone):
        glide = "j" if last_phone in {"i", "ɪ", "eɪ", "aɪ", "ɔɪ"} else "w" if last_phone in {"u", "ʊ", "oʊ", "aʊ"} else "glide"
        return {
            "type": "v_v_glide",
            "description": f"vowel-vowel transition may use a light /{glide}/ glide",
            "previous_final_phone": last_phone,
            "next_initial_phone": first_phone,
        }
    if last_phone == first_phone and last_phone:
        return {
            "type": "same_consonant",
            "description": f"same-consonant boundary /{last_phone}/ can be held smoothly rather than separated",
            "previous_final_phone": last_phone,
            "next_initial_phone": first_phone,
        }
    if last_phone in {"r", "ɝ"} and is_vowel_phone(first_phone):
        return {
            "type": "r_linking",
            "description": "r-linking before a vowel",
            "previous_final_phone": last_phone,
            "next_initial_phone": first_phone,
        }
    return None


def syllable_windows_from_alignment(ali: WordAlignment, stress_pattern: Sequence[int]) -> List[Tuple[float, float]]:
    vowels = [p for p in ali.phones if is_vowel_phone(p.phone) and p.start is not None and p.end is not None]
    if len(vowels) == len(stress_pattern):
        return [(float(v.start), float(v.end)) for v in vowels]
    return []


def equal_syllable_windows(start: float, end: float, n: int) -> List[Tuple[float, float]]:
    if n <= 0 or end <= start:
        return []
    step = (end - start) / n
    return [(start + i * step, start + (i + 1) * step) for i in range(n)]


def zscores(values: Sequence[Any]) -> List[float]:
    arr = np.asarray([float(v) if v is not None and math.isfinite(float(v)) else np.nan for v in values], dtype=float)
    if np.all(np.isnan(arr)):
        return [0.0 for _ in values]
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr))
    if std < 1e-8:
        return [0.0 if not np.isnan(v) else float("nan") for v in arr]
    return [float((v - mean) / std) if not np.isnan(v) else float("nan") for v in arr]


def normalized_pairwise_variability(durations: Sequence[float]) -> Optional[float]:
    vals = [float(x) for x in durations if x and x > 0]
    if len(vals) < 2:
        return None
    terms = []
    for a, b in zip(vals, vals[1:]):
        terms.append(abs(a - b) / ((a + b) / 2.0))
    return float(100.0 * np.mean(terms)) if terms else None


def sequence_similarity_smith_waterman(ref: Sequence[str], hyp: Sequence[str], match: float = 2.0, mismatch: float = -1.0, gap: float = -1.0) -> float:
    if not ref or not hyp:
        return 0.0
    n, m = len(ref), len(hyp)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    best = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score = match if ref[i - 1] == hyp[j - 1] else mismatch
            dp[i][j] = max(0.0, dp[i - 1][j - 1] + score, dp[i - 1][j] + gap, dp[i][j - 1] + gap)
            best = max(best, dp[i][j])
    denom = max(1.0, min(len(ref), len(hyp)) * match)
    return max(0.0, min(1.0, best / denom))



def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None
