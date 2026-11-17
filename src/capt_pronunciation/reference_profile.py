"""Build comparable acoustic-prosodic profiles for imitation analysis."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .audio_io import AudioProcessor
from .contracts import (
    AlignmentResult,
    AudioInfo,
    BoundaryFeature,
    CaptSettings,
    ReferenceProfile,
    SentenceProsodyFeature,
    WordAcousticFeature,
    WordAlignment,
)
from .diagnostics import (
    EvidenceAnalyzer,
    equal_syllable_windows,
    linking_opportunity,
    pitch_slope,
    syllable_windows_from_alignment,
    track_mean,
    zscores,
)


class ReferenceProfileBuilder:
    """Converts aligned audio into reference-comparable evidence targets."""

    def __init__(self, settings: CaptSettings):
        self.settings = settings
        self.audio = AudioProcessor(settings)
        # Reuse the existing prosody summarizer; it does not load heavy phoneme
        # models unless phoneme recognition is explicitly requested elsewhere.
        self._analyzer = EvidenceAnalyzer(settings)

    def build(
        self,
        audio_info: AudioInfo,
        alignment: AlignmentResult,
        reference_text: str,
        *,
        reference_text_source: str = "user",
        asr_text: str = "",
    ) -> ReferenceProfile:
        waveform = self.audio.load_wave(audio_info.path, sr=audio_info.sample_rate)
        acoustic = self.audio.summarize_acoustics(waveform, audio_info.vad_segments, len(alignment.reference_tokens))
        prosody = self._analyzer.compute_prosody_features(acoustic, alignment, audio_info.duration_s)
        tracks = prosody.raw.get("tracks", {}) if isinstance(prosody.raw, dict) else {}
        word_features = self._word_features(alignment, tracks)
        boundary_features = self._boundary_features(alignment, tracks)
        sentence_features = self._sentence_features(alignment, tracks, word_features)
        return ReferenceProfile(
            reference_text=reference_text,
            reference_text_source=reference_text_source,  # type: ignore[arg-type]
            asr_text=asr_text,
            audio=audio_info,
            alignment=alignment,
            prosody=prosody,
            word_features=word_features,
            boundary_features=boundary_features,
            sentence_features=sentence_features,
            raw={
                "profile_version": "reference_audio_v1",
                "alignment_source": alignment.source,
                "acoustic": {
                    "speech_rate_wpm": acoustic.get("speech_rate_wpm"),
                    "articulation_rate_wpm": acoustic.get("articulation_rate_wpm"),
                    "pause_count": acoustic.get("pause_count"),
                    "total_pause_s": acoustic.get("total_pause_s"),
                },
            },
        )

    def _word_features(self, alignment: AlignmentResult, tracks: Dict[str, Any]) -> List[WordAcousticFeature]:
        token_by_index = {t.index: t for t in alignment.reference_tokens}
        pitch = tracks.get("pitch", {}) if isinstance(tracks, dict) else {}
        intensity = tracks.get("intensity", {}) if isinstance(tracks, dict) else {}
        ptimes = pitch.get("times", []) or []
        f0s = pitch.get("f0", []) or []
        itimes = intensity.get("times", []) or []
        dbs = intensity.get("db", []) or []
        median_f0 = _median_positive(f0s)

        rows: List[Dict[str, Any]] = []
        for ali in alignment.word_alignments:
            if ali.ref_index is None or ali.start is None or ali.end is None or ali.operation == "delete":
                continue
            token = token_by_index.get(ali.ref_index)
            if token is None:
                continue
            start = float(ali.start)
            end = float(ali.end)
            if end <= start:
                continue
            segment_f0 = _track_values(ptimes, f0s, start, end, positive=True)
            f0_mean = float(np.mean(segment_f0)) if segment_f0 else None
            f0_peak = float(np.percentile(segment_f0, 90)) if len(segment_f0) >= 3 else f0_mean
            f0_range_st = _range_st(segment_f0, median_f0)
            intensity_mean = track_mean(itimes, dbs, start, end)
            syllables, prominent_syllable, stress_contrast = self._syllable_features(
                ali,
                token.stress_pattern,
                ptimes,
                f0s,
                itimes,
                dbs,
                median_f0,
            )
            rows.append(
                {
                    "ref_index": int(ali.ref_index),
                    "word": token.normalized,
                    "start": start,
                    "end": end,
                    "duration": max(0.001, end - start),
                    "intensity_mean_db": intensity_mean,
                    "f0_mean_hz": f0_mean,
                    "f0_peak_hz": f0_peak,
                    "f0_range_st": f0_range_st,
                    "prominent_syllable": prominent_syllable,
                    "stress_contrast": stress_contrast,
                    "syllables": syllables,
                }
            )

        dur_z = zscores([x["duration"] for x in rows])
        int_z = zscores([x["intensity_mean_db"] for x in rows])
        f0_z = zscores([x["f0_peak_hz"] for x in rows])
        range_z = zscores([x["f0_range_st"] for x in rows])
        out: List[WordAcousticFeature] = []
        for i, row in enumerate(rows):
            vals = [dur_z[i], int_z[i], f0_z[i], range_z[i]]
            vals = [v for v in vals if math.isfinite(v)]
            row["prominence"] = float(np.mean(vals)) if vals else 0.0
            out.append(WordAcousticFeature(**_round_nested(row)))
        return out

    def _syllable_features(
        self,
        ali: WordAlignment,
        stress_pattern: List[int],
        ptimes: List[float],
        f0s: List[float],
        itimes: List[float],
        dbs: List[float],
        median_f0: Optional[float],
    ) -> Tuple[List[Dict[str, Any]], Optional[int], Optional[float]]:
        if ali.ref_index is None or ali.start is None or ali.end is None:
            return [], None, None
        if not stress_pattern:
            return [], None, None
        windows = syllable_windows_from_alignment(ali, stress_pattern) if stress_pattern else []
        if not windows and stress_pattern:
            windows = equal_syllable_windows(float(ali.start), float(ali.end), len(stress_pattern))
        if not windows:
            return [], None, None

        rows: List[Dict[str, Any]] = []
        for start, end in windows:
            seg_f0 = _track_values(ptimes, f0s, start, end, positive=True)
            f0_peak = float(np.percentile(seg_f0, 90)) if len(seg_f0) >= 3 else float(np.mean(seg_f0)) if seg_f0 else None
            rows.append(
                {
                    "start": start,
                    "end": end,
                    "duration": max(0.001, end - start),
                    "intensity_db": track_mean(itimes, dbs, start, end),
                    "f0_peak_hz": f0_peak,
                    "f0_range_st": _range_st(seg_f0, median_f0),
                }
            )
        dur_z = zscores([x["duration"] for x in rows])
        int_z = zscores([x["intensity_db"] for x in rows])
        f0_z = zscores([x["f0_peak_hz"] for x in rows])
        range_z = zscores([x["f0_range_st"] for x in rows])
        scores: List[float] = []
        for i, row in enumerate(rows):
            vals = [
                0.35 * dur_z[i],
                0.35 * int_z[i],
                0.20 * f0_z[i],
                0.10 * range_z[i],
            ]
            vals = [v for v in vals if math.isfinite(v)]
            score = float(sum(vals)) if vals else 0.0
            row["prominence"] = score
            scores.append(score)
        if not scores:
            return [_round_nested(x) for x in rows], None, None
        best = int(np.argmax(scores))
        runner_up = max([s for i, s in enumerate(scores) if i != best] or [scores[best]])
        return [_round_nested(x) for x in rows], best + 1, float(scores[best] - runner_up)

    def _boundary_features(self, alignment: AlignmentResult, tracks: Dict[str, Any]) -> List[BoundaryFeature]:
        token_by_index = {t.index: t for t in alignment.reference_tokens}
        seq = [
            a
            for a in alignment.word_alignments
            if a.ref_index is not None and a.operation != "delete" and a.start is not None and a.end is not None
        ]
        seq.sort(key=lambda a: int(a.ref_index or 0))
        out: List[BoundaryFeature] = []
        for cur, nxt in zip(seq, seq[1:]):
            if cur.ref_index is None or nxt.ref_index is None or cur.end is None or nxt.start is None:
                continue
            cur_tok = token_by_index.get(cur.ref_index)
            nxt_tok = token_by_index.get(nxt.ref_index)
            if cur_tok is None or nxt_tok is None:
                continue
            gap = max(0.0, float(nxt.start) - float(cur.end))
            opp = linking_opportunity(cur_tok, nxt_tok)
            energy = max(0.0, min(1.0, 1.0 - gap / 0.25))
            voicing = _voicing_continuity(tracks, float(cur.end), float(nxt.start))
            has_linking = bool(
                opp
                and cur_tok.boundary_after not in {"phrase", "clause", "sentence"}
                and gap <= self.settings.imitation_linking_ref_gap_s
                and energy >= 0.55
            )
            out.append(
                BoundaryFeature(
                    left_index=int(cur.ref_index),
                    right_index=int(nxt.ref_index),
                    left_word=cur_tok.normalized,
                    right_word=nxt_tok.normalized,
                    start=float(cur.end),
                    end=float(nxt.start),
                    gap_s=round(gap, 3),
                    boundary_after=cur_tok.boundary_after,
                    linking_opportunity=opp,
                    energy_continuity=round(energy, 3),
                    voicing_continuity=round(voicing, 3) if voicing is not None else None,
                    has_linking=has_linking,
                )
            )
        return out

    def _sentence_features(
        self,
        alignment: AlignmentResult,
        tracks: Dict[str, Any],
        word_features: List[WordAcousticFeature],
    ) -> List[SentenceProsodyFeature]:
        pitch = tracks.get("pitch", {}) if isinstance(tracks, dict) else {}
        times = pitch.get("times", []) or []
        f0 = pitch.get("f0", []) or []
        median_f0 = _median_positive(f0)
        if not times or not f0 or median_f0 is None:
            return []
        token_by_index = {t.index: t for t in alignment.reference_tokens}
        wf_by_index = {w.ref_index: w for w in word_features}
        groups: Dict[int, List[WordAlignment]] = {}
        for ali in alignment.word_alignments:
            tok = token_by_index.get(ali.ref_index)
            if tok is None or ali.start is None or ali.end is None or ali.operation == "delete":
                continue
            groups.setdefault(tok.sentence_index, []).append(ali)
        out: List[SentenceProsodyFeature] = []
        for sentence_index, words in groups.items():
            words.sort(key=lambda w: float(w.start or 0.0))
            start = float(words[0].start or 0.0)
            end = float(words[-1].end or start)
            if end <= start:
                continue
            contour_times: List[float] = []
            contour: List[float] = []
            for t, v in zip(times, f0):
                t = float(t)
                v = float(v)
                if start <= t <= end and v > 0 and math.isfinite(v):
                    contour_times.append(round(t - start, 3))
                    contour.append(float(12.0 * math.log2(v / median_f0)))
            slope_hz = pitch_slope(times, f0, max(start, end - 0.8), end)
            slope_st = None
            if slope_hz is not None and median_f0 > 0:
                slope_st = 12.0 / math.log(2) * slope_hz / median_f0
            pitch_range = float(np.percentile(contour, 95) - np.percentile(contour, 5)) if len(contour) >= 6 else None
            candidates = [
                wf_by_index.get(int(w.ref_index))
                for w in words
                if w.ref_index is not None and token_by_index.get(int(w.ref_index)) is not None and token_by_index[int(w.ref_index)].is_content_word
            ]
            candidates = [w for w in candidates if w is not None and w.prominence is not None]
            candidates.sort(key=lambda w: float(w.prominence or 0.0), reverse=True)
            keep = max(1, min(3, int(math.ceil(len(candidates) * 0.2)))) if candidates else 0
            out.append(
                SentenceProsodyFeature(
                    sentence_index=int(sentence_index),
                    start=round(start, 3),
                    end=round(end, 3),
                    final_slope_st_per_s=round(slope_st, 3) if slope_st is not None else None,
                    pitch_range_st=round(pitch_range, 3) if pitch_range is not None else None,
                    f0_contour_st=[round(x, 3) for x in _downsample(contour, 120)],
                    f0_times=[round(x, 3) for x in _downsample(contour_times, 120)],
                    prominent_words=[int(w.ref_index) for w in candidates[:keep]],
                )
            )
        return out


def _track_values(times: List[float], values: List[float], start: float, end: float, *, positive: bool = False) -> List[float]:
    out: List[float] = []
    for t, v in zip(times, values):
        try:
            t = float(t)
            v = float(v)
        except Exception:
            continue
        if start <= t <= end and math.isfinite(v) and (not positive or v > 0):
            out.append(v)
    return out


def _median_positive(values: List[float]) -> Optional[float]:
    voiced = [float(v) for v in values if v and float(v) > 0 and math.isfinite(float(v))]
    if not voiced:
        return None
    return float(np.median(voiced))


def _range_st(values: List[float], median_f0: Optional[float]) -> Optional[float]:
    if len(values) < 3 or median_f0 is None or median_f0 <= 0:
        return None
    st = [12.0 * math.log2(float(v) / median_f0) for v in values if v > 0]
    if len(st) < 3:
        return None
    return float(np.percentile(st, 95) - np.percentile(st, 5))


def _voicing_continuity(tracks: Dict[str, Any], left_end: float, right_start: float) -> Optional[float]:
    pitch = tracks.get("pitch", {}) if isinstance(tracks, dict) else {}
    times = pitch.get("times", []) or []
    f0 = pitch.get("f0", []) or []
    if not times or not f0:
        return None
    window_start = max(0.0, left_end - 0.04)
    window_end = right_start + 0.04
    vals = _track_values(times, f0, window_start, window_end, positive=False)
    if not vals:
        return None
    voiced = [v for v in vals if v > 0]
    return float(len(voiced) / max(1, len(vals)))


def _downsample(values: List[float], limit: int) -> List[float]:
    if len(values) <= limit:
        return list(values)
    idx = np.linspace(0, len(values) - 1, num=limit)
    return [float(values[int(round(i))]) for i in idx]


def _round_nested(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _round_nested(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round_nested(v) for v in value]
    return value
