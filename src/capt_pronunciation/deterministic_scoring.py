"""Continuous, parameterized scoring for Mode A and Mode B.

The scorer consumes only auditable evidence.  It never calls an LLM and never
uses the human label of the item being scored.  Every transform maps a raw
measurement to [0, 1], then combines available measurements with fixed weights
from a YAML file.  Missing evidence is reported through coverage and is not
silently treated as a perfect observation.
"""
from __future__ import annotations

import copy
import difflib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import yaml

from .contracts import AlignmentResult, EvidenceBundle, EvidenceIssue, PhonemeHypothesis


DEFAULT_SCORE_CONFIG: Dict[str, Any] = {
    "version": "continuous_v3_alignment_gop",
    "missing_quality": 0.50,
    "mode_a": {
        "accuracy_weights": {
            "word_alignment": 0.35,
            "asr_text_similarity": 0.10,
            "phone_edit": 0.15,
            "phone_gop": 0.20,
            "phone_confidence": 0.20,
        },
        "phoneme_weights": {
            "phone_edit": 0.35,
            "phone_gop": 0.45,
            "word_alignment": 0.20,
        },
        "fluency_weights": {
            "speech_rate": 0.55,
            "articulation_rate": 0.10,
            "pause_ratio": 0.15,
            "long_pause_ratio": 0.10,
            "mean_length_run": 0.05,
            "disfluency": 0.05,
        },
        "prosody_weights": {
            "pitch_range": 0.34,
            "rhythm_variability": 0.20,
            "duration_variability": 0.16,
            "stress_evidence": 0.15,
            "intonation_evidence": 0.15,
        },
        "bands": {
            "word_duration_cv": [0.05, 0.18, 0.62, 1.20],
        },
        "peaks": {
            "speech_rate_wpm": [45.0, 190.0, 280.0],
            "articulation_rate_wpm": [70.0, 280.0, 420.0],
        },
        "inverse_ramps": {
            "pause_ratio": [0.03, 0.35],
            "long_pause_ratio": [0.0, 0.20],
            "pause_rate_per_min": [3.0, 20.0],
            "filled_pause_rate_per_100_words": [0.0, 12.0],
        },
        "ramps": {
            "pitch_range_st": [1.5, 8.0],
            "npvi_word_duration": [18.0, 65.0],
            "mean_length_run_words": [1.0, 20.0],
        },
        "insertion_penalty": 0.65,
    },
    "mode_b": {
        "accuracy_weights": {
            "word_alignment": 0.34,
            "phone_reference_similarity": 0.38,
            "phone_canonical_similarity": 0.18,
            "ssl_similarity": 0.10,
        },
        "phoneme_weights": {
            "phone_reference_similarity": 0.55,
            "phone_canonical_similarity": 0.30,
            "word_alignment": 0.15,
        },
        "fluency_weights": {
            "speech_tempo_similarity": 0.24,
            "pause_similarity": 0.28,
            "duration_similarity": 0.28,
            "ssl_similarity": 0.20,
        },
        "prosody_weights": {
            "f0_dtw_similarity": 0.34,
            "f0_correlation": 0.18,
            "sentence_stress_overlap": 0.20,
            "word_stress_similarity": 0.14,
            "duration_similarity": 0.14,
        },
        "distance_scales": {
            "pause_mae_s": 0.18,
            "duration_mae": 0.55,
            "f0_dtw": 1.20,
            "ssl_soft_dtw": 0.42,
            "tempo_log_ratio": 0.32,
        },
    },
    "overall_weights": {
        "accuracy": 0.35,
        "phoneme": 0.20,
        "fluency": 0.25,
        "prosody": 0.20,
    },
}


@dataclass
class DeterministicScoreResult:
    scores: Dict[str, float]
    normalized_features: Dict[str, Optional[float]]
    raw_features: Dict[str, Optional[float]]
    coverage: Dict[str, float]
    config_version: str
    mode: str


def load_score_config(path: str | Path | None = None) -> Dict[str, Any]:
    config = copy.deepcopy(DEFAULT_SCORE_CONFIG)
    if not path:
        return config
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Deterministic score config not found: {source}")
    loaded = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Deterministic score config must contain a YAML mapping")
    return deep_merge(config, loaded)


def score_evidence(
    evidence: EvidenceBundle,
    alignment: AlignmentResult | None,
    *,
    mode: str = "modeA",
    config: Mapping[str, Any] | None = None,
) -> DeterministicScoreResult:
    cfg = copy.deepcopy(dict(config or DEFAULT_SCORE_CONFIG))
    mode_key = "mode_b" if str(mode).lower() in {"modeb", "mode_b", "imitation"} else "mode_a"
    raw = extract_common_features(evidence, alignment, cfg)
    normalized = normalize_mode_a_features(raw, evidence, cfg["mode_a"])
    if mode_key == "mode_b":
        raw.update(extract_mode_b_features(evidence))
        normalized.update(normalize_mode_b_features(raw, cfg["mode_b"]))
    section = cfg[mode_key]
    missing = clip01(float(cfg.get("missing_quality", 0.5)))

    dimensions: Dict[str, float] = {}
    dimension_coverage: Dict[str, float] = {}
    for dimension in ("accuracy", "phoneme", "fluency", "prosody"):
        weights = section[f"{dimension}_weights"]
        value, cov = weighted_available(normalized, weights, missing)
        dimensions[dimension] = round(100.0 * value, 3)
        dimension_coverage[dimension] = round(cov, 4)

    overall_weights = cfg.get("overall_weights", {})
    overall, _ = weighted_available(
        {key: value / 100.0 for key, value in dimensions.items()},
        overall_weights,
        missing,
    )
    dimensions["overall"] = round(100.0 * overall, 3)
    return DeterministicScoreResult(
        scores=dimensions,
        normalized_features={k: round_optional(v, 6) for k, v in normalized.items()},
        raw_features={k: round_optional(v, 6) for k, v in raw.items()},
        coverage=dimension_coverage,
        config_version=str(cfg.get("version", "continuous_v2")),
        mode="modeB" if mode_key == "mode_b" else "modeA",
    )


def extract_common_features(
    evidence: EvidenceBundle,
    alignment: AlignmentResult | None,
    cfg: Mapping[str, Any],
) -> Dict[str, Optional[float]]:
    mode_a = cfg["mode_a"]
    ref_count = len(alignment.reference_tokens) if alignment is not None else 0
    word_quality, text_similarity, insertion_rate = alignment_quality(
        alignment,
        insertion_penalty=float(mode_a.get("insertion_penalty", 0.65)),
    )
    phone_quality, phone_edit, phone_confidence, phone_gop, phone_coverage = phone_hypothesis_quality(
        evidence.phoneme_hypotheses,
        ref_count,
    )
    if word_quality is None and evidence.word_issues:
        word_quality = issue_quality(evidence.word_issues, max(1, ref_count))
    phone_issue_quality = (
        issue_quality(evidence.phoneme_issues, max(1, ref_count))
        if evidence.phoneme_issues
        else None
    )
    phone_effective: Optional[float]
    if phone_quality is None:
        phone_effective = phone_issue_quality if phone_issue_quality is not None else word_quality
    elif word_quality is None:
        phone_effective = phone_quality
    else:
        phone_effective = phone_coverage * phone_quality + (1.0 - phone_coverage) * word_quality

    prosody = evidence.prosody
    prosody_raw = prosody.raw if isinstance(prosody.raw, dict) else {}
    timed_count = 0
    if alignment is not None:
        timed_count = len(
            {
                item.ref_index
                for item in alignment.word_alignments
                if item.ref_index is not None
                and item.operation != "delete"
                and item.start is not None
                and item.end is not None
            }
        )
    timing_available = timed_count >= 2
    pitch_available = prosody.pitch_range_st is not None
    return {
        "word_alignment": word_quality,
        "asr_text_similarity": text_similarity,
        "insertion_rate": insertion_rate,
        "phone_quality": phone_quality,
        "phone_edit_quality": phone_edit,
        "phone_effective": phone_effective,
        "phone_confidence": phone_confidence,
        "phone_ctc_gop": phone_gop,
        "phone_coverage": phone_coverage,
        "speech_rate_wpm": finite_optional(prosody.speech_rate_wpm),
        "articulation_rate_wpm": finite_optional(prosody.articulation_rate_wpm),
        "pause_ratio": finite_optional(prosody.pause_ratio),
        "long_pause_ratio": finite_optional(prosody_raw.get("long_pause_ratio")),
        "pause_rate_per_min": finite_optional(prosody_raw.get("pause_rate_per_min")),
        "mean_pause_s": finite_optional(prosody_raw.get("mean_pause_s")),
        "filled_pause_rate_per_100_words": finite_optional(
            prosody_raw.get("filled_pause_rate_per_100_words")
        ),
        "mean_length_run_words": finite_optional(prosody.mean_length_run_words),
        "npvi_word_duration": finite_optional(prosody.npvi_word_duration),
        "word_duration_cv": finite_optional(prosody.word_duration_cv),
        "pitch_range_st": finite_optional(prosody.pitch_range_st),
        "boundary_evidence": (
            issue_quality(evidence.pause_issues + evidence.linking_issues, max(1, ref_count - 1))
            if timing_available
            else None
        ),
        "stress_evidence": (
            issue_quality(evidence.stress_issues + evidence.sentence_stress_issues, max(1, ref_count))
            if timing_available and pitch_available
            else None
        ),
        "intonation_evidence": (
            issue_quality(evidence.intonation_issues, 1) if pitch_available else None
        ),
        "rhythm_evidence": (
            issue_quality(evidence.rhythm_issues, max(1, ref_count))
            if prosody.word_duration_cv is not None
            else None
        ),
    }


def normalize_mode_a_features(
    raw: Mapping[str, Optional[float]],
    evidence: EvidenceBundle,
    cfg: Mapping[str, Any],
) -> Dict[str, Optional[float]]:
    bands = cfg["bands"]
    peaks = cfg["peaks"]
    inverse_ramps = cfg["inverse_ramps"]
    ramps = cfg["ramps"]
    rhythm = mean_available(
        [
            ramp_quality(raw.get("npvi_word_duration"), *ramps["npvi_word_duration"]),
            raw.get("rhythm_evidence"),
        ]
    )
    return {
        "word_alignment": raw.get("word_alignment"),
        "phone_effective": raw.get("phone_effective"),
        "phone_edit": raw.get("phone_edit_quality"),
        "phone_gop": raw.get("phone_ctc_gop"),
        "asr_text_similarity": raw.get("asr_text_similarity"),
        "phone_confidence": raw.get("phone_confidence"),
        "speech_rate": peak_quality(raw.get("speech_rate_wpm"), *peaks["speech_rate_wpm"]),
        "articulation_rate": peak_quality(
            raw.get("articulation_rate_wpm"), *peaks["articulation_rate_wpm"]
        ),
        "pause_ratio": inverse_ramp_quality(
            raw.get("pause_ratio"), *inverse_ramps["pause_ratio"]
        ),
        "long_pause_ratio": inverse_ramp_quality(
            raw.get("long_pause_ratio"), *inverse_ramps["long_pause_ratio"]
        ),
        "pause_rate": inverse_ramp_quality(
            raw.get("pause_rate_per_min"), *inverse_ramps["pause_rate_per_min"]
        ),
        "mean_length_run": ramp_quality(
            raw.get("mean_length_run_words"), *ramps["mean_length_run_words"]
        ),
        "disfluency": inverse_ramp_quality(
            raw.get("filled_pause_rate_per_100_words"),
            *inverse_ramps["filled_pause_rate_per_100_words"],
        ),
        "boundary_evidence": raw.get("boundary_evidence"),
        "pitch_range": ramp_quality(raw.get("pitch_range_st"), *ramps["pitch_range_st"]),
        "rhythm_variability": rhythm,
        "duration_variability": band_quality(raw.get("word_duration_cv"), *bands["word_duration_cv"]),
        "stress_evidence": raw.get("stress_evidence"),
        "intonation_evidence": raw.get("intonation_evidence"),
    }


def extract_mode_b_features(evidence: EvidenceBundle) -> Dict[str, Optional[float]]:
    source = evidence.continuous_features.get("mode_b", {})
    if not isinstance(source, dict):
        source = {}
    out = {
        key: finite_optional(source.get(key))
        for key in (
            "phone_reference_similarity",
            "phone_canonical_similarity",
            "phone_coverage",
            "tempo_log_ratio",
            "pause_mae_s",
            "duration_mae",
            "f0_dtw",
            "f0_correlation",
            "sentence_stress_overlap",
            "word_stress_similarity",
            "ssl_soft_dtw",
            "ssl_frame_coverage",
        )
    }
    # Backward-compatible recovery for reports created before continuous_v2.
    issue_groups: Iterable[EvidenceIssue] = (
        evidence.pause_issues
        + evidence.stress_issues
        + evidence.sentence_stress_issues
        + evidence.intonation_issues
        + evidence.rhythm_issues
        + evidence.phoneme_issues
    )
    evidence_rows = [i.evidence for i in issue_groups if isinstance(i.evidence, dict)]
    fallback_keys = {
        "f0_dtw": "f0_dtw_distance",
        "f0_correlation": "f0_contour_correlation",
        "sentence_stress_overlap": "overlap",
        "phone_reference_similarity": "learner_reference_similarity",
        "phone_canonical_similarity": "learner_canonical_similarity",
    }
    for target, source_key in fallback_keys.items():
        if out[target] is None:
            out[target] = mean_available([finite_optional(row.get(source_key)) for row in evidence_rows])
    return out


def normalize_mode_b_features(
    raw: Mapping[str, Optional[float]],
    cfg: Mapping[str, Any],
) -> Dict[str, Optional[float]]:
    scales = cfg["distance_scales"]
    ssl_similarity = exp_distance_quality(raw.get("ssl_soft_dtw"), scales["ssl_soft_dtw"])
    return {
        "word_alignment": raw.get("word_alignment"),
        "phone_reference_similarity": first_available(
            raw.get("phone_reference_similarity"),
            raw.get("phone_effective"),
        ),
        "phone_canonical_similarity": first_available(
            raw.get("phone_canonical_similarity"),
            raw.get("phone_effective"),
        ),
        "ssl_similarity": ssl_similarity,
        "speech_tempo_similarity": exp_distance_quality(abs_optional(raw.get("tempo_log_ratio")), scales["tempo_log_ratio"]),
        "pause_similarity": exp_distance_quality(raw.get("pause_mae_s"), scales["pause_mae_s"]),
        "duration_similarity": exp_distance_quality(raw.get("duration_mae"), scales["duration_mae"]),
        "f0_dtw_similarity": exp_distance_quality(raw.get("f0_dtw"), scales["f0_dtw"]),
        "f0_correlation": correlation_quality(raw.get("f0_correlation")),
        "sentence_stress_overlap": raw.get("sentence_stress_overlap"),
        "word_stress_similarity": raw.get("word_stress_similarity"),
    }


def alignment_quality(
    alignment: AlignmentResult | None,
    *,
    insertion_penalty: float,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if alignment is None or not alignment.reference_tokens:
        return None, None, None
    if not alignment.word_alignments:
        return None, None, None
    ref_count = len(alignment.reference_tokens)
    qualities = []
    reference_text = []
    hypothesis_text = []
    insertions = 0
    for item in alignment.word_alignments:
        if item.ref_word:
            reference_text.append(item.ref_word)
        if item.hyp_word:
            hypothesis_text.append(item.hyp_word)
        if item.operation == "insert":
            insertions += 1
            continue
        if item.ref_index is None:
            continue
        if item.operation == "equal":
            qualities.append(1.0)
        elif item.operation == "substitute":
            qualities.append(
                difflib.SequenceMatcher(
                    None,
                    str(item.ref_word or "").lower(),
                    str(item.hyp_word or "").lower(),
                ).ratio()
            )
        else:
            qualities.append(0.0)
    while len(qualities) < ref_count:
        qualities.append(0.0)
    base = float(np.mean(qualities[:ref_count])) if ref_count else None
    insertion_rate = insertions / max(1, ref_count)
    if base is not None:
        base = clip01(base - insertion_penalty * insertion_rate)
    text_similarity = difflib.SequenceMatcher(
        None,
        " ".join(reference_text).lower(),
        " ".join(hypothesis_text).lower(),
    ).ratio()
    return base, float(text_similarity), float(insertion_rate)


def phone_hypothesis_quality(
    hypotheses: Sequence[PhonemeHypothesis],
    ref_count: int,
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float], float]:
    qualities = []
    confidences = []
    gop_values = []
    observed = set()
    for index, hypothesis in enumerate(hypotheses):
        canonical = [str(x) for x in hypothesis.canonical_phones if str(x)]
        predicted = [str(x) for x in hypothesis.predicted_phones if str(x)]
        if not canonical and not predicted:
            continue
        edit_quality = global_sequence_similarity(canonical, predicted)
        gop = finite_optional(hypothesis.ctc_gop)
        qualities.append(edit_quality if gop is None else 0.70 * edit_quality + 0.30 * clip01(gop))
        if gop is not None:
            gop_values.append(clip01(gop))
        if hypothesis.confidence is not None and math.isfinite(float(hypothesis.confidence)):
            confidences.append(clip01(float(hypothesis.confidence)))
        observed.add(hypothesis.ref_index if hypothesis.ref_index is not None else (hypothesis.word, index))
    coverage = min(1.0, len(observed) / max(1, ref_count)) if ref_count else (1.0 if observed else 0.0)
    return (
        mean_available(qualities),
        mean_available(
            [
                global_sequence_similarity(
                    [str(x) for x in hypothesis.canonical_phones if str(x)],
                    [str(x) for x in hypothesis.predicted_phones if str(x)],
                )
                for hypothesis in hypotheses
                if hypothesis.canonical_phones or hypothesis.predicted_phones
            ]
        ),
        mean_available(confidences),
        mean_available(gop_values),
        float(coverage),
    )


def global_sequence_similarity(a: Sequence[str], b: Sequence[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, start=1):
        current = [i]
        for j, right in enumerate(b, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (0 if left == right else 1),
                )
            )
        previous = current
    return clip01(1.0 - previous[-1] / max(len(a), len(b), 1))


def issue_quality(issues: Sequence[EvidenceIssue], evaluable_units: int) -> Optional[float]:
    if not issues:
        return 1.0
    severity = {"info": 0.08, "low": 0.25, "medium": 0.58, "high": 1.0}
    penalties = []
    seen = set()
    for issue in issues:
        key = (issue.ref_index, str(issue.word or "").lower(), str(issue.type))
        if key in seen:
            continue
        seen.add(key)
        penalties.append(
            severity.get(str(getattr(issue.severity, "value", issue.severity)), 0.5)
            * clip01(float(issue.confidence))
        )
    return clip01(1.0 - sum(penalties) / max(1, evaluable_units, len(penalties)))


def weighted_available(
    values: Mapping[str, Optional[float]],
    weights: Mapping[str, float],
    missing: float,
) -> tuple[float, float]:
    numerator = 0.0
    available_weight = 0.0
    total_weight = sum(max(0.0, float(w)) for w in weights.values())
    for key, weight_value in weights.items():
        weight = max(0.0, float(weight_value))
        value = finite_optional(values.get(key))
        if value is None:
            continue
        numerator += weight * clip01(value)
        available_weight += weight
    if available_weight <= 0.0:
        return missing, 0.0
    return clip01(numerator / available_weight), available_weight / max(total_weight, 1e-8)


def band_quality(
    value: Optional[float],
    soft_low: float,
    ideal_low: float,
    ideal_high: float,
    soft_high: float,
) -> Optional[float]:
    value = finite_optional(value)
    if value is None:
        return None
    if ideal_low <= value <= ideal_high:
        return 1.0
    if value <= soft_low or value >= soft_high:
        return 0.0
    if value < ideal_low:
        return clip01((value - soft_low) / max(ideal_low - soft_low, 1e-8))
    return clip01((soft_high - value) / max(soft_high - ideal_high, 1e-8))


def ramp_quality(value: Optional[float], low: float, high: float) -> Optional[float]:
    value = finite_optional(value)
    if value is None:
        return None
    return clip01((value - low) / max(high - low, 1e-8))


def inverse_ramp_quality(
    value: Optional[float],
    good_max: float,
    bad_anchor: float,
) -> Optional[float]:
    value = finite_optional(value)
    if value is None:
        return None
    return clip01(1.0 - (value - good_max) / max(bad_anchor - good_max, 1e-8))


def peak_quality(
    value: Optional[float],
    low: float,
    peak: float,
    high: float,
) -> Optional[float]:
    """Continuous triangular quality without an information-destroying plateau."""
    value = finite_optional(value)
    if value is None:
        return None
    if value <= low or value >= high:
        return 0.0
    if value <= peak:
        return clip01((value - low) / max(peak - low, 1e-8))
    return clip01((high - value) / max(high - peak, 1e-8))


def exp_distance_quality(value: Optional[float], scale: float) -> Optional[float]:
    value = finite_optional(value)
    if value is None:
        return None
    return clip01(math.exp(-max(0.0, value) / max(float(scale), 1e-8)))


def correlation_quality(value: Optional[float]) -> Optional[float]:
    value = finite_optional(value)
    return None if value is None else clip01((value + 1.0) / 2.0)


def mean_available(values: Iterable[Optional[float]]) -> Optional[float]:
    finite = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(np.mean(finite)) if finite else None


def finite_optional(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def abs_optional(value: Optional[float]) -> Optional[float]:
    return None if value is None else abs(float(value))


def first_available(*values: Optional[float]) -> Optional[float]:
    for value in values:
        finite = finite_optional(value)
        if finite is not None:
            return finite
    return None


def round_optional(value: Optional[float], digits: int) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base
