"""Audio-to-audio comparison helpers for reference-imitation mode."""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar

import numpy as np

from .contracts import BoundaryFeature, SentenceProsodyFeature, WordAcousticFeature

T = TypeVar("T")


def by_ref_index(items: Iterable[T]) -> Dict[int, T]:
    out: Dict[int, T] = {}
    for item in items:
        ref_index = getattr(item, "ref_index", None)
        if ref_index is not None:
            out[int(ref_index)] = item
    return out


def boundary_key(feature: BoundaryFeature) -> Tuple[int, int]:
    return int(feature.left_index), int(feature.right_index)


def by_boundary(items: Iterable[BoundaryFeature]) -> Dict[Tuple[int, int], BoundaryFeature]:
    return {boundary_key(item): item for item in items}


def by_sentence(items: Iterable[SentenceProsodyFeature]) -> Dict[int, SentenceProsodyFeature]:
    return {int(item.sentence_index): item for item in items}


def speaking_duration(words: Sequence[WordAcousticFeature]) -> float:
    durations = [max(0.0, float(w.duration)) for w in words if w.duration is not None]
    return float(sum(durations))


def tempo_ratio(reference_words: Sequence[WordAcousticFeature], learner_words: Sequence[WordAcousticFeature]) -> float:
    ref = speaking_duration(reference_words)
    learner = speaking_duration(learner_words)
    if ref <= 1e-6 or learner <= 1e-6:
        return 1.0
    return max(0.35, min(3.0, learner / ref))


def normalized_duration(value: Optional[float], median_value: Optional[float]) -> Optional[float]:
    if value is None or median_value is None or median_value <= 1e-8:
        return None
    return float(value) / float(median_value)


def median_duration(words: Sequence[WordAcousticFeature]) -> Optional[float]:
    vals = [float(w.duration) for w in words if w.duration is not None and float(w.duration) > 0]
    if not vals:
        return None
    return float(np.median(vals))


def contour_dtw_distance(a: Sequence[float], b: Sequence[float], max_len: int = 120) -> Optional[float]:
    seq_a = _clean_contour(a)
    seq_b = _clean_contour(b)
    if len(seq_a) < 4 or len(seq_b) < 4:
        return None
    seq_a = _resample_if_needed(seq_a, max_len)
    seq_b = _resample_if_needed(seq_b, max_len)
    n, m = len(seq_a), len(seq_b)
    dp = np.full((n + 1, m + 1), np.inf, dtype=float)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(seq_a[i - 1] - seq_b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m] / max(1, n + m))


def contour_correlation(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    seq_a = _clean_contour(a)
    seq_b = _clean_contour(b)
    if len(seq_a) < 4 or len(seq_b) < 4:
        return None
    target = min(len(seq_a), len(seq_b), 120)
    seq_a = _resample(seq_a, target)
    seq_b = _resample(seq_b, target)
    mean_a = float(np.mean(seq_a))
    mean_b = float(np.mean(seq_b))
    da = [float(x - mean_a) for x in seq_a]
    db = [float(x - mean_b) for x in seq_b]
    denom_a = math.sqrt(sum(x * x for x in da))
    denom_b = math.sqrt(sum(x * x for x in db))
    if denom_a < 1e-8 or denom_b < 1e-8:
        return None
    return float(sum(x * y for x, y in zip(da, db)) / (denom_a * denom_b))


def top_overlap(a: Sequence[int], b: Sequence[int]) -> float:
    set_a = {int(x) for x in a}
    set_b = {int(x) for x in b}
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / max(1, len(set_a | set_b))


def _clean_contour(values: Sequence[float]) -> List[float]:
    out = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not out:
        return []
    median = float(np.median(out))
    return [float(v - median) for v in out]


def _resample_if_needed(values: Sequence[float], max_len: int) -> List[float]:
    if len(values) <= max_len:
        return list(values)
    return _resample(values, max_len)


def _resample(values: Sequence[float], length: int) -> List[float]:
    if length <= 0:
        return []
    if len(values) == length:
        return list(values)
    if len(values) == 1:
        return [float(values[0])] * length
    x_old = np.linspace(0.0, 1.0, num=len(values))
    x_new = np.linspace(0.0, 1.0, num=length)
    return [float(x) for x in np.interp(x_new, x_old, np.asarray(values, dtype=float))]
