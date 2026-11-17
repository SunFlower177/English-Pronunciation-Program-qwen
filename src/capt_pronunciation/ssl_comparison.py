"""Optional WavLM/HuBERT embedding comparison for Mode B."""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .audio_io import Waveform
from .contracts import CaptSettings

logger = logging.getLogger(__name__)


class SslEmbeddingComparator:
    """Compare reference and learner speech with speaker-normalized SSL frames.

    The model is loaded only when CAPT_IMITATION_USE_SSL=1.  Failures produce
    missing evidence instead of a fabricated similarity score.
    """

    def __init__(self, settings: CaptSettings):
        self.settings = settings
        self.processor: Any = None
        self.model: Any = None
        self._failed = False

    def compare(self, reference: Waveform, learner: Waveform) -> Dict[str, Optional[float]]:
        if not self.settings.imitation_use_ssl_embeddings:
            return {"ssl_soft_dtw": None, "ssl_frame_coverage": 0.0}
        ref = self._embed(reference)
        hyp = self._embed(learner)
        if ref is None or hyp is None or len(ref) < 2 or len(hyp) < 2:
            return {"ssl_soft_dtw": None, "ssl_frame_coverage": 0.0}
        distance = cosine_soft_dtw(ref, hyp)
        coverage = min(len(ref), len(hyp)) / max(len(ref), len(hyp), 1)
        return {
            "ssl_soft_dtw": None if distance is None else round(float(distance), 6),
            "ssl_frame_coverage": round(float(coverage), 6),
        }

    def _embed(self, waveform: Waveform) -> Optional[np.ndarray]:
        if not self._ensure_loaded():
            return None
        try:
            import torch  # type: ignore

            inputs = self.processor(
                waveform.samples,
                sampling_rate=waveform.sample_rate,
                return_tensors="pt",
                padding=True,
            )
            device = (
                self.settings.device
                if self.settings.device == "cuda" and torch.cuda.is_available()
                else "cpu"
            )
            inputs = {key: value.to(device) for key, value in inputs.items() if hasattr(value, "to")}
            with torch.no_grad():
                output = self.model(**inputs, output_hidden_states=True)
            hidden_states = output.hidden_states
            layer = max(0, min(int(self.settings.imitation_ssl_layer), len(hidden_states) - 1))
            frames = hidden_states[layer][0].detach().float().cpu().numpy()
            frames = normalize_frames(frames)
            return resample_frames(frames, int(self.settings.imitation_ssl_max_frames))
        except Exception as exc:
            logger.warning("SSL embedding extraction failed: %s", exc)
            return None

    def _ensure_loaded(self) -> bool:
        if self._failed:
            return False
        if self.processor is not None and self.model is not None:
            return True
        try:
            import torch  # type: ignore
            from transformers import AutoFeatureExtractor, AutoModel  # type: ignore

            self.processor = AutoFeatureExtractor.from_pretrained(self.settings.imitation_ssl_model)
            self.model = AutoModel.from_pretrained(self.settings.imitation_ssl_model)
            device = (
                self.settings.device
                if self.settings.device == "cuda" and torch.cuda.is_available()
                else "cpu"
            )
            self.model.to(device)
            self.model.eval()
            return True
        except Exception as exc:
            logger.warning("SSL comparison model unavailable: %s", exc)
            self.processor = None
            self.model = None
            self._failed = True
            return False


def normalize_frames(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 2 or frames.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    # Remove utterance-level speaker/channel offset, then L2 normalize frames.
    frames = frames - np.mean(frames, axis=0, keepdims=True)
    norms = np.linalg.norm(frames, axis=1, keepdims=True)
    return frames / np.maximum(norms, 1e-8)


def resample_frames(frames: np.ndarray, max_frames: int) -> np.ndarray:
    if max_frames <= 0 or len(frames) <= max_frames:
        return frames
    indices = np.linspace(0, len(frames) - 1, num=max_frames).round().astype(int)
    return frames[indices]


def cosine_soft_dtw(
    reference: Sequence[Sequence[float]] | np.ndarray,
    learner: Sequence[Sequence[float]] | np.ndarray,
    gamma: float = 0.10,
) -> Optional[float]:
    left = np.asarray(reference, dtype=float)
    right = np.asarray(learner, dtype=float)
    if left.ndim != 2 or right.ndim != 2 or not len(left) or not len(right):
        return None
    cost = 1.0 - np.clip(left @ right.T, -1.0, 1.0)
    n, m = cost.shape
    dp = np.full((n + 1, m + 1), np.inf, dtype=float)
    dp[0, 0] = 0.0
    gamma = max(float(gamma), 1e-6)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            previous = np.asarray([dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]])
            finite = previous[np.isfinite(previous)]
            if not len(finite):
                continue
            pivot = float(np.min(finite))
            soft_min = pivot - gamma * math.log(
                float(np.sum(np.exp(-(finite - pivot) / gamma)))
            )
            dp[i, j] = float(cost[i - 1, j - 1]) + soft_min
    if not math.isfinite(float(dp[n, m])):
        return None
    return max(0.0, float(dp[n, m]) / max(1, n + m))
