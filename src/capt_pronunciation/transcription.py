"""ASR and word timestamping.

Functional module 2/7.

Primary backend: WhisperX. Fallback backend: faster-whisper. The rest of the
system only sees the normalized AsrResult contract, so future backends can be
added without touching diagnostics or feedback generation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .contracts import AsrResult, AsrSegment, AsrWord, CaptSettings

logger = logging.getLogger(__name__)


class TranscriptionError(RuntimeError):
    pass


class WhisperXTranscriber:
    def __init__(self, settings: CaptSettings):
        self.settings = settings
        self._whisperx_model = None
        self._whisperx_align_cache: Dict[str, Any] = {}
        self._faster_model = None
        self._transformers_pipeline = None

    def transcribe(self, audio_path: str | Path) -> AsrResult:
        audio_path = str(audio_path)
        if self.settings.asr_backend == "faster_whisper":
            try:
                return self._transcribe_faster_whisper(audio_path)
            except Exception as exc:
                raise TranscriptionError(f"faster-whisper failed: {exc}") from exc
        if self.settings.asr_backend == "transformers_whisper":
            try:
                return self._transcribe_transformers_whisper(audio_path)
            except Exception as exc:
                raise TranscriptionError(f"transformers Whisper failed: {exc}") from exc
        try:
            return self._transcribe_whisperx(audio_path)
        except Exception as whisperx_exc:
            logger.warning("WhisperX failed; trying faster-whisper fallback: %s", whisperx_exc)
            try:
                return self._transcribe_faster_whisper(audio_path)
            except Exception as faster_exc:
                raise TranscriptionError(
                    "Both WhisperX and faster-whisper failed. Install whisperx, or set up faster-whisper. "
                    f"WhisperX error: {whisperx_exc}; faster-whisper error: {faster_exc}"
                ) from faster_exc

    def _transcribe_whisperx(self, audio_path: str) -> AsrResult:
        import whisperx  # type: ignore

        device = self.settings.device
        compute_type = self.settings.compute_type
        if device == "cpu" and compute_type in {"float16", "bfloat16"}:
            compute_type = "int8"
        if self._whisperx_model is None:
            logger.info("Loading WhisperX model=%s device=%s compute_type=%s", self.settings.whisperx_model, device, compute_type)
            self._whisperx_model = whisperx.load_model(
                self.settings.whisperx_model,
                device=device,
                compute_type=compute_type,
                language=self.settings.language,
            )
        audio = whisperx.load_audio(audio_path)
        result = self._whisperx_model.transcribe(audio, batch_size=self.settings.whisperx_batch_size)

        if self.settings.use_whisperx_alignment:
            language = result.get("language") or self.settings.language
            try:
                if language not in self._whisperx_align_cache:
                    self._whisperx_align_cache[language] = whisperx.load_align_model(language_code=language, device=device)
                align_model, metadata = self._whisperx_align_cache[language]
                result = whisperx.align(
                    result["segments"],
                    align_model,
                    metadata,
                    audio,
                    device,
                    return_char_alignments=False,
                )
                result["language"] = language
            except Exception as exc:
                logger.warning("WhisperX alignment failed; using segment timestamps only: %s", exc)

        return self._normalize_whisperx_result(result)

    def _normalize_whisperx_result(self, raw: Dict[str, Any]) -> AsrResult:
        segments: List[AsrSegment] = []
        flat_words: List[AsrWord] = []
        for seg in raw.get("segments", []) or []:
            words: List[AsrWord] = []
            for w in seg.get("words", []) or []:
                token = str(w.get("word", "")).strip()
                if not token:
                    continue
                word = AsrWord(
                    word=token,
                    start=_safe_float(w.get("start")),
                    end=_safe_float(w.get("end")),
                    score=_safe_float(w.get("score") or w.get("confidence")),
                    source="whisperx",
                )
                words.append(word)
                flat_words.append(word)
            segments.append(
                AsrSegment(
                    text=str(seg.get("text", "")).strip(),
                    start=_safe_float(seg.get("start")),
                    end=_safe_float(seg.get("end")),
                    words=words,
                )
            )
        text = " ".join(seg.text for seg in segments).strip()
        return AsrResult(text=text, language=raw.get("language", "en"), segments=segments, words=flat_words, raw=_compact_raw(raw))

    def _transcribe_faster_whisper(self, audio_path: str) -> AsrResult:
        from faster_whisper import WhisperModel  # type: ignore

        device = "cuda" if self.settings.device == "cuda" else "cpu"
        compute_type = self.settings.compute_type
        if device == "cpu" and compute_type in {"float16", "bfloat16"}:
            compute_type = "int8"
        if self._faster_model is None:
            logger.info("Loading faster-whisper model=%s", self.settings.whisperx_model)
            self._faster_model = WhisperModel(self.settings.whisperx_model, device=device, compute_type=compute_type)
        segments_iter, info = self._faster_model.transcribe(
            audio_path,
            language=self.settings.language,
            vad_filter=True,
            word_timestamps=True,
            beam_size=5,
        )
        segments: List[AsrSegment] = []
        flat_words: List[AsrWord] = []
        for seg in segments_iter:
            words: List[AsrWord] = []
            for w in getattr(seg, "words", []) or []:
                token = str(getattr(w, "word", "")).strip()
                if not token:
                    continue
                word = AsrWord(
                    word=token,
                    start=_safe_float(getattr(w, "start", None)),
                    end=_safe_float(getattr(w, "end", None)),
                    score=_safe_float(getattr(w, "probability", None)),
                    source="faster_whisper",
                )
                words.append(word)
                flat_words.append(word)
            segments.append(
                AsrSegment(
                    text=str(getattr(seg, "text", "")).strip(),
                    start=_safe_float(getattr(seg, "start", None)),
                    end=_safe_float(getattr(seg, "end", None)),
                    words=words,
                )
            )
        text = " ".join(s.text for s in segments).strip()
        return AsrResult(
            text=text,
            language=getattr(info, "language", self.settings.language),
            segments=segments,
            words=flat_words,
            raw={"backend": "faster_whisper"},
        )

    def _transcribe_transformers_whisper(self, audio_path: str) -> AsrResult:
        """Transformers Whisper fallback that avoids ffmpeg by passing numpy audio."""
        import librosa  # type: ignore
        import torch  # type: ignore
        from transformers import pipeline  # type: ignore

        if self._transformers_pipeline is None:
            device = 0 if self.settings.device == "cuda" and torch.cuda.is_available() else -1
            logger.info("Loading transformers Whisper model=%s", self.settings.transformers_whisper_model)
            self._transformers_pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.settings.transformers_whisper_model,
                device=device,
            )
        y, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
        try:
            raw = self._transformers_pipeline(
                {"array": y, "sampling_rate": sample_rate},
                return_timestamps="word",
            )
        except Exception:
            raw = self._transformers_pipeline(
                {"array": y, "sampling_rate": sample_rate},
                return_timestamps=True,
            )
        chunks = raw.get("chunks", []) or []
        words: List[AsrWord] = []
        for chunk in chunks:
            token = str(chunk.get("text", "")).strip()
            if not token:
                continue
            start, end = _timestamp_pair(chunk.get("timestamp"))
            words.append(AsrWord(word=token, start=start, end=end, source="transformers_whisper"))
        text = str(raw.get("text", "")).strip()
        end_time = max([w.end or 0.0 for w in words], default=None)
        segment = AsrSegment(text=text, start=0.0 if text else None, end=end_time, words=words)
        return AsrResult(
            text=text,
            language=self.settings.language,
            segments=[segment] if text else [],
            words=words,
            raw={"backend": "transformers_whisper", "model": self.settings.transformers_whisper_model},
        )


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _timestamp_pair(value: Any) -> tuple[Optional[float], Optional[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None, None
    return _safe_float(value[0]), _safe_float(value[1])


def _compact_raw(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Avoid storing huge arrays in API responses."""
    out: Dict[str, Any] = {"backend": "whisperx"}
    if "language" in raw:
        out["language"] = raw["language"]
    out["segment_count"] = len(raw.get("segments", []) or [])
    return out
