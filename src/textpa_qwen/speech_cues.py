from __future__ import annotations

import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import librosa
import numpy as np
import torch

from .utils import normalize_text, text_to_words

SILENCE_LABELS = {"sil", "sp", "spn", "pau", "pause", "<pad>", "[pad]", "<s>", "</s>", ""}


def configure_espeak_for_windows() -> None:
    """Help phonemizer find eSpeak NG from the default Windows installer path."""
    if os.name != "nt" or os.getenv("PHONEMIZER_ESPEAK_LIBRARY"):
        return
    candidate = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "eSpeak NG" / "libespeak-ng.dll"
    if candidate.exists():
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(candidate)
        os.environ["PATH"] = str(candidate.parent) + os.pathsep + os.environ.get("PATH", "")


@dataclass
class PhoneSegment:
    label: str
    start: float
    end: float


@dataclass
class SpeechCues:
    utt_id: str
    audio_path: str
    transcript: str
    phonemes_cmu: str
    phonemes_ipa: str
    canonical_ipa_from_transcript: str
    canonical_cmu_from_transcript: str
    asr_backend: str
    ipa_model: str
    cmu_aligner: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_audio_16k(audio_path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    audio, _sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    # Avoid pathological zero-length audio after loading.
    if audio.size == 0:
        raise ValueError(f"Empty audio: {audio_path}")
    return audio.astype(np.float32)


def normalize_phone_token(token: str) -> str:
    token = token.strip()
    token = token.replace("ː", ":")
    token = re.sub(r"\s+", "", token)
    return token


def normalize_phone_sequence(seq: str) -> str:
    seq = seq.replace("|", " ").replace("_", " ")
    seq = re.sub(r"\s+", " ", seq.strip())
    return seq


def canonical_ipa(text: str, language: str = "en-us") -> str:
    """Map text to canonical IPA using phonemizer/eSpeak, as in the paper.

    Stress markers are removed to make Smith-Waterman token comparison less brittle.
    """
    configure_espeak_for_windows()
    try:
        from phonemizer.backend import EspeakBackend
        from phonemizer.separator import Separator
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install phonemizer and espeak-ng to use canonical IPA mapping") from exc

    backend = EspeakBackend(language, preserve_punctuation=False, with_stress=False)
    words = text_to_words(text)
    if not words:
        return ""
    # Use a distinct word separator for newer phonemizer versions, then normalize it away.
    phones = backend.phonemize(
        [" ".join(words)],
        separator=Separator(phone=" ", word=" | ", syllable=""),
        strip=True,
        njobs=1,
    )[0]
    phones = phones.replace("ˈ", "").replace("ˌ", "")
    phones = normalize_phone_sequence(phones)
    return phones


def canonical_cmu(text: str) -> str:
    """Map text to CMU/ARPAbet labels using g2p_en.

    This is used for the CMU match scoring ablation. The main TextPA prompt uses the
    recognized CMU sequence from Charsiu, not this canonical sequence.
    """
    try:
        from g2p_en import G2p
    except ImportError:
        # g2p_en is optional. Return empty rather than failing all TextPA runs.
        return ""
    g2p = G2p()
    tokens = g2p(" ".join(text_to_words(text)))
    phones = [t for t in tokens if re.fullmatch(r"[A-Z]+[0-2]?", t)]
    return " ".join(phones)


class WhisperTranscriber:
    """ASR wrapper for Whisper.

    backend='whisper' uses the openai-whisper package. backend='transformers-whisper'
    uses a Hugging Face ASR pipeline. Both are supported because package availability
    differs across lab machines.
    """

    def __init__(
        self,
        backend: str = "whisper",
        model_name: str = "large-v3",
        hf_model: str = "openai/whisper-large-v3",
        language: str = "en",
        device: str | None = None,
    ) -> None:
        self.backend = backend
        self.model_name = model_name
        self.hf_model = hf_model
        self.language = language
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Any | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if self.backend == "whisper":
            try:
                import whisper  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "Install openai-whisper or set asr_backend=transformers-whisper"
                ) from exc
            self._model = whisper.load_model(self.model_name, device=self.device)
        elif self.backend == "transformers-whisper":
            from transformers import pipeline

            dev = 0 if self.device == "cuda" else -1
            self._model = pipeline(
                "automatic-speech-recognition",
                model=self.hf_model,
                device=dev,
            )
        else:
            raise ValueError(f"Unknown ASR backend: {self.backend}")

    def transcribe(self, audio_path: str | Path) -> str:
        self._load()
        if self.backend == "whisper":
            result = self._model.transcribe(
                str(audio_path),
                language=self.language,
                task="transcribe",
                fp16=(self.device == "cuda"),
                verbose=False,
            )
            return normalize_text(result.get("text", ""))
        audio = load_audio_16k(audio_path, sample_rate=16000)
        generate_kwargs = {}
        if not self.hf_model.endswith(".en"):
            generate_kwargs = {"language": self.language, "task": "transcribe"}
        result = self._model({"array": audio, "sampling_rate": 16000}, generate_kwargs=generate_kwargs)
        return normalize_text(result.get("text", ""))


class IpaPhonemeRecognizer:
    """IPA phoneme recognizer based on facebook/wav2vec2-lv-60-espeak-cv-ft."""

    def __init__(self, model_id: str = "facebook/wav2vec2-lv-60-espeak-cv-ft", device: str | None = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor: Any | None = None
        self._model: Any | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        configure_espeak_for_windows()
        from transformers import AutoModelForCTC, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForCTC.from_pretrained(self.model_id).to(self.device)
        self._model.eval()

    @torch.inference_mode()
    def recognize(self, audio_path: str | Path, sample_rate: int = 16000) -> str:
        self._load()
        audio = load_audio_16k(audio_path, sample_rate=sample_rate)
        inputs = self._processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self._model(**inputs).logits
        ids = torch.argmax(logits, dim=-1)
        decoded = self._processor.batch_decode(ids)[0]
        return normalize_phone_sequence(decoded)


def _coerce_charsiu_alignment(raw: Any) -> list[PhoneSegment]:
    """Normalize likely Charsiu alignment outputs into PhoneSegment objects.

    Charsiu has had minor API differences across commits. This function accepts common
    formats: list[(label, start, end)], list[(start, end, label)], dicts, and TextGrid-ish
    dict objects.
    """
    segments: list[PhoneSegment] = []
    if raw is None:
        return segments

    if isinstance(raw, dict):
        if "phones" in raw:
            raw = raw["phones"]
        elif "segments" in raw:
            raw = raw["segments"]
        else:
            raw = list(raw.values())

    for item in raw:
        label: str | None = None
        start: float | None = None
        end: float | None = None
        if isinstance(item, PhoneSegment):
            segments.append(item)
            continue
        if isinstance(item, dict):
            label = str(item.get("phone") or item.get("label") or item.get("text") or item.get("mark") or "")
            start = item.get("start") or item.get("xmin") or item.get("begin")
            end = item.get("end") or item.get("xmax") or item.get("finish")
        elif isinstance(item, (tuple, list)) and len(item) >= 3:
            # Common forms: (phone, start, end) or (start, end, phone)
            a, b, c = item[0], item[1], item[2]
            if isinstance(a, str):
                label, start, end = str(a), float(b), float(c)
            elif isinstance(c, str):
                start, end, label = float(a), float(b), str(c)
            else:
                # Sometimes: (start, phone, end)
                start, label, end = float(a), str(b), float(c)
        if label is not None and start is not None and end is not None:
            try:
                segments.append(PhoneSegment(label=label.strip(), start=float(start), end=float(end)))
            except Exception:
                continue
    segments.sort(key=lambda s: (s.start, s.end))
    return segments


def format_cmu_with_pauses(
    segments: Sequence[PhoneSegment],
    min_pause_s: float = 0.03,
    keep_silence_labels: bool = False,
) -> str:
    """Format Charsiu phone segments as TextPA CMU string with pause annotations."""
    out: list[str] = []
    prev_end: float | None = None
    pending_pause = 0.0

    for seg in segments:
        label = seg.label.strip()
        low = label.lower()
        low_clean = re.sub(r"[^a-z0-9]", "", low)
        dur = max(0.0, seg.end - seg.start)
        if (
            low in SILENCE_LABELS
            or low_clean in SILENCE_LABELS
            or re.fullmatch(r"sil[0-9]*", low_clean or "")
        ):
            pending_pause += dur
            if keep_silence_labels:
                out.append(label.upper())
            continue
        if prev_end is not None:
            gap = max(0.0, seg.start - prev_end)
            pending_pause += gap
        if pending_pause >= min_pause_s and out:
            out.append(f"({pending_pause:.2f}s pause)")
        pending_pause = 0.0
        clean = re.sub(r"[^A-Za-z0-9]", "", label).upper()
        if clean:
            out.append(clean)
        prev_end = max(prev_end or 0.0, seg.end)
    return " ".join(out)


class CharsiuCMURecognizer:
    """Charsiu predictive aligner wrapper for recognized CMU phones and pauses."""

    def __init__(self, model_id: str = "charsiu/en_w2v2_fc_10ms", lang: str = "en"):
        self.model_id = model_id
        self.lang = lang
        self._aligner: Any | None = None

    def _load(self) -> None:
        if self._aligner is not None:
            return
        charsiu_path = os.getenv("CHARSIU_PATH")
        if charsiu_path and charsiu_path not in sys.path:
            sys.path.append(str(Path(charsiu_path) / "src"))
            sys.path.append(str(Path(charsiu_path)))
        try:
            from Charsiu import charsiu_predictive_aligner  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Charsiu import failed. Clone https://github.com/lingjzhu/charsiu and set "
                "CHARSIU_PATH=/path/to/charsiu, or enable allow_cmu_fallback for debugging."
            ) from exc
        self._aligner = charsiu_predictive_aligner(aligner=self.model_id, lang=self.lang)

    def recognize(self, audio_path: str | Path) -> str:
        self._load()
        raw = self._aligner.align(audio=str(audio_path))
        segments = _coerce_charsiu_alignment(raw)
        if not segments:
            raise RuntimeError("Charsiu returned no phone segments; inspect its raw output/API version.")
        return format_cmu_with_pauses(segments)


class FallbackCMURecognizer:
    """Debug-only fallback: canonical CMU from ASR transcript without acoustic pauses.

    This does not reproduce the paper. It is useful only for smoke tests when Charsiu is
    not installed. Reported experiments should keep allow_cmu_fallback=false.
    """

    def recognize(self, audio_path: str | Path, transcript: str = "") -> str:
        return canonical_cmu(transcript)


class SpeechCueExtractor:
    def __init__(
        self,
        asr_backend: str = "whisper",
        whisper_model: str = "large-v3",
        whisper_hf_model: str = "openai/whisper-large-v3",
        ipa_model: str = "facebook/wav2vec2-lv-60-espeak-cv-ft",
        charsiu_model: str = "charsiu/en_w2v2_fc_10ms",
        language: str = "en",
        sample_rate: int = 16000,
        allow_cmu_fallback: bool = False,
        device: str | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.allow_cmu_fallback = allow_cmu_fallback
        self.asr = WhisperTranscriber(
            backend=asr_backend,
            model_name=whisper_model,
            hf_model=whisper_hf_model,
            language=language,
            device=device,
        )
        self.ipa = IpaPhonemeRecognizer(model_id=ipa_model, device=device)
        self.cmu = CharsiuCMURecognizer(model_id=charsiu_model, lang=language)
        self.cmu_fallback = FallbackCMURecognizer()
        self.asr_backend = asr_backend
        self.ipa_model = ipa_model
        self.charsiu_model = charsiu_model

    def extract(self, utt_id: str, audio_path: str | Path) -> SpeechCues:
        transcript = self.asr.transcribe(audio_path)
        ipa_seq = self.ipa.recognize(audio_path, sample_rate=self.sample_rate)
        try:
            cmu_seq = self.cmu.recognize(audio_path)
            cmu_aligner = self.charsiu_model
        except Exception:
            if not self.allow_cmu_fallback:
                raise
            cmu_seq = self.cmu_fallback.recognize(audio_path, transcript=transcript)
            cmu_aligner = "fallback_canonical_cmu_not_paper_faithful"
        return SpeechCues(
            utt_id=utt_id,
            audio_path=str(audio_path),
            transcript=transcript,
            phonemes_cmu=cmu_seq,
            phonemes_ipa=ipa_seq,
            canonical_ipa_from_transcript=canonical_ipa(transcript),
            canonical_cmu_from_transcript=canonical_cmu(transcript),
            asr_backend=self.asr_backend,
            ipa_model=self.ipa_model,
            cmu_aligner=cmu_aligner,
        )
