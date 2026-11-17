"""Audio ingestion, preprocessing, VAD-lite, and acoustic feature extraction.

Functional module 1/7.

Responsibilities
----------------
- Convert user audio to 16 kHz mono wav for all downstream models.
- Provide a lightweight VAD/silence detector even when WhisperX VAD is absent.
- Extract pitch/intensity/duration features needed by pause, stress, and
  intonation diagnostics.
"""
from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .contracts import AudioInfo, CaptSettings


class AudioProcessingError(RuntimeError):
    pass


@dataclass
class Waveform:
    samples: np.ndarray
    sample_rate: int
    path: Optional[str] = None

    @property
    def duration_s(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return float(len(self.samples) / self.sample_rate)


class AudioProcessor:
    """Prepares and analyzes audio files.

    The pipeline stores a canonical wav file per analysis. Heavy models have
    strict input expectations, so all branches use this prepared wav.
    """

    def __init__(self, settings: CaptSettings):
        self.settings = settings

    def prepare_audio(self, input_path: str | Path, analysis_id: str) -> AudioInfo:
        input_path = Path(input_path).expanduser().resolve()
        if not input_path.exists():
            raise AudioProcessingError(f"Audio file not found: {input_path}")

        out_dir = self.settings.work_dir / analysis_id
        out_dir.mkdir(parents=True, exist_ok=True)
        output_wav = out_dir / "audio_16k_mono.wav"

        self._convert_to_wav(input_path, output_wav, self.settings.sample_rate)
        waveform = self.load_wave(output_wav, sr=self.settings.sample_rate)
        vad_segments = self.detect_speech_segments(waveform.samples, waveform.sample_rate)
        rms = float(np.sqrt(np.mean(np.square(waveform.samples)))) if waveform.samples.size else 0.0
        return AudioInfo(
            path=str(output_wav),
            original_path=str(input_path),
            sample_rate=waveform.sample_rate,
            duration_s=waveform.duration_s,
            channels=1,
            vad_segments=vad_segments,
            rms=rms,
        )

    def _convert_to_wav(self, input_path: Path, output_wav: Path, sample_rate: int) -> None:
        """Use ffmpeg when available; otherwise try soundfile/librosa fallback."""
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            cmd = [
                ffmpeg,
                "-y",
                "-i",
                str(input_path),
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-vn",
                "-f",
                "wav",
                str(output_wav),
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0:
                raise AudioProcessingError(f"ffmpeg failed: {proc.stderr[-2000:]}")
            return

        # Fallback path. This supports wav/flac/ogg if soundfile is installed.
        try:
            import librosa  # type: ignore
            import soundfile as sf  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise AudioProcessingError("ffmpeg is not installed and librosa/soundfile fallback is unavailable") from exc

        y, sr = librosa.load(str(input_path), sr=sample_rate, mono=True)
        sf.write(str(output_wav), y, sample_rate)

    def load_wave(self, path: str | Path, sr: Optional[int] = None) -> Waveform:
        path = Path(path)
        try:
            import librosa  # type: ignore

            y, sample_rate = librosa.load(str(path), sr=sr, mono=True)
            y = np.asarray(y, dtype=np.float32)
            return Waveform(y, int(sample_rate), str(path))
        except Exception:
            try:
                import soundfile as sf  # type: ignore

                y, sample_rate = sf.read(str(path), always_2d=False)
                if y.ndim > 1:
                    y = np.mean(y, axis=1)
                if sr and int(sample_rate) != int(sr):
                    raise AudioProcessingError("soundfile fallback cannot resample; install librosa or use ffmpeg")
                return Waveform(np.asarray(y, dtype=np.float32), int(sample_rate), str(path))
            except Exception as exc:
                raise AudioProcessingError(f"Cannot load audio: {path}") from exc

    def crop(self, waveform: Waveform, start: Optional[float], end: Optional[float], pad_s: float = 0.04) -> Waveform:
        if start is None or end is None:
            return Waveform(np.array([], dtype=np.float32), waveform.sample_rate, waveform.path)
        s = max(0, int((start - pad_s) * waveform.sample_rate))
        e = min(len(waveform.samples), int((end + pad_s) * waveform.sample_rate))
        return Waveform(waveform.samples[s:e].copy(), waveform.sample_rate, waveform.path)

    def save_temp_wave(self, waveform: Waveform, suffix: str = ".wav") -> Path:
        try:
            import soundfile as sf  # type: ignore
        except Exception as exc:
            raise AudioProcessingError("soundfile is required to save temporary segments") from exc
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.close()
        sf.write(tmp.name, waveform.samples, waveform.sample_rate)
        return Path(tmp.name)

    def detect_speech_segments(
        self,
        samples: np.ndarray,
        sample_rate: int,
        frame_ms: float = 30.0,
        hop_ms: float = 10.0,
        energy_percentile: float = 35.0,
        min_speech_s: float = 0.10,
        min_silence_s: float = 0.12,
    ) -> List[Dict[str, float]]:
        """Lightweight energy VAD.

        This is intentionally simple: WhisperX has its own VAD, but diagnostics
        still need a dependency-light way to infer pauses and speech duration.
        """
        if samples.size == 0:
            return []
        frame = max(1, int(sample_rate * frame_ms / 1000.0))
        hop = max(1, int(sample_rate * hop_ms / 1000.0))
        if len(samples) < frame:
            return [{"start": 0.0, "end": len(samples) / sample_rate}]

        energies: List[float] = []
        starts: List[int] = []
        for start in range(0, len(samples) - frame + 1, hop):
            x = samples[start : start + frame]
            energies.append(float(np.sqrt(np.mean(x * x) + 1e-12)))
            starts.append(start)
        e = np.asarray(energies)
        noise = np.percentile(e, energy_percentile)
        threshold = max(float(noise * 1.8), 0.008)
        speech = e > threshold

        segments: List[Tuple[float, float]] = []
        active_start: Optional[int] = None
        last_active: Optional[int] = None
        max_silence_frames = max(1, int(min_silence_s / (hop_ms / 1000.0)))
        silence_count = 0

        for i, is_speech in enumerate(speech):
            if is_speech:
                if active_start is None:
                    active_start = starts[i]
                last_active = starts[i] + frame
                silence_count = 0
            elif active_start is not None:
                silence_count += 1
                if silence_count >= max_silence_frames:
                    end = last_active if last_active is not None else starts[i]
                    if (end - active_start) / sample_rate >= min_speech_s:
                        segments.append((active_start / sample_rate, end / sample_rate))
                    active_start = None
                    last_active = None
                    silence_count = 0
        if active_start is not None:
            end = last_active if last_active is not None else len(samples)
            if (end - active_start) / sample_rate >= min_speech_s:
                segments.append((active_start / sample_rate, min(len(samples) / sample_rate, end / sample_rate)))

        # Merge very close segments.
        merged: List[Tuple[float, float]] = []
        for start, end in segments:
            if not merged or start - merged[-1][1] > min_silence_s:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], end)
        return [{"start": round(s, 3), "end": round(e, 3)} for s, e in merged]

    def silence_between_words(self, word_times: Iterable[Tuple[Optional[float], Optional[float]]]) -> List[Dict[str, float]]:
        """Return time gaps between consecutive timed words."""
        timed = [(s, e) for s, e in word_times if s is not None and e is not None]
        gaps: List[Dict[str, float]] = []
        for (s1, e1), (s2, e2) in zip(timed, timed[1:]):
            assert e1 is not None and s2 is not None
            gap = float(s2 - e1)
            if gap > 0:
                gaps.append({"start": float(e1), "end": float(s2), "duration": gap})
        return gaps

    def extract_pitch_track(self, waveform: Waveform, f0_min: float = 65.0, f0_max: float = 450.0) -> Dict[str, List[float]]:
        """Extract pitch track with parselmouth when possible, else librosa.yin."""
        if waveform.samples.size == 0:
            return {"times": [], "f0": []}
        try:
            import parselmouth  # type: ignore

            snd = parselmouth.Sound(waveform.samples, waveform.sample_rate)
            pitch = snd.to_pitch(pitch_floor=f0_min, pitch_ceiling=f0_max)
            times = [pitch.get_time_from_frame_number(i + 1) for i in range(pitch.get_number_of_frames())]
            f0 = [pitch.get_value_in_frame(i + 1) or 0.0 for i in range(pitch.get_number_of_frames())]
            return {"times": [float(t) for t in times], "f0": [float(v) for v in f0]}
        except Exception:
            pass
        try:
            import librosa  # type: ignore

            hop_length = int(0.01 * waveform.sample_rate)
            frame_length = int(0.04 * waveform.sample_rate)
            f0 = librosa.yin(
                waveform.samples.astype(np.float32),
                fmin=f0_min,
                fmax=f0_max,
                sr=waveform.sample_rate,
                frame_length=frame_length,
                hop_length=hop_length,
            )
            times = librosa.frames_to_time(np.arange(len(f0)), sr=waveform.sample_rate, hop_length=hop_length)
            f0 = np.where(np.isfinite(f0), f0, 0.0)
            return {"times": [float(x) for x in times], "f0": [float(x) for x in f0]}
        except Exception:
            return {"times": [], "f0": []}

    def extract_intensity_track(self, waveform: Waveform, frame_ms: float = 25.0, hop_ms: float = 10.0) -> Dict[str, List[float]]:
        if waveform.samples.size == 0:
            return {"times": [], "db": []}
        frame = max(1, int(waveform.sample_rate * frame_ms / 1000.0))
        hop = max(1, int(waveform.sample_rate * hop_ms / 1000.0))
        vals: List[float] = []
        times: List[float] = []
        for start in range(0, max(1, len(waveform.samples) - frame + 1), hop):
            x = waveform.samples[start : start + frame]
            rms = float(np.sqrt(np.mean(x * x) + 1e-12))
            vals.append(20.0 * math.log10(rms + 1e-8))
            times.append((start + frame / 2) / waveform.sample_rate)
        return {"times": times, "db": vals}

    def summarize_acoustics(self, waveform: Waveform, vad_segments: List[Dict[str, float]], word_count: int) -> Dict[str, float | int | None | Dict[str, List[float]]]:
        duration = waveform.duration_s
        speech_s = sum(max(0.0, seg["end"] - seg["start"]) for seg in vad_segments)
        pauses = self._pauses_from_vad(vad_segments, duration)
        pitch = self.extract_pitch_track(waveform)
        voiced = [x for x in pitch["f0"] if x and x > 0]
        intensity = self.extract_intensity_track(waveform)
        return {
            "speech_rate_wpm": (word_count / duration * 60.0) if duration > 0 else None,
            "articulation_rate_wpm": (word_count / speech_s * 60.0) if speech_s > 0 else None,
            "pause_count": len(pauses),
            "total_pause_s": round(sum(p["duration"] for p in pauses), 3),
            "pitch_mean_hz": round(float(np.mean(voiced)), 2) if voiced else None,
            "pitch_range_hz": round(float(np.percentile(voiced, 95) - np.percentile(voiced, 5)), 2) if len(voiced) >= 5 else None,
            "intensity_mean_db": round(float(np.mean(intensity["db"])), 2) if intensity["db"] else None,
            "tracks": {"pitch": pitch, "intensity": intensity},
        }

    def _pauses_from_vad(self, vad_segments: List[Dict[str, float]], duration_s: float) -> List[Dict[str, float]]:
        pauses: List[Dict[str, float]] = []
        cursor = 0.0
        for seg in vad_segments:
            start = float(seg["start"])
            if start - cursor > 0.12:
                pauses.append({"start": cursor, "end": start, "duration": start - cursor})
            cursor = max(cursor, float(seg["end"]))
        if duration_s - cursor > 0.12:
            pauses.append({"start": cursor, "end": duration_s, "duration": duration_s - cursor})
        return pauses
