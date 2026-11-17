from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import soundfile as sf
from tqdm import tqdm

from .utils import ensure_dir, normalize_text


@dataclass(frozen=True)
class Utterance:
    utt_id: str
    audio_path: Path
    reference_text: str | None
    human_accuracy: float | None
    human_fluency: float | None
    human_prosodic: float | None = None
    human_total: float | None = None
    speaker: str | None = None
    split: str | None = None


def _stable_utt_id(index: int, text: str, audio_path: str | None = None) -> str:
    seed = f"{index}|{text}|{audio_path or ''}".encode("utf-8")
    return hashlib.sha1(seed).hexdigest()[:12]


def prepare_speechocean762(
    output_dir: str | Path,
    split: str = "test",
    limit: int | None = None,
    dataset_name: str = "mispeech/speechocean762",
    overwrite_audio: bool = False,
    local_root: str | Path | None = None,
) -> Path:
    """Download/load Speechocean762 via Hugging Face datasets and build a manifest.

    The paper uses the public Speechocean test set for zero-shot evaluation. This helper
    keeps the exact split visible in a CSV and stores audio as 16-kHz wav files so that
    Whisper/Charsiu/phoneme recognition can all read the same paths.
    """
    if local_root is not None:
        return prepare_local_speechocean762(
            local_root=local_root,
            output_dir=output_dir,
            split=split,
            limit=limit,
        )

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install datasets: pip install datasets") from exc

    out = ensure_dir(output_dir)
    audio_dir = ensure_dir(out / "audio" / split)
    manifest_path = out / f"{split}_manifest.csv"

    ds = load_dataset(dataset_name, split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    rows: list[dict[str, Any]] = []
    for i, sample in enumerate(tqdm(ds, desc=f"Preparing {dataset_name}:{split}")):
        text = normalize_text(str(sample.get("text", "")))
        audio_obj = sample["audio"]
        sr = int(audio_obj["sampling_rate"])
        arr = audio_obj["array"]
        source_path = audio_obj.get("path") or ""
        utt_id = Path(source_path).stem if source_path else _stable_utt_id(i, text)
        wav_path = audio_dir / f"{utt_id}.wav"
        if overwrite_audio or not wav_path.exists():
            sf.write(wav_path, arr, sr)

        rows.append(
            {
                "utt_id": utt_id,
                "split": split,
                "audio_path": str(wav_path),
                "reference_text": text,
                "human_accuracy": sample.get("accuracy"),
                "human_fluency": sample.get("fluency"),
                "human_prosodic": sample.get("prosodic"),
                "human_completeness": sample.get("completeness"),
                "human_total": sample.get("total"),
                "speaker": sample.get("speaker"),
                "gender": sample.get("gender"),
                "age": sample.get("age"),
            }
        )

    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def _read_kaldi_mapping(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            rows[parts[0]] = parts[1] if len(parts) > 1 else ""
    return rows


def prepare_local_speechocean762(
    local_root: str | Path,
    output_dir: str | Path,
    split: str = "test",
    limit: int | None = None,
) -> Path:
    """Build a manifest from an already-downloaded Speechocean762 directory.

    Expected layout is the public corpus layout:

    ``local_root/{train,test}/{text,wav.scp,utt2spk}``, ``local_root/WAVE/...``, and
    ``local_root/resource/scores.json``.
    """
    root = Path(local_root)
    split_dir = root / split
    scores_path = root / "resource" / "scores.json"
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing Speechocean split directory: {split_dir}")
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing Speechocean scores file: {scores_path}")

    out = ensure_dir(output_dir)
    manifest_path = out / f"{split}_manifest.csv"
    text = _read_kaldi_mapping(split_dir / "text")
    wav_scp = _read_kaldi_mapping(split_dir / "wav.scp")
    utt2spk = _read_kaldi_mapping(split_dir / "utt2spk")
    spk2gender = _read_kaldi_mapping(split_dir / "spk2gender")
    spk2age = _read_kaldi_mapping(split_dir / "spk2age")
    with scores_path.open("r", encoding="utf-8") as f:
        scores = json.load(f)

    rows: list[dict[str, Any]] = []
    utt_ids = list(text.keys())
    if limit is not None:
        utt_ids = utt_ids[:limit]
    for utt_id in utt_ids:
        if utt_id not in wav_scp:
            raise ValueError(f"Missing wav.scp entry for utterance {utt_id}")
        rel_audio = Path(wav_scp[utt_id])
        audio_path = rel_audio if rel_audio.is_absolute() else root / rel_audio
        score = scores.get(utt_id, {})
        speaker = utt2spk.get(utt_id)
        rows.append(
            {
                "utt_id": utt_id,
                "split": split,
                "audio_path": str(audio_path),
                "reference_text": normalize_text(str(score.get("text") or text.get(utt_id, ""))),
                "human_accuracy": score.get("accuracy"),
                "human_fluency": score.get("fluency"),
                "human_prosodic": score.get("prosodic"),
                "human_completeness": score.get("completeness"),
                "human_total": score.get("total"),
                "speaker": speaker,
                "gender": spk2gender.get(speaker or ""),
                "age": spk2age.get(speaker or ""),
            }
        )

    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def _score_label_to_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)", str(value))
    return float(match.group(1)) if match else None


def prepare_multipa(
    local_root: str | Path,
    output_dir: str | Path,
    split: str = "test",
    limit: int | None = None,
) -> Path:
    """Build a manifest from the MultiPA open-response pronunciation dataset.

    MultiPA stores five annotator rows per audio clip in ``annotation.csv``. This
    helper aggregates them to one row per clip using the mean sentence-level scores.
    """
    root = Path(local_root)
    annotation_path = root / "annotation.csv"
    wav_dir = root / "wav"
    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing MultiPA annotation file: {annotation_path}")
    if not wav_dir.exists():
        raise FileNotFoundError(f"Missing MultiPA wav directory: {wav_dir}")

    out = ensure_dir(output_dir)
    manifest_path = out / f"{split}_manifest.csv"
    df = pd.read_csv(annotation_path)
    required = {"audio", "accuracy", "fluency"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing MultiPA annotation columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    grouped = df.groupby("audio", sort=False)
    for audio, group in grouped:
        audio_name = str(audio)
        audio_path = wav_dir / audio_name
        if not audio_path.exists():
            raise FileNotFoundError(f"Missing MultiPA audio file: {audio_path}")

        scores = {
            "accuracy": [_score_label_to_float(v) for v in group["accuracy"]],
            "fluency": [_score_label_to_float(v) for v in group["fluency"]],
            "prosodic": [_score_label_to_float(v) for v in group["prosody"]] if "prosody" in group else [],
        }
        means = {
            key: float(pd.Series([v for v in vals if v is not None]).mean()) if vals else None
            for key, vals in scores.items()
        }
        utt_id = Path(audio_name).stem
        speaker = utt_id.split("---", 1)[0] if "---" in utt_id else None
        total_values = [means["accuracy"], means["fluency"], means["prosodic"]]
        total_values = [v for v in total_values if v is not None]
        rows.append(
            {
                "utt_id": utt_id,
                "split": split,
                "audio_path": str(audio_path),
                "reference_text": "",
                "human_accuracy": means["accuracy"],
                "human_fluency": means["fluency"],
                "human_prosodic": means["prosodic"],
                "human_total": float(pd.Series(total_values).mean()) if total_values else None,
                "speaker": speaker,
                "annotator_count": int(len(group)),
            }
        )
        if limit is not None and len(rows) >= limit:
            break

    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def load_manifest(path: str | Path, limit: int | None = None) -> list[Utterance]:
    df = pd.read_csv(path, dtype={"utt_id": str, "speaker": str, "split": str})
    if limit is not None:
        df = df.head(limit)
    utterances: list[Utterance] = []
    for _, row in df.iterrows():
        utterances.append(
            Utterance(
                utt_id=str(row["utt_id"]),
                audio_path=Path(str(row["audio_path"])),
                reference_text=None if pd.isna(row.get("reference_text")) else str(row.get("reference_text")),
                human_accuracy=None if pd.isna(row.get("human_accuracy")) else float(row.get("human_accuracy")),
                human_fluency=None if pd.isna(row.get("human_fluency")) else float(row.get("human_fluency")),
                human_prosodic=None if pd.isna(row.get("human_prosodic")) else float(row.get("human_prosodic")),
                human_total=None if pd.isna(row.get("human_total")) else float(row.get("human_total")),
                speaker=None if pd.isna(row.get("speaker")) else str(row.get("speaker")),
                split=None if pd.isna(row.get("split")) else str(row.get("split")),
            )
        )
    return utterances


def write_custom_manifest(
    rows: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Write a custom manifest for MultiPA-like or self-collected open-response data.

    Required columns: utt_id, audio_path. Optional columns: reference_text,
    human_accuracy, human_fluency, human_prosodic, human_total, speaker, split.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p, index=False)
    return p
