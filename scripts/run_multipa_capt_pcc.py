#!/usr/bin/env python
"""Evaluate this project's CAPT Mode A/B evidence on MultiPA with Qwen PCC.

This is intentionally separate from the TextPA reproduction pipeline. It uses
capt_pronunciation reports as the evidence source, asks Qwen for MultiPA-style
scores, then computes PCC against MultiPA human annotations.

Important experimental boundary:
- Mode A needs a reference text for each learner audio.
- Mode B needs a reference recording for each learner audio.
- ``MultiPA/multipa_reference_texts_reviewed.csv`` supplies the reviewed
  reference transcript for each MultiPA audio. Its transcription provenance
  and review status are recorded in the result metadata, so ASR-assisted text
  is not mislabeled as fully listening-verified reference text.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_multipa_pcc import compute_pcc, load_multipa_gold, resolve_audio_id  # noqa: E402


SCORING_PROMPT_VERSION = "multipa-continuous-v2"


MULTIPA_RUBRIC = """
MultiPA sentence-level scoring rubric, continuous 1.00-5.00:

Accuracy:
5 Excellent: nearly perfect sentence pronunciation with accurate articulation.
4 Good: minor pronunciation errors, highly understandable.
3 Fair: noticeable pronunciation errors, occasional listener effort.
2 Poor: pronounced pronunciation issues, listener effort needed.
1 Extremely poor: very few words are recognizable.

Fluency:
5 Fluent without noticeable pauses or stammering.
4 Mostly fluent with occasional minor pauses, repetition, or stammering.
3 Moderately fluent with noticeable hesitations but coherent communication.
2 Not very fluent, intermittent disruptions and pauses affect flow.
1 Not fluent, communication is challenging.

Prosody:
5 Expressive and well-modulated tone with effective pitch/rhythm variation.
4 Generally well-controlled pitch and rhythm.
3 Noticeable pitch/rhythm variation, not always aligned with content.
2 Somewhat monotonous, limited expressiveness.
1 Robotic or monotonous.

Phoneme:
Return a project-internal 1.00-5.00 phoneme clarity score from the evidence, but note
that MultiPA has no strict phoneme-level human gold for PCC.
""".strip()


def audio_aliases(value: Any) -> list[str]:
    text = str(value or "").strip().strip('"').strip("'")
    if not text:
        return []
    base = text.replace("\\", "/").rsplit("/", 1)[-1]
    aliases = [base]
    if Path(base).suffix:
        aliases.append(Path(base).stem)
    return list(dict.fromkeys(x for x in aliases if x))


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", str(value))
    if not match:
        return None
    out = float(match.group(0))
    return out if math.isfinite(out) else None


def format_score(value: Any) -> str:
    score = safe_float(value)
    return "" if score is None else f"{score:.2f}"


def nested_get(obj: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(part)
    return default if cur is None else cur


def compact_issue(issue: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "issue_id",
        "type",
        "severity",
        "ref_index",
        "word",
        "observed",
        "expected",
        "start",
        "end",
        "confidence",
        "evidence",
    }
    return {k: issue.get(k) for k in keep if k in issue and issue.get(k) not in (None, "", [])}


def build_evidence_payload(report: dict[str, Any]) -> dict[str, Any]:
    evidence = report.get("evidence") or {}
    feedback = report.get("feedback") or {}
    issue_groups = [
        "word_issues",
        "phoneme_issues",
        "pause_issues",
        "linking_issues",
        "stress_issues",
        "sentence_stress_issues",
        "intonation_issues",
        "rhythm_issues",
    ]
    issues: list[dict[str, Any]] = []
    for group in issue_groups:
        for issue in evidence.get(group) or []:
            if isinstance(issue, dict):
                item = compact_issue(issue)
                item["group"] = group
                issues.append(item)
    issues = sorted(
        issues,
        key=lambda x: (
            {"high": 3, "medium": 2, "low": 1, "info": 0}.get(str(x.get("severity")), 0),
            float(x.get("confidence") or 0.0),
        ),
        reverse=True,
    )
    return {
        "analysis_id": report.get("analysis_id"),
        "mode": report.get("mode"),
        "reference_text_source": report.get("reference_text_source"),
        "reference_text": report.get("reference_text") or "",
        "asr_text": nested_get(report, "asr.text", ""),
        "audio_duration_s": nested_get(report, "audio.duration_s"),
        "deterministic_scores_0_100": feedback.get("overall_scores") or {},
        "prosody_features": evidence.get("prosody") or {},
        "textpa_cues": evidence.get("textpa") or {},
        "issue_counts": {
            group: len(evidence.get(group) or [])
            for group in issue_groups
        },
        "top_issues": issues[:40],
        "phoneme_hypotheses": (evidence.get("phoneme_hypotheses") or [])[:20],
    }


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object found in LLM response")
    obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON is not an object")
    return obj


def qwen_score_payload(
    payload: dict[str, Any],
    *,
    api_key: str | None,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    mock: bool,
) -> tuple[dict[str, Any], str]:
    if mock:
        scores = payload.get("deterministic_scores_0_100") or {}
        out = {
            key: max(1.0, min(5.0, 1.0 + (float(scores.get(key, 50.0)) / 100.0) * 4.0))
            for key in ("accuracy", "phoneme", "fluency", "prosody")
        }
        out["rationale"] = {key: "mock score from deterministic score" for key in out}
        raw = json.dumps(out, ensure_ascii=False)
        return out, raw

    if not api_key:
        raise RuntimeError("QWEN_API_KEY or DASHSCOPE_API_KEY is required unless --mock-llm is set")
    system = (
        "You are a strict pronunciation assessment researcher. "
        "Use only the provided structured evidence. Do not invent errors. "
        "Return valid JSON only."
    )
    user = {
        "task": "score_multipa_from_project_capt_evidence",
        "rubric": MULTIPA_RUBRIC,
        "instructions": [
            "Score Accuracy, Phoneme, Fluency, and Prosody on a continuous 1.00-5.00 scale.",
            "Interpolate between rubric anchors and use the full continuous range; values such as 2.73 or 4.18 are expected.",
            "Return exactly two decimal places for every score. Do not collapse scores to integers or half-point bins unless the evidence lands exactly on that anchor.",
            "Preserve fine-grained differences between samples because the scores will be evaluated with Pearson correlation.",
            "Accuracy should reflect word/pronunciation understandability from evidence.",
            "Phoneme should reflect project-internal phone-level clarity from evidence.",
            "Fluency should reflect pauses, rhythm, repetitions, and smoothness.",
            "Prosody should reflect stress, intonation, pitch/rhythm naturalness.",
            "Use deterministic_scores_0_100 as evidence, not as a mandatory linear mapping.",
            "If evidence is weak because the reference was ASR-derived, lower confidence in rationale but still score.",
        ],
        "output_schema": {
            "Accuracy": "number from 1.00 to 5.00 with exactly two decimal places",
            "Phoneme": "number from 1.00 to 5.00 with exactly two decimal places",
            "Fluency": "number from 1.00 to 5.00 with exactly two decimal places",
            "Prosody": "number from 1.00 to 5.00 with exactly two decimal places",
            "Reasoning": {
                "Accuracy": "short evidence-grounded rationale",
                "Phoneme": "short evidence-grounded rationale",
                "Fluency": "short evidence-grounded rationale",
                "Prosody": "short evidence-grounded rationale",
            },
        },
        "evidence": payload,
    }
    request_body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")[:2000]
        raise RuntimeError(f"Qwen API HTTP {exc.code}: {body}") from exc
    raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return extract_json_object(raw), raw


def sanitize_scores(parsed: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ("Accuracy", "Phoneme", "Fluency", "Prosody"):
        value = safe_float(parsed.get(key))
        if value is None:
            lower = key.lower()
            value = safe_float(parsed.get(lower))
        if value is None:
            continue
        if value > 5.0 and value <= 100.0:
            value = 1.0 + (value / 100.0) * 4.0
        out[key.lower()] = round(max(1.0, min(5.0, value)), 2)
    return out


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("audio"):
                cache[str(row["audio"])] = row
    return cache


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_reference_map(
    path: Path | None,
    alias_to_audio: dict[str, str],
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    if path is None:
        return {}, {"path": None, "row_count": 0, "matched_audio_count": 0}
    if not path.exists():
        raise FileNotFoundError(f"Reference-text CSV not found: {path}")
    out: dict[str, dict[str, str]] = {}
    unmatched: list[str] = []
    duplicate_audio: list[str] = []
    transcription_models: set[str] = set()
    review_status_counts: dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"No CSV header found in reference-text file: {path}")
        if not {"audio", "utt_id", "id"} & fieldnames:
            raise ValueError(
                "Reference-text CSV needs an audio identifier column named audio, utt_id, or id. "
                f"Available columns: {sorted(fieldnames)}"
            )
        if not {"reference_text", "text"} & fieldnames:
            raise ValueError(
                "Reference-text CSV needs a reference_text or text column. "
                f"Available columns: {sorted(fieldnames)}"
            )
        row_count = 0
        for row in reader:
            row_count += 1
            audio = resolve_audio_id(row.get("audio") or row.get("utt_id") or row.get("id"), alias_to_audio)
            if not audio:
                unmatched.append(str(row.get("audio") or row.get("utt_id") or row.get("id") or ""))
                continue
            if audio in out:
                duplicate_audio.append(audio)
                continue
            transcription_model = str(row.get("transcription_model") or "").strip()
            if transcription_model:
                transcription_models.add(transcription_model)
            review_status = str(row.get("transcription_review_status") or "").strip()
            if review_status:
                review_status_counts[review_status] = review_status_counts.get(review_status, 0) + 1
            out[audio] = {
                "reference_text": str(row.get("reference_text") or row.get("text") or "").strip(),
                "reference_audio": str(row.get("reference_audio") or row.get("ref_audio") or "").strip(),
                "mean_accuracy": str(row.get("mean_accuracy") or "").strip(),
                "mean_fluency": str(row.get("mean_fluency") or "").strip(),
                "mean_prosody": str(row.get("mean_prosody") or "").strip(),
            }
    if duplicate_audio:
        raise ValueError(
            "Reference-text CSV contains duplicate audio rows: " + ", ".join(sorted(set(duplicate_audio))[:10])
        )
    return out, {
        "path": str(path),
        "row_count": row_count,
        "matched_audio_count": len(out),
        "unmatched_row_count": len(unmatched),
        "unmatched_row_examples": unmatched[:10],
        "has_mean_human_scores": all(
            name in fieldnames for name in ("mean_accuracy", "mean_fluency", "mean_prosody")
        ),
        "transcription_models": sorted(transcription_models),
        "review_status_counts": dict(sorted(review_status_counts.items())),
        "reference_text_origin": "asr_transcription" if transcription_models else "provided_reference_text",
    }


def validate_reference_map(
    reference_map: dict[str, dict[str, str]],
    gold: dict[str, dict[str, Any]],
    audio_names: list[str],
    *,
    require_reference_text: bool,
) -> None:
    missing = [audio for audio in audio_names if audio not in reference_map]
    if missing:
        raise ValueError(
            "Reference-text CSV does not cover every requested MultiPA audio. Missing "
            f"{len(missing)} item(s), for example: {', '.join(missing[:5])}"
        )
    if require_reference_text:
        empty = [audio for audio in audio_names if not reference_map[audio].get("reference_text")]
        if empty:
            raise ValueError(
                "Reference-text CSV has empty reference_text values for "
                f"{len(empty)} item(s), for example: {', '.join(empty[:5])}"
            )

    # The supplied file includes per-audio human-score means. Check that those
    # values agree with the five-annotator source used as PCC gold, but never
    # pass them to Qwen or include them in the CAPT evidence.
    score_mismatches: list[str] = []
    for audio in audio_names:
        for metric in ("accuracy", "fluency", "prosody"):
            supplied = safe_float(reference_map[audio].get(f"mean_{metric}"))
            if supplied is None:
                continue
            expected = safe_float(gold[audio].get(f"human_{metric}"))
            if expected is None or not math.isclose(supplied, expected, abs_tol=1e-6):
                score_mismatches.append(
                    f"{audio} ({metric}: reference CSV={supplied}, annotation.csv={expected})"
                )
    if score_mismatches:
        raise ValueError(
            "Human-score means in the reference-text CSV do not match annotation.csv. "
            "Examples: " + "; ".join(score_mismatches[:5])
        )


def report_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def preflight_asr_backend(backend: str) -> None:
    if backend == "faster_whisper":
        if not module_available("faster_whisper"):
            raise SystemExit(
                "Missing ASR backend package: faster-whisper.\n"
                "Install it with:\n"
                "  python -m pip install faster-whisper\n"
                "Then rerun with --asr-backend faster_whisper."
            )
        return
    if backend == "transformers_whisper":
        missing = [name for name in ("transformers", "torch", "librosa") if not module_available(name)]
        if missing:
            raise SystemExit(
                "Missing packages for --asr-backend transformers_whisper: "
                + ", ".join(missing)
                + "\nInstall project dependencies with:\n"
                '  python -m pip install -e ".[capt]"'
            )
        return
    if backend == "whisperx":
        if shutil.which("ffmpeg") is None:
            raise SystemExit(
                "Missing ffmpeg.exe in PATH. WhisperX requires the ffmpeg CLI to load audio.\n"
                "Install it with:\n"
                "  winget install Gyan.FFmpeg\n"
                "Then reopen PowerShell and verify:\n"
                "  ffmpeg -version"
            )
        if module_available("whisperx") or module_available("faster_whisper"):
            return
        raise SystemExit(
            "CAPT_ASR_BACKEND defaults to whisperx, but neither whisperx nor faster-whisper is installed.\n"
            "Recommended Windows fix:\n"
            "  python -m pip install faster-whisper\n"
            "  rerun with --asr-backend faster_whisper\n"
            "Full optional install:\n"
            '  python -m pip install -e ".[capt]"'
        )
    raise SystemExit(f"Unknown ASR backend: {backend}")


def report_path(report_root: Path, mode: str, audio: str) -> Path:
    analysis_id = f"multipa_{mode}_{Path(audio).stem}"
    return report_root / analysis_id / "report.json"


def index_existing_reports(report_root: Path, alias_to_audio: dict[str, str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not report_root.exists():
        return out
    for path in sorted(report_root.rglob("report.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        candidates = [
            nested_get(report, "audio.original_path"),
            nested_get(report, "audio.path"),
            report.get("analysis_id"),
        ]
        for candidate in candidates:
            audio = resolve_audio_id(candidate, alias_to_audio)
            if audio:
                out.setdefault(audio, path)
                break
    return out


def run_reports(
    *,
    mode: str,
    gold_audio_names: list[str],
    wav_dir: Path,
    report_root: Path,
    reference_map: dict[str, dict[str, str]],
    reference_source: str,
    allow_self_reference: bool,
    force: bool,
) -> None:
    from capt_pronunciation.runtime import CaptPipeline
    from capt_pronunciation.contracts import CaptSettings

    settings = CaptSettings.from_env()
    settings.work_dir = report_root.resolve()
    settings.work_dir.mkdir(parents=True, exist_ok=True)
    pipeline = CaptPipeline(settings)

    for index, audio in enumerate(gold_audio_names, start=1):
        out_path = report_path(report_root, mode, audio)
        ref = reference_map.get(audio, {})
        if out_path.exists() and not force:
            existing_reference = ""
            if mode == "modeA" and reference_source == "map":
                try:
                    existing_report = json.loads(out_path.read_text(encoding="utf-8"))
                    existing_reference = str(existing_report.get("reference_text") or "").strip()
                except (OSError, json.JSONDecodeError):
                    pass
                if existing_reference != ref.get("reference_text", ""):
                    print(f"[{index}/{len(gold_audio_names)}] reference text changed; regenerate report: {audio}")
                else:
                    print(f"[{index}/{len(gold_audio_names)}] skip existing report: {audio}")
                    continue
            else:
                print(f"[{index}/{len(gold_audio_names)}] skip existing report: {audio}")
                continue
        audio_path = wav_dir / audio
        if not audio_path.exists():
            raise FileNotFoundError(f"Missing MultiPA wav: {audio_path}")
        analysis_id = f"multipa_{mode}_{Path(audio).stem}"
        if mode == "modeA":
            reference_text = ref.get("reference_text", "")
            if not reference_text and reference_source == "asr":
                print(f"[{index}/{len(gold_audio_names)}] ASR pseudo-reference: {audio}")
                prepared = pipeline.audio_processor.prepare_audio(audio_path, f"{analysis_id}_pseudo_ref")
                reference_text = pipeline.transcriber.transcribe(prepared.path).text.strip()
            if not reference_text:
                raise ValueError(
                    f"Missing reference_text for {audio}. Provide --reference-map or use --reference-source asr."
                )
            print(f"[{index}/{len(gold_audio_names)}] Mode A analyze_reading: {audio}")
            pipeline.analyze_reading(reference_text, audio_path, analysis_id=analysis_id)
        else:
            reference_audio = ref.get("reference_audio", "")
            reference_text = ref.get("reference_text", "")
            if not reference_audio and allow_self_reference:
                reference_audio = str(audio_path)
            if not reference_audio:
                raise ValueError(
                    f"Missing reference_audio for {audio}. Raw MultiPA cannot run strict Mode B without a reference map."
                )
            print(f"[{index}/{len(gold_audio_names)}] Mode B analyze_imitation: {audio}")
            pipeline.analyze_imitation(reference_audio, audio_path, reference_text=reference_text, analysis_id=analysis_id)


def score_reports(
    *,
    mode: str,
    gold_audio_names: list[str],
    report_root: Path,
    alias_to_audio: dict[str, str],
    output_dir: Path,
    qwen_model: str,
    qwen_base_url: str,
    qwen_api_key: str | None,
    temperature: float,
    max_tokens: int,
    mock_llm: bool,
    force_llm: bool,
    sleep_s: float,
) -> dict[str, dict[str, float]]:
    cache_path = output_dir / f"qwen_scores_{mode}.jsonl"
    cache = load_cache(cache_path)
    report_index = index_existing_reports(report_root, alias_to_audio)
    predictions: dict[str, dict[str, float]] = {}
    rows_for_csv: list[dict[str, Any]] = []

    for index, audio in enumerate(gold_audio_names, start=1):
        path = report_path(report_root, mode, audio)
        if not path.exists():
            path = report_index.get(audio, path)
        if not path.exists():
            print(f"[{index}/{len(gold_audio_names)}] missing report, skip scoring: {audio}")
            continue
        signature = report_sha256(path)
        cached = cache.get(audio)
        if (
            cached
            and not force_llm
            and cached.get("report_sha256") == signature
            and cached.get("qwen_model") == qwen_model
            and cached.get("temperature") == temperature
            and cached.get("max_tokens") == max_tokens
            and cached.get("scoring_prompt_version") == SCORING_PROMPT_VERSION
        ):
            row = cached
        else:
            report = json.loads(path.read_text(encoding="utf-8"))
            payload = build_evidence_payload(report)
            print(f"[{index}/{len(gold_audio_names)}] Qwen scoring: {audio}")
            parsed, raw = qwen_score_payload(
                payload,
                api_key=qwen_api_key,
                base_url=qwen_base_url,
                model=qwen_model,
                temperature=temperature,
                max_tokens=max_tokens,
                mock=mock_llm,
            )
            scores = sanitize_scores(parsed)
            row = {
                "audio": audio,
                "analysis_id": report.get("analysis_id"),
                "mode": mode,
                "report_sha256": signature,
                "qwen_model": qwen_model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "scoring_prompt_version": SCORING_PROMPT_VERSION,
                "scores": scores,
                "raw_response": raw[:4000],
            }
            append_jsonl(cache_path, row)
            if sleep_s:
                time.sleep(sleep_s)
        scores = row.get("scores") or {}
        predictions[audio] = {k: float(v) for k, v in scores.items() if safe_float(v) is not None}
        rows_for_csv.append(
            {
                "audio": audio,
                "accuracy": format_score(predictions[audio].get("accuracy")),
                "phoneme": format_score(predictions[audio].get("phoneme")),
                "fluency": format_score(predictions[audio].get("fluency")),
                "prosody": format_score(predictions[audio].get("prosody")),
                "analysis_id": row.get("analysis_id"),
            }
        )

    csv_path = output_dir / f"predictions_{mode}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio", "accuracy", "phoneme", "fluency", "prosody", "analysis_id"])
        writer.writeheader()
        writer.writerows(rows_for_csv)
    return predictions


def main() -> int:
    if sys.version_info < (3, 10):
        raise SystemExit(
            "This project requires Python >= 3.10 for CAPT analysis. "
            "You are running Python "
            f"{sys.version_info.major}.{sys.version_info.minor}. "
            "Use .venv\\Scripts\\python.exe after rebuilding the venv, "
            "or create a new Python 3.10+ environment."
        )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", default="MultiPA/annotation.csv")
    parser.add_argument("--wav-dir", default="MultiPA/wav")
    parser.add_argument("--output-dir", default="runs/capt_multipa_project_pcc")
    parser.add_argument(
        "--report-root",
        default=None,
        help="Directory containing report.json files. Defaults to <output-dir>/reports.",
    )
    parser.add_argument("--mode", choices=["modeA", "modeB"], required=True)
    parser.add_argument("--run-analysis", action="store_true", help="Generate CAPT report.json files before scoring")
    parser.add_argument("--force-analysis", action="store_true")
    parser.add_argument(
        "--reference-map",
        "--reference-texts",
        dest="reference_map",
        default="MultiPA/multipa_reference_texts_reviewed.csv",
        help=(
            "CSV with audio and reference_text columns. Defaults to "
            "MultiPA/multipa_reference_texts_reviewed.csv. The legacy --reference-map spelling is retained."
        ),
    )
    parser.add_argument(
        "--reference-source",
        choices=["map", "asr"],
        default="map",
        help="For Mode A only: use the reference-text CSV (default) or an on-the-fly ASR pseudo reference",
    )
    parser.add_argument(
        "--allow-self-reference",
        action="store_true",
        help="Debug only for Mode B: use the same MultiPA audio as reference and learner",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mock-llm", action="store_true", help="Dry run without Qwen API")
    parser.add_argument("--force-llm", action="store_true")
    parser.add_argument("--sleep-between-llm-calls", type=float, default=0.0)
    parser.add_argument(
        "--asr-backend",
        choices=["whisperx", "faster_whisper", "transformers_whisper"],
        default=os.getenv("CAPT_ASR_BACKEND", "whisperx"),
        help="ASR backend used when --run-analysis is set",
    )
    parser.add_argument(
        "--whisper-model",
        default=None,
        help="Whisper model name for whisperx/faster-whisper, e.g. small.en or large-v3",
    )
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Override CAPT_DEVICE")
    parser.add_argument("--compute-type", default=None, help="Override CAPT_COMPUTE_TYPE, e.g. int8 or float16")
    parser.add_argument("--qwen-model", default=os.getenv("QWEN_MODEL", "qwen3.6-plus"))
    parser.add_argument(
        "--qwen-base-url",
        default=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    parser.add_argument("--qwen-api-key", default=os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--include-word-proxy", action="store_true")
    args = parser.parse_args()

    os.environ["CAPT_ASR_BACKEND"] = args.asr_backend
    if args.whisper_model:
        os.environ["CAPT_WHISPERX_MODEL"] = args.whisper_model
        os.environ["CAPT_TRANSFORMERS_WHISPER_MODEL"] = args.whisper_model
    if args.device:
        os.environ["CAPT_DEVICE"] = args.device
    if args.compute_type:
        os.environ["CAPT_COMPUTE_TYPE"] = args.compute_type

    if args.run_analysis:
        preflight_asr_backend(args.asr_backend)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_root = Path(args.report_root) if args.report_root else output_dir / "reports"
    gold, alias_to_audio = load_multipa_gold(Path(args.annotation))
    audio_names = list(gold.keys())
    if args.limit is not None:
        audio_names = audio_names[: args.limit]
    reference_map_path = Path(args.reference_map) if args.reference_map else None
    reference_map, reference_metadata = load_reference_map(reference_map_path, alias_to_audio)
    if args.mode == "modeA" and args.reference_source == "map":
        validate_reference_map(
            reference_map,
            gold,
            audio_names,
            require_reference_text=True,
        )
    has_mode_b_references = bool(reference_map) and all(
        bool(reference_map.get(audio, {}).get("reference_audio")) for audio in audio_names
    )

    metadata = {
        "mode": args.mode,
        "annotation": args.annotation,
        "reference_texts": reference_metadata,
        "scoring_prompt_version": SCORING_PROMPT_VERSION,
        "score_precision": "continuous_1.00_to_5.00_two_decimals",
        "n_audio_requested": len(audio_names),
        "reference_source": args.reference_source if args.mode == "modeA" else "reference_map",
        "allow_self_reference": bool(args.allow_self_reference),
        "strict_experiment": bool(
            (
                args.mode == "modeA"
                and args.reference_source == "map"
                and reference_map
                and reference_metadata.get("reference_text_origin") != "asr_transcription"
            )
            or (args.mode == "modeB" and has_mode_b_references and not args.allow_self_reference)
        ),
        "warning": "",
    }
    if args.mode == "modeA" and args.reference_source == "asr":
        metadata["warning"] = "Mode A uses ASR transcript as pseudo reference because raw MultiPA has no reference text."
    elif (
        args.mode == "modeA"
        and args.reference_source == "map"
        and reference_metadata.get("reference_text_origin") == "asr_transcription"
    ):
        metadata["warning"] = (
            "The supplied reference-text CSV is ASR-assisted and may contain pending_audio_review rows. "
            "This remains a fixed transcript-reference adaptation, not a fully listening-verified or prompt-grounded strict Mode A test."
        )
    if args.mode == "modeB" and args.allow_self_reference:
        metadata["warning"] = "Mode B self-reference is only a debug run and is not a valid paper experiment."
    if args.mode == "modeB" and not has_mode_b_references and not args.allow_self_reference:
        metadata["warning"] = (
            "MultiPA reference-text CSV has no reference_audio for every learner recording; "
            "Mode B cannot be evaluated strictly without a separate reference-audio map."
        )

    if args.run_analysis:
        run_reports(
            mode=args.mode,
            gold_audio_names=audio_names,
            wav_dir=Path(args.wav_dir),
            report_root=report_root,
            reference_map=reference_map,
            reference_source=args.reference_source,
            allow_self_reference=args.allow_self_reference,
            force=args.force_analysis,
        )

    predictions = score_reports(
        mode=args.mode,
        gold_audio_names=audio_names,
        report_root=report_root,
        alias_to_audio=alias_to_audio,
        output_dir=output_dir,
        qwen_model=args.qwen_model,
        qwen_base_url=args.qwen_base_url,
        qwen_api_key=args.qwen_api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        mock_llm=args.mock_llm,
        force_llm=args.force_llm,
        sleep_s=args.sleep_between_llm_calls,
    )
    metrics = compute_pcc(gold, predictions, include_word_proxy=args.include_word_proxy)
    result = {
        "metadata": metadata,
        "matched_predictions": len(predictions),
        "metrics": metrics,
        "predictions_csv": str(output_dir / f"predictions_{args.mode}.csv"),
        "score_cache": str(output_dir / f"qwen_scores_{args.mode}.jsonl"),
        "report_root": str(report_root),
    }
    result_path = output_dir / f"pcc_{args.mode}.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
