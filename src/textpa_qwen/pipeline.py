from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from .data import load_manifest
from .prompts import Ablation, build_textpa_prompt
from .qwen_client import make_qwen_client, run_llm_with_json_retries
from .scoring import MatchScoringConfig, add_textpa_scores, evaluate_predictions
from .speech_cues import SpeechCueExtractor
from .utils import append_jsonl, atomic_jsonl_write, ensure_dir, read_jsonl, write_json


def _cache_map(rows: Iterable[dict[str, Any]], key: str = "utt_id") -> dict[str, dict[str, Any]]:
    return {str(r[key]): r for r in rows if key in r}


def _valid_llm_cache_map(rows: Iterable[dict[str, Any]], key: str = "utt_id") -> dict[str, dict[str, Any]]:
    valid: dict[str, dict[str, Any]] = {}
    for row in rows:
        if key not in row or not row.get("valid_json", False):
            continue
        try:
            if not (
                np.isfinite(float(row.get("llm_accuracy")))
                and np.isfinite(float(row.get("llm_fluency")))
            ):
                continue
        except (TypeError, ValueError):
            continue
        valid[str(row[key])] = row
    return valid


def make_extractor_from_config(cfg: dict[str, Any]) -> SpeechCueExtractor:
    models = cfg.get("models", {})
    audio = cfg.get("audio", {})
    exp = cfg.get("experiment", {})
    return SpeechCueExtractor(
        asr_backend=models.get("asr_backend", "whisper"),
        whisper_model=models.get("whisper_model", "large-v3"),
        whisper_hf_model=models.get("whisper_hf_model", "openai/whisper-large-v3"),
        ipa_model=models.get("ipa_model", "facebook/wav2vec2-lv-60-espeak-cv-ft"),
        charsiu_model=models.get("charsiu_model", "charsiu/en_w2v2_fc_10ms"),
        language=audio.get("language", "en"),
        sample_rate=int(audio.get("sample_rate", 16000)),
        allow_cmu_fallback=bool(exp.get("allow_cmu_fallback", False)),
        device=models.get("device"),
    )


def extract_cues_for_manifest(cfg: dict[str, Any]) -> Path:
    paths = cfg.get("paths", {})
    exp = cfg.get("experiment", {})
    manifest = Path(paths.get("manifest", "data/speechocean762/test_manifest.csv"))
    output_dir = ensure_dir(paths.get("output_dir", "runs/textpa_qwen"))
    cues_path = output_dir / "cues.jsonl"
    limit = exp.get("limit")
    limit = int(limit) if limit is not None else None
    utterances = load_manifest(manifest, limit=limit)

    existing = _cache_map(read_jsonl(cues_path)) if exp.get("cache_cues", True) else {}
    extractor = make_extractor_from_config(cfg)

    all_rows: list[dict[str, Any]] = []
    for utt in tqdm(utterances, desc="Extracting TextPA cues"):
        if utt.utt_id in existing:
            row = existing[utt.utt_id]
        else:
            cues = extractor.extract(utt_id=utt.utt_id, audio_path=utt.audio_path)
            row = cues.to_dict()
            append_jsonl(row, cues_path)
        all_rows.append(row)

    # Rewrite cache in manifest order for reproducibility.
    atomic_jsonl_write(all_rows, cues_path)
    return cues_path


def _merge_cues_with_manifest(cues_rows: list[dict[str, Any]], manifest_path: str | Path) -> pd.DataFrame:
    cues_df = pd.DataFrame(cues_rows)
    manifest_df = pd.read_csv(manifest_path, dtype={"utt_id": str, "speaker": str, "split": str})
    keep_cols = [
        c
        for c in [
            "utt_id",
            "reference_text",
            "human_accuracy",
            "human_fluency",
            "human_prosodic",
            "human_total",
            "speaker",
            "split",
        ]
        if c in manifest_df.columns
    ]
    return cues_df.merge(manifest_df[keep_cols], on="utt_id", how="left")


def run_llm_for_cues(cfg: dict[str, Any], ablation: Ablation = "all") -> Path:
    paths = cfg.get("paths", {})
    exp = cfg.get("experiment", {})
    output_dir = ensure_dir(paths.get("output_dir", "runs/textpa_qwen"))
    cues_path = output_dir / "cues.jsonl"
    if not cues_path.exists():
        raise FileNotFoundError(f"Missing cues cache: {cues_path}. Run extract_cues first.")
    llm_path = output_dir / f"llm_{ablation}.jsonl"
    results_csv = output_dir / f"results_{ablation}.csv"

    cues_rows = read_jsonl(cues_path)
    existing = _valid_llm_cache_map(read_jsonl(llm_path)) if exp.get("cache_llm", True) else {}
    client = make_qwen_client(cfg)

    all_llm_rows: list[dict[str, Any]] = []
    for cues in tqdm(cues_rows, desc=f"Running Qwen LLM ({ablation})"):
        utt_id = str(cues["utt_id"])
        prompt = build_textpa_prompt(
            cues,
            ablation=ablation,
            basic=bool(exp.get("basic_prompt", True)),
            qwen_strict_json=bool(exp.get("qwen_strict_json", False)),
        )
        cached = existing.get(utt_id)
        if cached:
            llm_row = cached
        else:
            res = run_llm_with_json_retries(
                client,
                prompt,
                max_retries=int(exp.get("max_llm_retries", 5)),
                sleep_s=float(exp.get("sleep_between_llm_calls", 0.0)),
            )
            llm_row = {
                "utt_id": utt_id,
                "ablation": ablation,
                "llm_accuracy": res.accuracy,
                "llm_fluency": res.fluency,
                "llm_reasoning": res.reasoning,
                "raw_response": res.raw_response,
                "valid_json": res.valid_json,
                "attempts": res.attempts,
            }
            append_jsonl(llm_row, llm_path)
            if exp.get("sleep_between_llm_calls"):
                time.sleep(float(exp.get("sleep_between_llm_calls", 0.0)))
        all_llm_rows.append(llm_row)

    atomic_jsonl_write(all_llm_rows, llm_path)
    cues_df = _merge_cues_with_manifest(cues_rows, paths.get("manifest"))
    llm_df = pd.DataFrame(all_llm_rows)
    merged = cues_df.merge(llm_df, on="utt_id", how="left")
    merged.to_csv(results_csv, index=False)
    return results_csv


def evaluate_results(cfg: dict[str, Any], results_csv: str | Path) -> tuple[Path, dict[str, float]]:
    df = pd.read_csv(results_csv, dtype={"utt_id": str, "speaker": str, "split": str})
    sc = cfg.get("scoring", {})
    scored = add_textpa_scores(
        df,
        MatchScoringConfig(
            match=float(sc.get("sw_match", 2.0)),
            mismatch=float(sc.get("sw_mismatch", -1.0)),
            gap=float(sc.get("sw_gap", -1.0)),
        ),
    )
    scored_csv = Path(results_csv).with_name(Path(results_csv).stem + "_scored.csv")
    scored.to_csv(scored_csv, index=False)
    metrics = evaluate_predictions(scored)
    metrics_path = Path(results_csv).with_name(Path(results_csv).stem + "_metrics.json")
    write_json(metrics, metrics_path)
    return metrics_path, metrics


def run_experiment(cfg: dict[str, Any], ablations: list[Ablation] | None = None) -> dict[str, Any]:
    """End-to-end run: cues -> Qwen scoring -> TextPA scoring -> metrics."""
    extract_cues_for_manifest(cfg)
    exp = cfg.get("experiment", {})
    if ablations is None:
        a = exp.get("ablation", "all")
        if a == "paper_ablation":
            ablations = ["transcript", "transcript_ipa", "transcript_cmu", "all"]
        else:
            ablations = [a]
    summary: dict[str, Any] = {}
    for ablation in ablations:
        results = run_llm_for_cues(cfg, ablation=ablation)
        metrics_path, metrics = evaluate_results(cfg, results)
        summary[ablation] = {"results": str(results), "metrics_path": str(metrics_path), "metrics": metrics}
    output_dir = ensure_dir(cfg.get("paths", {}).get("output_dir", "runs/textpa_qwen"))
    write_json(summary, output_dir / "summary.json")
    return summary
