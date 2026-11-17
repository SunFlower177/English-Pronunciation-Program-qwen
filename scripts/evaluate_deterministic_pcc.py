"""Re-score CAPT reports and compute PCC without Qwen.

Example:
  .venv/Scripts/python scripts/evaluate_deterministic_pcc.py +    --run-config configs/deterministic_pcc.yaml
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from capt_pronunciation.contracts import (  # noqa: E402
    AlignmentResult,
    CaptSettings,
    EvidenceBundle,
)
from capt_pronunciation.deterministic_scoring import (  # noqa: E402
    load_score_config,
    score_evidence,
)
from capt_pronunciation.runtime import CaptPipeline  # noqa: E402
from evaluate_multipa_pcc import (  # noqa: E402
    compute_pcc,
    load_multipa_gold,
    pearsonr,
    resolve_audio_id,
)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--run-config", default=None)
    known, _ = bootstrap.parse_known_args(argv)
    loaded: Dict[str, Any] = {}
    if known.run_config:
        config_path = Path(known.run_config)
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError("PCC run config must be a YAML mapping")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", default=known.run_config)
    parser.add_argument("--annotation", default=loaded.get("annotation", "MultiPA/annotation.csv"))
    parser.add_argument("--wav-dir", default=loaded.get("wav_dir", "MultiPA/wav"))
    parser.add_argument("--report-dir", default=loaded.get("report_dir"))
    parser.add_argument("--reference-map", default=loaded.get("reference_map"))
    parser.add_argument("--mode", choices=["modeA", "modeB"], default=loaded.get("mode", "modeA"))
    parser.add_argument(
        "--score-config",
        default=loaded.get("score_config", "configs/deterministic_scoring.yaml"),
    )
    parser.add_argument("--output-dir", default=loaded.get("output_dir", "runs/deterministic_pcc"))
    parser.add_argument(
        "--aggregate",
        choices=["median", "mean", "best"],
        default=loaded.get("aggregate", "median"),
        help="Aggregate multiple reference reports for the same learner audio",
    )
    parser.add_argument("--bootstrap", type=int, default=int(loaded.get("bootstrap", 2000)))
    parser.add_argument("--seed", type=int, default=int(loaded.get("seed", 20260713)))
    parser.add_argument(
        "--include-word-proxy",
        action="store_true",
        default=bool(loaded.get("include_word_proxy", False)),
    )
    parser.add_argument(
        "--strict-reference",
        action="store_true",
        default=bool(loaded.get("strict_reference", False)),
        help="Reject ModeB reports without a separate reference recording",
    )
    parser.add_argument(
        "--run-analysis",
        action="store_true",
        default=bool(loaded.get("run_analysis", False)),
        help="Regenerate acoustic evidence before deterministic PCC",
    )
    parser.add_argument(
        "--force-analysis",
        action="store_true",
        default=bool(loaded.get("force_analysis", False)),
    )
    args = parser.parse_args(argv)
    args.runtime = loaded.get("runtime", {})
    args.evaluation_split = loaded.get("evaluation_split", {})
    if not args.report_dir:
        parser.error("--report-dir is required, either directly or in --run-config")
    return args


def iter_reports(report_dir: Path) -> Iterable[Path]:
    direct = sorted(report_dir.glob("*.json"))
    nested = sorted(report_dir.rglob("report.json"))
    seen = set()
    for path in direct + nested:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            yield path


def load_reference_map(
    path: Path,
    alias_to_audio: Mapping[str, str],
) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            identifier = next(
                (
                    str(row.get(key) or "").strip()
                    for key in ("audio", "filename", "file", "audio_id", "id")
                    if str(row.get(key) or "").strip()
                ),
                "",
            )
            audio = resolve_audio_id(identifier, dict(alias_to_audio))
            if not audio:
                continue
            reference_text = str(row.get("reference_text") or row.get("text") or "").strip()
            raw_audio = str(row.get("reference_audio") or row.get("ref_audio") or "").strip()
            reference_audios = [
                resolve_reference_path(value.strip(), path.parent)
                for value in re.split(r"[|;]", raw_audio)
                if value.strip()
            ]
            rows[audio] = {
                "reference_text": reference_text,
                "reference_audios": reference_audios,
            }
    return rows


def resolve_reference_path(value: str, map_dir: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    workspace_path = ROOT / path
    if workspace_path.exists():
        return str(workspace_path)
    return str(map_dir / path)


def regenerate_reports(
    *,
    gold_audio: Iterable[str],
    wav_dir: Path,
    report_dir: Path,
    reference_map_path: Path,
    alias_to_audio: Mapping[str, str],
    mode: str,
    score_config_path: str,
    force: bool,
    strict_reference: bool,
    runtime: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    reference_map = load_reference_map(reference_map_path, alias_to_audio)
    settings = CaptSettings.from_env()
    allowed_runtime = {
        "asr_backend",
        "whisperx_model",
        "device",
        "compute_type",
        "use_whisperx_alignment",
        "use_mfa",
        "use_ctc_alignment",
        "phoneme_model",
        "use_phoneme_model",
        "phoneme_require_equal_word",
        "phoneme_skip_function_words",
        "phoneme_min_word_duration_s",
    }
    for key, value in dict(runtime or {}).items():
        if key not in allowed_runtime:
            raise ValueError(f"Unsupported runtime override: {key}")
        setattr(settings, key, value)
    settings.work_dir = report_dir.resolve()
    settings.work_dir.mkdir(parents=True, exist_ok=True)
    settings.use_qwen_text = False
    settings.use_qwen2_audio = False
    settings.deterministic_score_config = str(Path(score_config_path).resolve())
    pipeline = CaptPipeline(settings)
    generated = 0
    skipped = 0
    failures: list[str] = []

    audio_items = list(gold_audio)
    for position, audio in enumerate(audio_items, start=1):
        learner_path = wav_dir / audio
        if not learner_path.exists():
            failures.append(f"{audio}: learner audio not found at {learner_path}")
            continue
        reference = reference_map.get(audio, {})
        reference_text = str(reference.get("reference_text") or "").strip()
        if mode == "modeA":
            if not reference_text:
                failures.append(f"{audio}: reference_text is missing")
                continue
            analysis_id = stable_analysis_id(mode, audio)
            report_path = report_dir / analysis_id / "report.json"
            if report_path.exists() and not force:
                print(f"[analysis {position}/{len(audio_items)}] existing report: {audio}", flush=True)
                skipped += 1
                continue
            try:
                print(f"[analysis {position}/{len(audio_items)}] ModeA: {audio}", flush=True)
                pipeline.analyze_reading(reference_text, learner_path, analysis_id=analysis_id)
                generated += 1
            except Exception as exc:
                failures.append(f"{audio}: {exc}")
            continue

        reference_audios = list(reference.get("reference_audios") or [])
        if not reference_audios:
            failures.append(f"{audio}: reference_audio is missing")
            continue
        for reference_index, reference_audio in enumerate(reference_audios, start=1):
            reference_path = Path(reference_audio)
            if not reference_path.exists():
                failures.append(f"{audio}: reference audio not found at {reference_path}")
                continue
            if strict_reference and reference_path.resolve() == learner_path.resolve():
                failures.append(f"{audio}: learner audio cannot be its own strict reference")
                continue
            analysis_id = stable_analysis_id(mode, audio, reference_index)
            report_path = report_dir / analysis_id / "report.json"
            if report_path.exists() and not force:
                print(
                    f"[analysis {position}/{len(audio_items)} ref {reference_index}] existing report: {audio}",
                    flush=True,
                )
                skipped += 1
                continue
            try:
                print(
                    f"[analysis {position}/{len(audio_items)} ref {reference_index}] ModeB: {audio}",
                    flush=True,
                )
                pipeline.analyze_imitation(
                    reference_path,
                    learner_path,
                    reference_text=reference_text or None,
                    analysis_id=analysis_id,
                )
                generated += 1
            except Exception as exc:
                failures.append(f"{audio}/reference_{reference_index}: {exc}")
    return {
        "generated_reports": generated,
        "skipped_existing_reports": skipped,
        "analysis_failure_count": len(failures),
        "analysis_failure_examples": failures[:20],
        "analysis_runtime": {
            "asr_backend": settings.asr_backend,
            "whisper_model": settings.whisperx_model,
            "device": settings.device,
            "compute_type": settings.compute_type,
            "phoneme_model": settings.phoneme_model,
            "phoneme_model_enabled": settings.use_phoneme_model,
            "mfa_enabled": settings.use_mfa,
            "ctc_forced_alignment_enabled": settings.use_ctc_alignment,
            "qwen_text_enabled": settings.use_qwen_text,
            "qwen_audio_enabled": settings.use_qwen2_audio,
        },
    }


def stable_analysis_id(mode: str, audio: str, reference_index: int = 0) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(audio).stem)[:48]
    digest = hashlib.sha1(f"{mode}|{audio}|{reference_index}".encode("utf-8")).hexdigest()[:10]
    suffix = f"_ref{reference_index}" if reference_index else ""
    return f"det_{mode}_{stem}_{digest}{suffix}"


def load_and_score_reports(
    report_dir: Path,
    alias_to_audio: Mapping[str, str],
    *,
    mode: str,
    score_config: Mapping[str, Any],
    aggregate: str,
    strict_reference: bool,
) -> tuple[Dict[str, Dict[str, float]], list[Dict[str, Any]], Dict[str, Any]]:
    grouped: Dict[str, list[Dict[str, Any]]] = {}
    unmatched: list[str] = []
    rejected: list[str] = []
    report_count = 0
    for path in iter_reports(report_dir):
        report_count += 1
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            report_mode = str(data.get("mode", ""))
            if mode == "modeA" and report_mode not in {"reading", ""}:
                continue
            if mode == "modeB" and report_mode != "imitation":
                continue
            if strict_reference and mode == "modeB" and not data.get("reference_audio"):
                rejected.append(str(path))
                continue
            candidates = [
                data.get("analysis_id"),
                nested_get(data, "audio.original_path"),
                nested_get(data, "audio.path"),
            ]
            audio = next(
                (
                    resolved
                    for candidate in candidates
                    if (resolved := resolve_audio_id(candidate, dict(alias_to_audio)))
                ),
                None,
            )
            if not audio:
                unmatched.append(str(path))
                continue
            evidence = EvidenceBundle(**(data.get("evidence") or {}))
            alignment = AlignmentResult(**(data.get("alignment") or {}))
            result = score_evidence(
                evidence,
                alignment,
                mode=mode,
                config=score_config,
            )
            grouped.setdefault(audio, []).append(
                {
                    "path": str(path),
                    "analysis_id": data.get("analysis_id"),
                    "scores": result.scores,
                    "coverage": result.coverage,
                    "raw_features": result.raw_features,
                    "normalized_features": result.normalized_features,
                }
            )
        except Exception as exc:
            rejected.append(f"{path}: {exc}")

    predictions: Dict[str, Dict[str, float]] = {}
    rows: list[Dict[str, Any]] = []
    for audio, items in sorted(grouped.items()):
        scores = {
            metric: aggregate_values(
                [float(item["scores"][metric]) for item in items],
                aggregate,
            )
            for metric in ("accuracy", "phoneme", "fluency", "prosody", "overall")
        }
        coverage = {
            metric: aggregate_values(
                [float(item["coverage"].get(metric, 0.0)) for item in items],
                "median",
            )
            for metric in ("accuracy", "phoneme", "fluency", "prosody")
        }
        predictions[audio] = {key: value for key, value in scores.items() if key != "overall"}
        rows.append(
            {
                "audio": audio,
                **scores,
                "accuracy_coverage": coverage["accuracy"],
                "phoneme_coverage": coverage["phoneme"],
                "fluency_coverage": coverage["fluency"],
                "prosody_coverage": coverage["prosody"],
                "reference_count": len(items),
                "analysis_ids": "|".join(str(item["analysis_id"] or "") for item in items),
            }
        )
    metadata = {
        "report_count": report_count,
        "matched_audio": len(predictions),
        "unmatched_count": len(unmatched),
        "rejected_count": len(rejected),
        "unmatched_examples": unmatched[:20],
        "rejected_examples": rejected[:20],
    }
    return predictions, rows, metadata


def add_distribution_metrics(
    metrics: Dict[str, Dict[str, Any]],
    gold: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, float]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> None:
    mapping = {
        "accuracy": ("accuracy", "human_accuracy"),
        "fluency": ("fluency", "human_fluency"),
        "prosody": ("prosody", "human_prosody"),
    }
    rng = np.random.default_rng(seed)
    for metric, (prediction_key, gold_key) in mapping.items():
        pairs = [
            (float(prediction[prediction_key]), float(gold[audio][gold_key]))
            for audio, prediction in predictions.items()
            if audio in gold
            and prediction_key in prediction
            and gold[audio].get(gold_key) is not None
        ]
        if not pairs or metric not in metrics:
            continue
        pred = np.asarray([pair[0] for pair in pairs], dtype=float)
        target = np.asarray([pair[1] for pair in pairs], dtype=float)
        pred_for_error = pred
        error_scale = "native"
        if float(np.nanmedian(pred)) > 10.0 and float(np.nanmax(target)) <= 5.5:
            pred_for_error = 1.0 + 4.0 * np.clip(pred, 0.0, 100.0) / 100.0
            error_scale = "human_1_to_5"
        metrics[metric]["mae"] = round(float(np.mean(np.abs(pred_for_error - target))), 6)
        metrics[metric]["rmse"] = round(
            float(np.sqrt(np.mean((pred_for_error - target) ** 2))),
            6,
        )
        metrics[metric]["error_scale"] = error_scale
        metrics[metric]["spearman"] = round(spearman(pred, target), 6)
        interval = bootstrap_pcc(pred, target, bootstrap_samples, rng)
        metrics[metric]["pcc_95ci"] = interval


def bootstrap_pcc(
    prediction: np.ndarray,
    target: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> Optional[list[float]]:
    if len(prediction) < 4 or samples <= 0:
        return None
    values = []
    for _ in range(samples):
        indices = rng.integers(0, len(prediction), len(prediction))
        value = pearsonr(prediction[indices].tolist(), target[indices].tolist())
        if value is not None and math.isfinite(value):
            values.append(float(value))
    if not values:
        return None
    low, high = np.percentile(values, [2.5, 97.5])
    return [round(float(low), 6), round(float(high), 6)]


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = rankdata(left)
    right_rank = rankdata(right)
    value = pearsonr(left_rank.tolist(), right_rank.tolist())
    return float(value) if value is not None else float("nan")


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + end - 1) / 2.0 + 1.0
        index = end
    return ranks


def aggregate_values(values: list[float], method: str) -> float:
    if not values:
        return float("nan")
    if method == "mean":
        return round(float(np.mean(values)), 6)
    if method == "best":
        return round(float(np.max(values)), 6)
    return round(float(np.median(values)), 6)


def nested_get(data: Mapping[str, Any], path: str, default: Any = None) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def write_predictions(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "audio", "evaluation_split", "accuracy", "phoneme", "fluency", "prosody", "overall",
        "accuracy_coverage", "phoneme_coverage", "fluency_coverage",
        "prosody_coverage", "reference_count", "analysis_ids",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def deterministic_group_split(
    audio_names: Iterable[str],
    config: Mapping[str, Any],
) -> Dict[str, str]:
    """Create a stable group-disjoint dev/test split from filename speaker proxies."""
    separator = str(config.get("group_separator", "---"))
    dev_fraction = min(0.95, max(0.05, float(config.get("dev_fraction", 0.70))))
    seed = str(config.get("seed", 20260716))
    out: Dict[str, str] = {}
    group_split: Dict[str, str] = {}
    for audio in sorted(audio_names):
        group = str(audio).split(separator, 1)[0]
        if group not in group_split:
            digest = hashlib.sha1(f"{seed}|{group}".encode("utf-8")).digest()
            ratio = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
            group_split[group] = "dev" if ratio < dev_fraction else "test"
        out[str(audio)] = group_split[group]
    return out


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    gold, alias_to_audio = load_multipa_gold(Path(args.annotation))
    score_config = load_score_config(args.score_config)
    analysis_meta: Dict[str, Any] = {}
    if args.run_analysis:
        if not args.reference_map:
            raise ValueError("--reference-map is required when --run-analysis is enabled")
        analysis_meta = regenerate_reports(
            gold_audio=gold.keys(),
            wav_dir=Path(args.wav_dir),
            report_dir=Path(args.report_dir),
            reference_map_path=Path(args.reference_map),
            alias_to_audio=alias_to_audio,
            mode=args.mode,
            score_config_path=args.score_config,
            force=args.force_analysis,
            strict_reference=args.strict_reference,
            runtime=args.runtime,
        )
    predictions, rows, report_meta = load_and_score_reports(
        Path(args.report_dir),
        alias_to_audio,
        mode=args.mode,
        score_config=score_config,
        aggregate=args.aggregate,
        strict_reference=args.strict_reference,
    )
    metrics = compute_pcc(gold, predictions, args.include_word_proxy)
    add_distribution_metrics(
        metrics,
        gold,
        predictions,
        bootstrap_samples=args.bootstrap,
        seed=args.seed,
    )
    split_metrics: Dict[str, Any] = {}
    split_counts: Dict[str, int] = {}
    if isinstance(args.evaluation_split, Mapping) and args.evaluation_split.get("enabled"):
        split_by_audio = deterministic_group_split(gold.keys(), args.evaluation_split)
        for row in rows:
            row["evaluation_split"] = split_by_audio.get(str(row["audio"]), "")
        for split_name in ("dev", "test"):
            split_gold = {
                audio: item
                for audio, item in gold.items()
                if split_by_audio.get(audio) == split_name
            }
            split_predictions = {
                audio: item
                for audio, item in predictions.items()
                if split_by_audio.get(audio) == split_name
            }
            current = compute_pcc(split_gold, split_predictions, args.include_word_proxy)
            add_distribution_metrics(
                current,
                split_gold,
                split_predictions,
                bootstrap_samples=args.bootstrap,
                seed=args.seed + (0 if split_name == "dev" else 1),
            )
            split_metrics[split_name] = current
            split_counts[split_name] = len(split_predictions)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"deterministic_predictions_{args.mode}.csv"
    write_predictions(predictions_path, rows)
    result = {
        "metadata": {
            "mode": args.mode,
            "score_source": "deterministic_continuous_evidence",
            "qwen_used": False,
            "score_config": str(Path(args.score_config)),
            "score_config_version": score_config.get("version"),
            "annotation": args.annotation,
            "report_dir": args.report_dir,
            "multi_reference_aggregate": args.aggregate,
            "strict_reference": bool(args.strict_reference),
            "bootstrap_samples": args.bootstrap,
            "analysis_regenerated": bool(args.run_analysis),
            "evaluation_split": args.evaluation_split,
            "evaluation_split_counts": split_counts,
            **analysis_meta,
            **report_meta,
        },
        "metrics": metrics,
        "metrics_by_split": split_metrics,
        "predictions_csv": str(predictions_path),
    }
    result_path = output_dir / f"deterministic_pcc_{args.mode}.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
