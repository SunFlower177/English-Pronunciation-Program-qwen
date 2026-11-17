#!/usr/bin/env python
"""Compute PCC between project predictions and MultiPA human ratings.

The MultiPA annotations contain sentence-level human ratings for accuracy,
fluency, and prosody. They also contain word-level labels, but not phoneme-level
gold labels. This script therefore evaluates phoneme only when explicitly asked
to use the word-label aggregate as an exploratory proxy.

Examples:

  python scripts/evaluate_multipa_pcc.py --annotation MultiPA/annotation.csv

  python scripts/evaluate_multipa_pcc.py \
    --annotation MultiPA/annotation.csv \
    --predictions runs/multipa_predictions.csv \
    --id-col audio \
    --accuracy-col accuracy --fluency-col fluency --prosody-col prosody

  python scripts/evaluate_multipa_pcc.py \
    --annotation MultiPA/annotation.csv \
    --report-dir runs \
    --score-source overall
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SENTENCE_GOLD = {
    "accuracy": "human_accuracy",
    "fluency": "human_fluency",
    "prosody": "human_prosody",
}

DEFAULT_ID_COLUMNS = [
    "audio",
    "audio_path",
    "utt_id",
    "analysis_id",
    "id",
    "file",
    "filename",
    "path",
]

DEFAULT_PREDICTION_COLUMNS = {
    "accuracy": [
        "pred_accuracy",
        "accuracy",
        "overall_scores.accuracy",
        "feedback.overall_scores.accuracy",
        "qwen_rubric_scores.accuracy",
        "feedback.qwen_rubric_scores.accuracy",
        "textpa_accuracy",
        "llm_accuracy",
    ],
    "fluency": [
        "pred_fluency",
        "fluency",
        "overall_scores.fluency",
        "feedback.overall_scores.fluency",
        "qwen_rubric_scores.fluency",
        "feedback.qwen_rubric_scores.fluency",
        "textpa_fluency",
        "llm_fluency",
    ],
    "prosody": [
        "pred_prosody",
        "prosody",
        "overall_scores.prosody",
        "feedback.overall_scores.prosody",
        "qwen_rubric_scores.prosody",
        "feedback.qwen_rubric_scores.prosody",
        "textpa_prosody",
        "llm_prosody",
    ],
    "phoneme": [
        "pred_phoneme",
        "phoneme",
        "overall_scores.phoneme",
        "feedback.overall_scores.phoneme",
        "qwen_rubric_scores.phoneme",
        "feedback.qwen_rubric_scores.phoneme",
        "textpa_phoneme",
        "llm_phoneme",
    ],
}


def leading_score(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    text = str(value).strip()
    if not text:
        return None
    match = re.match(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))", text)
    if not match:
        return None
    out = float(match.group(1))
    return out if math.isfinite(out) else None


def mean(values: Iterable[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def audio_aliases(value: Any) -> list[str]:
    text = str(value or "").strip().strip('"').strip("'")
    if not text:
        return []
    base = text.replace("\\", "/").rsplit("/", 1)[-1]
    aliases = [base]
    suffix = Path(base).suffix
    if suffix:
        aliases.append(Path(base).stem)
    return list(dict.fromkeys(a for a in aliases if a))


def parse_word_label_score(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        spans = json.loads(str(value))
    except json.JSONDecodeError:
        return None
    scores: list[float] = []
    if not isinstance(spans, list):
        return None
    for span in spans:
        if not isinstance(span, dict):
            continue
        labels = span.get("labels") or []
        if isinstance(labels, str):
            labels = [labels]
        for label in labels:
            score = leading_score(label)
            if score is not None:
                scores.append(score)
                break
    return mean(scores)


def load_multipa_gold(annotation_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    with annotation_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {annotation_path}")
    required = {"audio", "accuracy", "fluency", "prosody", "word-label"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"Missing MultiPA annotation columns: {sorted(missing)}")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["audio"])].append(row)

    gold: dict[str, dict[str, Any]] = {}
    alias_to_audio: dict[str, str] = {}
    for audio, group in grouped.items():
        item = {
            "audio": audio,
            "human_accuracy": mean(leading_score(row.get("accuracy")) for row in group),
            "human_fluency": mean(leading_score(row.get("fluency")) for row in group),
            "human_prosody": mean(leading_score(row.get("prosody")) for row in group),
            "human_word_label": mean(parse_word_label_score(row.get("word-label")) for row in group),
            "annotator_count": len(group),
        }
        gold[audio] = item
        for alias in audio_aliases(audio):
            alias_to_audio[alias] = audio
    return gold, alias_to_audio


def resolve_audio_id(value: Any, alias_to_audio: dict[str, str]) -> str | None:
    for alias in audio_aliases(value):
        if alias in alias_to_audio:
            return alias_to_audio[alias]
    return None


def choose_column(fieldnames: list[str], explicit: str | None, candidates: list[str]) -> str | None:
    if explicit:
        if explicit not in fieldnames:
            raise ValueError(f"Column '{explicit}' not found. Available columns: {fieldnames}")
        return explicit
    lower_to_name = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        found = lower_to_name.get(candidate.lower())
        if found:
            return found
    return None


def load_prediction_csv(
    path: Path,
    alias_to_audio: dict[str, str],
    id_col: str | None,
    metric_cols: dict[str, str | None],
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"No CSV header found in {path}")
        resolved_id_col = choose_column(fieldnames, id_col, DEFAULT_ID_COLUMNS)
        if not resolved_id_col:
            raise ValueError(
                "Cannot infer an id column. Pass --id-col with one of: "
                + ", ".join(fieldnames)
            )
        resolved_metric_cols = {
            metric: choose_column(fieldnames, metric_cols.get(metric), DEFAULT_PREDICTION_COLUMNS[metric])
            for metric in DEFAULT_PREDICTION_COLUMNS
        }
        predictions: dict[str, dict[str, float]] = {}
        unmatched: list[str] = []
        for row in reader:
            audio = resolve_audio_id(row.get(resolved_id_col), alias_to_audio)
            if not audio:
                unmatched.append(str(row.get(resolved_id_col, "")))
                continue
            item = predictions.setdefault(audio, {})
            for metric, col in resolved_metric_cols.items():
                if not col:
                    continue
                score = leading_score(row.get(col))
                if score is not None:
                    item[metric] = score
    meta = {
        "source": str(path),
        "id_col": resolved_id_col,
        "metric_cols": resolved_metric_cols,
        "unmatched_count": len(unmatched),
        "unmatched_examples": unmatched[:20],
    }
    return predictions, meta


def nested_get(obj: dict[str, Any], path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def load_report_predictions(
    report_dir: Path,
    alias_to_audio: dict[str, str],
    score_source: str,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    report_paths = sorted(report_dir.rglob("report.json"))
    predictions: dict[str, dict[str, float]] = {}
    unmatched: list[str] = []
    for report_path in report_paths:
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            unmatched.append(f"{report_path}: {exc}")
            continue
        candidates = [
            data.get("analysis_id"),
            nested_get(data, "audio.original_path"),
            nested_get(data, "audio.path"),
        ]
        audio = None
        for candidate in candidates:
            audio = resolve_audio_id(candidate, alias_to_audio)
            if audio:
                break
        if not audio:
            unmatched.append(str(report_path))
            continue

        if score_source == "qwen":
            scores = nested_get(data, "feedback.qwen_rubric_scores") or {}
        else:
            scores = nested_get(data, "feedback.overall_scores") or {}
        item = predictions.setdefault(audio, {})
        for metric in ("accuracy", "phoneme", "fluency", "prosody"):
            score = leading_score(scores.get(metric) if isinstance(scores, dict) else None)
            if score is not None:
                item[metric] = score

    meta = {
        "source": str(report_dir),
        "score_source": score_source,
        "report_count": len(report_paths),
        "unmatched_count": len(unmatched),
        "unmatched_examples": unmatched[:20],
    }
    return predictions, meta


def pearsonr(xs: list[float], ys: list[float]) -> float | None:
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 2:
        return None
    x_vals = [p[0] for p in pairs]
    y_vals = [p[1] for p in pairs]
    x_mean = sum(x_vals) / len(x_vals)
    y_mean = sum(y_vals) / len(y_vals)
    num = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    den_x = math.sqrt(sum((x - x_mean) ** 2 for x in x_vals))
    den_y = math.sqrt(sum((y - y_mean) ** 2 for y in y_vals))
    if den_x == 0.0 or den_y == 0.0:
        return None
    return num / (den_x * den_y)


def compute_pcc(
    gold: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, float]],
    include_word_proxy: bool,
) -> dict[str, dict[str, Any]]:
    targets: list[tuple[str, str, str]] = [
        ("accuracy", "accuracy", "human_accuracy"),
        ("fluency", "fluency", "human_fluency"),
        ("prosody", "prosody", "human_prosody"),
    ]
    if include_word_proxy:
        targets.append(("phoneme_word_label_proxy", "phoneme", "human_word_label"))

    metrics: dict[str, dict[str, Any]] = {}
    for name, pred_key, gold_key in targets:
        pred_values: list[float] = []
        gold_values: list[float] = []
        for audio, pred in predictions.items():
            if audio not in gold or pred_key not in pred:
                continue
            g = gold[audio].get(gold_key)
            p = pred.get(pred_key)
            if g is None or p is None:
                continue
            pred_values.append(float(p))
            gold_values.append(float(g))
        pcc = pearsonr(pred_values, gold_values)
        metrics[name] = {
            "pcc": None if pcc is None else round(pcc, 6),
            "n": len(pred_values),
            "prediction": pred_key,
            "gold": gold_key,
        }
    return metrics


def write_gold_csv(path: Path, gold: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "audio",
        "utt_id",
        "human_accuracy",
        "human_fluency",
        "human_prosody",
        "human_word_label",
        "annotator_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for audio, item in sorted(gold.items()):
            row = {key: item.get(key) for key in fieldnames}
            row["utt_id"] = Path(audio).stem
            writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", default="MultiPA/annotation.csv", help="Path to MultiPA annotation.csv")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--predictions", help="Prediction CSV to evaluate")
    source.add_argument("--report-dir", help="Directory containing CAPT report.json files")
    parser.add_argument("--score-source", choices=["overall", "qwen"], default="overall")
    parser.add_argument("--id-col", default=None, help="Prediction CSV id/audio column")
    parser.add_argument("--accuracy-col", default=None, help="Prediction CSV accuracy column")
    parser.add_argument("--fluency-col", default=None, help="Prediction CSV fluency column")
    parser.add_argument("--prosody-col", default=None, help="Prediction CSV prosody column")
    parser.add_argument("--phoneme-col", default=None, help="Prediction CSV phoneme column")
    parser.add_argument(
        "--include-word-proxy",
        action="store_true",
        help="Also correlate predicted phoneme with the MultiPA word-label aggregate. This is not phoneme gold.",
    )
    parser.add_argument("--write-gold", default=None, help="Optional path to write aggregated MultiPA gold CSV")
    parser.add_argument("--output-json", default=None, help="Optional path to write the JSON result")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    annotation_path = Path(args.annotation)
    gold, alias_to_audio = load_multipa_gold(annotation_path)

    if args.write_gold:
        write_gold_csv(Path(args.write_gold), gold)

    result: dict[str, Any] = {
        "gold": {
            "annotation": str(annotation_path),
            "n_audio": len(gold),
            "supported_sentence_targets": list(SENTENCE_GOLD),
            "unsupported_strict_targets": ["phoneme"],
            "word_label_proxy_available": True,
            "note": (
                "MultiPA word-label is word-level recognizability/stress, "
                "not phoneme-level expected/observed gold."
            ),
        },
        "prediction_source": None,
        "metrics": {},
    }

    predictions: dict[str, dict[str, float]] = {}
    source_meta: dict[str, Any] | None = None
    if args.predictions:
        metric_cols = {
            "accuracy": args.accuracy_col,
            "fluency": args.fluency_col,
            "prosody": args.prosody_col,
            "phoneme": args.phoneme_col,
        }
        predictions, source_meta = load_prediction_csv(
            Path(args.predictions),
            alias_to_audio,
            args.id_col,
            metric_cols,
        )
    elif args.report_dir:
        predictions, source_meta = load_report_predictions(
            Path(args.report_dir),
            alias_to_audio,
            args.score_source,
        )

    if source_meta is not None:
        result["prediction_source"] = source_meta
        result["matched_predictions"] = len(predictions)
        result["metrics"] = compute_pcc(gold, predictions, args.include_word_proxy)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
