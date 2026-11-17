"""Evaluation utilities for the paper-version system.

Functional module 7/7.

This module is intentionally independent from the runtime pipeline. It computes
standard metrics for ASR, word-level error detection, phoneme-level MDD, and
human feedback quality annotations. Use it in notebooks or scripts for the
paper experiments.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .contracts import AnalysisReport
from .diagnostics import simplify_phone
from .reference_alignment import normalize_word


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


@dataclass
class AsrMetrics:
    wer: float
    substitutions: int
    deletions: int
    insertions: int
    reference_words: int


# ---------- Generic edit distance ----------


def edit_ops(ref: Sequence[str], hyp: Sequence[str]) -> List[Tuple[str, Optional[str], Optional[str]]]:
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[(0, 0, "")] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = (i - 1, 0, "delete")
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = (0, j - 1, "insert")
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            same = ref[i - 1] == hyp[j - 1]
            candidates = [
                (dp[i - 1][j - 1] + (0 if same else 1), i - 1, j - 1, "equal" if same else "substitute"),
                (dp[i - 1][j] + 1, i - 1, j, "delete"),
                (dp[i][j - 1] + 1, i, j - 1, "insert"),
            ]
            best = min(candidates, key=lambda x: x[0])
            dp[i][j] = best[0]
            back[i][j] = (best[1], best[2], best[3])
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        pi, pj, op = back[i][j]
        if op in {"equal", "substitute"}:
            ops.append((op, ref[i - 1], hyp[j - 1]))
        elif op == "delete":
            ops.append((op, ref[i - 1], None))
        else:
            ops.append((op, None, hyp[j - 1]))
        i, j = pi, pj
    ops.reverse()
    return ops


def word_error_rate(reference_text: str, hypothesis_text: str) -> AsrMetrics:
    ref = [normalize_word(x) for x in reference_text.split() if normalize_word(x)]
    hyp = [normalize_word(x) for x in hypothesis_text.split() if normalize_word(x)]
    ops = edit_ops(ref, hyp)
    subs = sum(1 for op, _, _ in ops if op == "substitute")
    dels = sum(1 for op, _, _ in ops if op == "delete")
    ins = sum(1 for op, _, _ in ops if op == "insert")
    denom = max(1, len(ref))
    return AsrMetrics(wer=(subs + dels + ins) / denom, substitutions=subs, deletions=dels, insertions=ins, reference_words=len(ref))


def phone_error_rate(reference_phones: Sequence[str], hypothesis_phones: Sequence[str]) -> float:
    ref = [simplify_phone(x) for x in reference_phones if simplify_phone(x)]
    hyp = [simplify_phone(x) for x in hypothesis_phones if simplify_phone(x)]
    ops = edit_ops(ref, hyp)
    errors = sum(1 for op, _, _ in ops if op != "equal")
    return errors / max(1, len(ref))


# ---------- Detection metrics ----------


def precision_recall_f1(predicted: Iterable[Any], gold: Iterable[Any]) -> PRF:
    p = set(predicted)
    g = set(gold)
    tp = len(p & g)
    fp = len(p - g)
    fn = len(g - p)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return PRF(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn)


def word_issue_keys(report: AnalysisReport) -> List[Tuple[int, str]]:
    keys = []
    for issue in report.evidence.word_issues:
        if issue.ref_index is not None:
            keys.append((issue.ref_index, issue.type.value))
    return keys


def phoneme_issue_keys(report: AnalysisReport) -> List[Tuple[int, str, str]]:
    keys = []
    for issue in report.evidence.phoneme_issues:
        if issue.ref_index is not None:
            keys.append((issue.ref_index, issue.expected or "", issue.observed or ""))
    return keys


def evaluate_against_gold_csv(report_json_paths: Sequence[str | Path], gold_csv: str | Path) -> Dict[str, Any]:
    """Evaluate predicted issues against a simple gold CSV.

    Gold CSV schema:
    - analysis_id
    - level: "word" or "phoneme"
    - ref_index: integer
    - type: for word-level issues, e.g. word_omission
    - expected: for phoneme-level issues
    - observed: for phoneme-level issues
    """
    gold_word: Dict[str, List[Tuple[int, str]]] = {}
    gold_phone: Dict[str, List[Tuple[int, str, str]]] = {}
    with Path(gold_csv).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            aid = row["analysis_id"]
            level = row.get("level", "word")
            idx = int(row["ref_index"])
            if level == "phoneme":
                gold_phone.setdefault(aid, []).append((idx, simplify_phone(row.get("expected", "")), simplify_phone(row.get("observed", ""))))
            else:
                gold_word.setdefault(aid, []).append((idx, row.get("type", "")))

    pred_word_all: List[Tuple[str, int, str]] = []
    gold_word_all: List[Tuple[str, int, str]] = []
    pred_phone_all: List[Tuple[str, int, str, str]] = []
    gold_phone_all: List[Tuple[str, int, str, str]] = []

    for path in report_json_paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        report = AnalysisReport(**data)
        aid = report.analysis_id
        pred_word_all.extend((aid, idx, typ) for idx, typ in word_issue_keys(report))
        pred_phone_all.extend((aid, idx, exp, obs) for idx, exp, obs in phoneme_issue_keys(report))
        gold_word_all.extend((aid, idx, typ) for idx, typ in gold_word.get(aid, []))
        gold_phone_all.extend((aid, idx, exp, obs) for idx, exp, obs in gold_phone.get(aid, []))

    return {
        "word_detection": precision_recall_f1(pred_word_all, gold_word_all).__dict__,
        "phoneme_detection": precision_recall_f1(pred_phone_all, gold_phone_all).__dict__,
        "n_reports": len(report_json_paths),
    }


# ---------- Feedback annotation aggregation ----------


def aggregate_feedback_ratings(csv_path: str | Path) -> Dict[str, float]:
    """Aggregate human feedback ratings.

    CSV schema can contain columns such as correctness, specificity,
    actionability, hallucination_risk, helpfulness with 1--5 ratings.
    """
    cols: Dict[str, List[float]] = {}
    with Path(csv_path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k in {"analysis_id", "issue_id", "annotator"}:
                    continue
                try:
                    cols.setdefault(k, []).append(float(v))
                except Exception:
                    continue
    return {k: round(sum(v) / max(1, len(v)), 3) for k, v in cols.items()}
