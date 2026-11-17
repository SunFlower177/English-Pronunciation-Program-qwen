from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .utils import pearsonr_safe


@dataclass
class MatchScoringConfig:
    match: float = 2.0
    mismatch: float = -1.0
    gap: float = -1.0


def phone_tokens(seq: str, *, remove_pauses: bool = True) -> list[str]:
    if seq is None:
        return []
    text = str(seq)
    if remove_pauses:
        text = re.sub(r"\([0-9.]+s\s+pause\)", " ", text)
    text = text.replace("|", " ").replace("_", " ")
    text = text.replace("ː", ":")
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    return [tok for tok in text.split(" ") if tok]


def smith_waterman_score(
    a: Sequence[str],
    b: Sequence[str],
    config: MatchScoringConfig = MatchScoringConfig(),
) -> float:
    """Local-alignment score for two phone sequences.

    The TextPA paper states that Smith-Waterman is used, but does not publish the exact
    match/mismatch/gap weights. This implementation makes them configurable and defaults
    to a standard simple scheme.
    """
    if not a or not b:
        return 0.0
    h = np.zeros((len(a) + 1, len(b) + 1), dtype=np.float32)
    best = 0.0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            diag = h[i - 1, j - 1] + (config.match if a[i - 1] == b[j - 1] else config.mismatch)
            delete = h[i - 1, j] + config.gap
            insert = h[i, j - 1] + config.gap
            h[i, j] = max(0.0, diag, delete, insert)
            if h[i, j] > best:
                best = float(h[i, j])
    return best


def normalized_sw_similarity(
    recognized: str,
    canonical: str,
    config: MatchScoringConfig = MatchScoringConfig(),
) -> float:
    rec = phone_tokens(recognized)
    can = phone_tokens(canonical)
    if not rec or not can:
        return float("nan")
    score = smith_waterman_score(rec, can, config=config)
    max_possible = min(len(rec), len(can)) * config.match
    if max_possible <= 0:
        return float("nan")
    return float(score / max_possible)


def fixed_scale_score_norm(values: pd.Series, *, min_score: float = 1.0, max_score: float = 5.0) -> pd.Series:
    """Map absolute LLM rubric scores to [0, 1] without using batch min/max."""
    vals = pd.to_numeric(values, errors="coerce")
    norm = (vals - min_score) / (max_score - min_score)
    return norm.clip(0.0, 1.0)


def similarity_norm(values: pd.Series) -> pd.Series:
    """Keep phoneme similarity on the absolute [0, 1] scale."""
    return pd.to_numeric(values, errors="coerce").clip(0.0, 1.0)


def add_textpa_scores(df: pd.DataFrame, config: MatchScoringConfig = MatchScoringConfig()) -> pd.DataFrame:
    """Add phoneme match scores, fixed-scale normalized scores, and final TextPA predictions."""
    out = df.copy()
    out["ipa_match"] = [
        normalized_sw_similarity(r, c, config=config)
        for r, c in zip(out.get("phonemes_ipa", []), out.get("canonical_ipa_from_transcript", []))
    ]
    if "phonemes_cmu" in out and "canonical_cmu_from_transcript" in out:
        out["cmu_match"] = [
            normalized_sw_similarity(r, c, config=config)
            for r, c in zip(out["phonemes_cmu"], out["canonical_cmu_from_transcript"])
        ]
    else:
        out["cmu_match"] = np.nan

    out["llm_accuracy_norm"] = fixed_scale_score_norm(out["llm_accuracy"])
    out["llm_fluency_norm"] = fixed_scale_score_norm(out["llm_fluency"])
    out["ipa_match_norm"] = similarity_norm(out["ipa_match"])
    out["cmu_match_norm"] = similarity_norm(out["cmu_match"])
    out["textpa_accuracy"] = np.nanmean(
        np.vstack([out["llm_accuracy_norm"].to_numpy(float), out["ipa_match_norm"].to_numpy(float)]),
        axis=0,
    )
    # Paper: IPA match scoring only refines accuracy; fluency stays LLM-based.
    out["textpa_fluency"] = out["llm_fluency_norm"]
    return out


def evaluate_predictions(df: pd.DataFrame) -> dict[str, float]:
    """Compute Pearson correlations for LLM-only, match-only, and TextPA predictions."""
    metrics: dict[str, float] = {}
    if "human_accuracy" in df:
        metrics["pcc_llm_accuracy"] = pearsonr_safe(df["llm_accuracy"], df["human_accuracy"])
        metrics["pcc_textpa_accuracy"] = pearsonr_safe(df["textpa_accuracy"], df["human_accuracy"])
        metrics["pcc_ipa_match_accuracy"] = pearsonr_safe(df["ipa_match"], df["human_accuracy"])
        metrics["pcc_cmu_match_accuracy"] = pearsonr_safe(df["cmu_match"], df["human_accuracy"])
    if "human_fluency" in df:
        metrics["pcc_llm_fluency"] = pearsonr_safe(df["llm_fluency"], df["human_fluency"])
        metrics["pcc_textpa_fluency"] = pearsonr_safe(df["textpa_fluency"], df["human_fluency"])
    metrics["n"] = float(len(df))
    metrics["valid_llm_ratio"] = float(np.mean(np.isfinite(df["llm_accuracy"].to_numpy(float)))) if len(df) else math.nan
    return metrics


def aggregate_ablation_table(results_by_ablation: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in results_by_ablation.items():
        scored = add_textpa_scores(df)
        m = evaluate_predictions(scored)
        rows.append(
            {
                "ablation": name,
                "Accuracy_PCC_TextPA": m.get("pcc_textpa_accuracy"),
                "Accuracy_PCC_LLM": m.get("pcc_llm_accuracy"),
                "Fluency_PCC": m.get("pcc_textpa_fluency"),
                "IPA_match_PCC": m.get("pcc_ipa_match_accuracy"),
                "CMU_match_PCC": m.get("pcc_cmu_match_accuracy"),
                "n": m.get("n"),
            }
        )
    return pd.DataFrame(rows)
