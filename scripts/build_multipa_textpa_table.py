#!/usr/bin/env python
"""Build a paper-style MultiPA PCC table from textpa-run-experiment output.

Expected input is the summary JSON written by:

  textpa-run-experiment --config configs/multipa.yaml --ablations transcript transcript_ipa transcript_cmu all

The script prints a Markdown table by default and can also write CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def fmt(value: Any) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "-"
    if num != num:
        return "-"
    return f"{num:.3f}"


def metric(summary: dict[str, Any], ablation: str, name: str) -> Any:
    return summary.get(ablation, {}).get("metrics", {}).get(name)


def build_rows(summary: dict[str, Any], model_label: str) -> list[dict[str, str]]:
    rows = [
        {
            "Method": f"TextPA ({model_label})",
            "Accuracy": fmt(metric(summary, "all", "pcc_textpa_accuracy")),
            "Fluency": fmt(metric(summary, "all", "pcc_textpa_fluency")),
        },
        {
            "Method": "LLM: all",
            "Accuracy": fmt(metric(summary, "all", "pcc_llm_accuracy")),
            "Fluency": fmt(metric(summary, "all", "pcc_llm_fluency")),
        },
        {
            "Method": "LLM: trans.+cmu",
            "Accuracy": fmt(metric(summary, "transcript_cmu", "pcc_llm_accuracy")),
            "Fluency": fmt(metric(summary, "transcript_cmu", "pcc_llm_fluency")),
        },
        {
            "Method": "LLM: trans.+ipa",
            "Accuracy": fmt(metric(summary, "transcript_ipa", "pcc_llm_accuracy")),
            "Fluency": fmt(metric(summary, "transcript_ipa", "pcc_llm_fluency")),
        },
        {
            "Method": "LLM: transcript",
            "Accuracy": fmt(metric(summary, "transcript", "pcc_llm_accuracy")),
            "Fluency": fmt(metric(summary, "transcript", "pcc_llm_fluency")),
        },
        {
            "Method": "IPA match scoring",
            "Accuracy": fmt(metric(summary, "all", "pcc_ipa_match_accuracy")),
            "Fluency": "-",
        },
        {
            "Method": "CMU match scoring",
            "Accuracy": fmt(metric(summary, "all", "pcc_cmu_match_accuracy")),
            "Fluency": "-",
        },
    ]
    return rows


def print_markdown(rows: list[dict[str, str]]) -> None:
    print("| MultiPA data | Accuracy | Fluency |")
    print("|---|---:|---:|")
    for row in rows:
        print(f"| {row['Method']} | {row['Accuracy']} | {row['Fluency']} |")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Method", "Accuracy", "Fluency"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        default="runs/textpa_qwen_multipa_qwen36plus/summary.json",
        help="Path to summary.json written by textpa-run-experiment",
    )
    parser.add_argument("--model-label", default="qwen", help="Label shown in the TextPA row")
    parser.add_argument("--output-csv", default=None, help="Optional CSV output path")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = build_rows(summary, args.model_label)
    print_markdown(rows)
    if args.output_csv:
        write_csv(Path(args.output_csv), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
