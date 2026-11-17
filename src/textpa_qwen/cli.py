from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import prepare_multipa, prepare_speechocean762
from .pipeline import evaluate_results, extract_cues_for_manifest, run_experiment, run_llm_for_cues
from .prompts import Ablation
from .utils import load_config


def _print_json(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")


def prepare_speechocean_main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Speechocean762 test manifest and wav files")
    parser.add_argument("--output-dir", default="data/speechocean762")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dataset-name", default="mispeech/speechocean762")
    parser.add_argument("--overwrite-audio", action="store_true")
    parser.add_argument(
        "--local-root",
        default=None,
        help="Path to an already-downloaded Speechocean762 directory. If set, no HF download is used.",
    )
    args = parser.parse_args()
    path = prepare_speechocean762(
        args.output_dir,
        split=args.split,
        limit=args.limit,
        dataset_name=args.dataset_name,
        overwrite_audio=args.overwrite_audio,
        local_root=args.local_root,
    )
    print(path)


def prepare_multipa_main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MultiPA manifest from local files")
    parser.add_argument("--local-root", default="MultiPA", help="Path to the local MultiPA directory")
    parser.add_argument("--output-dir", default="data/multipa")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    path = prepare_multipa(
        local_root=args.local_root,
        output_dir=args.output_dir,
        split=args.split,
        limit=args.limit,
    )
    print(path)


def extract_cues_main() -> None:
    parser = argparse.ArgumentParser(description="Extract TextPA textual acoustic cues")
    add_config_arg(parser)
    args = parser.parse_args()
    cfg = load_config(args.config)
    path = extract_cues_for_manifest(cfg)
    print(path)


def run_llm_main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen LLM scoring over extracted TextPA cues")
    add_config_arg(parser)
    parser.add_argument(
        "--ablation",
        choices=["transcript", "transcript_ipa", "transcript_cmu", "all"],
        default=None,
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    ablation: Ablation = args.ablation or cfg.get("experiment", {}).get("ablation", "all")
    path = run_llm_for_cues(cfg, ablation=ablation)
    print(path)


def evaluate_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TextPA-Qwen predictions")
    add_config_arg(parser)
    parser.add_argument("--results", required=True, help="CSV from textpa-run-llm")
    args = parser.parse_args()
    cfg = load_config(args.config)
    metrics_path, metrics = evaluate_results(cfg, Path(args.results))
    _print_json({"metrics_path": str(metrics_path), "metrics": metrics})


def run_experiment_main() -> None:
    parser = argparse.ArgumentParser(description="Run complete TextPA-Qwen reproduction experiment")
    add_config_arg(parser)
    parser.add_argument(
        "--ablations",
        nargs="*",
        choices=["transcript", "transcript_ipa", "transcript_cmu", "all"],
        default=None,
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    summary = run_experiment(cfg, ablations=args.ablations)
    _print_json(summary)


if __name__ == "__main__":
    run_experiment_main()
