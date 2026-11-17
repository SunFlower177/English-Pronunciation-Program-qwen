#!/usr/bin/env bash
set -euo pipefail
: "${QWEN_API_KEY:?Set QWEN_API_KEY or DASHSCOPE_API_KEY before running.}"

textpa-prepare-speechocean --output-dir data/speechocean762 --split test
textpa-run-experiment --config configs/default.yaml --ablations transcript transcript_ipa transcript_cmu all
