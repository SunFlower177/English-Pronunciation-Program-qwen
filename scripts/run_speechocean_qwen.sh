#!/usr/bin/env bash
set -euo pipefail

# Required for DashScope/Model Studio API. For a local vLLM server, set:
#   export QWEN_BACKEND=openai_compatible
#   export QWEN_BASE_URL=http://localhost:8000/v1
#   export QWEN_MODEL=Qwen/Qwen2.5-7B-Instruct
#   export QWEN_API_KEY=EMPTY
: "${QWEN_API_KEY:?Set QWEN_API_KEY or DASHSCOPE_API_KEY before running.}"

textpa-prepare-speechocean --output-dir data/speechocean762 --split test
textpa-run-experiment --config configs/default.yaml --ablations all
