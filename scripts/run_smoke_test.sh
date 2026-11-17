#!/usr/bin/env bash
set -euo pipefail

# A cheap smoke test. It uses the mock LLM and allows the CMU fallback, so it is NOT a
# paper-faithful result. Use this only to test the plumbing.
textpa-prepare-speechocean --output-dir data/speechocean762 --split test --limit 3
textpa-run-experiment --config configs/debug.yaml --ablations all
