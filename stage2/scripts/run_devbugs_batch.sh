#!/usr/bin/env bash
# Devbugs local-agent batch — NO Docker required.
#
# Runs hand-written Python bug-fix tasks in temp directories via subprocess,
# calling Qwen3-8B on a remote Colab A100 (same tunnel setup as run_pilot_batch.sh).
# Output is SWE-agent-compatible .traj JSON for the existing ingest/projection pipeline.
#
# Prerequisites:
#   1. pip install -e . pytest
#   2. Remote vLLM endpoint (notebooks/serve_qwen_colab.ipynb)
#
# Environment (same as run_pilot_batch.sh):
#   MODEL_API_BASE   e.g. https://<tunnel>.trycloudflare.com/v1
#   MODEL_API_KEY    default EMPTY
#   MODEL_NAME       default Qwen3-8B (vLLM served name; hosted_vllm/ prefix stripped)
#   OUTPUT_DIR       default data/trajectories/devbugs_run_<timestamp>
#   SKIP_PREFLIGHT   set to 1 to skip endpoint check
#
# Usage:
#   export MODEL_API_BASE="https://..."
#   bash scripts/run_devbugs_batch.sh
#   bash scripts/run_devbugs_batch.sh config/devbugs_instances.txt
#   bash scripts/run_devbugs_batch.sh --instance-id mini_add_001   # single smoke run

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_API_BASE="${MODEL_API_BASE:-${VLLM_URL:-http://localhost:8000/v1}}"
MODEL_API_KEY="${MODEL_API_KEY:-EMPTY}"
MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-data/trajectories/devbugs_run_${TIMESTAMP}}"

if ! python -c "import pytest" >/dev/null 2>&1; then
  echo "Installing pytest..."
  pip install -q pytest
fi

ARGS=()
if [[ "${1:-}" == "--instance-id" ]]; then
  ARGS+=("$@")
elif [[ -n "${1:-}" ]]; then
  ARGS+=(--instances "$1")
else
  ARGS+=(--instances config/devbugs_instances.txt)
fi

PREFLIGHT_FLAG=""
if [[ "${SKIP_PREFLIGHT:-0}" == "1" ]]; then
  PREFLIGHT_FLAG="--skip-preflight"
fi

echo "=== Devbugs local-agent batch (no Docker) ==="
echo "Model endpoint: $MODEL_API_BASE"
echo "Model name:     $MODEL_NAME"
echo "Output dir:     $OUTPUT_DIR"
echo ""

MODEL_API_BASE="$MODEL_API_BASE" \
MODEL_API_KEY="$MODEL_API_KEY" \
MODEL_NAME="$MODEL_NAME" \
python -m stage2.devbugs.run_batch \
  --output-dir "$OUTPUT_DIR" \
  --api-base "$MODEL_API_BASE" \
  --api-key "$MODEL_API_KEY" \
  --model "$MODEL_NAME" \
  $PREFLIGHT_FLAG \
  "${ARGS[@]}"

echo ""
echo "=== Done ==="
echo "Trajectories: $OUTPUT_DIR"
echo "Ingest:  python -m stage2.trajectories.ingest_batch --traj-dir $OUTPUT_DIR"
echo "Project: run project_steps.py on the A100 (see notebooks/project_and_analyze_colab.ipynb)"
