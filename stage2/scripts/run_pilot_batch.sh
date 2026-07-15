#!/usr/bin/env bash
# Run a SWE-bench Lite pilot batch with SWE-agent locally (Docker/WSL2) while
# Qwen3-8B inference runs on a REMOTE GPU (e.g. Colab A100) behind an
# OpenAI-compatible endpoint.
#
# Topology:
#   - This machine (WSL2): SWE-agent orchestration + Docker containers + tests.
#   - Remote A100: vLLM serving Qwen/Qwen3-8B, exposed via a public tunnel.
#     See notebooks/serve_qwen_colab.ipynb for the server side.
#
# All neural-net inference happens remotely; only Docker/test execution is local.
#
# Prerequisites (local):
#   1. Docker daemon running and reachable from WSL2.
#   2. pip install sweagent   (or: pip install -e ".[swe]")
#   3. A reachable remote model endpoint (set MODEL_API_BASE).
#
# Environment:
#   MODEL_API_BASE   Remote OpenAI-compatible base URL, e.g.
#                    https://<your-tunnel>.trycloudflare.com/v1
#                    (default http://localhost:8000/v1 for an all-local run)
#   MODEL_API_KEY    API key for the endpoint (default: EMPTY; vLLM ignores it
#                    unless started with --api-key).
#   MODEL_NAME       litellm model id (default: hosted_vllm/Qwen3-8B)
#   OUTPUT_DIR       default data/trajectories/run_<timestamp>
#   SKIP_PREFLIGHT   set to 1 to skip the endpoint connectivity check
#
# Any extra args after the instances file are forwarded to `sweagent run-batch`.

set -euo pipefail
cd "$(dirname "$0")/.."

# Back-compat: VLLM_URL is the old name for MODEL_API_BASE.
MODEL_API_BASE="${MODEL_API_BASE:-${VLLM_URL:-http://localhost:8000/v1}}"
MODEL_API_KEY="${MODEL_API_KEY:-EMPTY}"
MODEL_NAME="${MODEL_NAME:-hosted_vllm/Qwen3-8B}"
INSTANCES_FILE="${1:-config/pilot_instances.txt}"
shift || true
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-data/trajectories/run_${TIMESTAMP}}"

if ! command -v sweagent >/dev/null 2>&1; then
  echo "ERROR: sweagent not found. Install with: pip install sweagent"
  exit 1
fi

if [[ ! -f "$INSTANCES_FILE" ]]; then
  echo "ERROR: instances file not found: $INSTANCES_FILE"
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker is not reachable. Start Docker Desktop / the daemon in WSL2."
  exit 1
fi

# Fail fast if the remote model endpoint is unreachable, before spinning up Docker.
if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  echo "Preflight: checking model endpoint ${MODEL_API_BASE%/}/models ..."
  if ! curl -fsS -m 15 -H "Authorization: Bearer ${MODEL_API_KEY}" \
       "${MODEL_API_BASE%/}/models" >/dev/null 2>&1; then
    echo "ERROR: cannot reach ${MODEL_API_BASE%/}/models"
    echo "  - Is the remote vLLM server up? (notebooks/serve_qwen_colab.ipynb)"
    echo "  - Is the tunnel URL current and does it include the /v1 suffix?"
    echo "  - Set SKIP_PREFLIGHT=1 to bypass this check."
    exit 1
  fi
  echo "Preflight: endpoint reachable."
fi

# Build filter regex from instance IDs (one per non-empty, non-comment line).
FILTER="$(grep -vE '^\s*(#|$)' "$INSTANCES_FILE" | paste -sd'|' -)"

mkdir -p "$OUTPUT_DIR"

# Render a resolved config with the remote endpoint baked in, so the run is
# deterministic regardless of what the committed config defaults to.
RESOLVED_CONFIG="$OUTPUT_DIR/swe_agent_resolved.yaml"
sed -E \
  -e "s#^([[:space:]]*name:).*#\1 ${MODEL_NAME}#" \
  -e "s#^([[:space:]]*api_base:).*#\1 ${MODEL_API_BASE}#" \
  -e "s#^([[:space:]]*api_key:).*#\1 ${MODEL_API_KEY}#" \
  config/swe_agent_qwen.yaml > "$RESOLVED_CONFIG"

echo "=== SWE-agent pilot batch (local Docker -> remote model) ==="
echo "Model endpoint: $MODEL_API_BASE"
echo "Model name:     $MODEL_NAME"
echo "Instances:      $INSTANCES_FILE ($(grep -cvE '^\s*(#|$)' "$INSTANCES_FILE") IDs)"
echo "Resolved cfg:   $RESOLVED_CONFIG"
echo "Output dir:     $OUTPUT_DIR"
echo ""

sweagent run-batch \
  --config "$RESOLVED_CONFIG" \
  --instances.type swe_bench \
  --instances.subset lite \
  --instances.split test \
  --instances.filter "$FILTER" \
  --output_dir "$OUTPUT_DIR" \
  "$@"

echo ""
echo "=== Done ==="
echo "Trajectories in: $OUTPUT_DIR"
echo "Next: python -m stage2.trajectories.ingest_batch --traj-dir $OUTPUT_DIR"
echo "Then run the GPU projection step (stage2.extract.project_steps) ON THE A100,"
echo "not locally — it needs raw residual-stream activations, which the API cannot give."
