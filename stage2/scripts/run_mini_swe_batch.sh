#!/usr/bin/env bash
# Real SWE-bench batch with mini-swe-agent driving on-policy Qwen3-8B.
#
# This is the primary generation path for paper data. It replaces the legacy
# SWE-agent path (scripts/run_pilot_batch.sh + config/swe_agent_qwen.yaml),
# which is kept only until this path is verified end to end.
#
# Topology:
#   - This machine (WSL2): mini-swe-agent + Docker containers + tests.
#   - Remote A100: vLLM serving Qwen/Qwen3-8B behind a public tunnel.
#     See notebooks/serve_qwen_colab.ipynb for the server side.
# All neural-net inference happens remotely; only Docker/test execution is local.
#
# Prerequisites (local):
#   1. Docker daemon running and reachable from WSL2.
#   2. pip install -e ".[swe]"   (installs mini-swe-agent + the swebench harness)
#   3. A reachable remote model endpoint (set MODEL_API_BASE).
#
# Environment:
#   MODEL_API_BASE   Remote OpenAI-compatible base URL, e.g.
#                    https://<your-tunnel>.trycloudflare.com/v1
#   MODEL_API_KEY    API key (default: EMPTY; vLLM ignores it unless started with --api-key)
#   MODEL_NAME       litellm model id (default: hosted_vllm/Qwen3-8B)
#   SUBSET           SWE-bench subset: verified | lite | <dataset path> (default: verified)
#   SPLIT            dataset split (default: test)
#   WORKERS          parallel workers (default: 1)
#   STEP_LIMIT       override the base config's agent.step_limit (optional)
#   OUTPUT_DIR       default data/trajectories/mini_swe_run_<timestamp>
#   SKIP_PREFLIGHT   set to 1 to skip the endpoint connectivity check
#
# Usage:
#   export MODEL_API_BASE="https://<tunnel>.trycloudflare.com/v1"
#   bash scripts/run_mini_swe_batch.sh                      # filter by config/pilot_instances.txt
#   bash scripts/run_mini_swe_batch.sh config/pilot_instances.txt
#   SUBSET=verified WORKERS=4 bash scripts/run_mini_swe_batch.sh
# Any extra args after the instances file are forwarded to `mini-extra swebench`.

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_API_BASE="${MODEL_API_BASE:-${VLLM_URL:-http://localhost:8000/v1}}"
MODEL_API_KEY="${MODEL_API_KEY:-EMPTY}"
MODEL_NAME="${MODEL_NAME:-hosted_vllm/Qwen3-8B}"
SUBSET="${SUBSET:-verified}"
SPLIT="${SPLIT:-test}"
WORKERS="${WORKERS:-1}"
STEP_LIMIT="${STEP_LIMIT:-}"
INSTANCES_FILE="${1:-config/pilot_instances.txt}"
shift || true
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-data/trajectories/mini_swe_run_${TIMESTAMP}}"
OVERRIDE_CFG="config/mini_swe_qwen.yaml"

if ! command -v mini-extra >/dev/null 2>&1; then
  echo "ERROR: mini-extra not found. Install with: pip install -e \".[swe]\""
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

mkdir -p "$OUTPUT_DIR"
RESOLVED_CONFIG="$OUTPUT_DIR/mini_swe_resolved.yaml"

# Resolve the run config: installed base <- our override layer <- env values.
# Done in Python because the merge is a deep merge and bash can't do YAML.
MODEL_API_BASE="$MODEL_API_BASE" MODEL_API_KEY="$MODEL_API_KEY" \
MODEL_NAME="$MODEL_NAME" STEP_LIMIT="$STEP_LIMIT" \
OVERRIDE_CFG="$OVERRIDE_CFG" RESOLVED_CONFIG="$RESOLVED_CONFIG" \
python - <<'PY'
import os, pathlib, yaml
import minisweagent

base_path = pathlib.Path(minisweagent.__file__).parent / "config" / "benchmarks" / "swebench.yaml"
base = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
override = yaml.safe_load(pathlib.Path(os.environ["OVERRIDE_CFG"]).read_text(encoding="utf-8")) or {}


def deep_merge(a: dict, b: dict) -> dict:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            deep_merge(a[k], v)
        else:
            a[k] = v
    return a


cfg = deep_merge(base, override)
model = cfg.setdefault("model", {})
model_kwargs = model.setdefault("model_kwargs", {})
model["model_name"] = os.environ["MODEL_NAME"]
model_kwargs["api_base"] = os.environ["MODEL_API_BASE"]
model_kwargs["api_key"] = os.environ["MODEL_API_KEY"]

step_limit = os.environ.get("STEP_LIMIT", "").strip()
if step_limit:
    cfg.setdefault("agent", {})["step_limit"] = int(step_limit)

out = pathlib.Path(os.environ["RESOLVED_CONFIG"])
out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

thinking = (
    model_kwargs.get("extra_body", {})
    .get("chat_template_kwargs", {})
    .get("enable_thinking")
)
print(f"Resolved config -> {out}")
print(f"  base:            {base_path}")
print(f"  model_name:      {model['model_name']}")
print(f"  api_base:        {model_kwargs['api_base']}")
print(f"  enable_thinking: {thinking}  (must be True)")
if thinking is not True:
    raise SystemExit("Refusing to run: enable_thinking is not True in the resolved config.")
PY

# Build a --filter regex from an instance-id file, if one is present.
FILTER_ARGS=()
if [[ -f "$INSTANCES_FILE" ]]; then
  FILTER="$(grep -vE '^\s*(#|$)' "$INSTANCES_FILE" | paste -sd'|' -)"
  if [[ -n "$FILTER" ]]; then
    FILTER_ARGS=(--filter "$FILTER")
    echo "Filtering to $(grep -cvE '^\s*(#|$)' "$INSTANCES_FILE") instance ids from $INSTANCES_FILE"
  fi
else
  echo "No instances file at $INSTANCES_FILE; running the full $SUBSET/$SPLIT subset."
fi

echo "=== mini-swe-agent SWE-bench batch (local Docker -> remote model) ==="
echo "Model endpoint: $MODEL_API_BASE"
echo "Model name:     $MODEL_NAME"
echo "Subset/split:   $SUBSET / $SPLIT   workers=$WORKERS"
echo "Resolved cfg:   $RESOLVED_CONFIG"
echo "Output dir:     $OUTPUT_DIR"
echo ""

mini-extra swebench \
  -c "$RESOLVED_CONFIG" \
  -m "$MODEL_NAME" \
  --subset "$SUBSET" \
  --split "$SPLIT" \
  -w "$WORKERS" \
  -o "$OUTPUT_DIR" \
  "${FILTER_ARGS[@]}" \
  "$@"

echo ""
echo "=== Done ==="
echo "Trajectories + predictions in: $OUTPUT_DIR (per-instance *.traj.json + preds.json)"
echo ""
echo "Next — turn predictions into resolved/unresolved labels, then ingest:"
echo "  python -m swebench.harness.run_evaluation \\"
echo "    --dataset_name princeton-nlp/SWE-bench_Verified \\"
echo "    --predictions_path $OUTPUT_DIR/preds.json --run_id <run_id>"
echo "  # place the harness report at $OUTPUT_DIR/results.json (resolved/unresolved), then:"
echo "  python -m stage2.trajectories.ingest_batch --traj-dir $OUTPUT_DIR --format mini-swe-agent"
echo ""
echo "Then run the GPU projection step (stage2.extract.project_steps) ON THE A100,"
echo "not locally — it needs raw residual-stream activations, which the API cannot give."
