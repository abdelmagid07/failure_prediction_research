#!/usr/bin/env bash
# Serve Qwen3-8B with vLLM on a self-hosted GPU box (AWS EC2, etc.), no tunnel.
#
# This is the AWS / single-VM analogue of notebooks/serve_qwen_colab.ipynb: it
# launches the SAME vLLM stack with the SAME parser flags, so the response
# contract (reasoning_content + tool_calls) that stage2's parser/projection were
# built against is byte-for-byte identical. The only difference from Colab is
# there is no cloudflared tunnel — the server listens on localhost and the
# mini-swe-agent batch (scripts/run_mini_swe_batch.sh) talks to it directly.
#
# Prereqs on the box (see docs/aws_runbook.md for the full sequence):
#   pip install "vllm==0.11.0" "transformers==4.55.4" "numpy<2.3"
#   huggingface-cli login          # needed once to pull Qwen/Qwen3-8B
#
# Env overrides (all optional; defaults are the real-experiment values):
#   MODEL          HF model id             (default: Qwen/Qwen3-8B)
#   SERVED_NAME    OpenAI-API model name   (default: basename of MODEL)
#   MAX_MODEL_LEN  context window          (default: 32768)
#   PORT           listen port             (default: 8000)
#   API_KEY        require a key if != EMPTY (default: EMPTY)
#   DTYPE          float16 | bfloat16 | auto (default: auto-detect from GPU)
#   REASONING_PARSER  (default: qwen3; use deepseek_r1 if this vLLM lacks qwen3)
#
# Usage:
#   bash stage2/scripts/serve_qwen.sh                 # real run: Qwen3-8B, 32k ctx
#   MODEL=Qwen/Qwen3-1.7B MAX_MODEL_LEN=8192 bash stage2/scripts/serve_qwen.sh   # smoke test
#
# Runs in the foreground; use tmux/screen (or nohup ... &) to keep it alive while
# you run the batch in another shell.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
SERVED_NAME="${SERVED_NAME:-$(basename "$MODEL")}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-EMPTY}"
DTYPE="${DTYPE:-auto}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"

# Auto-pick dtype from the GPU's compute capability: Ampere+ (>= 8.0, e.g. A100,
# A10G, L4, L40S, H100) does bfloat16; Turing (7.5, e.g. T4) has no bf16 → float16.
if [[ "$DTYPE" == "auto" ]]; then
  CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ' || true)"
  if [[ -n "$CC" ]] && awk "BEGIN{exit !($CC >= 8.0)}"; then
    DTYPE="bfloat16"
  else
    DTYPE="float16"
  fi
  echo "Auto-selected --dtype $DTYPE (GPU compute capability: ${CC:-unknown})"
fi

cmd=(
  python -m vllm.entrypoints.openai.api_server
  --model "$MODEL"
  --served-model-name "$SERVED_NAME"
  --dtype "$DTYPE"
  --max-model-len "$MAX_MODEL_LEN"
  --port "$PORT"
  # Required for mini-SWE-agent tool calls.
  --enable-auto-tool-choice
  --tool-call-parser hermes
  # Splits the <think> block into reasoning_content so tool calls still parse
  # under thinking-on (vllm-project/vllm#20611). project_steps re-renders
  # reasoning_content + tool_calls through the chat template.
  --reasoning-parser "$REASONING_PARSER"
)
if [[ -n "$API_KEY" && "$API_KEY" != "EMPTY" ]]; then
  cmd+=(--api-key "$API_KEY")
fi

echo "=== Serving $MODEL as '$SERVED_NAME' on http://localhost:$PORT/v1 ==="
echo "${cmd[*]}"
echo "First run downloads the weights (~16GB for Qwen3-8B); then it stays up."
echo "Point the batch at it with:  export MODEL_API_BASE=http://localhost:$PORT/v1"
echo ""
exec "${cmd[@]}"
