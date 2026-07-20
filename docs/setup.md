# Setup

> Quick reference. For the full hands-on walkthrough (compute topology, running
> the pipeline end to end, troubleshooting), see [onboarding.md](onboarding.md).
> New to the project? Start at [HANDOFF.md](../HANDOFF.md).

## Requirements

- Python 3.11+
- GPU (Colab A100 recommended) for Stage 1 extraction, vLLM serving, and Stage 2 projection
- Anthropic API key for faithful ICRL generation (`--backend anthropic`)

## Install

```bash
pip install -e stage1
pip install -e stage2
pip install -e "stage2[devbugs]"   # optional: local dev bug-fix harness (no Docker)
pip install -e "stage2[swe]"       # optional: mini-swe-agent + swebench harness (real runs)
```

## Colab notebooks

| Notebook | Runtime | Purpose |
|----------|---------|---------|
| `stage1/notebooks/stage1_gpu_colab.ipynb` | A100 | ICRL extract + axis gate |
| `stage2/notebooks/serve_qwen_colab.ipynb` | A100 | vLLM + tunnel for agent inference |
| `stage2/notebooks/devbugs_agent_colab.ipynb` | CPU | Devbugs-agent batch + ingest |
| `stage2/notebooks/project_and_analyze_colab.ipynb` | A100 | Projection + analyses |

Open via: `https://colab.research.google.com/github/abdelmagid07/latent_failiure_prediction/blob/main/<path>`

## vLLM on Colab

Colab may ship a stale `vllm` binary with the wrong CUDA build. `serve_qwen_colab.ipynb` installs vLLM for the detected CUDA version and launches via `python -m vllm.entrypoints.openai.api_server`.

Stage 2 runs with `enable_thinking=True` (project decision 2026-07-17), passed per request as `chat_template_kwargs`. Launch the server with `--enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3` so the `<think>` text is split into `reasoning_content` and tool calls still parse; projection re-renders both through the chat template.

## Self-hosted GPU (AWS, no tunnel)

For long batches the Colab + cloudflared tunnel is fragile (per-request timeout, rotating URL). The alternative is a single EC2 GPU VM running vLLM + Docker + mini-swe-agent together, so the agent hits vLLM over `localhost`. Serve with `stage2/scripts/serve_qwen.sh` (same parser flags as Colab, auto-picks `--dtype`). Full sequence + compute-policy classification in [aws_runbook.md](aws_runbook.md).

METHOD.tex pins: mini-swe-agent **2.4.5**, step budget **60**, sampling 0.6/0.95/20, readout = mean cosine over generated tokens. See [method.md](method.md).

## Trajectory environments

**Devbugs harness** (`stage2/devbugs/`): hand-written bug-fix tasks, no Docker. Output is SWE-agent-compatible `.traj` JSON.

**SWE-bench** (`scripts/run_mini_swe_batch.sh`): mini-swe-agent + Docker locally, Qwen inference via vLLM — either a Colab tunnel or a self-hosted `localhost` endpoint ([aws_runbook.md](aws_runbook.md)). Ingest with `--format mini-swe-agent`. (Legacy `run_pilot_batch.sh` / SWE-agent kept until the mini path is verified.)

## Offline wiring tests

```bash
bash tests/integration/test_stage1_wiring.sh
bash tests/integration/test_stage2_wiring.sh
```

No GPU required.

## Artifacts (gitignored)

| File | Stage |
|------|-------|
| `stage1/data/value_axis.npy` | 1 (default preset) |
| `stage1/data/value_axis_proxy.npy` | 1 (dev preset) |
| `stage2/data/projections.parquet` | 2 |
| `stage2/data/analysis_report.json` | 2 |
