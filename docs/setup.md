# Setup

## Requirements

- Python 3.11+
- GPU (Colab A100 recommended) for Stage 1 extraction, vLLM serving, and Stage 2 projection
- Anthropic API key for faithful ICRL generation (`--backend anthropic`)

## Install

```bash
pip install -e stage1
pip install -e stage2
pip install -e "stage2[mini]"   # optional: mini-agent environment
```

## Colab notebooks

| Notebook | Runtime | Purpose |
|----------|---------|---------|
| `stage1/notebooks/stage1_gpu_colab.ipynb` | A100 | ICRL extract + axis gate |
| `stage2/notebooks/serve_qwen_colab.ipynb` | A100 | vLLM + tunnel for agent inference |
| `stage2/notebooks/mini_agent_colab.ipynb` | CPU | Mini-agent batch + ingest |
| `stage2/notebooks/project_and_analyze_colab.ipynb` | A100 | Projection + analyses |

Open via: `https://colab.research.google.com/github/abdelmagid07/latent_failiure_prediction/blob/main/<path>`

## vLLM on Colab

Colab may ship a stale `vllm` binary with the wrong CUDA build. `serve_qwen_colab.ipynb` installs vLLM for the detected CUDA version and launches via `python -m vllm.entrypoints.openai.api_server`.

Set `enable_thinking=False` in the chat template to match activation extraction.

## Trajectory environments

**Mini-agent** (`stage2/mini/`): hand-written bug-fix tasks, no Docker. Output is SWE-agent-compatible `.traj` JSON.

**SWE-bench** (`scripts/run_pilot_batch.sh`): SWE-agent + Docker locally, Qwen inference via vLLM tunnel.

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
