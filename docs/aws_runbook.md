# AWS single-VM runbook (Stage 2 generation + projection)

> vLLM, Docker, and mini-swe-agent — runs on **one** EC2 GPU instance, so the
> agent talks to vLLM over `localhost` and there is **no tunnel** (no 524s, no
> rotating URL). New to the project? Start at [../HANDOFF.md](../HANDOFF.md).

## Why this needs almost no code changes

The pipeline is already endpoint-agnostic. `run_mini_swe_batch.sh` reads the
endpoint from `MODEL_API_BASE` and **defaults to `http://localhost:8000/v1`**, and
`serve_qwen.sh` launches the **same** vLLM stack with the **same** parser flags as
`serve_qwen_colab.ipynb` (`--reasoning-parser qwen3 --tool-call-parser hermes`).
So the response contract (`reasoning_content` + `tool_calls`) that `schema.py`,
`parse_mini_swe_traj.py`, and `project_steps.py` were built against is identical.
Moving to AWS is a set of commands plus a GPU-appropriate `--dtype`, not a
refactor.

## Compute-policy classification (read before you spend)

Two workloads, two rules (see [../compute_policy.txt](../compute_policy.txt)):

- **Generation** (mini-swe-agent driving Qwen3-8B) is *inference*. Policy §7 says
  paid-GPU inference is not reimbursed **unless pre-approved**. Because the result
  requires on-policy generation with the exact `Qwen/Qwen3-8B` weights we also
  read internally, and SWE-bench needs Docker (a real VM), self-hosting is the
  defensible path — **pre-approve it on the Airtable form** first.
- **Projection** (`project_steps.py`) genuinely needs residual-stream activations
  no API exposes → legitimate self-hosted GPU work. Can run on this same box, or
  later on a free tier (Kaggle/Modal/Colab) since it needs no Docker.

Validate on a free T4 first (a 0.6B/1.7B smoke run); only move the real 8B run to
paid hardware once it works end to end.

## Instance choice

| Need | Instance | GPU / VRAM | Notes |
|------|----------|-----------|-------|
| Real Qwen3-8B run | `g6.2xlarge` or `g5.2xlarge` | L4 / A10G, 24 GB | 8 vCPU handles parallel Docker workers; bf16 fits 8B + 32k KV at 1 worker |
| Headroom / throughput | Colab A100 burst, or `g6e.xlarge` | A100-40 / L40S-48 | if you need bigger batch or many workers |
| Smoke test only | `g4dn.xlarge` | T4, 16 GB | validation with a small model, `float16` |

- **Disk:** SWE-bench images are large — attach **≥100 GB EBS** or pulls fail mid-batch.
- **AMI:** use a Deep Learning AMI (CUDA preinstalled) to skip driver setup.
- **Spot** for anything resumable is ~60-70% cheaper (policy §5).

## 1. Provision (once per instance)

```bash
# Docker for the SWE-bench test containers (they are CPU-only; the GPU is vLLM's).
sudo systemctl start docker
docker info >/dev/null && echo "docker OK"

git clone https://github.com/abdelmagid07/latent_failiure_prediction.git
cd latent_failiure_prediction

python -m venv .venv && source .venv/bin/activate
pip install -e "stage2[swe]"                       # mini-swe-agent + swebench harness
pip install "vllm==0.11.0" "transformers==4.55.4" "numpy<2.3"

huggingface-cli login                              # once, to pull Qwen/Qwen3-8B
```

## 2. Serve Qwen3-8B on localhost (in tmux/screen)

```bash
tmux new -s vllm
bash stage2/scripts/serve_qwen.sh                  # real run: Qwen3-8B, 32k ctx, bf16 auto
# smoke test instead:
#   MODEL=Qwen/Qwen3-1.7B MAX_MODEL_LEN=8192 bash stage2/scripts/serve_qwen.sh
# detach: Ctrl-b d
```

`serve_qwen.sh` auto-selects `bfloat16` on Ampere+ (A100/A10G/L4) and `float16`
on a T4. Wait until it logs that the model is loaded, then confirm:

```bash
curl -fsS http://localhost:8000/v1/models
```

## 3. Generate trajectories (another shell)

No tunnel, so `MODEL_API_BASE` already defaults to localhost:

```bash
source .venv/bin/activate
bash stage2/scripts/run_mini_swe_batch.sh                    # pilot filter
# or:  SUBSET=verified WORKERS=4 bash stage2/scripts/run_mini_swe_batch.sh
```

The preflight check now hits `localhost:8000/models` and passes instantly. Output
lands in `data/trajectories/mini_swe_run_<timestamp>/` (per-instance `*.traj.json`
+ `preds.json`).

## 4. Label + ingest + project (same box)

```bash
# Turn predictions into resolved/unresolved labels:
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path data/trajectories/mini_swe_run_<ts>/preds.json --run_id <run_id>
# place the harness report at .../results.json, then normalize:
python -m stage2.trajectories.ingest_batch \
  --traj-dir data/trajectories/mini_swe_run_<ts> --format mini-swe-agent

# Projection runs right here (the GPU is on this box):
python -m stage2.extract.project_steps ...
```

## 5. Cost hygiene (conditions of reimbursement)

- **Stop the instance the moment a session ends** — you are billed per wall-clock
  hour whether or not the GPU is busy. Idle A100/GPU time is the #1 way teams blow
  their budget (policy §9).
- A *stopped* instance still bills for its EBS volume. At project end, **terminate**
  and delete the volume/snapshots (policy §5).
- Watch utilization (`watch -n1 nvidia-smi`); a starved GPU means you over-sized
  the instance.

## Relationship to the other options

- **Azure managed endpoint** (if it passes the compat test) is preferred for
  *generation* — bigger credit pool, stable HTTPS, no idle-GPU management. This
  AWS box is the fallback if Azure's TGI serving doesn't reproduce the
  `reasoning_content` / `tool_calls` contract.
- **Colab + tunnel** (`serve_qwen_colab.ipynb`) stays intact for quick smoke
  tests; it is not suitable for long multi-instance batches (per-request timeout,
  ephemeral URL).
