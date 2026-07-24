# RUN GUIDE — code → paper results

End-to-end instructions for this repo. Methodology: [METHOD.tex](METHOD.tex).
Architecture / decisions: [HANDOFF.md](HANDOFF.md). Compute policy:
[compute_policy.txt](compute_policy.txt). AWS details: [docs/aws_runbook.md](docs/aws_runbook.md).

**Paper ↔ codebase map**

| Paper stage | What | Code |
|-------------|------|------|
| 1 | Rebuild + gate value axis | `stage1/` (**done** — need the frozen `.npy`) |
| 2 | Single-turn coding control | **Deferred** — skip |
| 3 | Agentic transfer (60×5) | `stage2/` generate → ingest → project → analyze |
| 4 | Verbalized confidence | `stage2/elicit/` + `analyze/internal_vs_elicited.py` |
| 5 | Fitted probes | `stage2/probes/fit_probes.py` |

**Frozen task set:** 25 easy + 35 medium SWE-bench Verified IDs in
`stage2/config/selected_instances.txt` (metadata: `selected_tasks.csv`;
provenance: `task_selection.sql` + root `tasks.csv`).

**Target run:** 60 tasks × 5 seeds = 300 trajectories (after infra exclusions).

---

## 0. Mental model (where each step runs)

```
Generation (agent + Docker)     Inference (Qwen3-8B)      Projection (activations)
─────────────────────────────   ─────────────────────     ─────────────────────────
Your machine OR same GPU VM     vLLM on GPU box           Same GPU box OR Colab A100
mini-swe-agent + Docker         localhost or tunnel       project_steps.py
```

**Recommended for the real 60×5:** one AWS GPU VM (vLLM + Docker + agent on
`localhost`) — see [docs/aws_runbook.md](docs/aws_runbook.md). Colab + tunnel is
OK for a tiny pilot; it is fragile for long batches.

Projection **cannot** use the chat API — it needs residual-stream hooks, so it
runs wherever the weights are loaded locally (same VM or a Colab notebook).

---

## 1. Prerequisites

| Need | Why |
|------|-----|
| Python **3.11+** | packages |
| Linux / WSL2 | Docker + agent |
| Docker | SWE-bench containers |
| GPU with ≥24 GB VRAM | Qwen3-8B bf16 + context (e.g. g6.2xlarge L4 / g5 A10G / A100) |
| ≥100 GB disk | SWE-bench images |
| Hugging Face login | download `Qwen/Qwen3-8B` |
| Frozen axis file | `stage1/data/value_axis.npy` (or `value_axis_proxy.npy` for smoke) |

> **Check the axis now.** `stage1/data/` is often empty in a fresh clone (artifacts
> are gitignored). Without `value_axis.npy` you can generate trajectories but
> cannot project. Copy the Stage‑1 artifact from wherever you ran the gate, or
> re-run Stage 1 (§8).

---

## 2. Install

```bash
cd failure_prediction_research
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e stage1
pip install -e "stage2[swe]"       # mini-swe-agent==2.4.5 + swebench harness

# On the GPU box that serves / projects:
pip install "vllm==0.11.0" "transformers==4.55.4" "numpy<2.3"
huggingface-cli login
```

**Offline sanity (no GPU):**

```bash
bash tests/integration/test_stage1_wiring.sh
bash tests/integration/test_stage2_wiring.sh
```

Both must pass before you spend GPU time.

---

## 3. Serve Qwen3-8B (thinking-ON + tools)

On the GPU machine, in `tmux`/`screen`:

```bash
bash stage2/scripts/serve_qwen.sh
# Wait until weights load, then:
curl -fsS http://localhost:8000/v1/models
```

Required server flags (already in `serve_qwen.sh`):

- `--enable-auto-tool-choice --tool-call-parser hermes`
- `--reasoning-parser qwen3` (use `deepseek_r1` if your vLLM build lacks `qwen3`)

Without the reasoning parser, thinking-ON + tool calls break (tool call stranded
inside `<think>`).

**Colab alternative:** `stage2/notebooks/serve_qwen_colab.ipynb` → copy the
tunnel URL (must end in `/v1`).

```bash
export MODEL_API_BASE="http://localhost:8000/v1"   # AWS / same-box
# export MODEL_API_BASE="https://<tunnel>.trycloudflare.com/v1"  # Colab
export MODEL_API_KEY="EMPTY"
```

---

## 4. Generate trajectories

Always work from `stage2/`:

```bash
cd stage2
export MODEL_API_BASE="${MODEL_API_BASE:-http://localhost:8000/v1}"
```

### 4a. Tiny pilot first (do not skip)

```bash
# 3–5 lines from config/pilot_instances.txt is fine; or a short custom list
ROLLOUTS=1 STEP_LIMIT=60 WORKERS=1 \
  bash scripts/run_mini_swe_batch.sh config/pilot_instances.txt
```

Confirm:

1. Preflight reaches `/models`.
2. Resolved config prints `enable_thinking: True`.
3. Each instance has `*.traj.json` with `reasoning_content` / `tool_calls`.
4. `preds.json` exists.

Then try multi-seed on the pilot:

```bash
ROLLOUTS=5 SEED_BASE=0 STEP_LIMIT=60 \
  bash scripts/run_mini_swe_batch.sh config/pilot_instances.txt
```

Layout: `data/trajectories/mini_swe_run_<ts>/r0/ … r4/` each with trajs + `preds.json`.

### 4b. Full paper set (60 × 5)

```bash
ROLLOUTS=5 SEED_BASE=0 STEP_LIMIT=60 WORKERS=1 \
  OUTPUT_DIR=data/trajectories/verified_60x5 \
  bash scripts/run_mini_swe_batch.sh config/selected_instances.txt
```

Useful env vars:

| Var | Default | Meaning |
|-----|---------|---------|
| `ROLLOUTS` | 1 | seeds per task (`METHOD.tex` = 5) |
| `SEED_BASE` | 0 | first seed; uses `SEED_BASE .. SEED_BASE+ROLLOUTS-1` |
| `STEP_LIMIT` | 60 | agent step budget |
| `SUBSET` | `verified` | must stay Verified |
| `WORKERS` | 1 | parallel Docker agents (raise carefully) |
| `OUTPUT_DIR` | timestamped | set explicitly for a stable path |
| `SKIP_PREFLIGHT` | 0 | set `1` only if curl is blocked |

Sampling (already in `config/mini_swe_qwen.yaml`): temp **0.6**, top_p **0.95**,
top_k **20**.

> Wall-clock note: 300 trajectories × up to 60 steps is a long job. Prefer a
> stable `localhost` endpoint. Stop/spot the VM when idle ([compute_policy.txt](compute_policy.txt)).

---

## 5. Label each rollout (SWE-bench harness)

Generation writes **predictions**, not pass/fail. Outcomes are **per seed**.

```bash
cd stage2
RUN=data/trajectories/verified_60x5   # or your OUTPUT_DIR

for d in "$RUN"/r*/; do
  python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --predictions_path "$d/preds.json" \
    --run_id "$(basename "$d")"
  # Copy/move the harness report so ingest finds it:
  #   $d/results.json
  # The harness writes under logs/…; place a JSON with `resolved` /
  # `unresolved` (or `resolved_ids` / `unresolved_ids`) at $d/results.json.
done
```

For a flat single-rollout run (`ROLLOUTS=1` without nesting), put one
`results.json` at the run root next to `preds.json`.

---

## 6. Ingest → normalized trajectories

```bash
cd stage2
python -m stage2.trajectories.ingest_batch \
  --traj-dir data/trajectories/verified_60x5 \
  --format mini-swe-agent
```

Writes `data/normalized/*.json` + `data/normalized/ingest_manifest.json`.

**Always read the printed `exit_status` histogram.**

- Genuine (keep, label from harness): `Submitted`, `LimitsExceeded`,
  `TimeExceeded`, `RepeatedFormatError`
- Everything else (e.g. `APIConnectionError`) = **infra crash** → excluded

### 6a. Regenerate infra exclusions

```bash
python scripts/list_regens.py --manifest data/normalized/ingest_manifest.json
```

It prints `SEED_BASE=<fresh> ROLLOUTS=1` commands for excluded `(task_id, seed)`
pairs. Re-run those, evaluate the new `r<seed>/`, then **re-ingest** the same
`--traj-dir` (or merge dirs carefully). Report exclusion counts in the paper.

You need **both** outcome classes (resolved and unresolved) before analyses are
meaningful.

---

## 7. Project the frozen axis (GPU)

Needs local weights + hooks. On the GPU box (or
`stage2/notebooks/project_and_analyze_colab.ipynb`):

```bash
cd stage2

# Prefer the real Stage-1 axis; proxy is for plumbing smoke only.
AXIS=../stage1/data/value_axis.npy
# AXIS=../stage1/data/value_axis_proxy.npy

python -m stage2.extract.project_steps \
  --traj-dir data/normalized \
  --axis-path "$AXIS" \
  --activations-npz data/activations.npz \
  --output data/projections.parquet
```

Defaults: thinking **ON**, all layers, Eq. 1 mean cosine over agent-generated
tokens `G_t`, plus `proj_final` robustness column. The script aborts on
thinking-mode mismatch or render-fidelity failure — do not bypass those guards.

Headline analyses use **layer 21** (`config/defaults.yaml` → `primary_layer`).

---

## 8. Analyze transfer (paper Stage 3)

```bash
cd stage2
python -m stage2.analyze.run_analyses \
  --projections data/projections.parquet \
  --primary-layer 21 \
  --n-bins 5 \
  --output-dir data/report
```

**Read first:** `data/report/analysis_report.json`

| Field | Meaning |
|-------|---------|
| `final_step_auroc` + `final_step_auroc_ci` | Headline separation + task-level BCa CI |
| `majority_baseline` | Compare AUROC to this, not 0.5 |
| `permutation_p` | Block permutation test |
| `auroc_by_position` / CSV | Full positional sweep |
| `within_task_*` | Mixed-outcome within-task contrast |
| `late_bin_separation_to_noise` | SNR sanity |

Figures: `final_step_separation.png`, `noise_by_token_type.png`, etc.

Interpretation notes: [docs/analyses.md](docs/analyses.md).

---

## 9. Verbalized confidence (paper Stage 4)

Needs a live OpenAI-compatible endpoint (same vLLM is fine):

```bash
cd stage2
python -m stage2.elicit.confidence \
  --traj-dir data/normalized \
  --api-base "$MODEL_API_BASE" \
  --output data/confidence.parquet

python -m stage2.analyze.internal_vs_elicited \
  --projections data/projections.parquet \
  --confidence data/confidence.parquet \
  --output-dir data/report
```

Prompt text: `stage2/config/elicitation_prompt.txt`.

---

## 10. Fitted probes (paper Stage 5)

Requires the NPZ from §7:

```bash
cd stage2
python -m stage2.probes.fit_probes \
  --activations data/activations.npz \
  --output-dir data/probe_report
```

Report the **full layer × position grid**, not only the max (METHOD.tex).

---

## 11. Artifact checklist (what “done” looks like)

| Artifact | Path |
|----------|------|
| Task freeze | `stage2/config/selected_instances.txt` + `selected_tasks.csv` |
| Raw trajs | `stage2/data/trajectories/<run>/r{0..4}/…/*.traj.json` + `preds.json` |
| Labels | `…/r*/results.json` |
| Normalized | `stage2/data/normalized/*.json` + `ingest_manifest.json` |
| Projections | `stage2/data/projections.parquet` |
| Activations | `stage2/data/activations.npz` |
| Transfer report | `stage2/data/report/analysis_report.json` + plots |
| Confidence | `stage2/data/confidence.parquet` |
| Probe grid | `stage2/data/probe_report/` |
| Frozen axis | `stage1/data/value_axis.npy` (+ optional `axis_manifest.json`) |

---

## 12. If you need to rebuild Stage 1 (axis missing)

Stage 1 is already validated (≈0.87 AUROC at L21/L22; gate floor 0.87). Only
re-run if the `.npy` is gone:

```bash
cd stage1
pip install -e .
# Full GPU path: see stage1/notebooks/stage1_gpu_colab.ipynb
# Offline gate smoke (needs existing ICRL + activations or proxy preset):
python -m stage1.pipeline.run_gate --preset dev --icrl data/icrl_proxy.json --skip-extract
```

Do **not** refit the axis after looking at agentic data.

---

## 13. Suggested order of operations (minimize waste)

1. Wiring tests (§2)  
2. Serve model (§3) + `curl /models`  
3. Pilot `ROLLOUTS=1` on a few Verified IDs (§4a)  
4. Eval + ingest pilot; confirm both outcome classes appear if possible  
5. Pilot `ROLLOUTS=5` → eval → ingest → **project one traj** (catch thinking/render bugs early)  
6. Full `selected_instances.txt` with `ROLLOUTS=5` (§4b)  
7. Eval all seeds → ingest → regens (§5–6)  
8. Project all + activations (§7)  
9. `run_analyses` (§8)  
10. Elicit + probes (§9–10)  

---

## 14. Troubleshooting (short)

| Symptom | Fix |
|---------|-----|
| `cannot reach …/models` | vLLM down / wrong URL / missing `/v1` |
| `mini-extra not found` | `pip install -e "stage2[swe]"` in active venv |
| Docker not reachable | start Docker; `docker info` from the same shell |
| No `reasoning_content` / tools as raw text | server missing hermes + reasoning parser; client must send `enable_thinking: true` |
| Ingest skips everything | missing per-rollout `results.json` |
| Unexpected exclusions | check `exit_status` histogram; extend `--genuine-statuses` only for real terminals |
| One outcome class only | expand / finish run; hard tasks were excluded from the 60 for this reason |
| `Thinking-mode mismatch` | regenerate thinking-ON (don’t disable the guard) |
| `value_axis.npy` missing | restore Stage‑1 artifact (§12) |
| Tunnel 524 / rotating URL | move generation to AWS localhost |

More detail: [docs/onboarding.md](docs/onboarding.md) §7.

---

## 15. What this guide does **not** cover

- Paper **Stage 2** (single-turn coding control) — deferred in [ROADMAP.md](ROADMAP.md).
- Writing figures into the paper / filling `[FILL IN …]` in `METHOD.tex`.
- Azure Foundry compat spike — optional alternative to self-hosted vLLM for
  *generation only* (projection still needs a GPU with weights).

When in doubt: [HANDOFF.md](HANDOFF.md) is the project entry point; this file is
the operator checklist to get from installed code to `analysis_report.json`.
