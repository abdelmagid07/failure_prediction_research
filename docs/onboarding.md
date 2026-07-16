# Onboarding — from zero to your first result

This is the hands-on setup and workflow guide. If you only want the short version
(what installs what), see [setup.md](setup.md). For *why* the project exists and
the research framing, see [method.md](method.md) and the top-level
[HANDOFF.md](../HANDOFF.md).

> **Mental model.** Neural-net inference is expensive and runs on a **remote
> GPU** (Colab A100 serving Qwen3-8B via vLLM behind a public tunnel). The
> **agent + Docker** run on **your local machine** (WSL2) and call that remote
> endpoint for every model step. The GPU **projection** step also runs on Colab.
> Your laptop never runs the model itself — it orchestrates and runs the
> SWE-bench Docker containers.

```
        your machine (WSL2)                         Colab A100
   ┌─────────────────────────────┐          ┌──────────────────────────┐
   │ mini-swe-agent + Docker      │  HTTPS   │ vLLM serving Qwen3-8B     │
   │ (runs SWE-bench tasks) ──────┼────────► │ + cloudflared tunnel      │
   │                              │  tunnel  │                           │
   │ ingest / analyses (CPU)      │          │ project_steps.py (GPU)    │
   └─────────────────────────────┘          └──────────────────────────┘
```

---

## 0. Prerequisites

| Requirement | Why | Notes |
|-------------|-----|-------|
| Python 3.11+ | everything | `python --version` |
| Git | clone + review | maintainer commits/pushes; contributors branch + PR |
| WSL2 (Windows) or Linux/macOS | run the agent + Docker | the maintainer develops on Windows + WSL2 |
| Docker | SWE-bench task execution | must be reachable from WSL2 (`docker info`) |
| A Colab account with A100 access | serve Qwen, run projection | Colab Pro / pay-as-you-go for A100 |
| (Stage 1 only) Anthropic API key | faithful ICRL generation | only if regenerating the axis with `--backend anthropic` |

You do **not** need a local GPU. You do **not** need to download Qwen locally for
generation (it's served on Colab).

---

## 1. Clone and install the Python packages

There are two installable packages, `stage1/` and `stage2/`, each with its own
`pyproject.toml`. Use a single virtualenv.

```bash
git clone <repo-url> failure_prediction_research
cd failure_prediction_research

python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -e stage1
pip install -e stage2
```

Then add the optional extras you need:

```bash
# Real SWE-bench runs: installs mini-swe-agent (the agent) + swebench (the
# evaluation harness that turns predictions into resolved/unresolved labels).
pip install -e "stage2[swe]"

# Local no-Docker dev harness (12 hand-written bugs) for offline smoke tests.
pip install -e "stage2[devbugs]"
```

> **Note on `[swe]`:** it installs **mini-swe-agent**, not the old SWE-agent. The
> old `sweagent` PyPI package is a dead 0.0.1 stub — ignore any doc that says
> `pip install sweagent`. The legacy SWE-agent path is kept in the repo only
> until the mini path is verified end to end.

> **Do not bump `transformers`.** Its version bound in `stage2/pyproject.toml` is
> intentionally set by the maintainer. Leave it alone even if a tool suggests
> raising it for Qwen3.

### Verify the install (offline, no GPU)

```bash
bash tests/integration/test_stage1_wiring.sh
bash tests/integration/test_stage2_wiring.sh
```

Both should end with `... wiring test passed.` The Stage 2 test runs the **real
primary path** on synthetic fixtures: parse a mini-swe-agent trajectory → ingest
(a crash stub is excluded, 1 success / 1 failure labeled) → projections →
analyses. If this passes, your code environment is correct and only the live
compute remains.

---

## 2. Set up the remote model server (Colab A100)

1. Open `stage2/notebooks/serve_qwen_colab.ipynb` in Colab, set the runtime to
   **A100 GPU**, and run it top to bottom.
2. It installs a CUDA-matched vLLM, serves `Qwen/Qwen3-8B`
   (`--served-model-name Qwen3-8B`), and opens a **cloudflared tunnel**. The
   notebook waits for the public URL and errors out loudly if the tunnel or
   server fails to come up (rather than silently printing `None`).
3. Copy the printed tunnel URL. Your endpoint is that URL **with `/v1`
   appended**, e.g. `https://something.trycloudflare.com/v1`.
4. **Keep the Colab tab open** for the whole run — closing it kills the tunnel.

Sanity-check the endpoint from your machine:

```bash
export MODEL_API_BASE="https://<your-tunnel>.trycloudflare.com/v1"
curl -fsS "$MODEL_API_BASE/models"        # should list Qwen3-8B
```

### The one required manual check: thinking is OFF over the wire

Thinking mode **must** be off (see [method.md](method.md) — it's a locked
decision, and mismatched thinking corrupts every projection). We set it via the
agent's request config, but there is a known litellm↔vLLM `extra_body` quirk
([litellm#4769](https://github.com/BerriAI/litellm/issues/4769)), so verify it
once against the live endpoint before generating any real data:

```bash
curl -s "$MODEL_API_BASE/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "Qwen3-8B",
        "messages": [{"role":"user","content":"Say hi in one word."}],
        "chat_template_kwargs": {"enable_thinking": false}
      }' | grep -c "<think>"     # want 0
```

If you see a `<think>` block, the fallback is Qwen's `/no_think` soft-switch in
the system prompt. Do **not** try to fix this with the vLLM server flag
`--default-chat-template-kwargs` — it isn't present in the pinned vllm 0.11.0 and
adding it breaks server boot.

---

## 3. Two ways to generate trajectories

### (a) Local dev harness — no Docker, no SWE-bench (fast smoke test)

Use this to shake out the endpoint and the pipeline without Docker. It runs 12
hand-written Python bug-fix tasks in temp dirs and emits SWE-agent-format `.traj`
files.

```bash
export MODEL_API_BASE="https://<tunnel>/v1"
cd stage2
bash scripts/run_devbugs_batch.sh                       # all instances
bash scripts/run_devbugs_batch.sh --instance-id mini_eventbus_001   # one instance
```

Then ingest with the swe-agent format (that's what the devbugs harness writes):

```bash
python -m stage2.trajectories.ingest_batch \
  --traj-dir data/trajectories/devbugs_run_<ts> --format swe-agent
```

### (b) Real SWE-bench — mini-swe-agent + Docker (the paper data)

Requires Docker running and reachable from WSL2.

```bash
export MODEL_API_BASE="https://<tunnel>/v1"
cd stage2
# runs the ids in config/pilot_instances.txt (or pass your own file / omit for full subset)
bash scripts/run_mini_swe_batch.sh config/pilot_instances.txt
```

Useful env vars (all optional): `SUBSET` (default `verified`), `SPLIT` (default
`test`), `WORKERS` (default 1), `STEP_LIMIT`, `MODEL_NAME` (default
`hosted_vllm/Qwen3-8B`), `OUTPUT_DIR`, `SKIP_PREFLIGHT=1`.

The runner deep-merges `config/mini_swe_qwen.yaml` (our model overrides,
including thinking-off) onto the installed mini-swe-agent base config, writes a
`*_resolved.yaml` you can inspect, refuses to launch unless thinking is off, and
produces per-instance `<id>/<id>.traj.json` plus a `preds.json`.

---

## 4. Label the SWE-bench predictions

mini-swe-agent produces predictions but **not** pass/fail labels — the SWE-bench
harness does that by applying each patch and running the repo's tests:

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path data/trajectories/mini_swe_run_<ts>/preds.json \
  --run_id my_run_id
```

Place the harness report (with `resolved` / `unresolved` lists) at
`data/trajectories/mini_swe_run_<ts>/results.json`. Ingest reads those keys.

---

## 5. Ingest → normalize → label

```bash
python -m stage2.trajectories.ingest_batch \
  --traj-dir data/trajectories/mini_swe_run_<ts> --format mini-swe-agent
```

This writes normalized, agent-agnostic JSON to `data/normalized/`, attaches the
resolved/unresolved label to each trajectory, and **excludes crashed-run stubs**
(infra/transport errors) so they don't pollute the failure labels.

**Always read the `exit_status` histogram it prints.** It's the safety net: mini
encodes crashes as the exception class name (e.g. `APIConnectionError`), which we
treat as "exclude", while genuine attempts are `Submitted` / `LimitsExceeded` /
`TimeExceeded` / `RepeatedFormatError`. If a new *genuine* status appears in the
histogram, add it via `--genuine-statuses` so it isn't wrongly excluded.

---

## 6. Project (on the A100) → analyze

Projection reads **raw residual-stream activations**, so it must run on the GPU
that has the model — the API endpoint cannot give you activations. Run it on
Colab (`stage2/notebooks/project_and_analyze_colab.ipynb`), or on the A100
directly:

```bash
# On the A100, with normalized trajectories + the frozen axis available:
python -m stage2.extract.project_steps \
  --traj-dir data/normalized \
  --axis-path ../stage1/data/value_axis_proxy.npy \
  --layer 21
# thinking stays OFF (default); project_steps will HARD-STOP if it detects a
# thinking-on trajectory, because that would misalign the activations.

python -m stage2.analyze.run_analyses \
  --projections data/projections.parquet \
  --output-dir data/report
```

Outputs: `projections.parquet` (one row per measured token, tagged `reasoning` vs
`tool_output`, with `rel_pos`), and `analysis_report.json` + figures. How to read
them: [analyses.md](analyses.md).

---

## 7. Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `curl $MODEL_API_BASE/models` fails | Colab tab closed, tunnel rotated, or missing `/v1` suffix. Re-copy the URL from the serve notebook. |
| Runner: "mini-extra not found" | `pip install -e ".[swe]"` in the active venv. |
| Runner: "Docker is not reachable" | Start Docker Desktop / the daemon; ensure WSL2 integration is on. `docker info` must succeed. |
| Response contains `<think>` | thinking-off not honored over the wire — see §2 (fallback: Qwen `/no_think`). |
| vLLM on Colab won't start / CUDA mismatch | the serve notebook reinstalls a CUDA-matched vLLM; re-run that cell. Colab sometimes ships a stale binary. |
| Ingest excludes trajectories you expected to keep | check the printed `exit_status` histogram; add genuine statuses via `--genuine-statuses`. |
| Only one outcome class after ingest | expected on tiny pilots; expand the instance list until you have both successes and failures. |
| `project_steps` aborts with "Thinking-mode mismatch" | the trajectories were generated thinking-ON. Regenerate thinking-off (don't bypass the guard). |
| devbugs smoke run: "Unknown instance ids: ['mini_add_001']" | that id isn't in the catalog (ids start at `mini_eventbus_001`); use a real id. |

---

## 8. Where things live

- **Config:** `stage2/config/` — `defaults.yaml` (model/layer/dtype),
  `mini_swe_qwen.yaml` (primary agent overrides), `presets/dev.yaml` (proxy axis
  for dev).
- **The axis:** `stage1/data/value_axis_proxy.npy` (dev) /
  `value_axis.npy` (default). Frozen — never refit on agent data.
- **Generated data:** `stage1/data/`, `stage2/data/` — **gitignored**. Don't
  commit artifacts.
- **Read-only baseline:** `reference/` — the original pipeline snapshot. Never
  edit it.

When in doubt about conventions (thinking-off, on-policy, git ownership, naming),
[HANDOFF.md §5](../HANDOFF.md) is authoritative.
