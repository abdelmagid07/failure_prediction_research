# HANDOFF — pick up here (2026-07-27 evening)

**For a new Cursor chat:** attach this file (`@HANDOFF_CURRENT.md`) and say
“continue from this handoff.” Prior long chat transcript (optional context):
`agent-transcripts/ede99c66-681a-4a34-b5b0-3c05aac47c37`.

**Stale docs:** root [`HANDOFF.md`](HANDOFF.md) still describes an older 8B /
60×5 plan. Prefer this file + [`stage2/docs/PRIMARY_RUN_200.md`](stage2/docs/PRIMARY_RUN_200.md)
for the live generation phase. Methodology: [`METHOD.tex`](METHOD.tex) (may
still say 60×5 — update later).

---

## 1. Project in one paragraph

Test whether Jiang et al.’s **value axis** (linear direction in residual stream
encoding “am I on track?”) transfers to **long-horizon SWE-bench agent
trajectories**. Stage 1 axis exists for **Qwen3-8B**. Current generation is
**on-policy Qwen3-32B** via Azure Foundry + **mini-swe-agent 2.4.5**, then later
rebuild/project a faithful 32B axis. Venue: NeurIPS mech-interp workshop style.

---

## 2. Where we left off (LIVE)

### Status
User is mid **primary generation**: **200 random SWE-bench Verified**,
`ROLLOUTS=1`, on a **WSL laptop** with Docker, calling **Azure AI Foundry**
`openai/qwen3-32b` with **8 managed GPU instances** (~$32/hr while up).

### Output directory
```text
stage2/data/trajectories/verified_200_primary/
```
(On the WSL machine: `~/projects/failure_prediction_research/stage2/...`.
Windows Cursor workspace `c:\Users\Abdel\ResearchWork\...` may **not** mirror
traj files — open the **WSL folder in Cursor** if you need to inspect trajs.)

### Last observed board (~22–23 min into a restart; numbers will have moved)
- Progress ~**14/200**
- Exit mix roughly: **Submitted ~7**, **RepeatedFormatError ~5**,
  **CalledProcessError ~2**
- Intermittent **LiteLLM / Azure upstream request timeouts** with **retry +
  exponential backoff** (4s → 32s…). Run was still advancing (trajs saving,
  new containers starting). Treat as “noisy but working” unless Playground dies.
- One task stuck long on **Step 1**: `astropy__astropy-14096` (~22+ min) —
  candidate to `docker stop` that container only (see §6).

### Config already applied (repo)
[`stage2/config/mini_swe_qwen.yaml`](stage2/config/mini_swe_qwen.yaml):
```yaml
environment:
  container_timeout: "8h"   # default mini was 2h → RuntimeError if container dies
  pull_timeout: 1800        # default was 120s → TimeoutExpired on parallel pulls
```
Confirm after each start in `OUTPUT_DIR/mini_swe_resolved.yaml`.

### Recommended concurrency (decision)
- Azure: **8** model instances  
- mini: **`WORKERS=8`** (match GPUs; minimize queued completions)  
- Do **not** use 12 for this setup (oversubscribe + timeout risk).  
- `PRIMARY_RUN_200.md` still shows some `WORKERS=12` examples — prefer **8**.

### Task freeze
- [`stage2/config/selected_instances_200.txt`](stage2/config/selected_instances_200.txt)
- Source CSV: [`random_tasks.csv`](random_tasks.csv) (SQL:
  [`random_tast_selection.sql`](random_tast_selection.sql) — filename typo kept)
- ~200 unique Verified; django-heavy (~50%); ~74 easy / 107 medium / 19 hard

---

## 3. Key decisions (do not reopen without asking)

| Topic | Decision |
|--------|----------|
| Primary N | **200 random** Verified, not old stratified 60 |
| Rollouts | **`ROLLOUTS=1`** for primary (one traj per task). Multi-seed later = hand-picked **20×8** contrast |
| Model | **Qwen3-32B** on Azure first (not self-hosted vLLM for agent calls) |
| Thinking | **ON** (Azure default). Do **not** send `chat_template_kwargs` / `enable_thinking` / `top_k` to Azure |
| Empty patches | Count as **unresolved** (model failure), not exclude |
| Infra crashes | **Exclude + regen** (`APIConnectionError`, `TimeoutExpired`, `CalledProcessError`, `RuntimeError`, etc.) |
| Workers vs GPUs | Prefer **WORKERS = Azure instance count (8)** |

---

## 4. Architecture (generation path)

```
Laptop (WSL): mini-swe-agent + Docker SWE-bench images
       │  OpenAI-compatible chat
       ▼
Azure Foundry Global Managed Compute: openai/qwen3-32b (8 replicas)
```

Provider switch:
- `MODEL_PROVIDER=azure|vllm`
- [`stage2/config/providers/azure.yaml`](stage2/config/providers/azure.yaml)
- Resolver: [`stage2/scripts/resolve_mini_config.py`](stage2/scripts/resolve_mini_config.py)
- Runner: [`stage2/scripts/run_mini_swe_batch.sh`](stage2/scripts/run_mini_swe_batch.sh)

Env (repo root `.env`, gitignored):
```bash
MODEL_PROVIDER=azure
MODEL_API_BASE=https://jonas-nrja-research.services.ai.azure.com/openai/v1
MODEL_NAME=openai/qwen3-32b
MODEL_API_KEY=...   # key_len ~84; rotate if ever pasted in chat
```

Azure quirks validated 2026-07-26:
- Rejects: `chat_template_kwargs`, `enable_thinking`, `top_k`
- Accepts: `temperature`, `top_p`, `seed`
- Reasoning often in `message.reasoning` → ingest maps to `reasoning_content`

Ingest: [`stage2/stage2/trajectories/ingest_batch.py`](stage2/stage2/trajectories/ingest_batch.py)  
Genuine mini exit statuses (keep): `Submitted`, `LimitsExceeded`, `TimeExceeded`,
`RepeatedFormatError`. Everything else (exception class names) = crash exclude.

---

## 5. Incidents already seen (and fixes)

| Symptom | Cause | Fix / policy |
|---------|--------|----------------|
| First wave all `TimeoutExpired` | `pull_timeout: 120` + many parallel large pulls | `pull_timeout: 1800` in `mini_swe_qwen.yaml` |
| `RuntimeError` after long traj | Container `sleep 2h` expired | `container_timeout: "8h"` |
| Completions/Playground **408**, deploy dead | Azure upstream timeout / wedged replicas | Restart deploy; don’t start 200 until Playground OK |
| LiteLLM `Timeout: upstream request timeout` + retries | Long thinking gens under load | Often recovers via backoff; if Playground dies or progress stalls → drop `WORKERS` to 6 or 4 |
| Smoke filter `500 → 3` | `head -8` on file with `#` comments | Use `grep -vE '^\s*(#|$)' ... \| head -8` |
| `RepeatedFormatError` | Model failed valid tool/format 3× | Genuine unresolved — keep |
| `CalledProcessError` | Docker/shell infra | Exclude + regen |
| tmux no scroll | Normal | `Ctrl-b [` then arrows; `q` exit. Optional `tmux set -g mouse on` |

---

## 6. Ops cheatsheet

### Attach / monitor
```bash
tmux attach -t swe200   # or whatever session name
RUN=data/trajectories/verified_200_primary   # from stage2/

find "$RUN" -name "*.traj.json" | wc -l
grep -iE '429|timeout|APIConnection|TimeoutExpired|upstream' "$RUN/minisweagent.log" | tail -40
docker ps | grep minisweagent
```

### Kill one stuck task (batch continues)
```bash
docker ps | grep -i 'astropy-14096'   # or other instance image slug
docker stop <id_or_name>              # or: docker rm -f ...
# Later: rm -rf $RUN/astropy__astropy-14096  and regen
```
If stuck on **Azure/litellm** (Step 1 forever), stopping Docker may not free the
worker until retries exhaust.

### Restart same OUTPUT_DIR (skip finished)
```bash
cd ~/projects/failure_prediction_research/stage2
source ../venv/bin/activate   # or repo venv path
set -a && source ../.env && set +a

ROLLOUTS=1 WORKERS=8 STEP_LIMIT=60 \
  OUTPUT_DIR=data/trajectories/verified_200_primary \
  bash scripts/run_mini_swe_batch.sh config/selected_instances_200.txt
```
Delete instance dirs for infra failures before they will retry cleanly.

### When ~200 trajs + preds.json done
1. **Stop Azure deploy** (stop ~$32/hr).
2. Harness eval → place/adapt report for ingest.
3. ```bash
   python -m stage2.trajectories.ingest_batch \
     --traj-dir data/trajectories/verified_200_primary \
     --format mini-swe-agent \
     --output-dir data/normalized/verified_200_primary
   ```
4. Expect ~**15–25%** resolved (~30–50/200) ballpark for base 32B + mini.

---

## 7. Expected science outcomes (generation)

- Resolve rate ballpark **15–25%** (SWE-Lego ~23% OpenHands; SambaNova mini ~15%).
- Need both success and failure for later AUROC; &lt;5% resolved → diagnose before projecting.
- Empty `model_patch` after `Submitted` → unresolved at ingest.

---

## 8. Not done yet (after this batch)

1. Finish / stabilize **verified_200_primary** (regen infra crashes).
2. SWE-bench harness eval + ingest.
3. Stage‑1 **faithful 32B value axis** — OpenRouter Opus 4.6 ICRL → Colab A100 extract (`enable_thinking=true`). See [`stage1/README.md`](stage1/README.md) § Qwen3-32B and [`stage1/notebooks/stage1_gpu_colab_32b.ipynb`](stage1/notebooks/stage1_gpu_colab_32b.ipynb).
4. Project trajs → analyses (final-step AUROC, task-level CIs, etc.).
5. Later: hand-picked **20×8** within-task contrast.
6. Update `METHOD.tex` / root `HANDOFF.md` for 200 random + 32B + Azure.
7. **Rotate Azure key** if it was ever exposed in chat.

---

## 9. What the next agent should do first

1. Ask user: **Is the 200-run still going?** Progress count? Playground OK?
2. If running: do **not** restart casually; help monitor / kill single stuck containers / interpret exit statuses.
3. If stopped uncleanly: check `OUTPUT_DIR` traj count, delete infra stubs, restart with `WORKERS=8` and confirm `pull_timeout: 1800` in resolved yaml.
4. Do **not** commit unless user asks. Do **not** start projection until ingest has both classes.
5. Prefer opening **WSL path** for live traj inspection.

### Paste-ready prompt for new chat
```text
Read @HANDOFF_CURRENT.md and continue from there.
I'm running (or paused) the Stage-2 primary 200-task SWE-bench generation with
Qwen3-32B on Azure + mini-swe-agent. Help me monitor / finish / ingest next.
Current status: <paste TUI board or traj count / any errors>.
```

---

## 10. File map (high value)

| Path | Role |
|------|------|
| `HANDOFF_CURRENT.md` | **This file** — live handoff |
| `stage2/docs/PRIMARY_RUN_200.md` | Ops checklist for 200-run |
| `stage2/config/mini_swe_qwen.yaml` | Shared mini override + timeouts |
| `stage2/config/providers/azure.yaml` | Azure-safe model kwargs |
| `stage2/config/selected_instances_200.txt` | Frozen task list |
| `stage2/scripts/run_mini_swe_batch.sh` | Batch entrypoint |
| `stage2/stage2/trajectories/ingest_batch.py` | Crash vs genuine labeling |
| `RUN_GUIDE.md` | Broader run guide (may lag decisions) |
| `METHOD.tex` | Methodology SoT (may lag 200/32B) |
