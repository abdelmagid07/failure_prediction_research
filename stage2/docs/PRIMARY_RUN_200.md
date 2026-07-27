# Primary run: 200 random Verified (ROLLOUTS=1)

Operational checklist for the paper’s main generation slice.
Task freeze: `config/selected_instances_200.txt` (from `random_tasks.csv`).

## Expected outcomes (ballpark — not a guarantee)

You are running **base** `Qwen3-32B` + **mini-swe-agent 2.4.5**, not a SWE-finetuned checkpoint.

| Source | Setup | Reported resolve |
|--------|--------|------------------|
| SWE-Lego (arXiv 2601.01426) | Qwen3-32B baseline, OpenHands | ~**23%** |
| SambaNova blog | Qwen3-32B, mini-swe-agent, 100 Verified | ~**15%** |
| Fine-tuned 32B agents | SFT/RL specialists | 50–60% (not you) |

**Plan for ~15–25% resolved on 200** → roughly **30–50 resolved**, **150–170 unresolved** (includes empty patches + `LimitsExceeded`).

Your 3-id smoke (0/3 resolved) is compatible with that rate (wide CI at n=3).

Hard tier in the freeze: 19× `1-4 hours` — expect almost all unresolved there. Django-heavy (~50%) is normal for random Verified.

**Science note:** You need **both** classes for AUROC. At ~20% success you should be fine; if you land &lt;5% resolved, expand or diagnose (context 40k, workers, Azure health).

---

## Pre-flight (do in order)

### Config
`config/mini_swe_qwen.yaml` sets:
- `environment.container_timeout: "8h"` (per-task Docker lifetime; default mini is `2h`)
- `environment.pull_timeout: 1800` (image pull seconds; default mini is `120` — too short for parallel first-wave pulls → `TimeoutExpired`)

Prefer `WORKERS=8` with 8 Azure instances. Confirm both settings appear in `OUTPUT_DIR/mini_swe_resolved.yaml` before a long batch.

### Azure
1. Deploy `qwen3-32b` with **8 model instances** (same deployment name / LB).
2. Playground: one short chat succeeds.
3. Confirm billing awareness: **8 × ~$3.95/hr ≈ $32/hr** while up — delete when idle.

### Laptop (Docker host)
```bash
cd ~/projects/failure_prediction_research   # or your path
source venv/bin/activate
docker info                                 # must work
set -a && source .env && set +a
echo "provider=$MODEL_PROVIDER base=$MODEL_API_BASE name=$MODEL_NAME key_len=${#MODEL_API_KEY}"
# expect: azure, .../openai/v1, openai/qwen3-32b, key_len~84

curl -sS -m 20 -H "Authorization: Bearer $MODEL_API_KEY" \
  "$MODEL_API_BASE/models" | head -c 200   # HTTP 200
```

Optional: `huggingface-cli login` / `HF_TOKEN` (dataset + image metadata).

### Disk / images
SWE-bench images are large. Prefer **≥200 GB** free. First django/sympy pulls dominate early wall time.

### Ramp (do not jump straight to 200×16)
```bash
cd stage2
# 1) 8 ids, 4 workers — sanity
head -8 config/selected_instances_200.txt > /tmp/smoke8.txt
ROLLOUTS=1 WORKERS=4 STEP_LIMIT=60 \
  OUTPUT_DIR=data/trajectories/ramp_smoke8 \
  bash scripts/run_mini_swe_batch.sh /tmp/smoke8.txt

# 2) If healthy: full 200
ROLLOUTS=1 WORKERS=12 STEP_LIMIT=60 \
  OUTPUT_DIR=data/trajectories/verified_200_primary \
  bash scripts/run_mini_swe_batch.sh config/selected_instances_200.txt
```
If no 429s after ~30–60 min, you can stop and restart with `WORKERS=16` **or** leave 12 running (safer). Use `--redo-existing` only if you intentionally want overwrites (our wrapper doesn’t pass it by default — finished ids are typically skipped by mini if present in `-o`).

**tmux/screen required** — this is a multi-hour / overnight job.

```bash
tmux new -s swe200
# ... run command ...
# detach: Ctrl-b d
# reattach: tmux attach -t swe200
```

---

## Monitoring (while it runs)

### A. Live TUI (same pane)
mini’s progress bar: `Overall Progress` + per-instance status (`Pulling/starting`, steps, cost).

Healthy: several instances active when `WORKERS>1`; statuses cycling; not stuck forever on one “Pulling”.

### B. Second pane — throughput
```bash
RUN=data/trajectories/verified_200_primary   # your OUTPUT_DIR

# How many traj files so far?
watch -n 60 'find '"$RUN"' -name "*.traj.json" | wc -l'

# Exit-status mix (updates as mini writes yaml)
watch -n 120 'ls -lt '"$RUN"'/exit_statuses*.yaml 2>/dev/null | head -1; \
  python3 -c "import yaml,glob,sys; fs=sorted(glob.glob(\"'"$RUN"'/exit_statuses*.yaml\"));
print(open(fs[-1]).read() if fs else \"no status file yet\")"'
```

### C. Azure / 429
- Foundry metrics: request rate, errors, GPU util if shown.
- In `minisweagent.log`, grep:
```bash
grep -iE '429|rate limit|timeout|APIConnection|Error' "$RUN/minisweagent.log" | tail -40
```
- **If 429s spike:** lower `WORKERS` (16→12→8). Don’t raise instances beyond what you paid for.
- **If many `APIConnectionError`:** infra — those must be **excluded + regen**, not counted as unresolved.

### D. Context / length risks (40k Azure template)
Long django trajs may hit context or look “stuck”. Check:
```bash
# Huge trajs (bytes) — possible runaway / long context
find "$RUN" -name "*.traj.json" -printf '%s %p\n' | sort -n | tail -10
```
`LimitsExceeded` is expected for hard tasks; empty `model_patch` after `Submitted` is also a failure (we map to unresolved).

### E. Cost clock
Wall-clock estimate: very rough **~1–4 hours per task** wall time with workers, but parallelized. With 12–16 workers, **200 tasks often takes overnight to ~1–2 days**, dominated by Docker pulls early and long trajs late. Azure: **~$32/hr × hours up**.

Stop the deploy when the batch finishes.

---

## What “done” looks like

Under `OUTPUT_DIR`:
- ~200 `*/\*.traj.json` (some missing only if hard crash)
- `preds.json` with 200 keys (some `model_patch: ""`)
- `exit_statuses_*.yaml`, `minisweagent.log`, `mini_swe_resolved.yaml`

Then **on a Docker machine**:

```bash
# Eval
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path "$RUN/preds.json" \
  --run_id verified_200_primary \
  --max_workers 4

# Copy/adapt harness report → $RUN/results.json with resolved + unresolved
# (empty_patch_ids → unresolved; ingest does this if you pass the raw harness JSON)

python -m stage2.trajectories.ingest_batch \
  --traj-dir "$RUN" --format mini-swe-agent \
  --output-dir data/normalized/verified_200_primary
```

Ingest histogram should show genuine statuses only; `N_success` in the ~30–50 band is a good sign; `N_success=0` means investigate before projection.

---

## Red flags → pause

| Signal | Action |
|--------|--------|
| Playground/API 500 | Pause batch; fix Azure; don’t burn GPU $ |
| Sustained 429 | Drop WORKERS |
| All `APIConnectionError` | Infra; fix endpoint; regen later |
| 0 trajs after 1h | Docker/image pull or filter bug |
| Disk full | Free space; mini will fail mid-batch |
| Only empty patches for hours | Check model/tools; inspect one traj |

---

## After primary (not now)

1. Report resolve rate + empty-patch / LimitsExceeded counts  
2. Stage‑1 axis on 32B (if not done) → project → analyses  
3. Later: hand-picked **20×8** within-task contrast  

Command to start the real job (after ramp smoke):

```bash
cd stage2 && set -a && source ../.env && set +a
tmux new -s swe200
ROLLOUTS=1 WORKERS=12 STEP_LIMIT=60 \
  OUTPUT_DIR=data/trajectories/verified_200_primary \
  bash scripts/run_mini_swe_batch.sh config/selected_instances_200.txt
```
