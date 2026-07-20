# HANDOFF

Last updated: 2026-07-20. This file is the single entry point for anyone (human
or agent) picking up this project. Read it top to bottom once, then use the
linked docs. **Methodology source of truth:** [METHOD.tex](METHOD.tex).

---

## 1. TL;DR — what this project is

We are testing whether a **"value axis"** — a single linear direction in
Qwen3-8B's residual stream (layers ~21–22) that encodes "do I believe I'm on
track?" — **transfers** from the single-turn setting it was built in to
**long-horizon, multi-step SWE-bench agent trajectories**. If a frozen axis,
read off a running agent's internal activations, separates eventual
success from failure, that's a cheap latent monitor for agent failure. Target
venue: a NeurIPS mechanistic-interpretability workshop.

- **Model:** `Qwen/Qwen3-8B` (everything is on-policy — the same model generates
  trajectories and is read internally).
- **Axis source:** Jiang, Kauvar, Lindsey (2026), arXiv
  [2606.17056](https://arxiv.org/abs/2606.17056).
- **Benchmark:** SWE-bench **Verified** (60 tasks × 5 seeds when selection is fixed).
- **Headline so far:** Stage 1 (rebuild + verify the axis) is **done** — 0.87
  AUROC at layers 21/22 (passes METHOD.tex gate: within 0.03 of published ~0.90).
  The Stage 2 pipeline matches METHOD.tex Eq. 1 (mean cosine over generated
  tokens), multi-rollout generation, and task-level stats — but has **not yet
  been run on real SWE-bench trajectories**.

Full research framing: [docs/method.md](docs/method.md) + [METHOD.tex](METHOD.tex).
Proposal: [PROPOSAL.tex](PROPOSAL.tex). Plan: [ROADMAP.md](ROADMAP.md).

---

## 2. Where we are right now

### Done
- **Stage 1** — ICRL → activation extract → value axis + AUROC gate. Axis frozen
  as `stage1/data/value_axis_proxy.npy` (dev preset) / `value_axis.npy`
  (default). Shape 36 layers × 4096 hidden. 0.87 AUROC at L21/L22. Gate floor is
  now `published_auroc - gate_tolerance` = 0.87 (METHOD.tex).
- **Stage 2 pipeline (METHOD.tex-aligned)** — offline wiring test green:
  - `trajectories/` — schema with `task_id` / `seed` / `exit_status`; multi-rollout
    ingest under `r<seed>/`; crash-stub exclusion + regen planner.
  - `extract/project_steps.py` — **mean cosine over `G_t`** (Eq. 1) + `proj_final`
    robustness; multi-layer; optional activations `.npz` for probes.
  - `analyze/` — final-step AUROC + task-level BCa CI + permutation; AUROC by
    position; majority baseline; within-task contrast.
  - `elicit/confidence.py` — post-hoc P(success) elicitation (Stage 3).
  - `probes/fit_probes.py` — task-level CV probe grid (Stage 4).
- **Agent:** mini-swe-agent **2.4.5**, thinking ON, sampling 0.6/0.95/20,
  step budget 60, `ROLLOUTS` seed loop in `run_mini_swe_batch.sh`.

### Not done yet
- A **real SWE-bench Verified run** (pilot → then 60×5 once selection criteria
  are fixed).
- Task selection criteria for the 60 Verified instances.
- Paper Stage 2 single-turn coding control (deferred).
- Live end-to-end projection on GPU with the new readout.

---

## 3. Architecture & data flow

```
stage1/                      # build + verify the value axis (DONE)
stage2/
  config/
    defaults.yaml            # primary_layer=21, step_limit=60, n_bins=5,
                             #   enable_thinking=true
    elicitation_prompt.txt   # verbatim P(success) prompt (paper appendix)
    mini_swe_qwen.yaml       # PRIMARY: mini-swe-agent model-override layer
    pilot_instances.txt      # SWE-bench Verified pilot ids
  scripts/
    run_mini_swe_batch.sh    # PRIMARY generator (ROLLOUTS seed loop, STEP_LIMIT=60)
    list_regens.py           # infra-excluded → fresh-seed regen commands
    serve_qwen.sh            # self-hosted vLLM (AWS / no tunnel)
  stage2/
    trajectories/            # schema (task_id/seed/exit_status), parsers, ingest
    extract/                 # project_steps (Eq. 1 mean-cosine), token_spans
    analyze/                 # stats, final_step, by_position, internal_vs_elicited
    elicit/                  # post-hoc confidence (codebase Stage 3)
    probes/                  # fitted logistic probes (codebase Stage 4)
    devbugs/                 # local no-Docker harness
tests/integration/
docs/                        # method.md mirrors METHOD.tex
```

**The pipeline, in order:**

```
1. GENERATE   run_mini_swe_batch.sh  ->  data/trajectories/mini_swe_run_<ts>/
                 per-instance <id>/<id>.traj.json  +  preds.json
2. LABEL      swebench.harness.run_evaluation on preds.json  ->  results.json
                 (resolved / unresolved per instance)
3. INGEST     ingest_batch --format mini-swe-agent  ->  data/normalized/*.json
                 (normalized, agent-agnostic; crash stubs excluded; labels attached)
4. PROJECT    project_steps.py  (ON A100 — needs raw activations)  ->  projections.parquet
5. ANALYZE    run_analyses.py  ->  analysis_report.json + figures
```

**Normalized schema** (`trajectories/schema.py`) is the contract everything
downstream depends on — it is deliberately **agent-agnostic**, which is why
swapping agents only meant writing one new parser:

```
TrajectoryRecord(trajectory_id, outcome, steps[], n_steps)   # n_steps is derived, never trusted
  TrajectoryStep(step_index, messages_before_gen[], assistant_response, observation,
                 assistant_message)   # assistant_message = native {content, reasoning_content,
                                      #   tool_calls}, authoritative for thinking-on replay
```

---

## 4. The SWE-agent → mini-swe-agent migration (most recent work)

**Why:** SWE-agent 1.x is git-only (PyPI `sweagent` is stuck at 0.0.1) and drags
in swe-rex dependency pain; it's in maintenance mode. mini-swe-agent is the
maintained successor, installs from PyPI, is ~100 lines (trivial to describe in
the paper), scores well on SWE-bench Verified, and lets us set the thinking mode
in a clean config block. **The science is unaffected** — both are on-policy
Qwen3-8B on real SWE-bench; the choice is purely engineering.

**What changed:**
- New parser `parse_mini_swe_traj.py`. mini stores one running `messages` list
  for the whole run; the parser reconstructs per-step context from it (each
  assistant message = one step; everything before it = `messages_before_gen`;
  the following user/tool message = `observation`). It drops mini's synthetic
  trailing `exit` message.
- `ingest_batch.py` is now format-aware: `--format {mini-swe-agent, swe-agent}`
  (mini default). **Crash guard:** mini sets `exit_status` to the exception class
  name on any infra crash (e.g. `APIConnectionError`), so we **allowlist** the
  genuine terminal statuses (`Submitted`, `LimitsExceeded`, `TimeExceeded`,
  `RepeatedFormatError`) and exclude everything else as a crashed stub before
  labeling — otherwise a dropped-tunnel crash gets mislabeled as a task failure.
  Ingest prints an `exit_status` histogram so the real vocabulary is visible.
- `config/mini_swe_qwen.yaml` + `scripts/run_mini_swe_batch.sh` — the runner
  deep-merges our model overrides onto the *installed* mini base config (so task
  prompts stay reproducible against a pinned version), sets thinking-on, and
  refuses to launch unless `enable_thinking is True`.
- **Rename:** the project's own toy dev harness `stage2/mini/` → `stage2/devbugs/`
  (package, `run_devbugs_batch.sh`, `devbugs_agent_colab.ipynb`,
  `test_devbugs_catalog.py`, `[devbugs]` extra). This kills the name collision
  with third-party mini-swe-agent. The `MiniInstance` class and `mini_*` instance
  ids were intentionally kept (data identifiers).
- Legacy SWE-agent files (`run_pilot_batch.sh`, `swe_agent_qwen.yaml`,
  `parse_swe_traj.py`) are **kept** and clearly marked LEGACY until the mini path
  is verified end to end on a real run.

---

## 4b. The thinking-OFF → thinking-ON migration (2026-07-17)

**Why:** maintainer's architectural decision — Stage 2 trajectories are now
generated AND projected with Qwen3 thinking **ON**, so the value axis is read off
the model's full reasoning state. Stage 1 and the frozen axis are **not** touched
(the axis was built thinking-off), so the Stage 2 result is explicitly a
**cross-mode transfer** claim; the paper must say so (see [docs/method.md](docs/method.md)).

**What changed:**
- **Serving** — `serve_qwen_colab.ipynb` launches vLLM with
  `--enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3`.
  The reasoning parser splits the generated `<think>` text into
  `reasoning_content`; without it, thinking-on + tool calls strands the tool call
  inside the think text (vllm#20611). Use `deepseek_r1` if a build lacks `qwen3`.
- **Configs** — `defaults.yaml` `enable_thinking: true`; `mini_swe_qwen.yaml` and
  `qwen_mini.yaml` send `chat_template_kwargs.enable_thinking: true`; the runner
  guard now refuses to launch unless thinking is ON.
- **Schema** (`trajectories/schema.py`) — `TrajectoryStep` gains
  `assistant_message`, the native turn (`content` + `reasoning_content` +
  `tool_calls`). `messages_before_gen` is now native dicts (assistant
  `tool_calls`, `tool`-role `tool_call_id`), so replayed context matches
  generation token-for-token. `assistant_response` stays as a flattened
  convenience string (`<think>` + content).
- **Parser** (`parse_mini_swe_traj.py`) — preserves the structured fields instead
  of flattening to `content`; drops provider `extra`; history keeps tool calls
  but drops prior-turn reasoning (Qwen3 strips it from context).
- **Projection** (`project_steps.py`) — replays `messages_before_gen + [native
  assistant turn]` through the Qwen3 template (`enable_thinking=True`), which
  re-emits `<think>...</think>` + content + `<tool_call>...`. Reads the last token
  of the whole assistant turn via the new `last_token_of_final_assistant`
  (robust to the trailing tool call). The thinking guard is now bidirectional,
  and a one-time **render-fidelity check** aborts if the template fails to
  reproduce `reasoning_content` / `tool_calls`. `--no-enable-thinking` still
  projects legacy thinking-off data. Validated against the real Qwen3 tokenizer.
- **devbugs** (`agent_loop.py`) — toy harness sends thinking-on and folds the
  server-split `reasoning_content` back into the recorded turn.

---

## 5. Conventions (read before editing)

- **Thinking mode is ON in Stage 2 (decision 2026-07-17).** Qwen3's
  `enable_thinking` must be `true` at generation AND projection, or the recorded
  tokens no longer match what the model computed. `project_steps.py` has a
  bidirectional guard (`assert_thinking_mode_matches`) plus a one-time
  render-fidelity check; do not "fix" a guard failure by disabling the guard.
  (Stage 1 built the axis thinking-OFF and is frozen — Stage 2 is a cross-mode
  transfer test, not a bug. Legacy thinking-off data projects with
  `--no-enable-thinking`.)
- **On-policy only.** The model that generates a trajectory is the model we read
  internally. Never mix generators.
- **The frozen axis is frozen.** No refitting on agent data.
- **`reference/` is read-only.** It's a snapshot of the original pipeline for
  comparison. Never edit it; never let searches/renames touch it.
- **Generated data is gitignored** (`data/trajectories/`, `data/normalized/`,
  `data/*_smoke*`, etc.). Keep test artifacts under those roots.
- **Git is the human's job.** The maintainer commits and pushes themselves — do
  not run `git commit`/`git push`. Staging a rename via `git mv` is fine; the
  current renames are staged but uncommitted, awaiting review.
- **Do not touch the `transformers` version bound** in `stage2/pyproject.toml`
  (maintainer's explicit call, even though Qwen3 technically wants ≥4.51).
- **The code IS the methods section.** Explain what/why/what-changed as you go,
  and keep names and comments paper-ready — the maintainer writes the paper
  directly from this codebase.
- **`devbugs` ≠ `mini-swe-agent`.** `stage2/devbugs/` is our home-grown toy
  no-Docker harness (12 hand-written bugs) for smoke tests. `mini-swe-agent` is
  the third-party real SWE-bench agent. They are unrelated despite the old name.

**Locked decisions** (from [docs/method.md](docs/method.md)):

| Decision | Value |
|----------|-------|
| Model | `Qwen/Qwen3-8B` |
| Thinking mode | Stage 2: `enable_thinking=True`. Stage 1 axis built `=False` → cross-mode transfer |
| Projection layer | 21 (verify from Stage 1 manifest) |
| Decoder layers | 36 (indices 0–35) |
| Trajectories | On-policy only |
| Read position | Last token of assistant output per step |
| Axis | Frozen after Stage 1; no refit |
| Benchmark | SWE-bench (primary); devbugs env for development |
| Baseline | Majority-class, not 0.5 |

---

## 6. Next steps (ordered)

1. **Finish the local environment** (WSL2 + Docker + `pip install -e ".[swe]"`).
   Full instructions: [docs/onboarding.md](docs/onboarding.md).
2. **LIVE thinking-on + tool-call check (do this before any real run).** Send one
   completion through the resolved mini config and confirm the response carries a
   `<think>` block / `reasoning_content` **and** a structured `tool_calls` array.
   There is a known litellm↔vLLM `extra_body` quirk
   ([litellm#4769](https://github.com/BerriAI/litellm/issues/4769)), and
   thinking-on + tool calls only parse when the server runs `--reasoning-parser`
   (vllm#20611). Two curls in [docs/onboarding.md](docs/onboarding.md) §2. This is
   the one unverified assumption in the whole path.
3. **Run a small pilot** (3–5 ids from `config/pilot_instances.txt`) through
   `run_mini_swe_batch.sh`, then the swebench harness for labels, then
   `ingest_batch --format mini-swe-agent`. **Inspect the exit_status histogram**
   the ingest prints and tune `--genuine-statuses` if a new genuine status shows.
4. **Project + analyze on the A100** and confirm real `.traj.json` files flow all
   the way to `analysis_report.json`. Fix whatever breaks on real data.
5. **Scale up:** the ~20-instance pilot list, then 150+ for the real Stage 2
   result. Report N_success / N_failure; expand ids if successes are thin.
6. **Add the majority-class baseline** to `analyze/final_step.py` and surface it
   in `analysis_report.json` (currently missing; the paper needs it).
7. **Stage 3 / Stage 4** — see [ROADMAP.md](ROADMAP.md) weeks 3–5.

---

## 7. Known issues & gotchas

- **litellm→vLLM `extra_body`** — see step 2 above. Unverified until a live run.
- **Thinking-on needs a reasoning parser for tool calls.** With
  `--tool-call-parser hermes` alone, a thinking-on turn can leave the tool call
  stranded inside the `<think>` text and `tool_calls` comes back empty
  (vllm#20611). The serve notebook adds `--reasoning-parser qwen3` to split the
  think text out. If a build lacks `qwen3`, use `deepseek_r1`.
- **vLLM 0.11.0 has no `--default-chat-template-kwargs`** flag, so thinking mode
  cannot be forced at the server level on the pinned version. This is why we set
  it per-request via the model config instead. Do not add that server flag — it
  breaks boot on 0.11.0.
- **`devbugs_agent_colab.ipynb` smoke run uses instance id `mini_add_001`**,
  which is NOT in the catalog (ids start at `mini_eventbus_001`). The single
  smoke run will fail "Unknown instance ids" until that id is corrected. Pre-
  existing; left for the maintainer to decide the right id.
- **`transformers` bound** is intentionally below what Qwen3 nominally wants — do
  not bump it.
- **Two "mini" names** historically — see the `devbugs` note in §5.

---

## 8. How to confirm you're set up

```bash
# Offline, no GPU — should both print "... wiring test passed."
bash tests/integration/test_stage1_wiring.sh
bash tests/integration/test_stage2_wiring.sh

# The stage 2 test exercises the PRIMARY path: mini parse -> ingest
# (crash stub excluded, 1 success / 1 failure) -> projections -> analyses.
```

If those pass, the code is wired correctly and you only need the live compute
(Colab vLLM + local Docker) to start generating real data.

---

## 9. Doc map

| Doc | What's in it |
|-----|--------------|
| [HANDOFF.md](HANDOFF.md) | This file — start here |
| [docs/onboarding.md](docs/onboarding.md) | Full environment setup, deps, running the pipeline, troubleshooting |
| [docs/setup.md](docs/setup.md) | Terse setup reference (install, notebooks, artifacts) |
| [docs/method.md](docs/method.md) | Research question, locked decisions, null-result policy |
| [docs/stage2_walkthrough.md](docs/stage2_walkthrough.md) | Plain-language walkthrough of Stage 2 |
| [docs/analyses.md](docs/analyses.md) | How to read the analysis outputs |
| [ROADMAP.md](ROADMAP.md) | Week-by-week plan and current status |
