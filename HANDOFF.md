# HANDOFF

Last updated: 2026-07-16. This file is the single entry point for anyone (human
or agent) picking up this project. Read it top to bottom once, then use the
linked docs.

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
- **Benchmark:** SWE-bench (real GitHub bug-fix tasks).
- **Headline so far:** Stage 1 (rebuild + verify the axis) is **done** — 0.87
  AUROC at layers 21/22 on held-out data. The Stage 2 pipeline is **built and
  green on offline tests** but has **not yet been run on real SWE-bench
  trajectories** — that's the next milestone.

Full research framing: [docs/method.md](docs/method.md). Proposal:
[PROPOSAL.tex](PROPOSAL.tex). Plan: [ROADMAP.md](ROADMAP.md).

---

## 2. Where we are right now

### Done
- **Stage 1** — ICRL → activation extract → value axis + AUROC gate. Axis frozen
  as `stage1/data/value_axis_proxy.npy` (dev preset) / `value_axis.npy`
  (default). Shape 36 layers × 4096 hidden. 0.87 AUROC at L21/L22.
- **Stage 2 pipeline plumbing** — end-to-end path exists and passes the offline
  wiring test:
  - `trajectories/` — normalized schema + two parsers + format-aware batch ingest.
  - `extract/project_steps.py` — replays each recorded turn through Qwen3-8B and
    reads the value-axis projection at the last token of assistant output (and of
    tool observations), tagging rows `reasoning` vs `tool_output`.
  - `analyze/` — final-step AUROC, SNR-by-position, noise-by-token-type.
- **Agent migration (this session)** — switched the real-run agent from
  **SWE-agent → mini-swe-agent** (the maintained lightweight successor SWE-bench
  is migrating to). See §4.

### Not done yet
- A **real SWE-bench run** through the mini-swe-agent path (the whole point of
  Week 1). Everything downstream has only been exercised on synthetic fixtures.
- **One live verification** that thinking-mode is actually OFF over the wire (see
  §6, the single most important pre-run check).
- Majority-class baseline surfaced in `analysis_report.json`.
- Stage 3 (projection-vs-position curves, elicited-confidence baseline) and
  Stage 4 (layer localization).

---

## 3. Architecture & data flow

```
stage1/                      # build + verify the value axis (DONE)
stage2/
  config/
    defaults.yaml            # model=Qwen/Qwen3-8B, layer=21, n_layers=36,
                             #   hidden_dim=4096, enable_thinking=false
    presets/dev.yaml         # dev preset -> proxy axis (value_axis_proxy.npy)
    mini_swe_qwen.yaml       # PRIMARY: mini-swe-agent model-override layer
    swe_agent_qwen.yaml      # LEGACY: SWE-agent config (kept until mini verified)
    pilot_instances.txt      # SWE-bench instance ids for a pilot batch
    devbugs_instances.txt    # ids for the local no-Docker dev harness
  scripts/
    run_mini_swe_batch.sh    # PRIMARY generator (mini-swe-agent + Docker)
    run_devbugs_batch.sh     # local dev harness (no Docker, hand-written bugs)
    run_pilot_batch.sh       # LEGACY (SWE-agent)
  stage2/
    trajectories/            # schema.py, parse_swe_traj.py,
                             #   parse_mini_swe_traj.py, ingest_batch.py
    extract/                 # project_steps.py, token_spans.py
    analyze/                 # run_analyses.py + final-step / SNR / token-type
    devbugs/                 # local dev bug-fix harness (was stage2/mini/)
    common/                  # paths, config, projection helpers
tests/integration/           # test_stage1_wiring.sh, test_stage2_wiring.sh
docs/                        # method, setup, onboarding, analyses, walkthrough
reference/                   # FROZEN read-only snapshot of the original pipeline
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
  TrajectoryStep(step_index, messages_before_gen[], assistant_response, observation)
```

---

## 4. The SWE-agent → mini-swe-agent migration (most recent work)

**Why:** SWE-agent 1.x is git-only (PyPI `sweagent` is stuck at 0.0.1) and drags
in swe-rex dependency pain; it's in maintenance mode. mini-swe-agent is the
maintained successor, installs from PyPI, is ~100 lines (trivial to describe in
the paper), scores well on SWE-bench Verified, and lets us set thinking-off in a
clean config block. **The science is unaffected** — both are on-policy Qwen3-8B
on real SWE-bench; the choice is purely engineering.

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
  prompts stay reproducible against a pinned version), sets thinking-off, and
  refuses to launch unless `enable_thinking is False`.
- **Rename:** the project's own toy dev harness `stage2/mini/` → `stage2/devbugs/`
  (package, `run_devbugs_batch.sh`, `devbugs_agent_colab.ipynb`,
  `test_devbugs_catalog.py`, `[devbugs]` extra). This kills the name collision
  with third-party mini-swe-agent. The `MiniInstance` class and `mini_*` instance
  ids were intentionally kept (data identifiers).
- Legacy SWE-agent files (`run_pilot_batch.sh`, `swe_agent_qwen.yaml`,
  `parse_swe_traj.py`) are **kept** and clearly marked LEGACY until the mini path
  is verified end to end on a real run.

---

## 5. Conventions (read before editing)

- **Thinking mode is OFF everywhere, locked.** Qwen3's `enable_thinking` must be
  `false` at generation AND projection, or the recorded tokens no longer match
  what the model computed. `project_steps.py` has a hard guard
  (`assert_thinking_mode_matches`) that stops the run if it detects `<think>`
  blocks under thinking-off. Do not "fix" a guard failure by disabling the guard.
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
| Thinking mode | `enable_thinking=False` everywhere |
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
2. **LIVE thinking-off check (do this before any real run).** Send one completion
   through the resolved mini config and confirm the response has **no `<think>`
   block**. There is a known litellm↔vLLM `extra_body` quirk
   ([litellm#4769](https://github.com/BerriAI/litellm/issues/4769)); if the
   nesting isn't honored, fall back to Qwen's `/no_think` soft-switch in the
   system prompt. This is the one unverified assumption in the whole path.
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
- **vLLM 0.11.0 has no `--default-chat-template-kwargs`** flag, so thinking-off
  cannot be forced at the server level on the pinned version. This is why we do
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
