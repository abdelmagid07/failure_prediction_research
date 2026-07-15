# Basic language walkthrough (stage 2 focused)

## The core idea 

When a language model reads text, it doesn't just output words — internally, at
each step, it holds a big list of numbers representing "what it currently
understands." For Qwen3-8B that list is **4096 numbers per token**, and the model
has **36 layers**, so there are 36 such lists stacked up as text passes through
it. These are the model's internal *activations* (or "hidden state"). Think of
them as the model's private scratch notes.

A prior paper (Jiang, Kauvar, Lindsey, 2026, arXiv 2606.17056) found that inside
those 4096 numbers there is **one particular direction** that tracks whether the
model "thinks it's on the right track." Measure how much the model's notes point
along that direction and you get a single number that correlates with the model
doing well vs. badly. They call it the **value axis**. It is literally a fixed
table of numbers (36 layers × 4096), computed once and then frozen.

They found this on **simple, single-question tasks**. This project asks one
question: does that same frozen direction still work when read off a model doing
something much harder and longer — a multi-step software-bug-fixing session?

- **If yes:** the "am I doing well?" signal survives into real agent work (useful
  for catching failures early).
- **If no:** it scopes the original result as narrow.
- **If partial:** characterize the degradation.

All three outcomes are publishable. Everything in the code is machinery to answer
this.

---

## Stage 1 — quick summary

Stage 1 **builds and validates the value axis, then freezes it.**

- Take conversations where the model is either "on track" or "off track" (the
  "ICRL" data).
- Run the model over them and record the internal notes (activations) per layer.
- Compute the value axis as **the difference between the average notes of the
  on-track examples and the average notes of the off-track examples** — the
  direction pointing from "failing" toward "succeeding" (`build_axis.py`).
- Check how well that direction separates *held-out* examples using **AUROC**, a
  0-to-1 score where 0.5 = coin flip and 1.0 = perfect separation
  (`eval_auroc.py`). We get **0.87** at layers 21–22 — a solid signal.
- Save as `value_axis.npy` (dev preset: `value_axis_proxy.npy`) and never change
  it again.

For Stage 2 that is all that matters: **Stage 1 produced a frozen 36×4096 table,
and layer 21 is the best row.** Stage 2 only *uses* this file.

---

## Stage 2 — deep dive

Three jobs, mapped onto the folders:

1. **Get trajectories** (`mini/`, plus the SWE-bench scripts) — recordings of the
   model doing multi-step coding tasks.
2. **Read the axis off them** (`trajectories/`, `extract/`) — per step, run the
   model, grab its internal notes, measure against the frozen axis.
3. **Analyze** (`analyze/`) — do the success-vs-failure numbers actually separate?

Four words, in plain terms:

- **Trajectory** — one complete recording of the model trying to fix one bug:
  every message it saw, everything it said, every command result, and whether it
  ultimately succeeded.
- **Step** — one turn in the back-and-forth (model reads context → writes a
  command → gets output).
- **Projection** — the single number you get by measuring the model's internal
  notes against the value axis at one token. Higher = internals look more "on
  track."
- **Outcome / label** — did this trajectory actually fix the bug? 1 = resolved,
  0 = unresolved. Comes from *running the tests*, not from the model.

### Job 1 — getting trajectories

Two ways to produce trajectories; both write the **same `.traj` file format**, so
the rest of the pipeline doesn't care which was used.

**(a) The real way: SWE-bench + SWE-agent** (`scripts/run_pilot_batch.sh`).

- **SWE-bench** is a standard benchmark: real GitHub bugs, each with the broken
  code and a hidden test that passes only if the bug is truly fixed.
- **SWE-agent** is a program that *drives* the model to fix one such bug: it shows
  the model the issue, lets it run shell commands in a real copy of the repo
  (inside Docker for isolation), feeds command output back, and loops until the
  model submits or gives up.
- This is the disk-heavy, Docker-required path; the model itself runs remotely on
  a Colab GPU (the script just talks to it over the network). This produces the
  actual paper results.

**(b) The development stand-in: the mini environment** (`mini/`). Exists so the
whole pipeline can be developed and tested on a laptop with **no Docker and no
SWE-bench download**. A tiny, self-contained imitation:

- `catalog.py` — **12 hand-written buggy Python programs** (event bus, paginator,
  LRU cache, CSV parser, …), graded easy/medium/hard. Each is a small broken file
  plus a `pytest` test that passes only when fixed. A miniature SWE-bench.
- `sandbox.py` — `materialize_repo` writes one buggy program to a temp folder;
  `run_command` runs a shell command there and captures output (with a timeout
  and output-size cap so a runaway command can't hang or flood the log).
- `agent_loop.py` — the loop: build the messages the model sees (system prompt +
  the issue + history), ask the model for its next move, run the requested
  command, feed the result back, and after each step **run the tests to check if
  it's fixed** (`evaluate_instance`). Stops when tests pass or the step limit is
  hit, then writes the `.traj` file.
- `parse_action.py` — the model replies with prose reasoning followed by a command
  in a fenced code block; this splits the "thought" from the "action."
- `evaluate.py` — runs the test command; `write_results_json` records which
  instances passed (`resolved_ids`) and which failed (`unresolved_ids`).
  **This is where labels come from.**
- `run_batch.py` — runs a whole list of mini instances and writes all `.traj`
  files plus one `results.json`.

Key point: **the mini agent produces `.traj` files that look exactly like
SWE-agent's**, so it's a drop-in rehearsal. The paper's results come from
SWE-bench; the mini env is only for verifying the plumbing. Caveat worth
remembering: the *same* model both generates and is measured ("on-policy"), a
locked decision in `method.md`.

### What a `.traj` file contains

See `tests/fixtures/sample.traj`. It's a list of steps; each step has:

- `query` — the full list of messages the model *saw* right before this turn
  (system prompt, the issue, and all prior back-and-forth),
- `response` — what the model said (reasoning + command),
- `observation` — what the command printed back,

and at the end an `info` block with the `instance_id` and exit status. This is a
raw, tool-specific format — which is why the next piece exists.

### Job 2, part 1 — `trajectories/` (normalize + label)

Converts raw `.traj` files into a clean standard form and attaches labels.

**`schema.py` — the standard format everything else reads.**

- `TrajectoryStep`: `step_index`, `messages_before_gen` (messages the model saw),
  `assistant_response` (what it said), `observation` (what came back).
- `TrajectoryRecord`: `trajectory_id`, `outcome` (0/1), the list of steps,
  `n_steps`.
- Plus `save_trajectory` and `load_trajectories_from_dir`.

Why keep this separate: everything downstream depends only on *this*, not on
SWE-agent. Swap agent tools later and you only rewrite the parser, not the
analysis. **Design choice for the methods section:** `n_steps` is always
recomputed from the actual number of steps, never trusted from the input — the
analysis divides by it to compute "how far along" a step is, so a wrong value
would distort that axis.

**`parse_swe_traj.py` — raw `.traj` → normalized record.** Maps each raw step onto
the clean schema (`query` → `messages_before_gen`, `response` →
`assistant_response`, `observation` stays). The subtlety: the **outcome is not in
the `.traj` file** — a trajectory is "what the model did"; whether it *worked* is
a separate verdict from running the tests. So the outcome is passed in, keeping
"what happened" and "did it succeed" cleanly separated. It also defensively
flattens message content that arrives as a list of blocks instead of a plain
string (real SWE-agent output sometimes does this).

**`ingest_batch.py` — do a whole run at once and attach labels.** The command you
run after generating trajectories. It:

- scans a run folder for all `.traj` files (flat, like the mini agent, or
  one-subfolder-per-instance, like real SWE-agent),
- reads `results.json` to learn which instance IDs were resolved (1) vs
  unresolved (0),
- writes one clean normalized JSON per trajectory into `data/normalized/`,
- prints **how many successes and failures** and warns if either class is empty
  (you can't measure separation with only one class), and writes a manifest
  recording exactly what was ingested and from where.

**For the paper:** this module is the "data preparation" section. The two
sentences that matter: *trajectories are normalized to a model-agnostic per-step
schema, and each is labeled resolved/unresolved by the benchmark's own test-based
verdict (not the model's self-report).*

### Job 2, part 2 — `extract/` (read the axis off a trajectory)

The heart of Stage 2, and the only part that needs the GPU. Goal: per step,
produce the projection number. `project_steps.py` orchestrates it with three
helpers.

**Which token to measure.** When the model processes a step, it reads the whole
conversation so far — hundreds of tokens. We don't want an average over all of
them; we want the internal state at a *meaningful moment*. Locked decision: **the
last token of the model's own response at each step** — right after it finished
"thinking" about what to do, when its "am I on track?" state is most fully
formed. The code reads two kinds of position per step and tags them:

- `reasoning` — last token of the model's own response (the primary signal),
- `tool_output` — last token of the command output it received (secondary,
  expected to be noisier).

The pieces:

- **`stage1/common/chat.py`** — formats a message list into the single exact
  string the model expects (Qwen's chat template), with `enable_thinking=False`
  (Qwen has an optional verbose "thinking" mode; turned off everywhere for
  consistency — a locked decision).
- **`extract/token_spans.py`** — after formatting, the text is one long string but
  the model works in *tokens* (text chunks). This does the bookkeeping to answer
  "which token number is the last character of the model's response?"
  (`last_token_of_suffix`) and "which token ends this observation?"
  (`last_token_of_message_content`). Fiddly string-to-token index matching;
  nothing conceptual, just necessary so we grab the right position.
- **`stage1/common/hooks.py` → `LayerActivationCapture`** — how we get the internal
  notes out. Normally a model only gives you output text; the per-layer
  activations are hidden. A "hook" is a listener attached to each of the 36 layers
  so that, when the model processes text, we snapshot the 4096 numbers at every
  layer. After one forward pass we have all 36 layers for every token, and pick
  layer 21.
- **`common/projection.py`** — takes the 4096 numbers at the chosen token+layer and
  measures them against the value axis using **cosine similarity** (how aligned
  two directions are, −1 to +1, ignoring magnitude). That number is the
  projection.

So `project_steps.py`, per trajectory, per step: format the messages → find the
right token → run the model with capture hooks → pull layer-21 notes at that
token → project onto the frozen axis → record a row. Each row is
`{trajectory_id, outcome, step_index, n_steps, rel_pos, projection, token_type,
layer}`, where `rel_pos` is how far into the trajectory the step is (0 = start,
1 = end). All rows go into `projections.parquet` — a table, one row per measured
token. **This table is the raw material for every figure in the paper.**

### Job 3 — `analyze/` (does the signal separate success from failure?)

`run_analyses.py` reads `projections.parquet`, runs three analyses, and writes
`analysis_report.json` plus two figures.

**`final_step.py` — the headline result.** For each trajectory, take the
projection at its **last step**, then ask whether successful trajectories score
systematically higher (or lower) than failed ones. Measured with AUROC; draws the
two histograms (`final_step_separation.png`). This directly answers the proposal's
Stage 2 question: *does the frozen axis separate resolved from unresolved at the
last step?* It reports `separability = max(auroc, 1 − auroc)`: if the axis points
the "wrong way" (failures score higher), an AUROC of 0.1 is still a strong signal
— 0.9 of separation, flipped. `separability` captures "how much information,
regardless of sign."

**`signal_to_noise.py` — is the signal real or just noise?** The honesty check.
Split steps into bins by how far along they are (early → late); in each bin
compute **how far apart the success and failure averages are, divided by how
spread-out the values are within each group** (essentially Cohen's *d*). A ratio
near or above 1 means the between-group gap is as big as the within-group noise —
trustworthy. Below 1 means the groups overlap too much to conclude anything. The
"late bin" value is the headline (near the end of the trajectory). This is what
`method.md` insists on: a null result only counts if the measurement was
sensitive enough — distinguishing "the axis genuinely doesn't separate outcomes"
from "our sample is just too noisy to tell."

**`noise_by_token_type.py` — sanity check on the read position.** Compares how
noisy projections are on `reasoning` vs `tool_output` tokens. Expectation:
reasoning tokens are cleaner (the model's own coherent words), tool outputs (raw
file dumps, tracebacks) noisier. If that holds, it justifies reading the signal on
reasoning tokens.

`run_analyses.py` bundles all this into `analysis_report.json` with counts
(`n_success`, `n_failure`), final-step AUROC, late-bin signal-to-noise,
per-token-type noise, and a short plain-English `interpretation` string.

> Note: the offline wiring test reports `final_step_auroc: 1.0`, but that is
> **synthetic mock data designed to separate perfectly**. It proves the machinery
> runs end-to-end, not a real result.

---

## One-paragraph version

Stage 1 found a frozen direction in Qwen's internals that means "I'm on track,"
validated at 0.87 AUROC on simple tasks. Stage 2 records the model doing
multi-step bug fixes (real SWE-bench, or the mini stand-in for development),
normalizes those recordings and labels each by whether the tests actually passed,
then for each step runs the model, grabs its layer-21 internal notes at the last
token of its reasoning, and measures them against the frozen direction. The
analyses ask whether that measurement separates eventual successes from failures —
with a built-in noise check so a null result is trustworthy rather than just
under-powered.
