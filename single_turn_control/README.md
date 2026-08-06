# Single-turn coding control (METHOD.tex Stage 2)

Projects the frozen value axis (Stage 1's `stage1/data/value_axis_32b.npy`)
onto **single-turn, non-agentic** coding correctness: given a LeetCode-style
problem and a solution, does the axis assign a higher projection to the
correct code than to a corrupted version? This isolates whether the Stage-2
agentic transfer null (`stage2/`) is driven by the *domain* shift (ICRL chat
→ code) or the *horizon* shift (single-turn → multi-step agentic) — the code
here never generates anything; it prefills the model with given code and
reads activations while it processes it, following the axis authors' own
precedent (`reference/latent_failure_prediction/value-axis/experiments/tasks/
code_correlation.py`, their Fig. 4).

Dataset: **DebugBench** (Tian et al., 2024), `Rtian/DebugBench` on Hugging
Face — Python problems only, each with a correct `solution` and a native
`buggy_code`. We additionally corrupt each solution three more ways
(`single_turn_control/corrupt.py`, reimplemented from `value_axis.md`'s
description since the reference's own `code_utils.py` is not in the
snapshot — not a byte-identical replication of the original paper's
corruptions):

- `buggy` — the dataset's native GPT-4-generated bug
- `syntax_error` — seeded removal of a block colon or de-indent
- `shuffled` — seeded random permutation of solution lines
- `obfuscated` — seeded rename of local variables to single letters

## Pipeline

Three stages, split so only the middle one needs a GPU:

```bash
# 1. Prepare (CPU) — DebugBench -> corrupted variants -> rendered prompts
python -m single_turn_control.prepare --n-problems 150 \
  --output data/problems_manifest.json

# 2. Project (GPU) — one forward pass per (problem, variant), all layers.
#    Resumable; --mirror-output/--mirror-every for Drive checkpoints on Colab.
python -m single_turn_control.run_control \
  --problems-manifest data/problems_manifest.json \
  --axis-path ../stage1/data/value_axis_32b.npy \
  --output data/projections.parquet

# 3. Analyze (CPU) — AUROC + BCa CI + permutation, per variant + pooled,
#    at the primary layer (from axis_manifest_32b.json) and a full layer sweep.
python -m single_turn_control.analyze \
  --projections data/projections.parquet \
  --output-dir data/report
```

Colab: [`notebooks/single_turn_control_colab.ipynb`](notebooks/single_turn_control_colab.ipynb)
runs all three in sequence on an A100 (same shape as
`stage2/notebooks/project_full_32b_colab.ipynb`).

## Statistics

Reuses `stage2.analyze.stats.auroc_with_ci` — the same AUROC + task-level BCa
95% CI machinery as the agentic transfer test (METHOD.tex "Statistical
reporting"), with the DebugBench problem `slug` as the resampling unit. Each
problem contributes an `outcome=1` (original) row and an `outcome=0`
(corrupted) row per variant it has; the frozen axis's convention (axis points
from failing toward succeeding) means `outcome=1` is expected to score
higher if the axis transfers.

The permutation test is **not** `stage2.analyze.stats.permutation_test_auroc`.
That one reassigns whole outcome-pattern *blocks* between tasks — correct
when tasks carry different patterns (true for agentic rollouts), but
degenerate here: every problem's block is the identical fixed pair `[1, 0]`
by construction, so block-swapping never perturbs anything (confirmed
empirically: it returned p=1.0 on synthetic data with AUROC~1.0).
`single_turn_control/stats_local.py` implements a within-task label
permutation instead (shuffle which of a problem's own rows carry which
label — a paired sign-flip for the 1-vs-1 per-variant frame).

**Note on the default `n_problems=150`**: a random seeded subset, not the
original paper's full 225 — this is a preliminary check, not the paper's
headline number.

## Requirements

Installs into the same venv as `stage1`/`stage2` (imports
`stage1.common.chat`, `stage1.common.hooks`, `stage2.analyze.stats`
directly, same convention as `stage2/stage2/common/projection.py`):

```bash
pip install -e stage1
pip install -e stage2
pip install -e single_turn_control
```

## Layout

```
single_turn_control/
  config/defaults.yaml   model, axis path, n_problems, variants, stats config
  single_turn_control/
    data.py              DebugBench loading/filtering/sampling
    corrupt.py            the three reimplemented corruption functions
    render.py              chat-template rendering + code char offsets
    prepare.py (CLI)       CPU: data + corrupt + render -> problems_manifest.json
    project.py              one forward pass -> per-layer projections
    run_control.py (CLI)    GPU: problems_manifest.json -> projections.parquet
    analyze.py (CLI)        CPU: projections.parquet -> report.json + plot
  notebooks/              Colab A100 run
  tests/                  offline unit + wiring tests (no GPU/network)
```

## Offline wiring test

```bash
bash ../tests/integration/test_single_turn_control_wiring.sh
```
