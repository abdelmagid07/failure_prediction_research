# Roadmap

Methodology source of truth: [METHOD.tex](METHOD.tex). Summary:
[docs/method.md](docs/method.md). Model: Qwen3-8B. Eval: SWE-bench **Verified**.
Scaffold: mini-swe-agent **2.4.5**, step budget **60**, sampling 0.6/0.95/20.

**Compute:** Prefer Azure managed endpoint (if compat passes) or AWS single-VM
([docs/aws_runbook.md](docs/aws_runbook.md)); Colab + tunnel is smoke-test only.
See [compute_policy.txt](compute_policy.txt).

Stage 1 is done: 0.87 AUROC at L21/L22 (passes gate floor 0.87 = published 0.90
− tolerance 0.03). Paper Stage 2 (single-turn coding control) is **deferred**;
codebase Stage 2 = agentic transfer.

---

## Current Status (updated 2026-07-20)

**Pipeline migrated to METHOD.tex (Chunks A–G):**
- ✅ Identity: `task_id` / `seed` / `exit_status` on every trajectory.
- ✅ Readout: mean cosine over agent-generated tokens `G_t` (Eq. 1); `proj_final`
  robustness; multi-layer; activations `.npz` for probes.
- ✅ Multi-rollout: `ROLLOUTS` + `SEED_BASE` in `run_mini_swe_batch.sh`; ingest
  walks `r<seed>/`; `list_regens.py` for infra regenerations.
- ✅ Stats: task-level BCa bootstrap, block permutation, majority baseline,
  AUROC-by-position, within-task contrast.
- ✅ Stage 3 (verbalized confidence): `elicit/confidence.py` +
  `analyze/internal_vs_elicited.py`.
- ✅ Stage 4 (fitted probes): `probes/fit_probes.py` layer × position grid.
- ✅ Pins: mini-swe-agent==2.4.5, step_limit=60, n_bins=5, gate_tolerance=0.03.

**Still open:**
- ⬜ Real SWE-bench Verified generation (pilot first; then 60×5).
- ⬜ Task selection criteria for the 60 instances.
- ⬜ Live GPU projection with the new readout on real trajectories.

---

## Week 1 — SWE-bench end-to-end  *(plumbing ✅; real run ⬜)*

1. Serve Qwen3-8B (Colab tunnel / AWS / Azure) with hermes + reasoning parser.
2. `ROLLOUTS=1 STEP_LIMIT=60 bash scripts/run_mini_swe_batch.sh` on a few
   Verified pilot ids; then `ROLLOUTS=5` on the pilot.
3. Per-rollout harness eval → `r<seed>/results.json` → `ingest_batch`.
4. `project_steps --layers … --activations-npz …` on GPU.
5. `run_analyses` → final-step AUROC + position sweep.

---

## Week 2 — Final-step + position transfer (METHOD Stage 3)

Pilot → full 60×5 once selection criteria exist. Headline: final-step AUROC with
BCa CI vs majority baseline + permutation; full positional AUROC sweep.

---

## Week 3 — Verbalized confidence (METHOD Stage 4 / codebase Stage 3)

```bash
python -m stage2.elicit.confidence --traj-dir data/normalized --api-base ...
python -m stage2.analyze.internal_vs_elicited \
  --projections data/projections.parquet --confidence data/confidence.parquet
```

---

## Week 4 — Fitted probes (METHOD Stage 5 / codebase Stage 4)

```bash
python -m stage2.probes.fit_probes --activations data/activations.npz
```

Report the full layer × position grid (never just the max).

---

## Week 5 — Figures and writing

1. Stage 1 reconstruction AUROC
2. Final-step separation + CI
3. Transfer AUROC by position
4. Internal vs elicited
5. Probe grid / layer localization

---

## Commands

Stage 1 (already run):

```bash
cd stage1 && pip install -e .
python -m stage1.pipeline.run_gate --preset dev --icrl data/icrl_proxy.json --skip-extract
```

Stage 2 pipeline:

```bash
export MODEL_API_BASE="https://<tunnel-or-localhost>/v1"
cd stage2
ROLLOUTS=5 STEP_LIMIT=60 bash scripts/run_mini_swe_batch.sh config/pilot_instances.txt

# Per-rollout harness eval (see script epilogue), then:
python -m stage2.trajectories.ingest_batch \
  --traj-dir data/trajectories/mini_swe_run_<ts> --format mini-swe-agent
# Infra exclusions → fresh seeds:
python scripts/list_regens.py --manifest data/normalized/ingest_manifest.json

# GPU projection (Eq. 1 mean-cosine + activations for probes):
python -m stage2.extract.project_steps \
  --traj-dir data/normalized \
  --axis-path ../stage1/data/value_axis_proxy.npy \
  --activations-npz data/activations.npz

python -m stage2.analyze.run_analyses --projections data/projections.parquet

# Stage 3 — verbalized confidence:
python -m stage2.elicit.confidence --traj-dir data/normalized --api-base "$MODEL_API_BASE"
python -m stage2.analyze.internal_vs_elicited \
  --projections data/projections.parquet --confidence data/confidence.parquet

# Stage 4 — fitted probes:
python -m stage2.probes.fit_probes --activations data/activations.npz
```

Wiring tests: `bash tests/integration/test_stage1_wiring.sh` and `test_stage2_wiring.sh`.

Artifacts in `stage1/data/` and `stage2/data/` (gitignored). See [docs/setup.md](docs/setup.md).





