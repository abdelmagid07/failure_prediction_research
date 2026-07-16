# Stage 2: Trajectory projection and analyses

Project the frozen value axis onto on-policy agent trajectories and run readability analyses.

**Prerequisite:** `../stage1/data/value_axis.npy` (or dev preset: `value_axis_proxy.npy`).

## Pipeline

```bash
pip install -e ../stage1 -e .

# 1. Generate trajectories (see environments below)
# 2. Ingest
python -m stage2.trajectories.ingest_batch --traj-dir data/trajectories/<run>

# 3. Project (GPU)
python -m stage2.extract.project_steps \
  --traj-dir data/normalized \
  --axis-path ../stage1/data/value_axis.npy

# 4. Analyze
python -m stage2.analyze.run_analyses --projections data/projections.parquet
```

See [docs/analyses.md](../docs/analyses.md) for output interpretation.

## Environments

### Devbugs harness (no Docker)

Colab: [notebooks/devbugs_agent_colab.ipynb](notebooks/devbugs_agent_colab.ipynb) + [serve_qwen_colab.ipynb](notebooks/serve_qwen_colab.ipynb).

```bash
bash scripts/run_devbugs_batch.sh
```

12 hand-written Python bug-fix tasks; SWE-agent-compatible `.traj` output. This
is the project's own dev/smoke harness — unrelated to the third-party
`mini-swe-agent` used for real runs below.

### SWE-bench (primary: mini-swe-agent)

```bash
export MODEL_API_BASE="https://<tunnel>/v1"
bash scripts/run_mini_swe_batch.sh config/pilot_instances.txt
```

Requires Docker. Qwen inference via remote vLLM tunnel. Ingest with
`--format mini-swe-agent`. The legacy SWE-agent path (`run_pilot_batch.sh` +
`swe_agent_qwen.yaml`) is kept only until this path is verified end to end.

## Colab projection

[notebooks/project_and_analyze_colab.ipynb](notebooks/project_and_analyze_colab.ipynb) — upload normalized trajectories + axis files, run projection and analyses on A100.

## Offline wiring test

```bash
bash ../tests/integration/test_stage2_wiring.sh
```

## Layout

```
stage2/
  config/           defaults.yaml, presets/dev.yaml, instance lists
  stage2/
    trajectories/   parse .traj, ingest
    extract/        project_steps, token_spans
    analyze/        SNR, final-step, token-type noise
    devbugs/        local dev bug-fix harness (no Docker)
  notebooks/
  scripts/          run_mini_swe_batch.sh (primary), run_devbugs_batch.sh, run_pilot_batch.sh (legacy)
  tests/fixtures/
```
