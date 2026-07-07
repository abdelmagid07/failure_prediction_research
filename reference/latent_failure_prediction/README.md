# Latent Failure Prediction in Long-Horizon Language Agents

We test whether the **value axis** from [Jiang et al. (2026)](https://arxiv.org/abs/2606.17056) transfers to on-policy agent trajectories on **Qwen3-8B**: freeze the axis built from single-turn ICRL, project it onto multi-step SWE-bench (or mini-agent) runs, and ask whether it separates eventually-successful from eventually-failed trajectories.

## Repository layout

| Path | Description |
|------|-------------|
| [docs/method.md](docs/method.md) | Research design and locked decisions |
| [docs/analyses.md](docs/analyses.md) | Stage 2 projection schema and analyses |
| [docs/setup.md](docs/setup.md) | Install, Colab, environments |
| [stage1/](stage1/) | ICRL generation, activation extract, axis build, AUROC gate |
| [stage2/](stage2/) | Trajectory ingest, projection, analyses |
| [value-axis/](value-axis/) | Upstream authors' reference code |

## Quick start

### Stage 1 — value axis

```bash
cd stage1
pip install -e .

# Generate ICRL (Anthropic API)
python -m stage1.icrl_gen.generate --n 300 --output data/icrl.json --resume

# Extract + gate (GPU — see stage1/notebooks/stage1_gpu_colab.ipynb)
python -m stage1.pipeline.extract_activations --icrl data/icrl.json --force
python -m stage1.pipeline.run_gate --icrl data/icrl.json --skip-extract
```

Gate: L21/L22 held-out AUROC ≥ 0.93 → `data/value_axis.npy`.

### Stage 2 — trajectories and projection

See [stage2/README.md](stage2/README.md). Summary:

1. Generate trajectories (mini-agent Colab notebook or SWE-bench + Docker)
2. `python -m stage2.trajectories.ingest_batch --traj-dir <run_dir>`
3. Project on GPU: `python -m stage2.extract.project_steps --traj-dir data/normalized`
4. `python -m stage2.analyze.run_analyses --projections data/projections.parquet`

## Development

- **Dev preset:** `python -m stage1.pipeline.run_gate --preset dev` uses Qwen-local ICRL and separate artifact files (0.75 gate). See [docs/method.md](docs/method.md).
- **Wiring tests:** `bash tests/integration/test_stage1_wiring.sh` and `test_stage2_wiring.sh`

## References

- Value axis paper: arXiv:2606.17056
- Authors' code: https://github.com/nickjiang2378/value-axis
