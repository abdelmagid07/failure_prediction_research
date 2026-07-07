# Trajectory readability analyses

Stage 2 produces a parquet of per-step projections. Run:

```bash
python -m stage2.analyze.run_analyses --projections data/projections.parquet
```

Outputs: `analysis_report.json`, `snr_by_position.csv`, `final_step_separation.png`, `noise_by_token_type.png`.

## Per-step schema

```python
{
    "trajectory_id": str,
    "outcome": int,           # 1 = resolved, 0 = failed
    "step_index": int,
    "n_steps": int,
    "rel_pos": float,         # step_index / (n_steps - 1)
    "projection": float,
    "token_type": str,        # "reasoning" | "tool_output"
    "layer": int,
}
```

## Analysis 1 — Signal-to-noise by position

For each relative-position bin, compute:

`|mean_success - mean_failure| / pooled_within_class_std`

Ratios above ~1 suggest the class gap is comparable to within-class jitter. Implemented in `stage2.analyze.signal_to_noise`.

## Analysis 2 — Final-step separation

Distribution of projections at the last step of each trajectory, success vs failure. Reports AUROC and `max(auroc, 1-auroc)` as separability. Implemented in `stage2.analyze.final_step`.

## Analysis 3 — Noise by token type

Compare projection standard deviation on `reasoning` vs `tool_output` tokens. Validates reading activations at the assistant's own output rather than echoed tool content. Implemented in `stage2.analyze.noise_by_token_type`.

## Interpretation

Small sample sizes give directional results only. These analyses test whether the projection is **readable** through trajectory noise; they do not by themselves establish transfer of the value axis to agentic outcomes.
