# Failure Prediction Research

Research workspace for latent failure prediction in long-horizon language agents.

## Layout

| Path | Description |
|------|-------------|
| `reference/latent_failure_prediction/` | Frozen snapshot of the Stage 1–2 pipeline (value axis, trajectories, analyses). Read-only baseline — no separate git history. |

New experiments and paper work live at the repo root as this project grows.

## Reference pipeline (quick pointer)

- **Stage 1:** ICRL → activation extract → value axis + AUROC gate (`reference/.../stage1/`)
- **Stage 2:** Trajectory ingest → projection → readability analyses (`reference/.../stage2/`)
- **Docs:** `reference/latent_failure_prediction/docs/`

See `reference/latent_failure_prediction/README.md` for commands and Colab notebooks.
