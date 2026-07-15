#!/usr/bin/env python
"""Generate synthetic projection rows for offline wiring tests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage2.common.paths import NORMALIZED_DIR, data_file
from stage2.extract.project_steps import rel_pos
from stage2.trajectories.schema import load_trajectories_from_dir


def mock_projections(
    traj_dir: Path,
    output_path: Path,
    *,
    layer: int = 21,
    seed: int = 42,
) -> pd.DataFrame:
    records = load_trajectories_from_dir(traj_dir)
    if not records:
        raise FileNotFoundError(f"No trajectories in {traj_dir}")

    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for record in records:
        base = 0.3 if record.outcome == 1 else -0.2
        for step in record.steps:
            rp = rel_pos(step.step_index, record.n_steps)
            noise = rng.normal(0, 0.15)

            if step.assistant_response.strip():
                rows.append(
                    {
                        "trajectory_id": record.trajectory_id,
                        "outcome": record.outcome,
                        "step_index": step.step_index,
                        "n_steps": record.n_steps,
                        "rel_pos": rp,
                        "projection": base + noise + 0.1 * rp,
                        "token_type": "reasoning",
                        "layer": layer,
                    }
                )

            if step.observation and step.observation.strip():
                rows.append(
                    {
                        "trajectory_id": record.trajectory_id,
                        "outcome": record.outcome,
                        "step_index": step.step_index,
                        "n_steps": record.n_steps,
                        "rel_pos": rp,
                        "projection": base + noise + rng.normal(0, 0.35),
                        "token_type": "tool_output",
                        "layer": layer,
                    }
                )

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Wrote {len(df)} mock rows to {output_path}", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj-dir", type=Path, default=NORMALIZED_DIR)
    ap.add_argument("--output", type=Path, default=data_file("projections.parquet"))
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    mock_projections(args.traj_dir, args.output, layer=args.layer, seed=args.seed)


if __name__ == "__main__":
    main()
