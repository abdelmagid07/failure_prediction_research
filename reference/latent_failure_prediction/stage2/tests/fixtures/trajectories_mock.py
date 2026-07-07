"""Synthetic trajectories for offline wiring tests."""

from __future__ import annotations

import json
from pathlib import Path

from stage2.trajectories.parse_swe_traj import parse_swe_traj
from stage2.trajectories.schema import TrajectoryRecord, TrajectoryStep, save_trajectory


def _make_record(trajectory_id: str, outcome: int, n_steps: int) -> TrajectoryRecord:
    steps = []
    for i in range(n_steps):
        steps.append(
            TrajectoryStep(
                step_index=i,
                messages_before_gen=[
                    {"role": "system", "content": "You are a coding agent."},
                    {"role": "user", "content": f"ISSUE: fix bug {trajectory_id} (step {i})"},
                ],
                assistant_response=f"Step {i}: inspect and patch.",
                observation=f"file_{i}.py\nline {i * 10}: def foo(): pass\n",
            )
        )
    return TrajectoryRecord(
        trajectory_id=trajectory_id,
        outcome=outcome,
        n_steps=n_steps,
        steps=steps,
    )


def write_smoke_fixtures(output_dir: Path, sample_traj: Path | None = None) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if sample_traj and sample_traj.exists():
        record = parse_swe_traj(sample_traj, outcome=1)
        path = output_dir / f"{record.trajectory_id}.json"
        save_trajectory(record, path)
        written.append(path)

    for tid, outcome, n_steps in [
        ("mock_success_a", 1, 4),
        ("mock_failure_b", 0, 5),
        ("mock_success_c", 1, 3),
    ]:
        record = _make_record(tid, outcome, n_steps)
        path = output_dir / f"{tid}.json"
        save_trajectory(record, path)
        written.append(path)

    manifest = {"n_trajectories": len(written), "paths": [str(p) for p in written]}
    with open(output_dir.parent / "smoke_fixtures.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return written
