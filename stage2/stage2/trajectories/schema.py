"""Normalized trajectory schema shared across Stage 2.

A *normalized* trajectory is the model-agnostic representation the projection
step reads: one record per agent run, one step per model turn. It is decoupled
from the raw SWE-agent ``.traj`` layout so downstream code (projection,
analyses, mock generators) depends only on this schema, not on the agent
harness that produced it.

On-disk form (one JSON file per trajectory) matches
``data/normalized_test/sample.json``::

    {
      "trajectory_id": "...",
      "outcome": 0 | 1,
      "n_steps": N,
      "steps": [
        {
          "step_index": i,
          "messages_before_gen": [{"role": ..., "content": ...}, ...],
          "assistant_response": "...",
          "observation": "..."
        },
        ...
      ]
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

Message = dict[str, str]


@dataclass
class TrajectoryStep:
    """One model turn: the context it saw, what it produced, what came back."""

    step_index: int
    messages_before_gen: list[Message]
    assistant_response: str
    observation: str

    def to_dict(self) -> dict:
        return {
            "step_index": self.step_index,
            "messages_before_gen": self.messages_before_gen,
            "assistant_response": self.assistant_response,
            "observation": self.observation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrajectoryStep":
        return cls(
            step_index=int(d["step_index"]),
            messages_before_gen=list(d.get("messages_before_gen", [])),
            assistant_response=d.get("assistant_response", "") or "",
            observation=d.get("observation", "") or "",
        )


@dataclass
class TrajectoryRecord:
    """A full agent run plus its resolved/unresolved label."""

    trajectory_id: str
    outcome: int
    steps: list[TrajectoryStep] = field(default_factory=list)
    n_steps: int = 0

    def __post_init__(self) -> None:
        # n_steps is derived, never trusted from the caller: keeping it in sync
        # with len(steps) means rel_pos in the projection step can't drift.
        self.n_steps = len(self.steps)

    def to_dict(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "outcome": self.outcome,
            "n_steps": self.n_steps,
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrajectoryRecord":
        return cls(
            trajectory_id=str(d["trajectory_id"]),
            outcome=int(d["outcome"]),
            steps=[TrajectoryStep.from_dict(s) for s in d.get("steps", [])],
        )


def save_trajectory(record: TrajectoryRecord, path: Path) -> Path:
    """Write a single normalized trajectory to ``path`` as pretty JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    return path


def load_trajectory(path: Path) -> TrajectoryRecord:
    """Load a single normalized trajectory JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return TrajectoryRecord.from_dict(data)


def _looks_like_trajectory(data: object) -> bool:
    return (
        isinstance(data, dict)
        and "trajectory_id" in data
        and "steps" in data
        and "outcome" in data
    )


def load_trajectories_from_dir(traj_dir: Path) -> list[TrajectoryRecord]:
    """Load every normalized trajectory JSON in ``traj_dir``.

    Files that don't look like normalized trajectories (e.g. a run manifest or
    ``results.json`` dropped alongside) are skipped rather than raising, so the
    directory can double as a run folder. Returned sorted by filename for
    deterministic downstream ordering.
    """
    traj_dir = Path(traj_dir)
    records: list[TrajectoryRecord] = []
    for path in sorted(traj_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not _looks_like_trajectory(data):
            continue
        records.append(TrajectoryRecord.from_dict(data))
    return records
