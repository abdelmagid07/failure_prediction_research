"""Parse a raw *mini-swe-agent* trajectory (``.traj.json``) into a normalized record.

``mini-swe-agent`` is the third-party lightweight SWE-bench agent we generate
paper data with. **It is not** the project's own ``stage2/devbugs`` dev harness
(the name is unfortunately similar). Its SWE-bench batch runner writes one file
per instance at ``{output_dir}/{instance_id}/{instance_id}.traj.json``::

    {
      "info": {
        "exit_status": "Submitted" | "LimitsExceeded" | "<ExceptionName>" | ...,
        "submission":  "<patch>",
        "model_stats": {...},
        "config":      {...},
        "mini_version": "..."
      },
      "messages": [
        {"role": "system",    "content": "..."},
        {"role": "user",      "content": "<task + instructions>"},
        {"role": "assistant", "content": "<raw model turn>", "extra": {"actions": [...]}},
        {"role": "user",      "content": "<observation>",     "extra": {"returncode": ...}},
        ...
        {"role": "exit",      "content": "...", "extra": {"exit_status": ...}}   # optional trailer
      ]
    }

Unlike SWE-agent's ``.traj`` (which stores a per-step ``query`` snapshot), mini
keeps a single running ``messages`` list for the whole run. We reconstruct the
normalized per-step view from it: each assistant message is one step, everything
before it is ``messages_before_gen``, and the following user/tool message is the
``observation``. This is exactly on-policy — the running prefix *is* what the
model saw when it produced that turn.

The resolved/unresolved *outcome* is not in this file; it comes from the
SWE-bench evaluation harness ``results.json`` and is passed in by the caller
(see :mod:`stage2.trajectories.ingest_batch`).

Thinking-mode note: under our locked thinking-OFF setup the assistant ``content``
is the complete raw response, so no reconstruction is needed. If thinking were
ever enabled, vLLM's reasoning parser could split the ``<think>`` block into a
separate ``extra.reasoning_content`` field that ``content`` would omit — which is
one more reason generation stays thinking-off (the ``project_steps`` guard only
inspects ``content``).
"""

from __future__ import annotations

import json
from pathlib import Path

from stage2.trajectories.parse_swe_traj import _coerce_content
from stage2.trajectories.schema import Message, TrajectoryRecord, TrajectoryStep

# Roles whose message, when it immediately follows an assistant turn, is the
# environment's response to that turn (the "observation").
_OBSERVATION_ROLES = frozenset({"user", "tool"})

# The synthetic terminal marker mini appends after the last real turn. The model
# never saw it, so it must never leak into any step's context.
_SYNTHETIC_ROLES = frozenset({"exit"})


def mini_instance_id(info: dict, traj_path: Path) -> str:
    """Best-effort instance id for a mini trajectory.

    mini's SWE-bench batch names files ``<instance_id>.traj.json`` and does not
    always duplicate the id inside the JSON, so the filename is the reliable
    source; ``info`` / ``info.config`` are tried first in case a future version
    records it there.
    """
    for key in ("instance_id", "instance"):
        val = info.get(key)
        if val:
            return str(val)
    config = info.get("config")
    if isinstance(config, dict):
        for key in ("instance_id", "instance"):
            val = config.get(key)
            if val:
                return str(val)
    name = traj_path.name
    for suffix in (".traj.json", ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return traj_path.stem


def load_messages(data: dict) -> list[dict]:
    """Return the running message list, tolerating minor layout differences."""
    messages = data.get("messages")
    if isinstance(messages, list):
        return messages
    # Fall back to a nested/older layout if a variant ever stores it elsewhere.
    trajectory = data.get("trajectory")
    if isinstance(trajectory, list):
        return trajectory
    return []


def _prefix_messages(messages: list) -> list[Message]:
    """Normalize a message prefix to ``{role, content}``, dropping synthetic turns."""
    out: list[Message] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", ""))
        if role in _SYNTHETIC_ROLES:
            continue
        out.append({"role": role, "content": _coerce_content(msg.get("content", ""))})
    return out


def parse_mini_swe_traj(traj_path: Path, outcome: int) -> TrajectoryRecord:
    """Parse one raw mini-swe-agent ``.traj.json`` into a normalized record."""
    traj_path = Path(traj_path)
    data = json.loads(traj_path.read_text(encoding="utf-8"))
    info = data.get("info", {}) or {}
    trajectory_id = mini_instance_id(info, traj_path)

    messages = load_messages(data)
    steps: list[TrajectoryStep] = []
    step_index = 0
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue

        observation = ""
        if i + 1 < len(messages):
            nxt = messages[i + 1]
            if isinstance(nxt, dict) and str(nxt.get("role", "")) in _OBSERVATION_ROLES:
                observation = _coerce_content(nxt.get("content", ""))

        steps.append(
            TrajectoryStep(
                step_index=step_index,
                messages_before_gen=_prefix_messages(messages[:i]),
                assistant_response=_coerce_content(msg.get("content", "")),
                observation=observation,
            )
        )
        step_index += 1

    return TrajectoryRecord(
        trajectory_id=trajectory_id,
        outcome=int(outcome),
        steps=steps,
    )
