"""Run pytest on a mini repo and write SWE-bench-style results.json."""

from __future__ import annotations

import json
from pathlib import Path

from stage2.devbugs.catalog import MiniInstance
from stage2.devbugs.sandbox import run_command


def evaluate_instance(instance: MiniInstance, repo_dir: Path) -> bool:
    code, output = run_command(instance.test_cmd, cwd=repo_dir, timeout_s=120)
    return code == 0


def write_results_json(
    outcomes: dict[str, bool],
    output_path: Path,
) -> None:
    resolved = sorted(iid for iid, ok in outcomes.items() if ok)
    unresolved = sorted(iid for iid, ok in outcomes.items() if not ok)
    payload = {
        "resolved_ids": resolved,
        "unresolved_ids": unresolved,
        "n_resolved": len(resolved),
        "n_unresolved": len(unresolved),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
