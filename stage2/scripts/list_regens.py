#!/usr/bin/env python
"""List infrastructure-excluded rollouts and print their regeneration commands.

METHOD.tex's failure taxonomy separates *model* failures (the agent tried and
did not resolve the task — these count as unresolved) from *infrastructure*
failures (API/transport error, dead container — these are excluded and
regenerated, with counts reported). ``ingest_batch`` already excludes the infra
crashes and records them in ``excluded_errors`` of ``ingest_manifest.json``.

This script reads that manifest and, for each excluded ``(task_id, seed)`` pair,
assigns a *fresh* seed (one not already used for that task) and prints a rerun
command that regenerates just those tasks into the run's ``r<seed>/`` layout, so
a follow-up evaluate + re-ingest folds them back in. Counts are echoed for the
paper's reporting.

    python scripts/list_regens.py --manifest data/normalized/ingest_manifest.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage2.common.paths import NORMALIZED_DIR  # noqa: E402


def _used_seeds(manifest: dict) -> dict[str, set[int]]:
    used: dict[str, set[int]] = defaultdict(set)
    for group in ("trajectories", "excluded_errors"):
        for entry in manifest.get(group, []):
            seed = entry.get("seed")
            if seed is not None:
                used[entry.get("task_id", entry.get("trajectory_id", ""))].add(int(seed))
    return used


def _next_free(used: set[int]) -> int:
    s = 0
    while s in used:
        s += 1
    used.add(s)
    return s


def plan_regens(manifest: dict) -> list[dict]:
    """Assign a fresh seed to every infra-excluded rollout."""
    used = _used_seeds(manifest)
    regens: list[dict] = []
    for entry in manifest.get("excluded_errors", []):
        task_id = entry.get("task_id", entry.get("trajectory_id", ""))
        fresh = _next_free(used[task_id])
        regens.append(
            {
                "task_id": task_id,
                "old_seed": entry.get("seed"),
                "exit_status": entry.get("exit_status"),
                "fresh_seed": fresh,
            }
        )
    return regens


def write_regen_files(
    regens: list[dict], out_dir: Path
) -> dict[int, tuple[Path, list[str]]]:
    """One instances file per fresh seed; returns {seed: (path, task_ids)}."""
    by_seed: dict[int, list[str]] = defaultdict(list)
    for r in regens:
        by_seed[r["fresh_seed"]].append(r["task_id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[int, tuple[Path, list[str]]] = {}
    for seed, tasks in sorted(by_seed.items()):
        path = out_dir / f"regen_seed{seed}.txt"
        path.write_text(
            "# Auto-generated regen list for infra-excluded rollouts (fresh seed).\n"
            + "\n".join(tasks)
            + "\n",
            encoding="utf-8",
        )
        written[seed] = (path, tasks)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=NORMALIZED_DIR / "ingest_manifest.json",
        help="Path to ingest_manifest.json (default: data/normalized/ingest_manifest.json)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write regen instance files (default: next to the manifest)",
    )
    args = ap.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"Manifest not found: {args.manifest}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    excluded = manifest.get("excluded_errors", [])
    print(
        f"Ingest manifest: {args.manifest}\n"
        f"  ingested={manifest.get('n_ingested')} "
        f"excluded(infra)={len(excluded)} "
        f"skipped={manifest.get('n_skipped')}"
    )
    if not excluded:
        print("No infrastructure-excluded rollouts to regenerate.")
        return

    regens = plan_regens(manifest)
    print("\nInfra-excluded rollouts to regenerate:")
    print(f"  {'task_id':45s} {'old_seed':>8s} {'->':^4s} {'fresh_seed':>10s}  exit_status")
    for r in regens:
        old = "flat" if r["old_seed"] is None else str(r["old_seed"])
        print(
            f"  {r['task_id']:45s} {old:>8s} {'->':^4s} {r['fresh_seed']:>10d}  {r['exit_status']}"
        )

    traj_dir = manifest.get("traj_dir", "<raw_run_dir>")
    out_dir = args.out_dir or args.manifest.parent / "regens"
    written = write_regen_files(regens, out_dir)

    print("\nRegeneration commands (run, then re-evaluate + re-ingest):")
    for seed, (path, tasks) in written.items():
        print(f"  # {len(tasks)} task(s) -> {traj_dir}/r{seed}/")
        print(
            f"  OUTPUT_DIR={traj_dir} SEED_BASE={seed} ROLLOUTS=1 \\\n"
            f"    bash scripts/run_mini_swe_batch.sh {path}"
        )
    print(
        "\nAfter each regen: evaluate its r<seed>/preds.json with the SWE-bench "
        "harness, place the report at r<seed>/results.json, then re-run "
        "ingest_batch on the run dir to fold the fresh rollouts in."
    )


if __name__ == "__main__":
    main()
