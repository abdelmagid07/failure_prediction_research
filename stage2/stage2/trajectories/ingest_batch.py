#!/usr/bin/env python
"""Ingest a batch of raw SWE-agent ``.traj`` files into normalized trajectories.

Reads a run directory (as produced by ``scripts/run_pilot_batch.sh`` or the
mini-agent ``run_batch``), labels each trajectory from the run's
``results.json``, and writes one normalized JSON per trajectory into the
normalized directory that the projection step consumes.

    python -m stage2.trajectories.ingest_batch --traj-dir data/trajectories/run_<ts>

Labels come from ``results.json`` (``resolved_ids`` / ``unresolved_ids``, or the
SWE-bench harness ``resolved`` / ``unresolved`` keys). A trajectory whose
instance id is in neither list is skipped with a warning: the transfer test is
supervised, so an unlabeled trajectory is unusable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage2.common.paths import NORMALIZED_DIR
from stage2.trajectories.parse_swe_traj import parse_swe_traj
from stage2.trajectories.schema import save_trajectory

_RESOLVED_KEYS = ("resolved_ids", "resolved", "resolved_instances")
_UNRESOLVED_KEYS = ("unresolved_ids", "unresolved", "unresolved_instances")


def _first_present(payload: dict, keys: tuple[str, ...]) -> tuple[set[str], bool]:
    """Return (ids, key_was_present) for the first matching key."""
    for key in keys:
        if key in payload:
            return set(payload[key] or []), True
    return set(), False


def load_labels(results_path: Path) -> tuple[dict[str, int], bool]:
    """Load instance_id -> outcome from a results.json.

    Returns the label map and whether an explicit unresolved list was present.
    When only a resolved list exists (common in SWE-bench reports), the caller
    treats any other instance as unresolved rather than dropping it.
    """
    payload = json.loads(Path(results_path).read_text(encoding="utf-8"))
    resolved, _ = _first_present(payload, _RESOLVED_KEYS)
    unresolved, had_unresolved = _first_present(payload, _UNRESOLVED_KEYS)

    labels = {iid: 1 for iid in resolved}
    for iid in unresolved:
        labels.setdefault(iid, 0)
    return labels, had_unresolved


def find_traj_files(traj_dir: Path) -> list[Path]:
    """All ``.traj`` files under ``traj_dir`` (SWE-agent nests them per instance;
    the mini-agent writes them flat), sorted for deterministic ordering."""
    return sorted(Path(traj_dir).rglob("*.traj"))


def ingest_batch(
    traj_dir: Path,
    *,
    results_path: Path | None = None,
    output_dir: Path = NORMALIZED_DIR,
) -> dict:
    traj_dir = Path(traj_dir)
    if not traj_dir.exists():
        raise SystemExit(f"Trajectory directory not found: {traj_dir}")

    results_path = Path(results_path) if results_path else traj_dir / "results.json"
    if not results_path.exists():
        raise SystemExit(
            f"Labels file not found: {results_path}\n"
            "Ingest needs resolved/unresolved labels. Point --results at the "
            "run's results.json (mini runs write one automatically; for "
            "SWE-bench, run the evaluation harness first)."
        )

    labels, had_unresolved = load_labels(results_path)
    traj_files = find_traj_files(traj_dir)
    if not traj_files:
        raise SystemExit(f"No .traj files found under {traj_dir}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[dict] = []
    skipped: list[str] = []
    n_success = 0
    n_failure = 0

    for traj_path in traj_files:
        # Peek at the instance id to resolve its label before a full parse.
        info = json.loads(traj_path.read_text(encoding="utf-8")).get("info", {}) or {}
        instance_id = str(info.get("instance_id") or traj_path.stem)

        if instance_id in labels:
            outcome = labels[instance_id]
        elif not had_unresolved:
            # Only a resolved list was given: absence means unresolved.
            outcome = 0
        else:
            print(f"  SKIP {instance_id}: no label in results.json", flush=True)
            skipped.append(instance_id)
            continue

        record = parse_swe_traj(traj_path, outcome=outcome)
        out_path = output_dir / f"{record.trajectory_id}.json"
        save_trajectory(record, out_path)
        written.append(
            {
                "trajectory_id": record.trajectory_id,
                "outcome": outcome,
                "n_steps": record.n_steps,
                "source": str(traj_path),
            }
        )
        n_success += int(outcome == 1)
        n_failure += int(outcome == 0)
        print(
            f"  {record.trajectory_id}: {record.n_steps} steps, "
            f"outcome={outcome} -> {out_path.name}",
            flush=True,
        )

    manifest = {
        "traj_dir": str(traj_dir),
        "results_path": str(results_path),
        "output_dir": str(output_dir),
        "n_ingested": len(written),
        "n_success": n_success,
        "n_failure": n_failure,
        "n_skipped": len(skipped),
        "skipped": skipped,
        "trajectories": written,
    }
    manifest_path = output_dir / "ingest_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"\nIngested {len(written)} trajectories "
        f"(N_success={n_success}, N_failure={n_failure}, skipped={len(skipped)}) "
        f"-> {output_dir}",
        flush=True,
    )
    if n_success == 0 or n_failure == 0:
        print(
            "  WARNING: only one outcome class present; "
            "expand the instance list before running the transfer analysis.",
            flush=True,
        )
    print(
        f"Next: python -m stage2.extract.project_steps --traj-dir {output_dir}",
        flush=True,
    )
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--traj-dir",
        type=Path,
        required=True,
        help="Run directory containing .traj files and results.json",
    )
    ap.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Labels file (default: <traj-dir>/results.json)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=NORMALIZED_DIR,
        help="Where to write normalized trajectory JSON (default: data/normalized)",
    )
    args = ap.parse_args()
    ingest_batch(
        args.traj_dir,
        results_path=args.results,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
