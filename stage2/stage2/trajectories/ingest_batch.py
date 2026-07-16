#!/usr/bin/env python
"""Ingest a batch of raw agent trajectories into normalized trajectories.

Reads a run directory (as produced by the ``mini-swe-agent`` SWE-bench batch
runner, or the ``stage2/devbugs`` dev harness), labels each trajectory from the
run's ``results.json``, and writes one normalized JSON per trajectory into the
directory the projection step consumes.

    python -m stage2.trajectories.ingest_batch --traj-dir data/trajectories/run_<ts>

Two raw formats are supported via ``--format``:

* ``mini-swe-agent`` (default): ``*.traj.json`` with a running ``messages`` list.
* ``swe-agent``: ``*.traj`` with a per-step ``trajectory``/``query`` layout.

Labels come from ``results.json`` (``resolved_ids`` / ``unresolved_ids``, or the
SWE-bench harness ``resolved`` / ``unresolved`` keys). A trajectory whose
instance id is in neither list is skipped with a warning: the transfer test is
supervised, so an unlabeled trajectory is unusable.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from stage2.common.paths import NORMALIZED_DIR
from stage2.trajectories.parse_mini_swe_traj import mini_instance_id, parse_mini_swe_traj
from stage2.trajectories.parse_swe_traj import parse_swe_traj
from stage2.trajectories.schema import TrajectoryRecord, save_trajectory

_RESOLVED_KEYS = ("resolved_ids", "resolved", "resolved_instances")
_UNRESOLVED_KEYS = ("unresolved_ids", "unresolved", "unresolved_instances")

# --- Crashed-run detection ------------------------------------------------
# A "crashed" run failed for infrastructure reasons (API/transport error, dead
# container) rather than because the model tried the task and failed. Its
# trajectory is a stub, and counting it as a task failure poisons the outcome
# labels, so we exclude it before labeling.
#
# The two agents encode this differently:
#   swe-agent  -> a single sentinel exit_status ("error"); use a DENYLIST.
#   mini       -> genuine terminations are a small fixed set, while crashes take
#                 the *exception class name* (e.g. "APIConnectionError"), which
#                 is open-ended; so we ALLOWLIST the genuine ones and treat
#                 everything else as a crash.
_SWEAGENT_ERROR_STATUSES = frozenset({"error"})
_MINI_GENUINE_STATUSES = frozenset(
    {"Submitted", "LimitsExceeded", "TimeExceeded", "RepeatedFormatError"}
)


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


@dataclass(frozen=True)
class FormatSpec:
    """How to read one raw trajectory format."""

    name: str
    glob: str
    parse: Callable[[Path, int], TrajectoryRecord]
    instance_id: Callable[[dict, Path], str]
    is_crash: Callable[[str], bool]


def _sweagent_instance_id(info: dict, path: Path) -> str:
    return str(info.get("instance_id") or path.stem)


def _mini_is_crash(exit_status: str, genuine: frozenset[str]) -> bool:
    status = (exit_status or "").strip()
    if not status:
        return False  # unknown; keep it and let the histogram surface it
    if status.startswith("Submitted"):
        return False  # e.g. "Submitted" / "Submitted (exit_command)"
    return status not in genuine


def build_format_spec(
    fmt: str,
    *,
    error_statuses: frozenset[str],
    genuine_statuses: frozenset[str],
) -> FormatSpec:
    if fmt == "mini-swe-agent":
        return FormatSpec(
            name=fmt,
            glob="*.traj.json",
            parse=parse_mini_swe_traj,
            instance_id=mini_instance_id,
            is_crash=lambda s: _mini_is_crash(s, genuine_statuses),
        )
    if fmt == "swe-agent":
        return FormatSpec(
            name=fmt,
            glob="*.traj",
            parse=parse_swe_traj,
            instance_id=_sweagent_instance_id,
            is_crash=lambda s: (s or "") in error_statuses,
        )
    raise SystemExit(f"Unknown --format {fmt!r}; expected mini-swe-agent or swe-agent")


def find_traj_files(traj_dir: Path, glob: str) -> list[Path]:
    """All raw trajectory files under ``traj_dir`` (agents nest them per
    instance), sorted for deterministic ordering."""
    return sorted(Path(traj_dir).rglob(glob))


def ingest_batch(
    traj_dir: Path,
    *,
    fmt: str = "mini-swe-agent",
    results_path: Path | None = None,
    output_dir: Path = NORMALIZED_DIR,
    error_statuses: frozenset[str] = _SWEAGENT_ERROR_STATUSES,
    genuine_statuses: frozenset[str] = _MINI_GENUINE_STATUSES,
    keep_error_stubs: bool = False,
) -> dict:
    traj_dir = Path(traj_dir)
    if not traj_dir.exists():
        raise SystemExit(f"Trajectory directory not found: {traj_dir}")

    spec = build_format_spec(
        fmt, error_statuses=error_statuses, genuine_statuses=genuine_statuses
    )

    results_path = Path(results_path) if results_path else traj_dir / "results.json"
    if not results_path.exists():
        raise SystemExit(
            f"Labels file not found: {results_path}\n"
            "Ingest needs resolved/unresolved labels. Point --results at the "
            "run's results.json (the devbugs harness writes one automatically; "
            "for mini-swe-agent/SWE-bench, run the evaluation harness first)."
        )

    labels, had_unresolved = load_labels(results_path)
    traj_files = find_traj_files(traj_dir, spec.glob)
    if not traj_files:
        raise SystemExit(
            f"No {spec.glob} files found under {traj_dir} (format={spec.name})"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[dict] = []
    skipped: list[str] = []
    excluded_errors: list[dict] = []
    exit_status_hist: Counter[str] = Counter()
    n_success = 0
    n_failure = 0

    for traj_path in traj_files:
        # Peek at info to resolve label and exit status before a full parse.
        info = json.loads(traj_path.read_text(encoding="utf-8")).get("info", {}) or {}
        instance_id = spec.instance_id(info, traj_path)

        exit_status = str(info.get("exit_status", ""))
        exit_status_hist[exit_status or "(none)"] += 1

        if not keep_error_stubs and spec.is_crash(exit_status):
            # Crashed run, not a real task failure: drop before labeling.
            print(
                f"  EXCLUDE {instance_id}: crashed run "
                f"(exit_status={exit_status!r}), not a task failure",
                flush=True,
            )
            excluded_errors.append(
                {"trajectory_id": instance_id, "exit_status": exit_status}
            )
            continue

        if instance_id in labels:
            outcome = labels[instance_id]
        elif not had_unresolved:
            # Only a resolved list was given: absence means unresolved.
            outcome = 0
        else:
            print(f"  SKIP {instance_id}: no label in results.json", flush=True)
            skipped.append(instance_id)
            continue

        record = spec.parse(traj_path, outcome)
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
        "format": spec.name,
        "traj_dir": str(traj_dir),
        "results_path": str(results_path),
        "output_dir": str(output_dir),
        "n_ingested": len(written),
        "n_success": n_success,
        "n_failure": n_failure,
        "n_skipped": len(skipped),
        "n_excluded_errors": len(excluded_errors),
        "exit_status_histogram": dict(sorted(exit_status_hist.items())),
        "skipped": skipped,
        "excluded_errors": excluded_errors,
        "trajectories": written,
    }
    manifest_path = output_dir / "ingest_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"\nIngested {len(written)} trajectories "
        f"(N_success={n_success}, N_failure={n_failure}, "
        f"skipped={len(skipped)}, excluded_errors={len(excluded_errors)}) "
        f"-> {output_dir}",
        flush=True,
    )
    # The histogram is the safety net for the crash guard: if a genuine
    # termination is being excluded (or a crash slipping through), it shows here.
    print("  exit_status histogram:", flush=True)
    for status, count in sorted(exit_status_hist.items()):
        print(f"    {status}: {count}", flush=True)
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
        help="Run directory containing raw trajectory files and results.json",
    )
    ap.add_argument(
        "--format",
        choices=["mini-swe-agent", "swe-agent"],
        default="mini-swe-agent",
        help="Raw trajectory format to ingest (default: mini-swe-agent)",
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
    ap.add_argument(
        "--error-statuses",
        nargs="+",
        default=sorted(_SWEAGENT_ERROR_STATUSES),
        help="[swe-agent] exit_status values to exclude as crashed runs "
        "(default: error).",
    )
    ap.add_argument(
        "--genuine-statuses",
        nargs="+",
        default=sorted(_MINI_GENUINE_STATUSES),
        help="[mini-swe-agent] exit_status values that mean a genuine task "
        "attempt; anything else (an exception class name) is treated as a "
        "crashed run and excluded. Inspect the ingest histogram and extend if "
        "a new genuine status appears.",
    )
    ap.add_argument(
        "--keep-error-stubs",
        action="store_true",
        help="Ingest crashed-run stubs instead of excluding them (not recommended: "
        "they corrupt the outcome labels)",
    )
    args = ap.parse_args()
    ingest_batch(
        args.traj_dir,
        fmt=args.format,
        results_path=args.results,
        output_dir=args.output_dir,
        error_statuses=frozenset(args.error_statuses),
        genuine_statuses=frozenset(args.genuine_statuses),
        keep_error_stubs=args.keep_error_stubs,
    )


if __name__ == "__main__":
    main()
