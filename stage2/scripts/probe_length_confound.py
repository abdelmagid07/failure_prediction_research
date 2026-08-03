#!/usr/bin/env python
"""Length-confound checks for Stage-5 probes (pilot / paper diagnostics).

1. Trajectory-level length-only AUROC (n_steps -> outcome).
2. Same task-level CV logistic probe using only n_steps as a feature (step rows).
3. Refit activation probes restricted to trajectories with n_steps in [lo, hi].

Usage (from stage2/ or repo root with PYTHONPATH):
  python -m stage2.scripts.probe_length_confound \\
    --activations path/to/agentic_meanpool.npz \\
    --output-dir path/to/out \\
    --min-steps 4 --max-steps 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Allow running as script without install.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO / "stage2"))

from stage2.probes.fit_probes import (  # noqa: E402
    _stratified_task_folds,
    _task_labels,
    fit_grid,
    load_activations_npz,
)


def _auroc(y, s) -> float:
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def traj_length_table(meta: pd.DataFrame) -> pd.DataFrame:
    """One row per trajectory: n_steps, outcome."""
    # n_steps is constant per traj; recover from max step_index + 1 if needed.
    g = meta.groupby("trajectory_id", sort=False)
    out = g.agg(
        task_id=("task_id", "first"),
        outcome=("outcome", "first"),
        max_step=("step_index", "max"),
    ).reset_index()
    out["n_steps"] = out["max_step"] + 1
    return out


def length_only_traj_auroc(traj: pd.DataFrame) -> dict:
    return {
        "unit": "trajectory",
        "feature": "n_steps",
        "n": int(len(traj)),
        "n_success": int((traj["outcome"] == 1).sum()),
        "n_failure": int((traj["outcome"] == 0).sum()),
        "auroc": _auroc(traj["outcome"], traj["n_steps"]),
        "note": "Higher n_steps predicting failure => AUROC of n_steps for success may be <0.5; "
        "separability = max(a, 1-a).",
        "separability": float(
            max(_auroc(traj["outcome"], traj["n_steps"]), 1.0 - _auroc(traj["outcome"], traj["n_steps"]))
            if not np.isnan(_auroc(traj["outcome"], traj["n_steps"]))
            else float("nan")
        ),
    }


def length_only_cv_auroc(
    meta: pd.DataFrame,
    *,
    layer: int,
    bin_idx: int,
    n_bins: int = 5,
    n_folds: int = 5,
    rng: np.random.Generator | None = None,
) -> dict:
    """Task-level CV AUROC using only n_steps as the feature (matched to probe cells)."""
    rng = rng or np.random.default_rng(0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(
        np.digitize(meta["rel_pos"].to_numpy(), edges[1:-1], right=False), 0, n_bins - 1
    )
    work = meta.copy()
    work["bin"] = bins
    # Attach n_steps per traj
    traj = traj_length_table(meta)
    n_map = traj.set_index("trajectory_id")["n_steps"].to_dict()
    work["n_steps"] = work["trajectory_id"].map(n_map)

    mask = (work["layer"].to_numpy() == layer) & (work["bin"].to_numpy() == bin_idx)
    sub = work.loc[mask]
    if len(sub) < 4 or sub["outcome"].nunique() < 2:
        return {"layer": layer, "bin": bin_idx, "auroc": float("nan"), "n": int(len(sub))}

    X = sub[["n_steps"]].to_numpy(dtype=float)
    y = sub["outcome"].to_numpy().astype(int)
    task_ids = sub["task_id"].to_numpy()
    task_y = _task_labels(sub)
    folds = _stratified_task_folds(task_ids, task_y, n_folds=n_folds, rng=rng)
    fold_scores: list[float] = []
    for train_m, test_m in folds:
        if train_m.sum() < 2 or test_m.sum() < 1:
            continue
        if len(np.unique(y[train_m])) < 2 or len(np.unique(y[test_m])) < 2:
            continue
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_m])
        X_test = scaler.transform(X[test_m])
        clf = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, class_weight="balanced"
        )
        clf.fit(X_train, y[train_m])
        scores = clf.predict_proba(X_test)[:, 1]
        fold_scores.append(float(roc_auc_score(y[test_m], scores)))
    return {
        "layer": layer,
        "bin": bin_idx,
        "feature": "n_steps",
        "auroc": float(np.mean(fold_scores)) if fold_scores else float("nan"),
        "n": int(len(sub)),
        "n_folds_used": int(len(fold_scores)),
    }


def run(
    activations_path: Path,
    *,
    output_dir: Path,
    min_steps: int = 4,
    max_steps: int = 12,
    layers: list[int] | None = None,
    n_bins: int = 5,
    n_folds: int = 5,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    acts, meta = load_activations_npz(activations_path)
    traj = traj_length_table(meta)

    # --- 1) length distribution + traj-level baseline ---
    length_auroc = length_only_traj_auroc(traj)
    by_out = traj.groupby("outcome")["n_steps"].describe().to_dict()
    long_fail = traj[traj["n_steps"] > max_steps]
    band = traj[(traj["n_steps"] >= min_steps) & (traj["n_steps"] <= max_steps)]

    # --- 2) length-only CV at L49 late bin (matched cell) ---
    length_cv_late = length_only_cv_auroc(meta, layer=49, bin_idx=n_bins - 1, n_bins=n_bins, n_folds=n_folds)
    length_cv_mid = length_only_cv_auroc(meta, layer=49, bin_idx=3, n_bins=n_bins, n_folds=n_folds)

    # --- 3) activation probes on length band ---
    keep_ids = set(band["trajectory_id"].tolist())
    row_mask = meta["trajectory_id"].isin(keep_ids).to_numpy()
    meta_b = meta.loc[row_mask].reset_index(drop=True)
    acts_b = acts[row_mask]

    if layers is not None:
        layer_mask = meta_b["layer"].isin(layers).to_numpy()
        meta_b = meta_b.loc[layer_mask].reset_index(drop=True)
        acts_b = acts_b[layer_mask]

    print(
        f"Length band [{min_steps},{max_steps}]: "
        f"{band['outcome'].eq(1).sum()} succ / {len(band)} trajs; "
        f"{len(meta_b)} activation rows; layers="
        f"{sorted(meta_b['layer'].unique().tolist())[:5]}...",
        flush=True,
    )
    grid = fit_grid(acts_b, meta_b, n_bins=n_bins, n_folds=n_folds)
    grid_path = output_dir / f"probe_auroc_grid_steps_{min_steps}_{max_steps}.csv"
    grid.to_csv(grid_path, index=False)
    valid = grid.dropna(subset=["auroc"])

    # Compare to full-grid summary if present beside activations
    full_summary = {
        "grid_mean_auroc": float(valid["auroc"].mean()) if len(valid) else float("nan"),
        "grid_max_auroc": float(valid["auroc"].max()) if len(valid) else float("nan"),
        "grid_median_auroc": float(valid["auroc"].median()) if len(valid) else float("nan"),
        "n_valid_cells": int(len(valid)),
        "l49_bins": valid[valid["layer"] == 49][["bin", "auroc", "n", "n_folds_used"]].to_dict(
            orient="records"
        )
        if (valid["layer"] == 49).any()
        else [],
    }

    report = {
        "activations": str(activations_path),
        "length_band": {"min_steps": min_steps, "max_steps": max_steps},
        "traj_counts": {
            "n_total": int(len(traj)),
            "n_success": int((traj["outcome"] == 1).sum()),
            "n_failure": int((traj["outcome"] == 0).sum()),
            "n_steps_by_outcome": by_out,
            "n_traj_above_max_steps": int(len(long_fail)),
            "n_success_above_max_steps": int((long_fail["outcome"] == 1).sum()),
            "n_band": int(len(band)),
            "n_success_in_band": int((band["outcome"] == 1).sum()),
            "n_failure_in_band": int((band["outcome"] == 0).sum()),
        },
        "length_only_traj_auroc": length_auroc,
        "length_only_cv_L49_late_bin": length_cv_late,
        "length_only_cv_L49_bin3": length_cv_mid,
        "activation_probes_in_band": full_summary,
        "grid_csv": str(grid_path),
        "interpretation_hints": [
            "If length-only separability is high, length is predictive of failure.",
            "If band-restricted probe grid_mean stays well above chance and above length-only CV, "
            "probes are not solely length.",
            "If band-restricted probes collapse to ~0.5, treat pilot probe hit as length-confounded.",
        ],
    }
    out_json = output_dir / "length_confound_report.json"
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str), flush=True)
    print(f"Wrote {out_json}", flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activations", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--min-steps", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help="Optional layer subset (default: all). Tip: 40-55 for a faster pilot.",
    )
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()
    run(
        args.activations,
        output_dir=args.output_dir,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
        layers=args.layers,
        n_bins=args.n_bins,
        n_folds=args.n_folds,
    )


if __name__ == "__main__":
    main()
