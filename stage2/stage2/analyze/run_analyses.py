#!/usr/bin/env python
"""Run trajectory readability analyses and write report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stage2.analyze.final_step import final_step_plot
from stage2.analyze.noise_by_token_type import noise_by_token_type
from stage2.analyze.signal_to_noise import headline_late_bin_snr, signal_to_noise_by_position
from stage2.common.config import load_defaults
from stage2.common.paths import data_file


def interpret_report(
    *,
    late_snr: float | None,
    separability: float,
    reasoning_std: float | None,
    tool_std: float | None,
) -> str:
    lines = []

    if late_snr is not None and late_snr >= 1.0:
        lines.append("Late-bin separation-to-noise is at or above 1.")
    elif late_snr is not None:
        lines.append("Late-bin separation-to-noise is below 1 at this sample size.")
    else:
        lines.append("Insufficient data for late-bin SNR.")

    if reasoning_std is not None and tool_std is not None and reasoning_std < tool_std:
        lines.append(
            f"Reasoning-token std ({reasoning_std:.3f}) < tool-output std ({tool_std:.3f})."
        )
    elif reasoning_std is not None and tool_std is not None:
        lines.append(
            f"Reasoning-token std ({reasoning_std:.3f}) vs tool-output std ({tool_std:.3f})."
        )

    return " ".join(lines)


def run_analyses(
    projections_path: Path,
    *,
    n_bins: int = 5,
    output_dir: Path | None = None,
) -> dict:
    output_dir = output_dir or data_file("").parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(projections_path)

    snr_df = signal_to_noise_by_position(df, n_bins=n_bins)
    snr_csv = output_dir / "snr_by_position.csv"
    snr_df.to_csv(snr_csv, index=False)

    final_results = final_step_plot(df, output_dir / "final_step_separation.png")
    token_summary = noise_by_token_type(df, output_dir / "noise_by_token_type.png")

    late_snr = headline_late_bin_snr(snr_df)
    std_map = {row["token_type"]: float(row["std"]) for _, row in token_summary.iterrows()}
    reasoning_std = std_map.get("reasoning")
    tool_std = std_map.get("tool_output")
    std_ratio = (
        reasoning_std / tool_std
        if reasoning_std is not None and tool_std and tool_std > 0
        else None
    )

    n_traj = df["trajectory_id"].nunique()
    n_success = int(df.groupby("trajectory_id")["outcome"].first().eq(1).sum())
    n_failure = int(n_traj - n_success)

    report = {
        "n_trajectories": int(n_traj),
        "n_success": n_success,
        "n_failure": n_failure,
        "n_projection_rows": len(df),
        "late_bin_separation_to_noise": late_snr,
        "final_step_auroc": final_results["auroc"],
        "final_step_separability": final_results["separability"],
        "reasoning_projection_std": reasoning_std,
        "tool_output_projection_std": tool_std,
        "reasoning_to_tool_std_ratio": std_ratio,
        "snr_by_position_csv": str(snr_csv),
        "final_step_plot": final_results["plot_path"],
        "noise_by_token_type_plot": str(output_dir / "noise_by_token_type.png"),
        "interpretation": interpret_report(
            late_snr=late_snr,
            separability=final_results.get("separability", float("nan")),
            reasoning_std=reasoning_std,
            tool_std=tool_std,
        ),
    }

    report_path = output_dir / "analysis_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2), flush=True)
    print(f"Report -> {report_path}", flush=True)
    return report


def main():
    defaults = load_defaults()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projections", type=Path, default=data_file("projections.parquet"))
    ap.add_argument("--n-bins", type=int, default=defaults["n_bins"])
    ap.add_argument("--output-dir", type=Path, default=data_file("").parent)
    args = ap.parse_args()

    if not args.projections.exists():
        raise SystemExit(f"Projections file not found: {args.projections}")

    run_analyses(args.projections, n_bins=args.n_bins, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
