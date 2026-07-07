"""Analysis 2: final-step projection distributions and AUROC."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def final_step_plot(
    df: pd.DataFrame,
    out_path: Path,
    *,
    token_type: str = "reasoning",
) -> dict:
    work = df.copy()
    if "token_type" in work.columns:
        work = work[work["token_type"] == token_type]

    finals = work.sort_values("step_index").groupby("trajectory_id").tail(1)
    finals = finals.dropna(subset=["projection"])

    succ = finals[finals["outcome"] == 1]["projection"]
    fail = finals[finals["outcome"] == 0]["projection"]

    fig, ax = plt.subplots(figsize=(7, 4))
    if len(finals) > 0:
        bins = np.histogram_bin_edges(finals["projection"], bins=min(12, max(3, len(finals))))
        ax.hist(fail, bins=bins, alpha=0.6, label=f"Failed (n={len(fail)})")
        ax.hist(succ, bins=bins, alpha=0.6, label=f"Succeeded (n={len(succ)})")
        if len(fail) > 0:
            ax.axvline(fail.mean(), linestyle="--", linewidth=1, color="C0")
        if len(succ) > 0:
            ax.axvline(succ.mean(), linestyle="--", linewidth=1, color="C1")

    ax.set_xlabel("Value-axis projection at final step")
    ax.set_ylabel("Count")
    ax.set_title("Final-step projection: success vs. failure")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    auroc = float("nan")
    separability = float("nan")
    if len(finals["outcome"].unique()) > 1 and len(finals) >= 2:
        auroc = float(roc_auc_score(finals["outcome"], finals["projection"]))
        separability = float(max(auroc, 1 - auroc))

    return {
        "n_final_steps": len(finals),
        "n_success": int(len(succ)),
        "n_failure": int(len(fail)),
        "auroc": auroc,
        "separability": separability,
        "plot_path": str(out_path),
    }
