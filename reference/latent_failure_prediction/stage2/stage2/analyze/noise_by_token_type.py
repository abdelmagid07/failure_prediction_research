"""Analysis 3: projection noise by token type."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def noise_by_token_type(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Compare projection std on reasoning vs tool_output tokens."""
    summary = (
        df.groupby("token_type")["projection"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(summary["token_type"], summary["std"])
    ax.set_ylabel("Projection std")
    ax.set_title("Within-token-type projection noise")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return summary
