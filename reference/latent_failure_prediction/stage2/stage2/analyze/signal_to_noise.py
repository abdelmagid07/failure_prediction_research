"""Analysis 1: signal-to-noise ratio by relative position bin."""

from __future__ import annotations

import numpy as np
import pandas as pd


def signal_to_noise_by_position(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """
    For each relative-position bin, compute separation-to-noise ratio:
        |mean_success - mean_failure| / pooled_within_class_std
    """
    work = df.copy()
    if "token_type" in work.columns:
        work = work[work["token_type"] == "reasoning"]

    work["bin"] = pd.cut(work["rel_pos"], bins=n_bins, labels=False)
    rows = []
    for b in sorted(work["bin"].dropna().unique()):
        sub = work[work["bin"] == b]
        succ = sub[sub["outcome"] == 1]["projection"]
        fail = sub[sub["outcome"] == 0]["projection"]
        if len(succ) < 2 or len(fail) < 2:
            continue
        between = abs(succ.mean() - fail.mean())
        within = np.sqrt(
            (
                (succ.var(ddof=1) * (len(succ) - 1))
                + (fail.var(ddof=1) * (len(fail) - 1))
            )
            / (len(succ) + len(fail) - 2)
        )
        ratio = between / within if within > 0 else np.nan
        rows.append(
            {
                "bin": int(b),
                "rel_pos_mid": (b + 0.5) / n_bins,
                "between_class_gap": between,
                "within_class_std": within,
                "separation_to_noise": ratio,
                "n_succ": len(succ),
                "n_fail": len(fail),
            }
        )
    return pd.DataFrame(rows)


def headline_late_bin_snr(snr_df: pd.DataFrame) -> float | None:
    """Return separation-to-noise in the latest available bin."""
    if snr_df.empty:
        return None
    late = snr_df.sort_values("rel_pos_mid").iloc[-1]
    return float(late["separation_to_noise"])
