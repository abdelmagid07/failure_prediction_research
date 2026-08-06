#!/usr/bin/env python
"""Generate synthetic projections.parquet for offline wiring tests.

Mirrors ``stage2/tests/fixtures/mock_projections.py``: skips the GPU forward
pass entirely and writes rows in exactly the schema ``run_control.py``
produces, so ``analyze.py`` can be exercised end to end offline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

VARIANTS = ["buggy", "syntax_error", "shuffled", "obfuscated"]


def mock_projections(
    output_path: Path,
    *,
    n_problems: int = 20,
    n_layers: int = 8,
    primary_layer: int = 5,
    signal_at_primary: float = 0.6,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic rows: separable at ``primary_layer``, near-chance elsewhere."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for p in range(n_problems):
        slug = f"mock-problem-{p:03d}"
        for layer in range(n_layers):
            gap = signal_at_primary if layer == primary_layer else 0.0
            orig = rng.normal(0.1 + gap, 0.15)
            rows.append({
                "slug": slug, "category": "mock", "subtype": "mock",
                "condition": "original", "outcome": 1, "layer": layer,
                "proj_mean": float(orig), "proj_final": float(orig + rng.normal(0, 0.1)),
                "n_tokens": int(rng.integers(20, 80)),
            })
            for variant in VARIANTS:
                corr = rng.normal(0.1, 0.15)
                rows.append({
                    "slug": slug, "category": "mock", "subtype": "mock",
                    "condition": variant, "outcome": 0, "layer": layer,
                    "proj_mean": float(corr), "proj_final": float(corr + rng.normal(0, 0.1)),
                    "n_tokens": int(rng.integers(20, 80)),
                })

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Wrote {len(df)} mock rows to {output_path}", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--n-problems", type=int, default=20)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--primary-layer", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    mock_projections(
        args.output, n_problems=args.n_problems, n_layers=args.n_layers,
        primary_layer=args.primary_layer, seed=args.seed,
    )


if __name__ == "__main__":
    main()
