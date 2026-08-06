#!/usr/bin/env python
"""GPU step: problems_manifest.json + frozen axis -> projections.parquet.

Mirrors ``stage2/stage2/extract/project_steps.py``'s resumable,
Drive-mirrorable checkpoint loop (``run()`` / ``_mirror_checkpoint()``,
project_steps.py:432-559) at the granularity of one DebugBench problem
(slug) instead of one trajectory: all of a slug's rows (the original plus
every corrupted variant) are computed and checkpointed together, so a
checkpoint is always slug-atomic.
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from single_turn_control.config import load_defaults
from single_turn_control.paths import data_file
from single_turn_control.project import run_forward
from stage1.common.hooks import unit_direction


def load_axis_directions(axis_path: Path, layers: list[int]) -> dict[int, torch.Tensor]:
    import numpy as np

    axis = np.load(axis_path)
    return {
        L: torch.tensor(unit_direction(axis[L].astype("float32")), dtype=torch.float32)
        for L in layers
    }


def _copy_replace(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".mirror_tmp" + dst.suffix)
    if tmp.exists():
        tmp.unlink()
    shutil.copy2(src, tmp)
    if dst.exists():
        dst.unlink()
    shutil.move(str(tmp), str(dst))


def _mirror_checkpoint(output_path: Path, mirror_output: Path | None) -> None:
    if mirror_output is not None and output_path.exists():
        _copy_replace(output_path, mirror_output)
        print(f"    mirrored parquet -> {mirror_output}", flush=True)


def rows_for_problem(
    entry: dict,
    model,
    tokenizer,
    *,
    n_layers: int,
    directions: dict[int, torch.Tensor],
    device,
) -> list[dict]:
    rows: list[dict] = []
    conditions = {"original": entry["original"], **entry["variants"]}
    for condition, payload in conditions.items():
        per_layer = run_forward(
            model,
            tokenizer,
            payload["full_text"],
            payload["code_char_start"],
            payload["code_char_end"],
            n_layers=n_layers,
            directions=directions,
            device=device,
        )
        if per_layer is None:
            print(f"    SKIP {entry['slug']}/{condition}: code span not found in tokenization",
                  flush=True)
            continue
        for layer, stats in per_layer.items():
            rows.append({
                "slug": entry["slug"],
                "category": entry.get("category", ""),
                "subtype": entry.get("subtype", ""),
                "condition": condition,
                "outcome": 1 if condition == "original" else 0,
                "layer": layer,
                "proj_mean": stats["mean"],
                "proj_final": stats["final"],
                "n_tokens": stats["n_tokens"],
            })
    return rows


def run(
    manifest_path: Path,
    *,
    axis_path: Path,
    layers: list[int],
    model_name: str,
    dtype: str,
    n_layers: int,
    output_path: Path,
    mirror_output: Path | None = None,
    mirror_every: int = 10,
) -> pd.DataFrame:
    with open(manifest_path) as f:
        problems = json.load(f)
    if not problems:
        raise FileNotFoundError(f"No problems in {manifest_path}")

    directions = load_axis_directions(axis_path, layers)

    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True,
    ).eval()
    device = next(model.parameters()).device

    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_slugs: set[str] = set()
    all_rows: list[dict] = []
    if output_path.exists():
        prev = pd.read_parquet(output_path)
        if len(prev):
            all_rows = prev.to_dict(orient="records")
            done_slugs = set(prev["slug"].astype(str).unique())
            print(f"Resuming: {len(done_slugs)} problems already in {output_path}", flush=True)

    n_total = len(problems)
    if mirror_output is not None:
        print(f"Drive/durable mirror every {mirror_every} problems (output={mirror_output})",
              flush=True)

    n_skipped = 0
    for entry in problems:
        slug = entry["slug"]
        n_done = len(done_slugs)
        if slug in done_slugs:
            n_skipped += 1
            continue
        if n_skipped:
            print(f"  (skipped {n_skipped} already-done problems)", flush=True)
            n_skipped = 0

        n_variants = len(entry["variants"])
        print(f"  [{n_done}/{n_total} done] {slug}: {n_variants} corrupted variant(s)",
              flush=True)
        rows = rows_for_problem(
            entry, model, tokenizer, n_layers=n_layers, directions=directions, device=device,
        )
        all_rows.extend(rows)
        done_slugs.add(slug)
        n_done = len(done_slugs)

        df = pd.DataFrame(all_rows)
        df.to_parquet(output_path, index=False)
        print(f"    checkpointed {len(df)} rows -> {output_path} [{n_done}/{n_total} done]",
              flush=True)

        if mirror_output is not None and mirror_every > 0 and n_done % mirror_every == 0:
            _mirror_checkpoint(output_path, mirror_output)

    if mirror_output is not None:
        print("Final mirror to durable storage...", flush=True)
        _mirror_checkpoint(output_path, mirror_output)

    return pd.DataFrame(all_rows)


def main():
    defaults = load_defaults()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problems-manifest", type=Path, default=data_file("problems_manifest.json"))
    ap.add_argument("--output", type=Path, default=data_file("projections.parquet"))
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                     help="Layers to project. Default: all layers 0..n_layers-1.")
    ap.add_argument("--mirror-output", type=Path, default=None,
                     help="Optional durable copy path for the projections parquet (e.g. Drive).")
    ap.add_argument("--mirror-every", type=int, default=10,
                     help="Copy to --mirror-output every N newly completed problems.")
    ap.add_argument("--model", default=defaults["model"])
    ap.add_argument("--axis-path", type=Path, default=defaults["axis_path"])
    ap.add_argument("--dtype", default=defaults["dtype"])
    ap.add_argument("--n-layers", type=int, default=defaults["n_layers"])
    args = ap.parse_args()

    if not args.axis_path.exists():
        raise SystemExit(f"Value axis not found at {args.axis_path}. Run Stage 1 first.")

    layers = args.layers if args.layers is not None else list(range(args.n_layers))

    print(f"Projecting {args.problems_manifest} over {len(layers)} layer(s)...", flush=True)
    df = run(
        args.problems_manifest,
        axis_path=args.axis_path,
        layers=layers,
        model_name=args.model,
        dtype=args.dtype,
        n_layers=args.n_layers,
        output_path=args.output,
        mirror_output=args.mirror_output,
        mirror_every=args.mirror_every,
    )
    print(f"Done. {len(df)} rows -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
