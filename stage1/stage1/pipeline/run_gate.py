#!/usr/bin/env python
"""Stage 1 pipeline stages: extract | build | gate | all.

Examples:
  python -m stage1.pipeline.run_gate --preset qwen32b --stage extract \\
    --icrl data/icrl_32b.json --activations-dir /path/to/activations_32b

  python -m stage1.pipeline.run_gate --preset qwen32b --stage build \\
    --activations-dir /path/to/activations_32b

  python -m stage1.pipeline.run_gate --preset qwen32b --stage gate \\
    --activations-dir /path/to/activations_32b
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from stage1.common.config import load_preset, load_split
from stage1.common.paths import data_file
from stage1.pipeline.build_axis import build_axis
from stage1.pipeline.eval_auroc import eval_auroc, plot_auroc
from stage1.pipeline.extract_activations import run as extract_run


def check_gate(auroc_by_layer: dict, gate_layers: list[int], threshold: float) -> bool:
    """Pass iff every gate layer's held-out AUROC is at least ``threshold``."""
    for layer in gate_layers:
        val = auroc_by_layer.get(str(layer), float("nan"))
        if np.isnan(val) or val < threshold:
            return False
    return True


def select_primary_layers(
    auroc_by_layer: dict,
    n_layers: int,
    *,
    band_lo_frac: float = 0.50,
    band_hi_frac: float = 0.85,
    top_k: int = 2,
) -> list[int]:
    """Pick top-k held-out AUROC layers in the middle–late band (Jiang: mid→late)."""
    lo = int(n_layers * band_lo_frac)
    hi = max(lo + 1, int(n_layers * band_hi_frac))
    scored: list[tuple[float, int]] = []
    for layer in range(lo, min(hi, n_layers)):
        val = auroc_by_layer.get(str(layer), float("nan"))
        if not np.isnan(val):
            scored.append((float(val), layer))
    scored.sort(reverse=True)
    return [layer for _, layer in scored[:top_k]]


def resolve_threshold(cfg: dict, override: float | None) -> float:
    if override is not None:
        return override
    if "published_auroc" in cfg and "gate_tolerance" in cfg:
        return float(cfg["published_auroc"]) - float(cfg["gate_tolerance"])
    return float(cfg["gate_threshold"])


def stage_extract(cfg: dict, icrl_path: Path, activations_dir: Path, *, force: bool, n_layers: int) -> None:
    print("extract_activations", flush=True)
    extract_run(
        icrl_path,
        model_name=cfg["model"],
        n_layers=n_layers,
        enable_thinking=cfg["enable_thinking"],
        dtype=cfg["dtype"],
        force=force,
        activations_dir=activations_dir,
    )
    n = len(list(activations_dir.glob("*.npz")))
    print(f"extract done. npz in {activations_dir}: {n}", flush=True)


def stage_build(cfg: dict, activations_dir: Path, n_layers: int) -> tuple[np.ndarray, dict]:
    print("build_axis", flush=True)
    split = load_split()
    train_criteria = set(split["train"])
    axis, meta = build_axis(activations_dir, train_criteria, n_layers)
    axis_path: Path = cfg["axis_path"]
    axis_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(axis_path, axis)
    print(f"axis -> {axis_path} shape={list(axis.shape)}", flush=True)
    print(
        f"  train convs: {meta['n_conversations']}, "
        f"pre tokens: {meta['n_pre_tokens']}, post: {meta['n_post_tokens']}",
        flush=True,
    )
    return axis, meta


def stage_gate(
    cfg: dict,
    activations_dir: Path,
    icrl_path: Path | None,
    *,
    n_layers: int,
    threshold: float,
    gate_floor: float,
    axis: np.ndarray | None = None,
    meta: dict | None = None,
    mock_only: bool = False,
) -> int:
    print("eval_auroc + gate", flush=True)
    axis_path: Path = cfg["axis_path"]
    if axis is None:
        if not axis_path.exists():
            raise SystemExit(f"Axis not found: {axis_path}. Run --stage build first.")
        axis = np.load(axis_path)
        meta = meta or {}

    split = load_split()
    held_out = set(split["held_out"])
    results = eval_auroc(axis, activations_dir, held_out, n_layers)
    actual_n_layers = int(axis.shape[0])

    gate_layers = list(cfg.get("gate_layers") or [])
    auto_selected = False
    if not gate_layers:
        gate_layers = select_primary_layers(results["auroc_by_layer"], actual_n_layers)
        auto_selected = True
        print(f"Auto-selected gate layers (mid-late AUROC): {gate_layers}", flush=True)
    if not gate_layers:
        raise SystemExit("No gate layers available (empty config and empty AUROC scores)")

    passed = check_gate(results["auroc_by_layer"], gate_layers, threshold)
    primary = gate_layers[0]
    primary_auroc = results["auroc_by_layer"].get(str(primary), float("nan"))

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preset": cfg["preset"],
        "model": cfg.get("model"),
        "enable_thinking": cfg.get("enable_thinking"),
        "output": str(axis_path),
        "shape": list(axis.shape),
        "gate_threshold": threshold,
        "gate_floor": gate_floor,
        "gate_layers": gate_layers,
        "gate_layers_auto_selected": auto_selected,
        "primary_layer": primary,
        "primary_auroc": primary_auroc,
        "gate_passed": passed,
        "auroc_by_layer": results["auroc_by_layer"],
        "n_held_out_conversations": results.get("n_held_out_conversations"),
        "icrl_path": str(icrl_path) if icrl_path else None,
        "activations_dir": str(activations_dir),
        **(meta or {}),
    }
    manifest_path: Path = cfg["manifest_path"]
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    auroc_out = {
        **results,
        "gate_threshold": threshold,
        "gate_floor": gate_floor,
        "gate_layers": gate_layers,
        "primary_layer": primary,
        "preset": cfg["preset"],
    }
    if cfg.get("preset") == "default" and "paper_targets" in cfg:
        auroc_out["paper_targets"] = {str(k): v for k, v in cfg["paper_targets"].items()}
    with open(cfg["auroc_path"], "w") as f:
        json.dump(auroc_out, f, indent=2)

    paper_targets = cfg.get("paper_targets") if cfg.get("preset") == "default" else None
    plot_auroc(results["auroc_by_layer"], cfg["plot_path"], paper_targets)

    print(f"manifest -> {manifest_path}", flush=True)
    print(f"auroc -> {cfg['auroc_path']}", flush=True)
    print(f"plot -> {cfg['plot_path']}", flush=True)
    for layer in gate_layers:
        val = results["auroc_by_layer"].get(str(layer), float("nan"))
        status = "pass" if not np.isnan(val) and val >= threshold else "fail"
        print(f"  L{layer}: {val:.4f} [{status}] (threshold {threshold})", flush=True)
    if (
        not passed
        and not np.isnan(primary_auroc)
        and primary_auroc >= gate_floor
        and primary_auroc < threshold
    ):
        print(
            f"  note: primary AUROC {primary_auroc:.4f} is above soft floor "
            f"{gate_floor} but below gate {threshold}",
            flush=True,
        )

    if passed or mock_only:
        return 0
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--preset",
        choices=["default", "dev", "qwen32b"],
        default="default",
        help="Config preset (default / dev / qwen32b)",
    )
    ap.add_argument(
        "--stage",
        choices=["extract", "build", "gate", "all"],
        default="all",
        help="Run only one stage, or all (default)",
    )
    ap.add_argument("--icrl", type=Path, default=None, help="ICRL JSON path (required for extract)")
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override gate threshold",
    )
    ap.add_argument("--skip-extract", action="store_true", help="Deprecated: use --stage build/gate")
    ap.add_argument("--skip-mock", action="store_true", help="Do not regenerate mock_icrl.json")
    ap.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract even if .npz caches exist",
    )
    ap.add_argument(
        "--activations-dir",
        type=Path,
        default=None,
        help="Override activation cache dir",
    )
    ap.add_argument("--n-layers", type=int, default=None)
    ap.add_argument(
        "--mock-only",
        action="store_true",
        help="Exit 0 even if gate fails (offline wiring test)",
    )
    args = ap.parse_args()

    # Back-compat: --skip-extract with default stage all → build+gate only
    stage = args.stage
    if args.skip_extract and stage == "all":
        stage = "gate"
        # old behavior ran build+gate; keep that by doing both via a local flag
        do_build_then_gate = True
    else:
        do_build_then_gate = False

    cfg = load_preset(args.preset)
    if args.icrl is not None:
        icrl_path = args.icrl
    elif args.preset == "default" and not args.skip_mock and stage in ("extract", "all"):
        icrl_path = data_file("mock_icrl.json")
    else:
        icrl_path = cfg["icrl_path"]
    activations_dir: Path = args.activations_dir or cfg["activations_dir"]
    activations_dir.mkdir(parents=True, exist_ok=True)
    n_layers = args.n_layers or cfg["n_layers"]
    threshold = resolve_threshold(cfg, args.threshold)
    gate_floor = float(cfg.get("gate_floor", threshold))

    if args.preset == "default" and not args.skip_mock and args.icrl is None and stage in ("extract", "all"):
        from tests.fixtures.icrl_mock import write_mock_icrl

        write_mock_icrl(icrl_path)
        print(f"Wrote mock ICRL -> {icrl_path}", flush=True)

    if stage in ("extract", "all"):
        if not icrl_path.exists():
            raise SystemExit(f"ICRL file not found: {icrl_path}")
        stage_extract(
            cfg,
            icrl_path,
            activations_dir,
            force=args.force_extract,
            n_layers=n_layers,
        )
        if stage == "extract":
            sys.exit(0)

    axis = None
    meta = None
    if stage in ("build", "all") or do_build_then_gate:
        axis, meta = stage_build(cfg, activations_dir, n_layers)
        if stage == "build":
            sys.exit(0)

    if stage in ("gate", "all") or do_build_then_gate:
        # If only --stage gate, load axis from disk; if all/build+gate, reuse in-memory.
        code = stage_gate(
            cfg,
            activations_dir,
            icrl_path if icrl_path.exists() else None,
            n_layers=n_layers,
            threshold=threshold,
            gate_floor=gate_floor,
            axis=axis,
            meta=meta,
            mock_only=args.mock_only,
        )
        sys.exit(code)

    sys.exit(0)


if __name__ == "__main__":
    main()
