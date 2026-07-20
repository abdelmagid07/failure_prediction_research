#!/usr/bin/env python
"""Stage 1 pipeline: extract activations, build axis, evaluate AUROC, apply gate."""

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
    """Pass iff every gate layer's held-out AUROC is at least ``threshold``.

    METHOD.tex: the reconstruction must land within ``gate_tolerance`` of the
    published held-out AUROC. The caller sets ``threshold = published_auroc -
    gate_tolerance`` (default 0.87 = 0.90 - 0.03).
    """
    for layer in gate_layers:
        val = auroc_by_layer.get(str(layer), float("nan"))
        if np.isnan(val) or val < threshold:
            return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--preset",
        choices=["default", "dev"],
        default="default",
        help="Config preset (default: METHOD.tex gate published-0.03; "
        "dev: separate artifacts, 0.75 gate)",
    )
    ap.add_argument("--icrl", type=Path, default=None, help="ICRL JSON path")
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override gate threshold (default: published_auroc - gate_tolerance)",
    )
    ap.add_argument("--skip-extract", action="store_true", help="Use cached activations only")
    ap.add_argument("--skip-mock", action="store_true", help="Do not regenerate mock_icrl.json")
    ap.add_argument("--force-extract", action="store_true")
    ap.add_argument("--n-layers", type=int, default=None)
    ap.add_argument(
        "--mock-only",
        action="store_true",
        help="Exit 0 even if gate fails (offline wiring test)",
    )
    args = ap.parse_args()

    cfg = load_preset(args.preset)
    if args.icrl is not None:
        icrl_path = args.icrl
    elif args.preset == "default" and not args.skip_mock:
        icrl_path = data_file("mock_icrl.json")
    else:
        icrl_path = cfg["icrl_path"]
    activations_dir: Path = cfg["activations_dir"]
    n_layers = args.n_layers or cfg["n_layers"]
    if args.threshold is not None:
        threshold = args.threshold
    elif "published_auroc" in cfg and "gate_tolerance" in cfg:
        threshold = float(cfg["published_auroc"]) - float(cfg["gate_tolerance"])
    else:
        threshold = cfg["gate_threshold"]

    if args.preset == "default" and not args.skip_mock and args.icrl is None:
        from tests.fixtures.icrl_mock import write_mock_icrl

        write_mock_icrl(icrl_path)
        print(f"Wrote mock ICRL -> {icrl_path}", flush=True)

    if not icrl_path.exists():
        raise SystemExit(f"ICRL file not found: {icrl_path}")

    if not args.skip_extract:
        print("extract_activations", flush=True)
        extract_run(
            icrl_path,
            model_name=cfg["model"],
            n_layers=n_layers,
            enable_thinking=cfg["enable_thinking"],
            dtype=cfg["dtype"],
            force=args.force_extract,
            activations_dir=activations_dir,
        )

    split = load_split()
    train_criteria = set(split["train"])
    held_out = set(split["held_out"])

    print("build_axis", flush=True)
    axis, meta = build_axis(activations_dir, train_criteria, n_layers)
    axis_path: Path = cfg["axis_path"]
    np.save(axis_path, axis)

    print("eval_auroc", flush=True)
    results = eval_auroc(axis, activations_dir, held_out, n_layers)
    passed = check_gate(results["auroc_by_layer"], cfg["gate_layers"], threshold)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preset": cfg["preset"],
        "output": str(axis_path),
        "shape": list(axis.shape),
        "gate_threshold": threshold,
        "gate_layers": cfg["gate_layers"],
        "gate_passed": passed,
        "auroc_by_layer": results["auroc_by_layer"],
        "n_held_out_conversations": results.get("n_held_out_conversations"),
        "icrl_path": str(icrl_path),
        "activations_dir": str(activations_dir),
        **meta,
    }
    manifest_path: Path = cfg["manifest_path"]
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    auroc_out = {
        **results,
        "gate_threshold": threshold,
        "gate_layers": cfg["gate_layers"],
        "preset": cfg["preset"],
    }
    if args.preset == "default":
        auroc_out["paper_targets"] = {str(k): v for k, v in cfg["paper_targets"].items()}
    with open(cfg["auroc_path"], "w") as f:
        json.dump(auroc_out, f, indent=2)

    paper_targets = cfg.get("paper_targets") if args.preset == "default" else None
    plot_auroc(results["auroc_by_layer"], cfg["plot_path"], paper_targets)

    print(f"axis -> {axis_path}", flush=True)
    print(f"manifest -> {manifest_path}", flush=True)
    for layer in cfg["gate_layers"]:
        val = results["auroc_by_layer"].get(str(layer), float("nan"))
        status = "pass" if not np.isnan(val) and val >= threshold else "fail"
        print(f"  L{layer}: {val:.4f} [{status}] (threshold {threshold})", flush=True)

    if passed:
        sys.exit(0)
    if args.mock_only:
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
