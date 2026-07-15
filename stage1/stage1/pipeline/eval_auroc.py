#!/usr/bin/env python
"""Evaluate held-out token-level AUROC per layer (Figure 2a reproduction)."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from stage1.common.config import load_defaults, load_split
from stage1.common.hooks import unit_direction
from stage1.common.paths import ACTIVATIONS_DIR, data_file


def eval_auroc(
    axis: np.ndarray,
    activation_dir: Path,
    held_out_criteria: set[str],
    n_layers: int,
) -> dict:
    # Discover actual layer count from activations (robust to model variations).
    actual_n_layers = axis.shape[0]
    if actual_n_layers != n_layers:
        print(f"Warning: config says n_layers={n_layers}, but axis has {actual_n_layers}. Using {actual_n_layers}.", flush=True)

    scores_by_layer: dict[int, list[float]] = {l: [] for l in range(actual_n_layers)}
    labels_by_layer: dict[int, list[int]] = {l: [] for l in range(actual_n_layers)}

    n_convs = 0
    for npz_path in sorted(activation_dir.glob("*.npz")):
        d = np.load(npz_path, allow_pickle=True)
        crit = str(d["criterion_id"])
        if crit not in held_out_criteria:
            continue

        acts = d["layer_activations"].astype(np.float32)
        labels = d["token_labels"]
        mask = (labels == 0) | (labels == 1)
        if not mask.any():
            continue

        n_convs += 1
        for layer in range(actual_n_layers):
            direction = unit_direction(axis[layer])
            h = acts[layer][mask]
            h_norm = h / np.linalg.norm(h, axis=1, keepdims=True).clip(min=1e-8)
            proj = h_norm @ direction
            scores_by_layer[layer].extend(proj.tolist())
            labels_by_layer[layer].extend(labels[mask].astype(int).tolist())

    auroc = {}
    for layer in range(actual_n_layers):
        y = labels_by_layer[layer]
        s = scores_by_layer[layer]
        if len(set(y)) < 2:
            auroc[str(layer)] = float("nan")
        else:
            auroc_val = roc_auc_score(y, s)
            auroc[str(layer)] = float(max(auroc_val, 1 - auroc_val))

    return {"auroc_by_layer": auroc, "n_held_out_conversations": n_convs}


def plot_auroc(auroc_by_layer: dict, output: Path, paper_targets: dict | None = None):
    layers = sorted(int(k) for k in auroc_by_layer.keys())
    values = [auroc_by_layer[str(l)] for l in layers]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layers, values, marker="o", markersize=3, label="held-out AUROC")
    if paper_targets:
        for layer, target in paper_targets.items():
            ax.axhline(y=target, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
            ax.annotate(f"L{layer} paper={target:.3f}", xy=(layers[-1], target), fontsize=8, alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC (max(auc, 1-auc))")
    ax.set_title("Value axis held-out criteria generalization")
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    defaults = load_defaults()
    split = load_split()
    held_out = set(split["held_out"])

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--axis", type=Path, default=data_file("value_axis.npy"))
    ap.add_argument("--activation-dir", type=Path, default=ACTIVATIONS_DIR)
    ap.add_argument("--output-json", type=Path, default=data_file("auroc_by_layer.json"))
    ap.add_argument("--output-plot", type=Path, default=data_file("auroc_by_layer.png"))
    ap.add_argument("--n-layers", type=int, default=defaults["n_layers"])
    args = ap.parse_args()

    axis = np.load(args.axis)
    results = eval_auroc(axis, args.activation_dir, held_out, args.n_layers)

    out = {
        **results,
        "paper_targets": {str(k): v for k, v in defaults["paper_targets"].items()},
        "gate_threshold": defaults["gate_threshold"],
        "gate_layers": defaults["gate_layers"],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)

    plot_auroc(results["auroc_by_layer"], args.output_plot, defaults["paper_targets"])

    print("AUROC by layer (selected):", flush=True)
    for layer in defaults["gate_layers"]:
        val = results["auroc_by_layer"].get(str(layer), float("nan"))
        print(f"  L{layer}: {val:.4f}  (gate >= {defaults['gate_threshold']})", flush=True)
    print(f"Saved {args.output_json} and {args.output_plot}", flush=True)


if __name__ == "__main__":
    main()
