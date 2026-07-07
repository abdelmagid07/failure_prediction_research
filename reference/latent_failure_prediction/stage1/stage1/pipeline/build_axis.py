#!/usr/bin/env python
"""Build value axis from train-split activation caches (Eq. 1 in paper)."""

import argparse
from pathlib import Path

import numpy as np

from stage1.common.config import load_defaults, load_split
from stage1.common.paths import ACTIVATIONS_DIR, data_file


def load_activation_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def build_axis(
    activation_dir: Path,
    train_criteria: set[str],
    n_layers: int,
) -> tuple[np.ndarray, dict]:
    post_sum = np.zeros((n_layers, 0), dtype=np.float64)
    pre_sum = np.zeros((n_layers, 0), dtype=np.float64)
    n_post = 0
    n_pre = 0
    used_convs = []

    hidden_dim = None
    actual_n_layers = None
    for npz_path in sorted(activation_dir.glob("*.npz")):
        d = load_activation_npz(npz_path)
        crit = str(d["criterion_id"])
        if crit not in train_criteria:
            continue

        acts = d["layer_activations"].astype(np.float64)  # (L, seq, H)
        if hidden_dim is None:
            hidden_dim = acts.shape[-1]
            actual_n_layers = acts.shape[0]
            # Use actual layer count from activations, not config (robust to model variations).
            if actual_n_layers != n_layers:
                print(f"Warning: config says n_layers={n_layers}, but activations have {actual_n_layers}. Using {actual_n_layers}.", flush=True)
            post_sum = np.zeros((actual_n_layers, hidden_dim), dtype=np.float64)
            pre_sum = np.zeros((actual_n_layers, hidden_dim), dtype=np.float64)

        labels = d["token_labels"]
        for layer in range(actual_n_layers):
            layer_h = acts[layer]
            pre_mask = labels == 0
            post_mask = labels == 1
            if pre_mask.any():
                pre_sum[layer] += layer_h[pre_mask].sum(axis=0)
                n_pre += int(pre_mask.sum())
            if post_mask.any():
                post_sum[layer] += layer_h[post_mask].sum(axis=0)
                n_post += int(post_mask.sum())

        used_convs.append(npz_path.stem)

    if n_pre == 0 or n_post == 0:
        raise RuntimeError(
            f"No train activations found (n_pre={n_pre}, n_post={n_post}). "
            "Check split.json and activation caches."
        )

    mean_post = post_sum / n_post
    mean_pre = pre_sum / n_pre
    axis = (mean_post - mean_pre).astype(np.float32)

    meta = {
        "n_pre_tokens": n_pre,
        "n_post_tokens": n_post,
        "n_conversations": len(used_convs),
        "conversation_ids": used_convs,
        "hidden_dim": hidden_dim,
    }
    return axis, meta


def main():
    defaults = load_defaults()
    split = load_split()
    train_criteria = set(split["train"])

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activation-dir", type=Path, default=ACTIVATIONS_DIR)
    ap.add_argument("--output", type=Path, default=data_file("value_axis.npy"))
    ap.add_argument("--n-layers", type=int, default=defaults["n_layers"])
    ap.add_argument("--manifest", type=Path, default=data_file("axis_manifest.json"))
    args = ap.parse_args()

    axis, meta = build_axis(args.activation_dir, train_criteria, args.n_layers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, axis)

    import json
    from datetime import datetime, timezone

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": str(args.output),
        "shape": list(axis.shape),
        **meta,
    }
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved axis {axis.shape} -> {args.output}", flush=True)
    print(f"  train convs: {meta['n_conversations']}, pre tokens: {meta['n_pre_tokens']}, post: {meta['n_post_tokens']}", flush=True)


if __name__ == "__main__":
    main()
