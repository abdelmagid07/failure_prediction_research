#!/usr/bin/env python
"""Jiang-style Fig 2: (a) AUROC-by-layer + (b) pairwise axis cosine similarity (32B)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

STAGE1 = Path(__file__).resolve().parents[1]
DATA = STAGE1 / "data"
FIG = DATA / "figures"
FIG.mkdir(parents=True, exist_ok=True)

DARKBLUE = (0.10, 0.20, 0.50)
ACCENT = (0.75, 0.22, 0.17)

# --- data ---
auroc = json.loads((DATA / "auroc_by_layer_32b.json").read_text())
by = {int(k): float(v) for k, v in auroc["auroc_by_layer"].items()}
layers = np.array(sorted(by), dtype=int)
vals = np.array([by[int(l)] for l in layers], dtype=float)
primary = int(auroc["primary_layer"])
primary_auroc = float(by[primary])
n_held = int(auroc["n_held_out_conversations"])

axis = np.load(DATA / "value_axis_32b.npy").astype(np.float64)
n_layers = axis.shape[0]
norms = np.linalg.norm(axis, axis=1, keepdims=True).clip(min=1e-12)
unit = axis / norms
sim = unit @ unit.T
np.save(FIG / "axis_layer_cosine_32b.npy", sim.astype(np.float32))

# Stats aligned with Jiang Fig 2b reporting style
early_idx = np.arange(0, 4)
late_idx = np.arange(50, n_layers)
mid_lo, mid_hi = 32, 55  # half-open: 32..54
early_m = unit[early_idx].mean(0)
early_m /= np.linalg.norm(early_m)
late_m = unit[late_idx].mean(0)
late_m /= np.linalg.norm(late_m)
early_late_cos = float(early_m @ late_m)
mid_block = sim[mid_lo:mid_hi, mid_lo:mid_hi]
mid_block_mean = float(mid_block.mean())
# first layer whose direction is within 0.5 of late mean (emergence proxy)
sim_to_late = unit @ late_m
emerge = int(np.argmax(sim_to_late > 0.5))

stats = {
    "n_layers": int(n_layers),
    "primary_layer": primary,
    "primary_auroc": primary_auroc,
    "n_held_out_conversations": n_held,
    "early_layers": "0-3",
    "late_layers": f"50-{n_layers - 1}",
    "early_vs_late_mean_dir_cosine": early_late_cos,
    "midlate_block": f"{mid_lo}-{mid_hi - 1}",
    "midlate_block_mean_cosine": mid_block_mean,
    "first_layer_cos_to_late_gt_0.5": emerge,
    "sim_full_range": [float(sim.min()), float(sim.max())],
}
(FIG / "axis_layer_cosine_32b_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)

# Panel size ~ half NeurIPS linewidth (Jiang uses 0.48\linewidth each)
panel_w, panel_h = 2.55, 2.15

# ========== (a) AUROC ==========
fig_a, ax = plt.subplots(figsize=(panel_w, panel_h))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.plot(layers, vals, color=DARKBLUE, lw=1.2, solid_capstyle="round", zorder=3)
mark = np.arange(0, len(layers), 5)
if (len(layers) - 1) not in mark:
    mark = np.append(mark, len(layers) - 1)
ax.plot(
    layers[mark],
    vals[mark],
    linestyle="None",
    marker="o",
    markersize=2.0,
    markerfacecolor="white",
    markeredgecolor=DARKBLUE,
    markeredgewidth=0.6,
    zorder=4,
)
ax.plot(
    primary,
    primary_auroc,
    linestyle="None",
    marker="D",
    markersize=4.5,
    markerfacecolor=ACCENT,
    markeredgecolor="white",
    markeredgewidth=0.4,
    zorder=5,
)
ax.set_xlim(-0.5, 63.5)
ax.set_ylim(0.50, 1.00)
ax.set_xlabel("Layer")
ax.set_ylabel("Held-out AUROC")
ax.set_xticks([0, 16, 32, 48, 63])
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.grid(True, axis="y", color="0.88", lw=0.4, zorder=0)
ax.set_axisbelow(True)
ax.legend(
    handles=[
        Line2D(
            [0],
            [0],
            color=DARKBLUE,
            lw=1.2,
            marker="o",
            markersize=2.5,
            markerfacecolor="white",
            markeredgecolor=DARKBLUE,
            label="Qwen3-32B",
        ),
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="D",
            markersize=4.5,
            markerfacecolor=ACCENT,
            markeredgecolor="white",
            label=f"L{primary} ({primary_auroc:.3f})",
        ),
    ],
    loc="lower right",
    frameon=False,
    handlelength=1.4,
    borderaxespad=0.15,
)
fig_a.tight_layout(pad=0.05)
fig_a.savefig(FIG / "fig_auroc_by_layer_32b_panel.pdf", format="pdf")
fig_a.savefig(FIG / "fig_auroc_by_layer_32b_panel.png", format="png", dpi=600)
plt.close(fig_a)

# ========== (b) heatmap ==========
fig_b, ax = plt.subplots(figsize=(panel_w, panel_h))
# White→darkblue (matches panel (a) / og_paper.tex darkblue)
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "paper_blues",
    [
        (0.00, "#ffffff"),
        (0.12, "#ffffff"),
        (0.30, "#d0d7e8"),
        (0.50, "#7f92b8"),
        (0.70, "#3d5a8c"),
        (0.85, "#1a3466"),
        (1.00, "#0a1433"),  # near darkblue (0.1, 0.2, 0.5)
    ],
    N=256,
)
im = ax.imshow(
    sim,
    origin="lower",
    cmap=cmap,
    vmin=0.0,
    vmax=1.0,
    aspect="equal",
    interpolation="nearest",
)
# Tick strategy: readable at 64 layers (Jiang had 37)
tick_pos = [0, 16, 32, 48, 63]
ax.set_xticks(tick_pos)
ax.set_yticks(tick_pos)
ax.set_xticklabels(tick_pos)
ax.set_yticklabels(tick_pos)
ax.set_xlabel("Layer")
ax.set_ylabel("Layer")
# light grid at major ticks only
ax.set_xticks(np.arange(-0.5, n_layers, 1), minor=True)
ax.set_yticks(np.arange(-0.5, n_layers, 1), minor=True)
ax.tick_params(which="minor", bottom=False, left=False, length=0)
ax.grid(False)
# colorbar
cbar = fig_b.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=6.5, width=0.5, length=2)
cbar.set_label("Cosine similarity", fontsize=7)
cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
# Mark primary (accent red — readable on blue heat)
ax.axhline(primary, color=ACCENT, ls="--", lw=0.75, alpha=0.95)
ax.axvline(primary, color=ACCENT, ls="--", lw=0.75, alpha=0.95)
fig_b.tight_layout(pad=0.05)
fig_b.savefig(FIG / "fig_axis_cosine_32b_panel.pdf", format="pdf")
fig_b.savefig(FIG / "fig_axis_cosine_32b_panel.png", format="png", dpi=600)
plt.close(fig_b)

# ========== combined single PDF (locked layout) ==========
fig, axes = plt.subplots(
    1,
    2,
    figsize=(5.5, 2.45),
    gridspec_kw={"width_ratios": [1.0, 1.08], "wspace": 0.32},
    layout="constrained",
)

ax = axes[0]
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.plot(layers, vals, color=DARKBLUE, lw=1.2, zorder=3)
ax.plot(
    layers[mark],
    vals[mark],
    linestyle="None",
    marker="o",
    markersize=2.0,
    markerfacecolor="white",
    markeredgecolor=DARKBLUE,
    markeredgewidth=0.6,
    zorder=4,
)
ax.plot(
    primary,
    primary_auroc,
    linestyle="None",
    marker="D",
    markersize=4.5,
    markerfacecolor=ACCENT,
    markeredgecolor="white",
    markeredgewidth=0.4,
    zorder=5,
)
ax.set_xlim(-0.5, 63.5)
ax.set_ylim(0.50, 1.00)
ax.set_xlabel("Layer")
ax.set_ylabel("Held-out AUROC")
ax.set_xticks([0, 16, 32, 48, 63])
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.grid(True, axis="y", color="0.88", lw=0.4, zorder=0)
ax.set_axisbelow(True)
ax.legend(
    handles=[
        Line2D(
            [0],
            [0],
            color=DARKBLUE,
            lw=1.2,
            marker="o",
            markersize=2.5,
            markerfacecolor="white",
            markeredgecolor=DARKBLUE,
            label="Qwen3-32B",
        ),
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="D",
            markersize=4.5,
            markerfacecolor=ACCENT,
            markeredgecolor="white",
            label=f"L{primary} ({primary_auroc:.3f})",
        ),
    ],
    loc="lower right",
    frameon=False,
    handlelength=1.3,
    borderaxespad=0.1,
)
ax.set_title(r"$\mathbf{(a)}$ Held-out AUROC", loc="left", fontsize=8, pad=4)

ax = axes[1]
im = ax.imshow(
    sim,
    origin="lower",
    cmap=cmap,
    vmin=0.0,
    vmax=1.0,
    aspect="equal",
    interpolation="nearest",
)
ax.set_xticks(tick_pos)
ax.set_yticks(tick_pos)
ax.set_xlabel("Layer")
ax.set_ylabel("Layer")
ax.axhline(primary, color=ACCENT, ls="--", lw=0.7, alpha=0.95)
ax.axvline(primary, color=ACCENT, ls="--", lw=0.7, alpha=0.95)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=6.5, width=0.5, length=2)
cbar.set_label("Cosine similarity", fontsize=7)
cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax.set_title(r"$\mathbf{(b)}$ Pairwise layer similarity", loc="left", fontsize=8, pad=4)

fig.savefig(FIG / "fig_probe_eval_32b.pdf", format="pdf")
fig.savefig(FIG / "fig_probe_eval_32b.png", format="png", dpi=600)
plt.close(fig)

# --- LaTeX include: Jiang minipage layout ---
(FIG / "fig_probe_eval_32b_include.tex").write_text(
    f"""% Jiang Fig.~2 layout (og_paper.tex): AUROC | layer-similarity heatmap
% Upload: fig_auroc_by_layer_32b_panel.pdf, fig_axis_cosine_32b_panel.pdf
%   OR single combined: fig_probe_eval_32b.pdf
%
% Optional preamble: \\usepackage{{flafter}}
% Placement: put a short prose paragraph BEFORE this float; use [ht] not [t].

% --- Option A: two separate panels (closest to Jiang source structure) ---
\\begin{{figure}}[ht]
  \\begin{{minipage}}[b]{{0.48\\linewidth}}
    \\centering
    \\includegraphics[width=\\linewidth]{{figures/fig_auroc_by_layer_32b_panel.pdf}}
    \\small\\textbf{{(a)}} Held-out AUROC by layer.
  \\end{{minipage}}
  \\hfill
  \\begin{{minipage}}[b]{{0.48\\linewidth}}
    \\centering
    \\includegraphics[width=\\linewidth]{{figures/fig_axis_cosine_32b_panel.pdf}}
    \\small\\textbf{{(b)}} Pairwise similarity across layers.
  \\end{{minipage}}
  \\caption{{\\textbf{{The Qwen3-32B value axis generalizes to held-out criteria and stabilizes in the middle-to-late layers.}}
  \\textbf{{(a)}} Held-out token AUROC peaks at layer {primary} (AUROC $= {primary_auroc:.3f}$; $N{{=}}{n_held}$).
  \\textbf{{(b)}} Pairwise cosine similarity of per-layer axis directions.
  Early (L0--3) and late (L50+) directions are nearly orthogonal (mean-dir.\\ cosine $= {early_late_cos:.2f}$);
  mid--late layers (L{mid_lo}--{mid_hi - 1}) are mutually similar (block mean $= {mid_block_mean:.2f}$).
  Red dashed lines mark the primary layer.}}
  \\label{{fig:probe_eval_32b}}
\\end{{figure}}

% --- Option B (alternative): single combined PDF ---
% \\begin{{figure}}[ht]
%   \\centering
%   \\includegraphics[width=\\linewidth]{{figures/fig_probe_eval_32b.pdf}}
%   \\caption{{...same caption...}}
%   \\label{{fig:probe_eval_32b}}
% \\end{{figure}}
""",
    encoding="utf-8",
)

# Update the older single-panel include to point at the two-panel figure
(FIG / "fig_auroc_by_layer_32b_include.tex").write_text(
    f"""% Deprecated single-panel include — use fig_probe_eval_32b_include.tex instead
% (AUROC + layer-similarity, Jiang Fig.~2 layout).

\\input{{figures/fig_probe_eval_32b_include.tex}}
""",
    encoding="utf-8",
)

print(json.dumps(stats, indent=2))
print("wrote panels + fig_probe_eval_32b.pdf + include")
