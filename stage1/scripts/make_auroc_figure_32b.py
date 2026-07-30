#!/usr/bin/env python
"""NeurIPS / Jiang-faithful held-out AUROC figure for Qwen3-32B (no gate lines)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

STAGE1 = Path(__file__).resolve().parents[1]
DATA = STAGE1 / "data"
ROOT = DATA / "figures"
ROOT.mkdir(parents=True, exist_ok=True)

data = json.loads((DATA / "auroc_by_layer_32b.json").read_text())
by = {int(k): float(v) for k, v in data["auroc_by_layer"].items()}
layers = np.array(sorted(by), dtype=int)
vals = np.array([by[int(l)] for l in layers], dtype=float)
primary = int(data["primary_layer"])
primary_auroc = float(by[primary])
n_held = int(data["n_held_out_conversations"])

# Coordinate table for PGFPlots
(ROOT / "auroc_by_layer_32b.dat").write_text(
    "layer auroc\n" + "\n".join(f"{l} {v:.10f}" for l, v in zip(layers, vals)) + "\n",
    encoding="utf-8",
)

# Standalone PGFPlots — compile with pdflatex for true Times / NeurIPS fonts
standalone = f"""% Standalone NeurIPS-style figure (conventions from og_paper.tex)
% Compile: pdflatex fig_auroc_by_layer_32b_standalone.tex
\\documentclass[border=2pt]{{standalone}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{times}}
\\usepackage{{pgfplots}}
\\pgfplotsset{{compat=1.18}}
\\definecolor{{darkblue}}{{rgb}}{{0.1,0.2,0.5}}
\\definecolor{{accentred}}{{rgb}}{{0.75,0.22,0.17}}

\\begin{{document}}
\\begin{{tikzpicture}}
\\begin{{axis}}[
  width=3.96in,
  height=2.45in,
  xlabel={{Layer}},
  ylabel={{Held-out AUROC}},
  xmin=-0.5, xmax=63.5,
  ymin=0.50, ymax=1.00,
  xtick={{0,16,32,48,63}},
  ytick={{0.5,0.6,0.7,0.8,0.9,1.0}},
  tick label style={{font=\\small}},
  label style={{font=\\small}},
  legend style={{
    font=\\footnotesize,
    at={{(0.03,0.97)}},
    anchor=north west,
    draw=none,
    fill=none,
    inner sep=1pt,
    row sep=0.5pt,
  }},
  grid=major,
  grid style={{line width=0.3pt, gray!25}},
  axis line style={{line width=0.6pt}},
  tick style={{line width=0.6pt}},
  axis x line*=bottom,
  axis y line*=left,
]
\\addplot[
  darkblue,
  line width=1.1pt,
  mark=*,
  mark size=1.05pt,
  mark options={{solid, fill=white, draw=darkblue, line width=0.6pt}},
  mark repeat=4,
]
table[x=layer, y=auroc] {{auroc_by_layer_32b.dat}};
\\addlegendentry{{Qwen3-32B (thinking ON)}}

\\addplot[
  only marks,
  mark=diamond*,
  mark size=2.6pt,
  accentred,
  mark options={{fill=accentred, draw=white, line width=0.4pt}},
] coordinates {{({primary},{primary_auroc:.10f})}};
\\addlegendentry{{Primary L{primary} ({primary_auroc:.3f})}}
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{document}}
"""
(ROOT / "fig_auroc_by_layer_32b_standalone.tex").write_text(standalone, encoding="utf-8")

# TikZ body for input into a NeurIPS main file (inherits document fonts)
tikz = f"""% Requires preamble colors/packages as in og_paper.tex:
%   \\usepackage{{tikz}}\\usepackage{{pgfplots}}\\pgfplotsset{{compat=1.18}}
%   \\definecolor{{darkblue}}{{rgb}}{{0.1,0.2,0.5}}
%   \\definecolor{{accentred}}{{rgb}}{{0.75,0.22,0.17}}
\\begin{{tikzpicture}}
\\begin{{axis}}[
  width=0.72\\linewidth,
  height=0.445\\linewidth,
  xlabel={{Layer}},
  ylabel={{Held-out AUROC}},
  xmin=-0.5, xmax=63.5,
  ymin=0.50, ymax=1.00,
  xtick={{0,16,32,48,63}},
  ytick={{0.5,0.6,0.7,0.8,0.9,1.0}},
  tick label style={{font=\\small}},
  label style={{font=\\small}},
  legend style={{
    font=\\footnotesize,
    at={{(0.03,0.97)}},
    anchor=north west,
    draw=none,
    fill=none,
    inner sep=1pt,
    row sep=0.5pt,
  }},
  grid=major,
  grid style={{line width=0.3pt, gray!25}},
  axis line style={{line width=0.6pt}},
  tick style={{line width=0.6pt}},
]
\\addplot[
  darkblue,
  line width=1.1pt,
  mark=*,
  mark size=1.05pt,
  mark options={{solid, fill=white, draw=darkblue, line width=0.6pt}},
  mark repeat=4,
]
table[x=layer, y=auroc] {{figures/auroc_by_layer_32b.dat}};
\\addlegendentry{{Qwen3-32B (thinking ON)}}

\\addplot[
  only marks,
  mark=diamond*,
  mark size=2.6pt,
  accentred,
  mark options={{fill=accentred, draw=white, line width=0.4pt}},
] coordinates {{({primary},{primary_auroc:.10f})}};
\\addlegendentry{{Primary L{primary} ({primary_auroc:.3f})}}
\\end{{axis}}
\\end{{tikzpicture}}
"""
(ROOT / "fig_auroc_by_layer_32b.tikz").write_text(tikz, encoding="utf-8")

# Matplotlib vector PDF (no gate / floor / shaded band)
DARKBLUE = (0.10, 0.20, 0.50)
ACCENT = (0.75, 0.22, 0.17)

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
    }
)

fig, ax = plt.subplots(figsize=(3.96, 2.45))
ax.plot(layers, vals, color=DARKBLUE, lw=1.35, solid_capstyle="round", zorder=3)
mark = np.arange(0, len(layers), 4)
if (len(layers) - 1) not in mark:
    mark = np.append(mark, len(layers) - 1)
ax.plot(
    layers[mark],
    vals[mark],
    linestyle="None",
    marker="o",
    markersize=2.4,
    markerfacecolor="white",
    markeredgecolor=DARKBLUE,
    markeredgewidth=0.7,
    zorder=4,
)
ax.plot(
    primary,
    primary_auroc,
    linestyle="None",
    marker="D",
    markersize=5.2,
    markerfacecolor=ACCENT,
    markeredgecolor="white",
    markeredgewidth=0.45,
    zorder=5,
)

ax.set_xlim(-0.5, 63.5)
ax.set_ylim(0.50, 1.00)
ax.set_xlabel("Layer")
ax.set_ylabel("Held-out AUROC")
ax.set_xticks([0, 16, 32, 48, 63])
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.grid(True, axis="y", color="0.88", lw=0.45, zorder=0)
ax.set_axisbelow(True)
ax.legend(
    handles=[
        Line2D(
            [0],
            [0],
            color=DARKBLUE,
            lw=1.35,
            marker="o",
            markersize=3,
            markerfacecolor="white",
            markeredgecolor=DARKBLUE,
            label="Qwen3-32B (thinking ON)",
        ),
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="D",
            markersize=5.2,
            markerfacecolor=ACCENT,
            markeredgecolor="white",
            label=f"Primary L{primary} ({primary_auroc:.3f})",
        ),
    ],
    loc="lower right",
    frameon=False,
    handlelength=1.6,
    handletextpad=0.4,
    borderaxespad=0.2,
)
fig.tight_layout(pad=0.05)

pdf = ROOT / "fig_auroc_by_layer_32b.pdf"
png = ROOT / "fig_auroc_by_layer_32b.png"
fig.savefig(pdf, format="pdf")
fig.savefig(png, format="png", dpi=600)
fig.savefig(DATA / "auroc_by_layer_32b_paper.pdf", format="pdf")
fig.savefig(DATA / "auroc_by_layer_32b_paper.png", format="png", dpi=600)
plt.close(fig)

(ROOT / "fig_auroc_by_layer_32b_include.tex").write_text(
    f"""% NeurIPS include (Jiang caption style from og_paper.tex).
% Preferred (font-perfect): compile fig_auroc_by_layer_32b_standalone.tex on Overleaf
%   with auroc_by_layer_32b.dat in the same folder, then include the resulting PDF.
% Fallback: upload fig_auroc_by_layer_32b.pdf as below.
%
% Preamble (as in og_paper.tex):
%   \\usepackage{{graphicx}} \\graphicspath{{{{figures/}}}}
%   \\usepackage{{caption}}
%   \\captionsetup{{font=small, aboveskip=4pt, belowskip=2pt}}

\\begin{{figure}}[t]
  \\centering
  \\includegraphics[width=0.72\\linewidth]{{figures/fig_auroc_by_layer_32b.pdf}}
  \\caption{{\\textbf{{The Qwen3-32B value axis generalizes to held-out ICRL criteria.}}
  Held-out token AUROC by layer for a difference-in-means value axis constructed
  with thinking ON ($N{{=}}{n_held}$ held-out conversations).
  Separation strengthens through the middle-to-late layers and peaks at layer {primary}
  (AUROC $= {primary_auroc:.3f}$), which we fix as the primary readout for subsequent analyses.}}
  \\label{{fig:auroc_32b}}
\\end{{figure}}
""",
    encoding="utf-8",
)

print(f"primary L{primary} = {primary_auroc:.6f}")
print(f"wrote {pdf}")
print(f"wrote {ROOT / 'fig_auroc_by_layer_32b_standalone.tex'}")
print(f"wrote {ROOT / 'fig_auroc_by_layer_32b_include.tex'}")
