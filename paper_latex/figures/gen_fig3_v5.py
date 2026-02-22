#!/usr/bin/env python3
"""
Generate Figure 3: GPS Benchmark - RWSE vs LapPE vs SRWE Across Datasets
Version 5: Final clean version — no overlapping ±std text, just error bars + value labels.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import sys

# ---- Configuration ----
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11.5,
    'ytick.labelsize': 10.5,
    'legend.fontsize': 11.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2,
    'axes.linewidth': 0.7,
    'text.color': '#222222',
    'axes.labelcolor': '#222222',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
})

# ---- Data ----
datasets = ['ZINC-subset', 'Peptides-func', 'Peptides-struct']
metric_types = ['MAE', 'AP', 'MAE']

values = {
    'ZINC-subset':     {'RWSE': 0.199, 'LapPE': 0.298, 'SRWE': 0.233},
    'Peptides-func':   {'RWSE': 0.263, 'LapPE': 0.258, 'SRWE': 0.276},
    'Peptides-struct': {'RWSE': 18.86, 'LapPE': 16.53, 'SRWE': 17.52},
}
stds = {
    'ZINC-subset':     {'RWSE': 0.013, 'LapPE': 0.009, 'SRWE': 0.020},
    'Peptides-func':   {'RWSE': 0.005, 'LapPE': 0.005, 'SRWE': 0.002},
    'Peptides-struct': {'RWSE': 0.87,  'LapPE': 0.54,  'SRWE': 0.45},
}

methods = ['RWSE', 'LapPE', 'SRWE']
colors = {'RWSE': '#4477AA', 'LapPE': '#EE7733', 'SRWE': '#228833'}
edge_colors = {'RWSE': '#355D88', 'LapPE': '#CC5F1E', 'SRWE': '#1A6628'}

# ---- Figure ----
fig, axes = plt.subplots(1, 3, figsize=(14, 6),
                          gridspec_kw={'width_ratios': [1, 1, 1], 'wspace': 0.40})

bar_width = 0.58

for idx, (dset, metric) in enumerate(zip(datasets, metric_types)):
    ax = axes[idx]

    vals = [values[dset][m] for m in methods]
    errs = [stds[dset][m] for m in methods]

    x = np.arange(len(methods))
    bars = ax.bar(x, vals, width=bar_width,
                  color=[colors[m] for m in methods],
                  edgecolor=[edge_colors[m] for m in methods],
                  linewidth=0.7, zorder=3,
                  yerr=errs, capsize=5,
                  error_kw={'linewidth': 1.2, 'capthick': 1.2, 'color': '#444444', 'zorder': 5})

    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontweight='medium')
    ax.tick_params(axis='x', length=0, pad=6)

    # Title
    ax.set_title(dset, fontsize=14, fontweight='bold', pad=14)

    # Y-axis label with arrow
    if metric == 'MAE':
        ax.set_ylabel('MAE  ↓', fontsize=13, fontweight='medium')
    else:
        ax.set_ylabel('AP  ↑', fontsize=13, fontweight='medium')

    # Y-axis limits and formatting
    if dset == 'ZINC-subset':
        ax.set_ylim(0, 0.40)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
        val_fmt = '.3f'
        label_gap = 0.010
        star_gap = 0.032
    elif dset == 'Peptides-func':
        ax.set_ylim(0.240, 0.300)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.01))
        val_fmt = '.3f'
        label_gap = 0.003
        star_gap = 0.009
    elif dset == 'Peptides-struct':
        ax.set_ylim(14, 22)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(1.0))
        val_fmt = '.2f'
        label_gap = 0.40
        star_gap = 1.8

    # Identify best
    if metric == 'MAE':
        best_idx = int(np.argmin(vals))
    else:
        best_idx = int(np.argmax(vals))

    # Value labels above each bar
    for i, (bar, val, err) in enumerate(zip(bars, vals, errs)):
        cx = bar.get_x() + bar.get_width() / 2
        top_y = val + err + label_gap

        fw = 'bold' if i == best_idx else 'medium'
        fs = 11 if i == best_idx else 10

        ax.text(cx, top_y, f'{val:{val_fmt}}',
                ha='center', va='bottom', fontsize=fs, fontweight=fw,
                color='#222222')

    # Gold star above best bar
    best_bar = bars[best_idx]
    bx = best_bar.get_x() + best_bar.get_width() / 2
    sy = vals[best_idx] + errs[best_idx] + star_gap

    ax.text(bx, sy, '★', ha='center', va='bottom',
            fontsize=15, color='#DAA520')

    # Grid & spines
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=0.22, linewidth=0.5, color='#aaaaaa')
    ax.grid(axis='x', visible=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for sp in ['left', 'bottom']:
        ax.spines[sp].set_linewidth(0.7)
        ax.spines[sp].set_color('#888888')

# Suptitle
fig.suptitle('GPS Graph Transformer Performance by Encoding Type',
             fontsize=16, fontweight='bold', y=1.02)

# Legend
from matplotlib.patches import Patch
legend_handles = [Patch(facecolor=colors[m], edgecolor=edge_colors[m],
                        linewidth=0.7, label=m) for m in methods]
leg = fig.legend(handles=legend_handles, loc='lower center', ncol=3,
                 frameon=True, fancybox=False, edgecolor='#cccccc',
                 bbox_to_anchor=(0.5, -0.04), fontsize=12,
                 handlelength=1.5, handletextpad=0.6, columnspacing=2.5)
leg.get_frame().set_linewidth(0.5)
leg.get_frame().set_facecolor('white')

# Direction note
fig.text(0.5, -0.09,
         'MAE: lower is better ↓   |   AP: higher is better ↑   (error bars = std over 3 seeds)',
         ha='center', va='top', fontsize=10, style='italic', color='#777777')

out = sys.argv[1]
plt.savefig(out, format='png', facecolor='white', edgecolor='none')
print(f"Saved to {out}")
