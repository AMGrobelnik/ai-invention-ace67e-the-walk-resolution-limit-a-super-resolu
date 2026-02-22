#!/usr/bin/env python3
"""
Generate Figure 3: GPS Benchmark - RWSE vs LapPE vs SRWE Across Datasets
Version 3: Refined star positioning, ±std labels, polished layout.
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
    'axes.labelsize': 12.5,
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

# ---- Figure: ~2:1 aspect ratio ----
fig, axes = plt.subplots(1, 3, figsize=(14, 6),
                          gridspec_kw={'width_ratios': [1, 1, 1], 'wspace': 0.40})

n_methods = len(methods)
bar_width = 0.58

for idx, (dset, metric) in enumerate(zip(datasets, metric_types)):
    ax = axes[idx]

    vals = [values[dset][m] for m in methods]
    errs = [stds[dset][m] for m in methods]

    x = np.arange(n_methods)
    bars = ax.bar(x, vals, width=bar_width,
                  color=[colors[m] for m in methods],
                  edgecolor=[edge_colors[m] for m in methods],
                  linewidth=0.7,
                  yerr=errs, capsize=5,
                  error_kw={'linewidth': 1.2, 'capthick': 1.2, 'color': '#444444', 'zorder': 5})

    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontweight='medium')

    # Title
    ax.set_title(dset, fontsize=14, fontweight='bold', pad=14)

    # Y-axis label with direction
    if metric == 'MAE':
        ax.set_ylabel('MAE  ↓', fontsize=13, fontweight='medium')
    else:
        ax.set_ylabel('AP  ↑', fontsize=13, fontweight='medium')

    # Y-axis limits and ticks
    if dset == 'ZINC-subset':
        ax.set_ylim(0, 0.42)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
    elif dset == 'Peptides-func':
        ax.set_ylim(0.235, 0.30)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.01))
    elif dset == 'Peptides-struct':
        ax.set_ylim(14, 22)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(1.0))

    # Identify best bar
    if metric == 'MAE':
        best_idx = int(np.argmin(vals))
    else:
        best_idx = int(np.argmax(vals))

    # Value labels + std on top of bars
    for i, (bar, val, err) in enumerate(zip(bars, vals, errs)):
        bar_center_x = bar.get_x() + bar.get_width() / 2

        if dset == 'Peptides-struct':
            main_label = f'{val:.2f}'
            std_label = f'±{err:.2f}'
            y_offset = 0.40
        elif dset == 'ZINC-subset':
            main_label = f'{val:.3f}'
            std_label = f'±{err:.3f}'
            y_offset = 0.010
        else:
            main_label = f'{val:.3f}'
            std_label = f'±{err:.3f}'
            y_offset = 0.0025

        y_pos = val + err + y_offset

        # Bold the best value
        fw = 'bold' if i == best_idx else 'semibold'
        fs = 10.5 if i == best_idx else 9.5

        ax.text(bar_center_x, y_pos, main_label,
                ha='center', va='bottom', fontsize=fs, fontweight=fw,
                color='#222222')
        ax.text(bar_center_x, y_pos - (y_offset * 0.15), std_label,
                ha='center', va='top', fontsize=7.5, fontweight='normal',
                color='#888888')

    # Gold star clearly centered above best bar
    best_bar = bars[best_idx]
    best_bar_x = best_bar.get_x() + best_bar.get_width() / 2
    best_val = vals[best_idx]
    best_err = errs[best_idx]

    if dset == 'Peptides-struct':
        star_y = best_val + best_err + 2.0
    elif dset == 'ZINC-subset':
        star_y = best_val + best_err + 0.035
    else:
        star_y = best_val + best_err + 0.010

    ax.text(best_bar_x, star_y, '★', ha='center', va='center',
            fontsize=15, color='#DAA520')

    # Grid
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=0.25, linewidth=0.5, color='#aaaaaa')
    ax.grid(axis='x', visible=False)

    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.7)
        ax.spines[spine].set_color('#888888')

# Suptitle
fig.suptitle('GPS Graph Transformer Performance by Encoding Type',
             fontsize=16, fontweight='bold', y=1.02)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[m], edgecolor=edge_colors[m],
                         linewidth=0.7, label=m) for m in methods]
leg = fig.legend(handles=legend_elements, loc='lower center', ncol=3,
                 frameon=True, fancybox=False, edgecolor='#cccccc',
                 bbox_to_anchor=(0.5, -0.04), fontsize=12,
                 handlelength=1.5, handletextpad=0.6, columnspacing=2.5)
leg.get_frame().set_linewidth(0.5)
leg.get_frame().set_facecolor('white')

# Annotation
fig.text(0.5, -0.09, 'MAE: lower is better ↓   |   AP: higher is better ↑',
         ha='center', va='top', fontsize=10.5, style='italic', color='#777777')

plt.savefig(sys.argv[1], format='png', facecolor='white', edgecolor='none')
print(f"Saved to {sys.argv[1]}")
