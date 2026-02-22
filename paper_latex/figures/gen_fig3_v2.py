#!/usr/bin/env python3
"""
Generate Figure 3: GPS Benchmark - RWSE vs LapPE vs SRWE Across Datasets
Grouped bar chart with separate subplots for NeurIPS publication.
Version 2: Improved layout, star positioning, and readability.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import numpy as np
import sys

# ---- Configuration ----
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 10.5,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2,
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.5,
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

# ---- Figure: 2:1 aspect ratio ----
fig, axes = plt.subplots(1, 3, figsize=(13, 5.8),
                          gridspec_kw={'width_ratios': [1, 1, 1], 'wspace': 0.38})

n_methods = len(methods)

for idx, (dset, metric) in enumerate(zip(datasets, metric_types)):
    ax = axes[idx]

    vals = [values[dset][m] for m in methods]
    errs = [stds[dset][m] for m in methods]

    x = np.arange(n_methods)
    bars = ax.bar(x, vals, width=0.56, color=[colors[m] for m in methods],
                  edgecolor=['#336688', '#CC6622', '#1A6628'],
                  linewidth=0.6,
                  yerr=errs, capsize=5,
                  error_kw={'linewidth': 1.3, 'capthick': 1.3, 'color': '#444444', 'zorder': 5})

    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontweight='medium')

    # Title with dataset name
    ax.set_title(dset, fontsize=14, fontweight='bold', pad=14)

    # Y-axis label
    if metric == 'MAE':
        ax.set_ylabel('MAE  ↓', fontsize=12.5, fontweight='medium')
    else:
        ax.set_ylabel('AP  ↑', fontsize=12.5, fontweight='medium')

    # Y-axis limits
    if dset == 'ZINC-subset':
        ax.set_ylim(0, 0.40)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
    elif dset == 'Peptides-func':
        ax.set_ylim(0.235, 0.295)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.01))
    elif dset == 'Peptides-struct':
        ax.set_ylim(14, 21.5)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(1.0))

    # Value labels on top of bars (with std offset)
    for i, (bar, val, err) in enumerate(zip(bars, vals, errs)):
        if dset == 'Peptides-struct':
            label = f'{val:.2f}'
            y_pos = val + err + 0.35
        elif dset == 'ZINC-subset':
            label = f'{val:.3f}'
            y_pos = val + err + 0.008
        else:
            label = f'{val:.3f}'
            y_pos = val + err + 0.002

        ax.text(bar.get_x() + bar.get_width()/2, y_pos, label,
                ha='center', va='bottom', fontsize=10, fontweight='semibold',
                color='#333333')

    # Identify best bar
    if metric == 'MAE':
        best_idx = np.argmin(vals)
    else:
        best_idx = np.argmax(vals)

    # Add gold star above the best performing bar
    best_val = vals[best_idx]
    best_err = errs[best_idx]
    if dset == 'Peptides-struct':
        star_y = best_val + best_err + 1.6
    elif dset == 'ZINC-subset':
        star_y = best_val + best_err + 0.028
    else:
        star_y = best_val + best_err + 0.008

    ax.annotate('★', xy=(best_idx, star_y),
                ha='center', va='center', fontsize=16, color='#DAA520',
                path_effects=[pe.withStroke(linewidth=0.5, foreground='#B8860B')])

    # Grid and spines
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=0.25, linewidth=0.5, color='#999999')
    ax.grid(axis='x', visible=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['bottom'].set_color('#666666')

# Suptitle
fig.suptitle('GPS Graph Transformer Performance by Encoding Type',
             fontsize=16, fontweight='bold', y=1.01)

# Legend at bottom center
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[m], edgecolor='#666666', linewidth=0.5, label=m) for m in methods]
leg = fig.legend(handles=legend_elements, loc='lower center', ncol=3,
                 frameon=True, fancybox=False, edgecolor='#cccccc',
                 bbox_to_anchor=(0.5, -0.04), fontsize=12,
                 handlelength=1.5, handletextpad=0.5, columnspacing=2.0)
leg.get_frame().set_linewidth(0.6)

# Direction annotation
fig.text(0.5, -0.09, 'MAE: lower is better ↓   |   AP: higher is better ↑',
         ha='center', va='top', fontsize=10.5, style='italic', color='#777777')

plt.savefig(sys.argv[1], format='png', facecolor='white', edgecolor='none')
print(f"Saved to {sys.argv[1]}")
