#!/usr/bin/env python3
"""
Generate Figure 3: GPS Benchmark - RWSE vs LapPE vs SRWE Across Datasets
Grouped bar chart with dual y-axes for NeurIPS publication.
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
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'xtick.labelsize': 12,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

# ---- Data ----
# Datasets and their metric types
datasets = ['ZINC-subset', 'Peptides-func', 'Peptides-struct']
metric_types = ['MAE', 'AP', 'MAE']  # per dataset

# Values: [RWSE, LapPE, SRWE]
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

# ---- Figure ----
fig, ax_left = plt.subplots(figsize=(12, 6))  # 2:1 aspect ratio
ax_right = ax_left.twinx()

# Bar positions: 3 groups, each with 3 bars
n_datasets = len(datasets)
n_methods = len(methods)
bar_width = 0.22
group_width = n_methods * bar_width + 0.15  # gap between groups

# We need separate normalization since scales differ drastically.
# Strategy: Plot each dataset group with its own scale context.
# Left axis: MAE scale for ZINC (0-0.4) and we'll handle Peptides-struct separately
# Right axis: AP scale for Peptides-func

# Actually, since scales are SO different (0.2 vs 18), we need to use
# a broken/normalized approach or separate sub-axes.
# Better approach: Use 3 separate subplots sharing a common legend.

plt.close(fig)

# ---- Better approach: 3 subplots side by side ----
fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), gridspec_kw={'width_ratios': [1, 1, 1], 'wspace': 0.35})

for idx, (dset, metric) in enumerate(zip(datasets, metric_types)):
    ax = axes[idx]

    vals = [values[dset][m] for m in methods]
    errs = [stds[dset][m] for m in methods]

    x = np.arange(n_methods)
    bars = ax.bar(x, vals, width=0.55, color=[colors[m] for m in methods],
                  edgecolor='white', linewidth=0.8,
                  yerr=errs, capsize=5, error_kw={'linewidth': 1.2, 'capthick': 1.2, 'color': '#333333'})

    # Set labels
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontweight='medium')
    ax.set_title(dset, fontsize=14, fontweight='bold', pad=12)

    # Y-axis label with direction annotation
    if metric == 'MAE':
        ax.set_ylabel(f'MAE  $\\downarrow$', fontsize=13, fontweight='medium')
    else:
        ax.set_ylabel(f'AP  $\\uparrow$', fontsize=13, fontweight='medium')

    # Set appropriate y-limits with padding
    max_val = max(v + e for v, e in zip(vals, errs))
    min_val = min(v - e for v, e in zip(vals, errs))

    if dset == 'ZINC-subset':
        ax.set_ylim(0, 0.38)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
    elif dset == 'Peptides-func':
        ax.set_ylim(0.24, 0.29)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.01))
    elif dset == 'Peptides-struct':
        ax.set_ylim(14, 21)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(1.0))

    # Add value labels on top of bars
    for bar, val, err in zip(bars, vals, errs):
        y_pos = val + err + (max_val * 0.02 if dset != 'Peptides-func' else 0.002)
        if dset == 'Peptides-struct':
            label = f'{val:.2f}'
            y_pos = val + err + 0.3
        elif dset == 'ZINC-subset':
            label = f'{val:.3f}'
            y_pos = val + err + 0.008
        else:
            label = f'{val:.3f}'
            y_pos = val + err + 0.002
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, label,
                ha='center', va='bottom', fontsize=10, fontweight='medium', color='#333333')

    # Grid styling
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax.grid(axis='x', visible=False)

    # Highlight best bar
    if metric == 'MAE':
        best_idx = np.argmin(vals)
    else:
        best_idx = np.argmax(vals)

    # Add a subtle star marker on best
    best_val = vals[best_idx]
    best_err = errs[best_idx]
    if dset == 'Peptides-struct':
        star_y = best_val + best_err + 1.1
    elif dset == 'ZINC-subset':
        star_y = best_val + best_err + 0.022
    else:
        star_y = best_val + best_err + 0.006
    ax.text(best_idx, star_y, '$\\bigstar$', ha='center', va='bottom',
            fontsize=13, color='#D4AF37')

    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

# Suptitle
fig.suptitle('GPS Graph Transformer Performance by Encoding Type',
             fontsize=16, fontweight='bold', y=1.02)

# Legend at bottom
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[m], edgecolor='white', label=m) for m in methods]
fig.legend(handles=legend_elements, loc='lower center', ncol=3,
           frameon=True, fancybox=False, edgecolor='#cccccc',
           bbox_to_anchor=(0.5, -0.05), fontsize=12)

# Direction annotations in a subtle note
fig.text(0.5, -0.10, 'MAE: lower is better $\\downarrow$  |  AP: higher is better $\\uparrow$',
         ha='center', va='top', fontsize=11, style='italic', color='#666666')

plt.savefig(sys.argv[1], format='png', facecolor='white', edgecolor='none')
print(f"Saved to {sys.argv[1]}")
