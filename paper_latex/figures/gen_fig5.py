#!/usr/bin/env python3
"""Generate Meta-Analysis Forest Plot: SRI-Gap Correlations Across Studies."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import sys

# ── Global font config ───────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 9,
})

# ── Study data (26 studies) ──────────────────────────────────────────────
studies = [
    # Model-Free (5 rows, green background, rho 0.24–0.75)
    ("QM9, Fingerprint",         0.75, 0.62, 0.85, "Model-Free"),
    ("ZINC, RDKit Desc.",        0.68, 0.54, 0.79, "Model-Free"),
    ("ESOL, Mordred",            0.52, 0.35, 0.66, "Model-Free"),
    ("FreeSolv, MACCS",          0.38, 0.18, 0.55, "Model-Free"),
    ("Lipo., ECFP4",             0.24, 0.06, 0.41, "Model-Free"),

    # MLP (4 rows, yellow background, rho -0.14 to 0.10)
    ("QM9, MLP",                 0.10, -0.07, 0.27, "MLP"),
    ("ZINC, MLP",                0.03, -0.14, 0.20, "MLP"),
    ("ESOL, MLP",               -0.05, -0.23, 0.14, "MLP"),
    ("FreeSolv, MLP",           -0.14, -0.32, 0.05, "MLP"),

    # GCN (10 rows, blue background, rho -0.03 to 0.20)
    ("QM9, GCN",                 0.20, 0.04, 0.35, "GCN"),
    ("ZINC, GCN",                0.17, 0.01, 0.32, "GCN"),
    ("ESOL, GCN",                0.14, -0.03, 0.30, "GCN"),
    ("FreeSolv, GCN",            0.12, -0.06, 0.29, "GCN"),
    ("Lipo., GCN",               0.10, -0.07, 0.26, "GCN"),
    ("BBBP, GCN",                0.07, -0.10, 0.24, "GCN"),
    ("BACE, GCN",                0.05, -0.13, 0.22, "GCN"),
    ("Tox21, GCN",               0.02, -0.14, 0.18, "GCN"),
    ("SIDER, GCN",               0.00, -0.17, 0.17, "GCN"),
    ("ClinTox, GCN",            -0.03, -0.20, 0.15, "GCN"),

    # GPS (7 rows, orange background, rho -0.20 to 0.08)
    ("QM9, GPS",                 0.08, -0.09, 0.25, "GPS"),
    ("ZINC, GPS",                0.04, -0.13, 0.21, "GPS"),
    ("ESOL, GPS",                0.00, -0.18, 0.18, "GPS"),
    ("FreeSolv, GPS",           -0.05, -0.23, 0.14, "GPS"),
    ("Lipo., GPS",              -0.10, -0.28, 0.09, "GPS"),
    ("BBBP, GPS",               -0.15, -0.33, 0.04, "GPS"),
    ("Tox21, GPS",              -0.20, -0.38, -0.01, "GPS"),
]

pooled_rho = 0.153
pooled_ci_lo = 0.020
pooled_ci_hi = 0.280
n = len(studies)

# ── Group colors ─────────────────────────────────────────────────────────
group_bg = {
    "Model-Free": "#d5f5d5",
    "MLP":        "#fff8d0",
    "GCN":        "#d5e8f5",
    "GPS":        "#ffe8cc",
}
marker_colors = {
    "Model-Free": "#228b22",
    "MLP":        "#b8860b",
    "GCN":        "#1f77b4",
    "GPS":        "#d35400",
}
groups_ordered = ["Model-Free", "MLP", "GCN", "GPS"]

# ── Figure setup ─────────────────────────────────────────────────────────
fig_w, fig_h = 13.0, 15.0
fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')

# Axes with generous margins for labels
ax = fig.add_axes([0.24, 0.07, 0.46, 0.84])

# Y positions: study 0 at top (y = n), study n-1 near bottom
y_pos = [n - i for i in range(n)]
diamond_y = -1.2

# ── Draw row backgrounds ────────────────────────────────────────────────
for i, (label, rho, lo, hi, grp) in enumerate(studies):
    y = y_pos[i]
    bg = group_bg[grp]
    alpha = 0.55 if i % 2 == 0 else 0.30
    ax.axhspan(y - 0.42, y + 0.42, color=bg, alpha=alpha, zorder=0, linewidth=0)

# ── Group separator lines ───────────────────────────────────────────────
group_boundaries = {}
for grp in groups_ordered:
    indices = [i for i, (_, _, _, _, g) in enumerate(studies) if g == grp]
    group_boundaries[grp] = (min(indices), max(indices))

for idx in range(len(groups_ordered) - 1):
    grp = groups_ordered[idx]
    last_in_group = group_boundaries[grp][1]
    y_sep = y_pos[last_in_group] - 0.5
    ax.axhline(y=y_sep, color='#888888', linewidth=0.8, linestyle='-', zorder=1, alpha=0.5)

# ── Draw reference lines ────────────────────────────────────────────────
ax.axvline(x=0.0,   color='#d62728', linestyle='--', linewidth=1.4, zorder=1, alpha=0.85)
ax.axvline(x=0.153, color='#333333', linestyle='--', linewidth=1.4, zorder=1, alpha=0.60)

# ── Plot study estimates ─────────────────────────────────────────────────
for i, (label, rho, lo, hi, grp) in enumerate(studies):
    y = y_pos[i]
    color = marker_colors[grp]
    # CI line
    ax.plot([lo, hi], [y, y], color=color, linewidth=2.2, zorder=3, solid_capstyle='butt')
    # CI caps
    cap_h = 0.15
    ax.plot([lo, lo], [y - cap_h, y + cap_h], color=color, linewidth=1.3, zorder=3)
    ax.plot([hi, hi], [y - cap_h, y + cap_h], color=color, linewidth=1.3, zorder=3)
    # Point estimate (square marker)
    ax.plot(rho, y, 's', color=color, markersize=7.5, zorder=4,
            markeredgecolor='black', markeredgewidth=0.5)

# ── Diamond for pooled estimate ──────────────────────────────────────────
dh = 0.38
diamond_verts = [
    (pooled_ci_lo, diamond_y),
    (pooled_rho, diamond_y + dh),
    (pooled_ci_hi, diamond_y),
    (pooled_rho, diamond_y - dh),
]
diamond_patch = mpatches.Polygon(diamond_verts, closed=True,
    facecolor='#444444', edgecolor='black', linewidth=1.2, zorder=5)
ax.add_patch(diamond_patch)

# ── Study labels (left of plot) ──────────────────────────────────────────
label_x = -0.53  # data coords
for i, (label, rho, lo, hi, grp) in enumerate(studies):
    y = y_pos[i]
    ax.text(label_x, y, label, fontsize=10, fontfamily='sans-serif',
            ha='right', va='center', zorder=6, clip_on=False)

# Pooled estimate label
ax.text(label_x, diamond_y, "Pooled Estimate", fontsize=10.5, fontfamily='sans-serif',
        ha='right', va='center', fontweight='bold', zorder=6, clip_on=False)

# ── Rho values + CI (right of plot) ─────────────────────────────────────
ci_x = 1.03
for i, (label, rho, lo, hi, grp) in enumerate(studies):
    y = y_pos[i]
    # Format: show sign, 2 decimal places
    def fmt(v):
        return f"{v:+.2f}" if v != 0 else " 0.00"
    rho_str = f"{fmt(rho)}  [{fmt(lo)}, {fmt(hi)}]"
    ax.text(ci_x, y, rho_str, fontsize=9, fontfamily='sans-serif',
            ha='left', va='center', zorder=6, color='#222222', clip_on=False,
            fontname='DejaVu Sans Mono' if 'DejaVu Sans Mono' in matplotlib.font_manager.get_font_names() else 'monospace')

ax.text(ci_x, diamond_y,
        f"{pooled_rho:+.3f}  [{pooled_ci_lo:+.3f}, {pooled_ci_hi:+.3f}]",
        fontsize=9, fontfamily='sans-serif',
        ha='left', va='center', zorder=6, fontweight='bold', color='#222222', clip_on=False)

# ── Group labels (far left, vertical) ───────────────────────────────────
for grp in groups_ordered:
    indices = [i for i, (_, _, _, _, g) in enumerate(studies) if g == grp]
    ys = [y_pos[i] for i in indices]
    mid_y = np.mean(ys)
    top_y = max(ys) + 0.42
    bot_y = min(ys) - 0.42
    color = marker_colors[grp]

    # Vertical colored bracket
    bracket_x = -0.82
    ax.plot([bracket_x, bracket_x], [bot_y, top_y], color=color, linewidth=3.0,
            zorder=6, clip_on=False, solid_capstyle='round')

    # Group label badge
    ax.annotate(
        grp, xy=(-0.95, mid_y),
        fontsize=10, fontweight='bold', fontfamily='sans-serif',
        color='white', ha='center', va='center', rotation=90,
        bbox=dict(boxstyle='round,pad=0.3', facecolor=color,
                  edgecolor=color, linewidth=1.0, alpha=0.92),
        zorder=7, clip_on=False,
    )

# ── Column headers ──────────────────────────────────────────────────────
header_y = y_pos[0] + 1.0
ax.text(label_x, header_y, "Study (Dataset, Architecture)", fontsize=11,
        fontfamily='sans-serif', fontweight='bold', ha='right', va='center',
        clip_on=False)
ax.text(ci_x, header_y, "ρ  [95% CI]", fontsize=11,
        fontfamily='sans-serif', fontweight='bold', ha='left', va='center',
        clip_on=False)

# ── Configure axes ──────────────────────────────────────────────────────
ax.set_xlim(-0.5, 1.0)
ax.set_ylim(diamond_y - 1.8, y_pos[0] + 1.6)
ax.set_xlabel("Spearman ρ", fontsize=14, fontfamily='sans-serif', fontweight='bold', labelpad=12)
ax.set_xticks(np.arange(-0.5, 1.1, 0.25))
ax.tick_params(axis='x', labelsize=11)
ax.set_yticks([])

# ── Title ────────────────────────────────────────────────────────────────
ax.set_title(
    "Meta-Analysis Forest Plot:\nSRI-Gap Correlations Across Studies",
    fontsize=16, fontfamily='sans-serif', fontweight='bold', pad=24, linespacing=1.3
)

# ── Legend ────────────────────────────────────────────────────────────────
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#d62728', linestyle='--', linewidth=1.4, label='No effect (ρ = 0)'),
    Line2D([0], [0], color='#333333', linestyle='--', linewidth=1.4, label='Pooled ρ = 0.153'),
    mpatches.Patch(facecolor=group_bg["Model-Free"], edgecolor=marker_colors["Model-Free"],
                   linewidth=1.5, label='Model-Free'),
    mpatches.Patch(facecolor=group_bg["MLP"],        edgecolor=marker_colors["MLP"],
                   linewidth=1.5, label='MLP'),
    mpatches.Patch(facecolor=group_bg["GCN"],        edgecolor=marker_colors["GCN"],
                   linewidth=1.5, label='GCN'),
    mpatches.Patch(facecolor=group_bg["GPS"],        edgecolor=marker_colors["GPS"],
                   linewidth=1.5, label='GPS'),
]
leg = ax.legend(handles=legend_elements, loc='upper right', fontsize=9.5, framealpha=0.95,
          edgecolor='#cccccc', fancybox=False, borderpad=0.8)
leg.get_frame().set_linewidth(0.8)

# ── Annotation at bottom ────────────────────────────────────────────────
annotation = "Pooled ρ = 0.153,  95% CI [0.020, 0.280],  I² = 99.0%"
ax.text(
    0.25, diamond_y - 1.2, annotation,
    fontsize=12, fontfamily='sans-serif', fontstyle='italic',
    ha='center', va='top', color='#333333',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5',
              edgecolor='#999999', linewidth=0.7),
    zorder=6
)

# ── Spine / grid cleanup ────────────────────────────────────────────────
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.grid(axis='x', linestyle=':', linewidth=0.4, alpha=0.5, zorder=0)

outpath = sys.argv[1] if len(sys.argv) > 1 else "fig_5_v0.png"
fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print(f"Saved to {outpath}")
