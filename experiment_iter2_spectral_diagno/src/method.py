#!/usr/bin/env python3
"""Spectral Diagnostics: Sparsity Validation, SRI Distribution Analysis,
Node-Level Resolution, and Vandermonde Conditioning.

Comprehensive spectral diagnostic experiment that validates four key aspects
of the walk resolution limit hypothesis.
"""

import json
import math
import os
import resource
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import psutil
from loguru import logger
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
WORKSPACE = Path(__file__).resolve().parent
LOG_DIR = WORKSPACE / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ---------------------------------------------------------------------------
# Hardware constants
# ---------------------------------------------------------------------------
def _cgroup_cpus():
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
    except (FileNotFoundError, ValueError):
        pass
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0:
            return math.ceil(q / p)
    except (FileNotFoundError, ValueError):
        pass
    return None

def _container_ram_gb():
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None

NUM_CPUS = _cgroup_cpus() or os.cpu_count() or 1
TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9
AVAILABLE_RAM_GB = min(psutil.virtual_memory().available / 1e9, TOTAL_RAM_GB)
logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f} GB RAM ({AVAILABLE_RAM_GB:.1f} GB available)")

# Resource limits
_ram_limit = int(min(40, TOTAL_RAM_GB * 0.85) * 1024**3)
try:
    resource.setrlimit(resource.RLIMIT_AS, (_ram_limit, _ram_limit))
except Exception:
    pass
try:
    resource.setrlimit(resource.RLIMIT_CPU, (3500, 3500))
except Exception:
    pass

_cpu_used = psutil.cpu_percent(interval=1)
NUM_WORKERS = max(1, int(NUM_CPUS * (1.0 - _cpu_used / 100.0) * 0.8))
logger.info(f"CPU usage: {_cpu_used}%, using {NUM_WORKERS} workers")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEP_DIR = Path("/opt/ai-inventor/aii_pipeline/runs/run__20260222_005317/3_invention_loop/iter_1/gen_art/data_id2_it1__opus")
FIGURES_DIR = WORKSPACE / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Plot defaults
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

K_VALUES = [2, 4, 8, 16, 20]
K_KEYS = [f'K={k}' for k in K_VALUES]

# ===================================================================
# DATA LOADING
# ===================================================================

def _parse_example(ex: dict) -> dict:
    """Parse a single example into our internal graph dict."""
    inp = json.loads(ex['input'])
    sp = inp['spectral']
    return {
        'eigenvalues': sp['eigenvalues'],
        'delta_min': sp['delta_min'],
        'sri': sp['sri'],
        'vandermonde_cond': sp['vandermonde_cond'],
        'rwse': sp['rwse'],
        'local_spectral': sp['local_spectral'],
        'num_nodes': inp.get('num_nodes', len(sp['rwse'])),
        'dataset_name': ex.get('metadata_source', ''),
        'metadata_pair_category': ex.get('metadata_pair_category', None),
    }


def load_data(max_per_dataset: int | None = None) -> dict[str, list[dict]]:
    """Load graph data from all 5 parts, grouped by dataset."""
    datasets: dict[str, list[dict]] = {}
    for i in range(1, 6):
        fpath = DEP_DIR / "data_out" / f"full_data_out_{i}.json"
        logger.info(f"Loading {fpath.name} ...")
        raw = json.loads(fpath.read_text())
        for ds_block in raw['datasets']:
            ds_name = ds_block['dataset']
            if ds_name not in datasets:
                datasets[ds_name] = []
            if max_per_dataset and len(datasets[ds_name]) >= max_per_dataset:
                continue
            for ex in ds_block['examples']:
                if max_per_dataset and len(datasets[ds_name]) >= max_per_dataset:
                    break
                datasets[ds_name].append(_parse_example(ex))
    for ds, gs in datasets.items():
        logger.info(f"  {ds}: {len(gs)} graphs loaded")
    return datasets


def load_mini_data() -> dict[str, list[dict]]:
    """Load mini dataset for quick testing."""
    datasets: dict[str, list[dict]] = {}
    fpath = DEP_DIR / "mini_data_out.json"
    logger.info(f"Loading {fpath.name} ...")
    raw = json.loads(fpath.read_text())
    for ds_block in raw['datasets']:
        ds_name = ds_block['dataset']
        datasets[ds_name] = []
        for ex in ds_block['examples']:
            datasets[ds_name].append(_parse_example(ex))
    for ds, gs in datasets.items():
        logger.info(f"  {ds}: {len(gs)} graphs loaded")
    return datasets

# ===================================================================
# ANALYSIS 1: SPECTRAL SPARSITY VALIDATION
# ===================================================================

def run_sparsity_analysis(datasets: dict[str, list[dict]]) -> tuple[dict, dict]:
    """Analysis 1: Spectral Sparsity Validation.
    Returns (sparsity_results, cached per-dataset arrays for plotting)."""
    logger.info("=== Analysis 1: Spectral Sparsity Validation ===")
    t0 = time.time()
    sparsity_results = {}
    sparsity_cache: dict[str, dict] = {}  # for plotting

    for ds_name, graphs in datasets.items():
        logger.info(f"  Processing {ds_name} ({len(graphs)} graphs)...")
        all_eff_rank_1pct = []
        all_eff_rank_5pct = []
        all_participation_ratios = []
        all_spectral_entropies = []
        all_n_eigenvalues = []

        for g in graphs:
            n_eigenvalues = len(g['eigenvalues'])
            all_n_eigenvalues.append(n_eigenvalues)
            for node_measures in g['local_spectral']:
                if not node_measures or len(node_measures) == 0:
                    continue
                weights = np.array([m[1] for m in node_measures], dtype=np.float64)
                total_w = weights.sum()
                if total_w < 1e-12:
                    continue

                eff_1 = int(np.sum(weights > 0.01 * total_w))
                all_eff_rank_1pct.append(eff_1)

                eff_5 = int(np.sum(weights > 0.05 * total_w))
                all_eff_rank_5pct.append(eff_5)

                pr = float(total_w ** 2 / np.sum(weights ** 2))
                all_participation_ratios.append(pr)

                p = weights / total_w
                p = p[p > 0]
                entropy = float(-np.sum(p * np.log2(p)))
                all_spectral_entropies.append(entropy)

        if len(all_eff_rank_1pct) == 0:
            logger.warning(f"  No valid nodes found for {ds_name}, skipping")
            continue

        arr_1 = np.array(all_eff_rank_1pct)
        arr_5 = np.array(all_eff_rank_5pct)
        arr_pr = np.array(all_participation_ratios)
        arr_ent = np.array(all_spectral_entropies)
        arr_n = np.array(all_n_eigenvalues)

        median_eff = float(np.median(arr_1))
        median_n = float(np.median(arr_n))
        ratio = median_eff / max(median_n, 1e-10)

        sparsity_cache[ds_name] = {
            'eff_rank_1pct': arr_1,
            'eff_rank_5pct': arr_5,
            'participation_ratios': arr_pr,
        }

        sparsity_results[ds_name] = {
            'eff_rank_1pct': {
                'mean': float(np.mean(arr_1)),
                'median': float(np.median(arr_1)),
                'std': float(np.std(arr_1)),
                'percentiles': {
                    '25': float(np.percentile(arr_1, 25)),
                    '75': float(np.percentile(arr_1, 75)),
                    '90': float(np.percentile(arr_1, 90)),
                },
                'histogram_counts': np.histogram(arr_1, bins=range(0, 12))[0].tolist(),
                'histogram_edges': list(range(0, 12)),
            },
            'eff_rank_5pct': {
                'mean': float(np.mean(arr_5)),
                'median': float(np.median(arr_5)),
                'std': float(np.std(arr_5)),
                'percentiles': {
                    '25': float(np.percentile(arr_5, 25)),
                    '75': float(np.percentile(arr_5, 75)),
                    '90': float(np.percentile(arr_5, 90)),
                },
            },
            'participation_ratio': {
                'mean': float(np.mean(arr_pr)),
                'median': float(np.median(arr_pr)),
                'std': float(np.std(arr_pr)),
                'percentiles': {
                    '25': float(np.percentile(arr_pr, 25)),
                    '75': float(np.percentile(arr_pr, 75)),
                    '90': float(np.percentile(arr_pr, 90)),
                },
            },
            'spectral_entropy': {
                'mean': float(np.mean(arr_ent)),
                'median': float(np.median(arr_ent)),
                'std': float(np.std(arr_ent)),
            },
            'n_nodes_analyzed': len(all_eff_rank_1pct),
            'median_graph_size': float(median_n),
            'sparsity_ratio': float(ratio),
            'sparsity_confirmed': bool(ratio < 0.3),
        }
        logger.info(f"    {ds_name}: median_eff_rank_1pct={median_eff:.1f}, "
                     f"median_n={median_n:.0f}, ratio={ratio:.4f}, "
                     f"confirmed={ratio < 0.3}")

    logger.info(f"  Sparsity analysis complete in {time.time()-t0:.1f}s")
    return sparsity_results, sparsity_cache


def plot_sparsity(sparsity_cache: dict):
    """Generate sparsity plots using cached data."""
    logger.info("  Plotting sparsity figures...")
    ds_names = sorted(sparsity_cache.keys())

    # PLOT 1a: Violin plot of effective rank per dataset
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    data_1 = [sparsity_cache[ds]['eff_rank_1pct'] for ds in ds_names]
    vp = axes[0].violinplot(data_1, positions=range(len(ds_names)), showmeans=True, showmedians=True)
    axes[0].set_xticks(range(len(ds_names)))
    axes[0].set_xticklabels([n.replace('-', '\n') for n in ds_names], fontsize=9)
    axes[0].axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Top-10 ceiling')
    axes[0].set_ylabel('Effective Rank')
    axes[0].set_title('Effective Rank (1% threshold)')
    axes[0].legend(fontsize=9)

    data_5 = [sparsity_cache[ds]['eff_rank_5pct'] for ds in ds_names]
    vp2 = axes[1].violinplot(data_5, positions=range(len(ds_names)), showmeans=True, showmedians=True)
    axes[1].set_xticks(range(len(ds_names)))
    axes[1].set_xticklabels([n.replace('-', '\n') for n in ds_names], fontsize=9)
    axes[1].axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Top-10 ceiling')
    axes[1].set_ylabel('Effective Rank')
    axes[1].set_title('Effective Rank (5% threshold)')
    axes[1].legend(fontsize=9)

    fig.suptitle('Effective Rank of Local Spectral Measures', fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig(str(FIGURES_DIR / 'sparsity_effective_rank.png'))
    plt.close(fig)

    # PLOT 1b: Participation ratio distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("husl", len(ds_names))
    for i, ds in enumerate(ds_names):
        vals = sparsity_cache[ds]['participation_ratios']
        if len(vals) > 10:
            sns.kdeplot(vals, ax=ax, label=ds, color=colors[i], fill=True, alpha=0.2)
        elif len(vals) > 0:
            ax.hist(vals, bins=10, alpha=0.5, label=ds, color=colors[i])
    ax.set_xlabel('Participation Ratio')
    ax.set_ylabel('Density')
    ax.set_title('Participation Ratio of Node Spectral Measures')
    ax.legend()
    plt.tight_layout()
    fig.savefig(str(FIGURES_DIR / 'sparsity_participation_ratio.png'))
    plt.close(fig)
    logger.info("  Sparsity plots saved.")


# ===================================================================
# ANALYSIS 2: SRI DISTRIBUTION ANALYSIS
# ===================================================================

def run_sri_analysis(datasets: dict[str, list[dict]]) -> tuple[dict, dict]:
    """Analysis 2: SRI Distribution Analysis with KS tests."""
    logger.info("=== Analysis 2: SRI Distribution Analysis ===")
    t0 = time.time()
    sri_results = {}
    sri_by_dataset: dict[str, dict] = {}

    for ds_name, graphs in datasets.items():
        logger.info(f"  Processing {ds_name} ({len(graphs)} graphs)...")
        delta_mins = np.array([g['delta_min'] for g in graphs], dtype=np.float64)
        sri_20 = np.array([g['sri']['K=20'] for g in graphs], dtype=np.float64)

        spectral_ranges = np.array([
            max(g['eigenvalues']) - min(g['eigenvalues']) for g in graphs
        ], dtype=np.float64)
        normalized_sri = delta_mins / np.maximum(spectral_ranges, 1e-10)

        sri_by_K = {}
        for k_key in K_KEYS:
            sri_by_K[k_key] = np.array([g['sri'][k_key] for g in graphs], dtype=np.float64)

        sri_by_dataset[ds_name] = {
            'delta_min': delta_mins,
            'sri_20': sri_20,
            'normalized_sri': normalized_sri,
            'spectral_range': spectral_ranges,
            'sri_by_K': sri_by_K,
        }

        sri_results[ds_name] = {
            'delta_min': {
                'mean': float(np.mean(delta_mins)),
                'median': float(np.median(delta_mins)),
                'std': float(np.std(delta_mins)),
                'min': float(np.min(delta_mins)),
                'max': float(np.max(delta_mins)),
            },
            'sri_20': {
                'mean': float(np.mean(sri_20)),
                'median': float(np.median(sri_20)),
                'std': float(np.std(sri_20)),
                'pct_below_1': float(np.mean(sri_20 < 1.0) * 100),
                'pct_above_5': float(np.mean(sri_20 > 5.0) * 100),
            },
            'normalized_sri': {
                'mean': float(np.mean(normalized_sri)),
                'median': float(np.median(normalized_sri)),
            },
            'spectral_range': {
                'mean': float(np.mean(spectral_ranges)),
                'median': float(np.median(spectral_ranges)),
            },
        }
        logger.info(f"    SRI(K=20): mean={np.mean(sri_20):.4f}, "
                     f"pct_below_1={np.mean(sri_20 < 1.0)*100:.1f}%")

    # Pairwise KS tests
    dataset_names = sorted(sri_by_dataset.keys())
    ks_results = {}
    for i, ds1 in enumerate(dataset_names):
        for ds2 in dataset_names[i + 1:]:
            stat_sri, p_sri = stats.ks_2samp(
                sri_by_dataset[ds1]['sri_20'],
                sri_by_dataset[ds2]['sri_20']
            )
            stat_dm, p_dm = stats.ks_2samp(
                sri_by_dataset[ds1]['delta_min'],
                sri_by_dataset[ds2]['delta_min']
            )
            stat_nsri, p_nsri = stats.ks_2samp(
                sri_by_dataset[ds1]['normalized_sri'],
                sri_by_dataset[ds2]['normalized_sri']
            )
            ks_results[f"{ds1}_vs_{ds2}"] = {
                'sri_20': {'statistic': float(stat_sri), 'p_value': float(p_sri)},
                'delta_min': {'statistic': float(stat_dm), 'p_value': float(p_dm)},
                'normalized_sri': {'statistic': float(stat_nsri), 'p_value': float(p_nsri)},
            }
            logger.info(f"    KS({ds1} vs {ds2}): SRI p={p_sri:.2e}, delta_min p={p_dm:.2e}")

    sri_results['ks_tests'] = ks_results
    logger.info(f"  SRI analysis complete in {time.time()-t0:.1f}s")
    return sri_results, sri_by_dataset


def plot_sri(sri_by_dataset: dict, datasets: dict[str, list[dict]]):
    """Generate SRI distribution plots."""
    logger.info("  Plotting SRI figures...")
    ds_names = sorted(sri_by_dataset.keys())
    colors = sns.color_palette("husl", len(ds_names))

    # PLOT 2a: Overlaid histograms
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, ds in enumerate(ds_names):
        vals = sri_by_dataset[ds]['delta_min']
        vals_pos = vals[vals > 0]
        if len(vals_pos) > 0:
            axes[0].hist(np.log10(vals_pos), bins=40, alpha=0.4, label=ds, color=colors[i])
        vals = sri_by_dataset[ds]['sri_20']
        vals_pos = vals[vals > 0]
        if len(vals_pos) > 0:
            axes[1].hist(np.log10(vals_pos), bins=40, alpha=0.4, label=ds, color=colors[i])
        vals = sri_by_dataset[ds]['normalized_sri']
        vals_pos = vals[vals > 0]
        if len(vals_pos) > 0:
            axes[2].hist(np.log10(vals_pos), bins=40, alpha=0.4, label=ds, color=colors[i])

    axes[0].set_xlabel('log\u2081\u2080(\u03b4_min)')
    axes[0].set_title('Minimum Eigenvalue Gap (\u03b4_min)')
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel('log\u2081\u2080(SRI at K=20)')
    axes[1].set_title('Spectral Resolution Index (K=20)')
    axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='SRI=1')
    axes[1].legend(fontsize=8)
    axes[2].set_xlabel('log\u2081\u2080(Normalized SRI)')
    axes[2].set_title('Normalized SRI (\u03b4_min / spectral range)')
    axes[2].legend(fontsize=8)

    fig.suptitle('SRI Distribution Comparison Across Datasets', fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig(str(FIGURES_DIR / 'sri_distributions.png'))
    plt.close(fig)

    # PLOT 2b: SRI vs K
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, ds_name in enumerate(ds_names):
        graphs = datasets[ds_name]
        median_sri = []
        q25_sri = []
        q75_sri = []
        for k_key in K_KEYS:
            sri_vals = [g['sri'][k_key] for g in graphs]
            median_sri.append(float(np.median(sri_vals)))
            q25_sri.append(float(np.percentile(sri_vals, 25)))
            q75_sri.append(float(np.percentile(sri_vals, 75)))
        ax.plot(K_VALUES, median_sri, '-o', label=ds_name, color=colors[i])
        ax.fill_between(K_VALUES, q25_sri, q75_sri, alpha=0.15, color=colors[i])

    ax.axhline(y=1.0, color='red', linestyle='--', label='Resolution limit (SRI=1)')
    ax.set_xlabel('Walk Length K')
    ax.set_ylabel('SRI(G, K)')
    ax.set_title('Spectral Resolution Index vs Walk Length K')
    ax.legend()
    ax.set_yscale('log')
    plt.tight_layout()
    fig.savefig(str(FIGURES_DIR / 'sri_vs_K.png'))
    plt.close(fig)
    logger.info("  SRI plots saved.")


# ===================================================================
# ANALYSIS 3: NODE-LEVEL vs GRAPH-LEVEL RESOLUTION
# ===================================================================

def _compute_node_deltas_for_graph(g: dict, thresh: float) -> tuple[list[float], float]:
    """Compute node-level delta_min values for a single graph.
    Returns (list of finite positive node deltas, graph_delta_min)."""
    graph_delta_min = g['delta_min']
    if graph_delta_min < 1e-15:
        return [], graph_delta_min

    node_deltas = []
    for node_measures in g['local_spectral']:
        if not node_measures or len(node_measures) == 0:
            continue
        eigenvals = np.array([m[0] for m in node_measures], dtype=np.float64)
        weights = np.array([m[1] for m in node_measures], dtype=np.float64)
        if len(weights) == 0 or weights.max() < 1e-12:
            continue
        mask = weights > thresh * weights.max()
        sig_eigenvals = eigenvals[mask]
        if len(sig_eigenvals) < 2:
            continue  # single component = effectively infinite separation
        sig_sorted = np.sort(sig_eigenvals)
        diffs = np.diff(sig_sorted)
        nonzero_diffs = diffs[diffs > 1e-15]
        if len(nonzero_diffs) > 0:
            node_deltas.append(float(np.min(nonzero_diffs)))

    return node_deltas, graph_delta_min


def run_node_vs_graph_analysis(datasets: dict[str, list[dict]]) -> tuple[dict, dict]:
    """Analysis 3: Node-level vs Graph-level resolution comparison.
    Returns (results, scatter_cache for plotting)."""
    logger.info("=== Analysis 3: Node-Level vs Graph-Level Resolution ===")
    t0 = time.time()
    WEIGHT_THRESHOLDS = [0.01, 0.05, 0.10]
    node_vs_graph_results = {}
    scatter_cache: dict[str, dict] = {}  # for plotting

    for ds_name, graphs in datasets.items():
        logger.info(f"  Processing {ds_name} ({len(graphs)} graphs)...")
        node_vs_graph_results[ds_name] = {}
        scatter_cache[ds_name] = {}

        for thresh in WEIGHT_THRESHOLDS:
            ratios = []
            node_sris = []
            graph_sris = []

            for g in graphs:
                node_deltas, graph_delta_min = _compute_node_deltas_for_graph(g, thresh)
                if not node_deltas or graph_delta_min < 1e-15:
                    continue
                median_node_delta = float(np.median(node_deltas))
                if median_node_delta > 0:
                    ratio = median_node_delta / graph_delta_min
                    ratios.append(ratio)
                    node_sris.append(median_node_delta * 20)
                    graph_sris.append(graph_delta_min * 20)

            # Cache for plotting
            scatter_cache[ds_name][f'{thresh}'] = {
                'node_sris': node_sris,
                'graph_sris': graph_sris,
                'ratios': ratios,
            }

            result = {'n_graphs_analyzed': len(ratios)}
            if ratios:
                ratios_arr = np.array(ratios)
                result.update({
                    'mean_ratio': float(np.mean(ratios_arr)),
                    'median_ratio': float(np.median(ratios_arr)),
                    'pct_node_better': float(np.mean(ratios_arr > 1.0) * 100),
                    'pct_node_10x_better': float(np.mean(ratios_arr > 10.0) * 100),
                })
                if len(graph_sris) > 2:
                    corr = stats.spearmanr(graph_sris, node_sris)
                    result['spearman_corr'] = float(corr.statistic)
                    result['spearman_p'] = float(corr.pvalue)
            else:
                result.update({
                    'mean_ratio': None,
                    'median_ratio': None,
                    'pct_node_better': None,
                    'pct_node_10x_better': None,
                })

            node_vs_graph_results[ds_name][f'threshold_{thresh}'] = result
            if ratios:
                logger.info(f"    {ds_name} thresh={thresh}: "
                             f"median_ratio={np.median(ratios):.2f}, "
                             f"pct_better={np.mean(np.array(ratios) > 1)*100:.1f}%")

    logger.info(f"  Node vs Graph analysis complete in {time.time()-t0:.1f}s")
    return node_vs_graph_results, scatter_cache


def plot_node_vs_graph(scatter_cache: dict):
    """Generate node vs graph plots using cached data."""
    logger.info("  Plotting Node vs Graph figures...")
    WEIGHT_THRESHOLDS = [0.01, 0.05, 0.10]
    ds_names = sorted(scatter_cache.keys())
    colors_map = dict(zip(ds_names, sns.color_palette("husl", len(ds_names))))

    # PLOT 3a: Scatter plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for tidx, thresh in enumerate(WEIGHT_THRESHOLDS):
        ax = axes[tidx]
        for ds_name in ds_names:
            cache = scatter_cache[ds_name].get(f'{thresh}', {})
            node_sris = cache.get('node_sris', [])
            graph_sris = cache.get('graph_sris', [])
            if len(node_sris) > 0:
                ax.scatter(graph_sris, node_sris, alpha=0.3, s=10,
                           label=ds_name, color=colors_map[ds_name])

        lims = ax.get_xlim()
        if lims[1] > 0:
            ax.plot([1e-4, max(lims[1], 100)], [1e-4, max(lims[1], 100)],
                    'k--', alpha=0.5, label='y=x')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Graph-level SRI (K=20)')
        ax.set_ylabel('Node-level SRI (K=20)')
        ax.set_title(f'Weight threshold = {thresh}')
        ax.legend(fontsize=7, markerscale=2)

    fig.suptitle('Node-Level vs Graph-Level Spectral Resolution', fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig(str(FIGURES_DIR / 'node_vs_graph_sri.png'))
    plt.close(fig)

    # PLOT 3b: Distribution of ratio
    fig, ax = plt.subplots(figsize=(8, 5))
    for ds_name in ds_names:
        cache = scatter_cache[ds_name].get('0.05', {})
        ratios = cache.get('ratios', [])
        if len(ratios) > 0:
            ratios_log = np.log10(np.array(ratios).clip(min=1e-3))
            ax.hist(ratios_log, bins=40, alpha=0.4, label=ds_name, color=colors_map[ds_name])

    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Ratio = 1')
    ax.set_xlabel('log\u2081\u2080(Node \u03b4_min / Graph \u03b4_min)')
    ax.set_ylabel('Count')
    ax.set_title('Eigenvector Localization Benefit: Node/Graph Resolution Ratio (threshold=0.05)')
    ax.legend()
    plt.tight_layout()
    fig.savefig(str(FIGURES_DIR / 'localization_ratio_distribution.png'))
    plt.close(fig)
    logger.info("  Node vs Graph plots saved.")


# ===================================================================
# ANALYSIS 4: VANDERMONDE CONDITION NUMBER ANALYSIS
# ===================================================================

def run_vandermonde_analysis(datasets: dict[str, list[dict]]) -> tuple[dict, dict, dict]:
    """Analysis 4: Vandermonde Condition Number Analysis.
    Returns (vander_results, reconstruction_results, scatter_cache)."""
    logger.info("=== Analysis 4: Vandermonde Conditioning Analysis ===")
    t0 = time.time()
    vander_results = {}
    reconstruction_results = {}
    vander_scatter_cache: dict[str, dict] = {}
    N_SUBSAMPLE = 200
    NOISE_LEVELS = [1e-8, 1e-6, 1e-4, 1e-2]
    MAX_COND = 1e15

    for ds_name, graphs in datasets.items():
        logger.info(f"  Processing {ds_name} ({len(graphs)} graphs)...")

        # (a) Growth rate analysis
        growth_rates = []
        sri_values_K20 = []
        all_log_conds = {k: [] for k in K_VALUES}

        for g in graphs:
            conds = [g['vandermonde_cond'][kk] for kk in K_KEYS]
            log_conds = np.log10(np.clip(np.array(conds, dtype=np.float64), 1.0, MAX_COND))
            slope, _, _, _, _ = stats.linregress(K_VALUES, log_conds)
            growth_rates.append(float(slope))
            sri_values_K20.append(g['sri']['K=20'])
            for ki, kv in enumerate(K_VALUES):
                all_log_conds[kv].append(float(log_conds[ki]))

        growth_rates_arr = np.array(growth_rates)
        sri_values_arr = np.array(sri_values_K20)

        # (b) Correlation: growth rate vs SRI
        corr_growth_sri = None
        if len(growth_rates_arr) > 2:
            try:
                corr = stats.spearmanr(growth_rates_arr, sri_values_arr)
                corr_growth_sri = {
                    'spearman_rho': float(corr.statistic),
                    'p_value': float(corr.pvalue),
                }
            except Exception:
                pass

        vander_results[ds_name] = {
            'growth_rate': {
                'mean': float(np.mean(growth_rates_arr)),
                'median': float(np.median(growth_rates_arr)),
                'std': float(np.std(growth_rates_arr)),
            },
            'corr_growth_rate_vs_sri': corr_growth_sri,
            'cond_number_stats': {},
            'sri_quartile_conds': {},
        }

        for kv in K_VALUES:
            log_conds_arr = np.array(all_log_conds[kv])
            vander_results[ds_name]['cond_number_stats'][str(kv)] = {
                'mean_log10': float(np.mean(log_conds_arr)),
                'median_log10': float(np.median(log_conds_arr)),
                'max_log10': float(np.max(log_conds_arr)),
            }

        # SRI quartile analysis
        if len(sri_values_arr) >= 4:
            quartiles = np.percentile(sri_values_arr, [25, 50, 75])
            bins = [-np.inf] + list(quartiles) + [np.inf]
            for kv in K_VALUES:
                lc = np.array(all_log_conds[kv])
                q_means = []
                for qi in range(4):
                    mask = (sri_values_arr >= bins[qi]) & (sri_values_arr < bins[qi + 1])
                    if mask.sum() > 0:
                        q_means.append(float(np.mean(lc[mask])))
                    else:
                        q_means.append(0.0)
                vander_results[ds_name]['sri_quartile_conds'][str(kv)] = q_means

        logger.info(f"    Growth rate: mean={np.mean(growth_rates_arr):.4f}, "
                     f"corr_with_sri={corr_growth_sri}")

        # (c) RWSE Reconstruction Error Experiment
        logger.info(f"    Running reconstruction experiment (subsample={N_SUBSAMPLE})...")
        np.random.seed(42)
        n_sample = min(N_SUBSAMPLE, len(graphs))
        sample_indices = np.random.choice(len(graphs), n_sample, replace=False)

        errors_by_noise = {eps: [] for eps in NOISE_LEVELS}
        cond_nums_recon = []
        recon_scatter_conds = []
        recon_scatter_errs = []

        for idx in sample_indices:
            g = graphs[idx]
            local_spectral = g['local_spectral']
            for node_idx in range(min(5, len(local_spectral))):
                measure = local_spectral[node_idx]
                if not measure or len(measure) < 2:
                    continue

                eigs = np.array([m[0] for m in measure], dtype=np.float64)
                weights_true = np.array([m[1] for m in measure], dtype=np.float64)
                if len(eigs) < 2 or np.linalg.norm(weights_true) < 1e-12:
                    continue

                K = 20
                V = np.array([[eig ** (k + 1) for eig in eigs] for k in range(K)],
                             dtype=np.float64)

                try:
                    cond = float(np.linalg.cond(V))
                except Exception:
                    cond = MAX_COND
                if not np.isfinite(cond):
                    cond = MAX_COND
                cond = min(cond, MAX_COND)
                cond_nums_recon.append(cond)

                m_true = V @ weights_true

                for eps in NOISE_LEVELS:
                    noise = eps * np.random.randn(K)
                    m_noisy = m_true + noise
                    try:
                        w_hat, _, _, _ = np.linalg.lstsq(V, m_noisy, rcond=None)
                        rel_error = float(np.linalg.norm(w_hat - weights_true) /
                                          max(np.linalg.norm(weights_true), 1e-12))
                        if not np.isfinite(rel_error):
                            rel_error = 1e10
                        errors_by_noise[eps].append(min(rel_error, 1e10))
                    except Exception:
                        errors_by_noise[eps].append(1e10)

                    # Cache for scatter plot at eps=1e-4
                    if eps == 1e-4:
                        recon_scatter_conds.append(np.log10(max(cond, 1.0)))
                        recon_scatter_errs.append(np.log10(max(errors_by_noise[1e-4][-1], 1e-15)))

        vander_scatter_cache[ds_name] = {
            'conds': recon_scatter_conds,
            'errors': recon_scatter_errs,
        }

        reconstruction_results[ds_name] = {}
        for eps in NOISE_LEVELS:
            errs = np.array(errors_by_noise[eps])
            if len(errs) > 0:
                reconstruction_results[ds_name][str(eps)] = {
                    'mean_error': float(np.mean(errs)),
                    'median_error': float(np.median(errs)),
                    'max_error': float(np.max(errs)),
                    'n_samples': len(errs),
                }
            else:
                reconstruction_results[ds_name][str(eps)] = {
                    'mean_error': None, 'median_error': None, 'max_error': None, 'n_samples': 0,
                }

        # (d) Correlation: cond vs error
        if len(cond_nums_recon) > 2 and len(errors_by_noise[1e-4]) > 2:
            min_len = min(len(cond_nums_recon), len(errors_by_noise[1e-4]))
            log_conds_r = np.log10(np.maximum(cond_nums_recon[:min_len], 1.0))
            log_errors_r = np.log10(np.maximum(errors_by_noise[1e-4][:min_len], 1e-15))
            valid = np.isfinite(log_conds_r) & np.isfinite(log_errors_r)
            if valid.sum() > 2:
                try:
                    corr = stats.spearmanr(log_conds_r[valid], log_errors_r[valid])
                    reconstruction_results[ds_name]['corr_cond_vs_error'] = {
                        'spearman_rho': float(corr.statistic),
                        'p_value': float(corr.pvalue),
                    }
                except Exception:
                    pass

        logger.info(f"    Reconstruction: {len(cond_nums_recon)} samples")

    logger.info(f"  Vandermonde analysis complete in {time.time()-t0:.1f}s")
    return vander_results, reconstruction_results, vander_scatter_cache


def plot_vandermonde(vander_results: dict, vander_scatter_cache: dict):
    """Generate Vandermonde plots using cached data."""
    logger.info("  Plotting Vandermonde figures...")
    ds_names = sorted(vander_results.keys())
    colors = sns.color_palette("husl", len(ds_names))

    # PLOT 4a: Condition number growth by SRI quartile
    fig, ax = plt.subplots(figsize=(9, 6))
    q_labels = ['Q1 (low SRI)', 'Q2', 'Q3', 'Q4 (high SRI)']
    q_colors = ['#e74c3c', '#f39c12', '#27ae60', '#2980b9']
    for qi in range(4):
        means_by_K = []
        for kv in K_VALUES:
            vals = []
            for ds in ds_names:
                qc = vander_results.get(ds, {}).get('sri_quartile_conds', {}).get(str(kv), None)
                if qc and qi < len(qc):
                    vals.append(qc[qi])
            means_by_K.append(np.mean(vals) if vals else 0.0)
        ax.plot(K_VALUES, means_by_K, '-o', label=q_labels[qi], color=q_colors[qi])

    ax.set_xlabel('Walk Length K')
    ax.set_ylabel('Mean log\u2081\u2080(\u03ba(V))')
    ax.set_title('Vandermonde Condition Number Growth by SRI Quartile')
    ax.legend()
    plt.tight_layout()
    fig.savefig(str(FIGURES_DIR / 'vandermonde_growth.png'))
    plt.close(fig)

    # PLOT 4b: Scatter of log10(kappa) vs log10(error)
    fig, ax = plt.subplots(figsize=(8, 6))
    all_cond = []
    all_err = []
    for di, ds_name in enumerate(ds_names):
        cache = vander_scatter_cache.get(ds_name, {})
        conds_p = cache.get('conds', [])
        errs_p = cache.get('errors', [])
        if conds_p:
            ax.scatter(conds_p, errs_p, alpha=0.2, s=8, label=ds_name, color=colors[di])
            all_cond.extend(conds_p)
            all_err.extend(errs_p)

    if len(all_cond) > 2:
        valid_mask = np.isfinite(all_cond) & np.isfinite(all_err)
        ac = np.array(all_cond)[valid_mask]
        ae = np.array(all_err)[valid_mask]
        if len(ac) > 2:
            slope, intercept, r, _, _ = stats.linregress(ac, ae)
            x_line = np.linspace(float(np.min(ac)), float(np.max(ac)), 100)
            ax.plot(x_line, slope * x_line + intercept, 'k--', alpha=0.7,
                    label=f'Fit: slope={slope:.2f}, R\u00b2={r**2:.3f}')

    ax.set_xlabel('log\u2081\u2080(\u03ba(V))')
    ax.set_ylabel('log\u2081\u2080(Relative Error)')
    ax.set_title('Vandermonde Conditioning Predicts RWSE Noise Sensitivity (\u03b5=1e-4)')
    ax.legend(fontsize=8, markerscale=2)
    plt.tight_layout()
    fig.savefig(str(FIGURES_DIR / 'cond_vs_recon_error.png'))
    plt.close(fig)

    # PLOT 4c: Heatmap
    fig, ax = plt.subplots(figsize=(10, max(3, len(ds_names) + 1)))
    data_mat = []
    ylabels = []
    for ds in ds_names:
        row = []
        for kv in K_VALUES:
            cs = vander_results.get(ds, {}).get('cond_number_stats', {}).get(str(kv), {})
            row.append(cs.get('median_log10', 0))
        data_mat.append(row)
        ylabels.append(ds)

    data_arr = np.array(data_mat)
    sns.heatmap(data_arr, annot=True, fmt='.1f', xticklabels=[f'K={k}' for k in K_VALUES],
                yticklabels=ylabels, cmap='YlOrRd', ax=ax)
    ax.set_title('Median Vandermonde Condition Number (log\u2081\u2080)')
    plt.tight_layout()
    fig.savefig(str(FIGURES_DIR / 'cond_heatmap.png'))
    plt.close(fig)
    logger.info("  Vandermonde plots saved.")


# ===================================================================
# ANALYSIS 5: BASELINE — EIGENVALUE CLUSTERING
# ===================================================================

def run_eigenvalue_clustering_baseline(datasets: dict[str, list[dict]]) -> dict:
    """Baseline analysis: eigenvalue clustering coefficient."""
    logger.info("=== Analysis 5: Eigenvalue Clustering Baseline ===")
    t0 = time.time()
    baseline_results = {}

    for ds_name, graphs in datasets.items():
        n_clusters_list = []
        spectral_gaps = []
        max_gaps = []

        for g in graphs:
            eigs = np.sort(g['eigenvalues'])
            n = len(eigs)
            if n < 2:
                continue

            clusters = 1
            for i in range(1, n):
                if abs(eigs[i] - eigs[i - 1]) > 0.01:
                    clusters += 1
            n_clusters_list.append(clusters)

            diffs = np.diff(eigs)
            nonzero = diffs[diffs > 1e-10]
            if len(nonzero) > 0:
                spectral_gaps.append(float(nonzero[0]))
                max_gaps.append(float(np.max(nonzero)))
            else:
                spectral_gaps.append(0.0)
                max_gaps.append(0.0)

        baseline_results[ds_name] = {
            'n_clusters': {
                'mean': float(np.mean(n_clusters_list)) if n_clusters_list else None,
                'median': float(np.median(n_clusters_list)) if n_clusters_list else None,
            },
            'spectral_gap': {
                'mean': float(np.mean(spectral_gaps)) if spectral_gaps else None,
                'median': float(np.median(spectral_gaps)) if spectral_gaps else None,
            },
            'max_gap': {
                'mean': float(np.mean(max_gaps)) if max_gaps else None,
                'median': float(np.median(max_gaps)) if max_gaps else None,
            },
            'n_graphs': len(n_clusters_list),
        }
        logger.info(f"  {ds_name}: median_clusters={np.median(n_clusters_list) if n_clusters_list else 'N/A'}")

    logger.info(f"  Baseline analysis complete in {time.time()-t0:.1f}s")
    return baseline_results


# ===================================================================
# JSON SERIALIZATION
# ===================================================================

def convert_for_json(obj):
    """Recursively convert numpy types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_for_json(i) for i in obj]
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def _per_graph_sparsity(g: dict) -> tuple[float, float]:
    """Compute per-graph sparsity metrics (median eff rank and participation ratio)."""
    eff_ranks = []
    prs = []
    for node_measures in g['local_spectral']:
        if not node_measures or len(node_measures) == 0:
            continue
        weights = np.array([m[1] for m in node_measures], dtype=np.float64)
        total_w = weights.sum()
        if total_w < 1e-12:
            continue
        eff_ranks.append(int(np.sum(weights > 0.01 * total_w)))
        prs.append(float(total_w ** 2 / np.sum(weights ** 2)))
    if eff_ranks:
        return float(np.median(eff_ranks)), float(np.median(prs))
    return 0.0, 0.0


def build_method_out(
    sparsity_results: dict,
    sri_results: dict,
    node_vs_graph_results: dict,
    vander_results: dict,
    reconstruction_results: dict,
    baseline_results: dict,
    datasets: dict[str, list[dict]],
) -> dict:
    """Build method_out.json in the exp_gen_sol_out schema."""

    ks_results = sri_results.get('ks_tests', {})
    any_separable = any(
        ks_results.get(pair, {}).get('sri_20', {}).get('p_value', 1.0) < 0.01
        for pair in ks_results
    )

    summary = {
        'sparsity_confirmed': {
            ds: sparsity_results.get(ds, {}).get('sparsity_confirmed', None)
            for ds in sparsity_results
        },
        'datasets_separable_by_sri': any_separable,
        'node_level_benefits_eigenvector_localization': {
            ds: node_vs_graph_results.get(ds, {}).get('threshold_0.05', {}).get('pct_node_better', None)
            for ds in node_vs_graph_results
        },
        'vandermonde_cond_predicts_error': {
            ds: reconstruction_results.get(ds, {}).get('corr_cond_vs_error', {}).get('spearman_rho', None)
            for ds in reconstruction_results
        },
    }

    metadata = {
        'method_name': 'Spectral Diagnostics for Walk Resolution Limit Hypothesis',
        'description': (
            'Comprehensive spectral diagnostic experiment validating 4 aspects: '
            '(1) spectral sparsity of local node measures, '
            '(2) SRI distribution differences with KS tests, '
            '(3) node-level vs graph-level resolution, '
            '(4) Vandermonde conditioning. '
            'Plus baseline eigenvalue clustering analysis.'
        ),
        'analysis_1_spectral_sparsity': convert_for_json(sparsity_results),
        'analysis_2_sri_distributions': convert_for_json(sri_results),
        'analysis_3_node_vs_graph_resolution': convert_for_json(node_vs_graph_results),
        'analysis_4_vandermonde_conditioning': convert_for_json(vander_results),
        'analysis_4_reconstruction': convert_for_json(reconstruction_results),
        'analysis_5_baseline_eigenvalue_clustering': convert_for_json(baseline_results),
        'summary': convert_for_json(summary),
        'figures': [
            'figures/sparsity_effective_rank.png',
            'figures/sparsity_participation_ratio.png',
            'figures/sri_distributions.png',
            'figures/sri_vs_K.png',
            'figures/node_vs_graph_sri.png',
            'figures/localization_ratio_distribution.png',
            'figures/vandermonde_growth.png',
            'figures/cond_vs_recon_error.png',
            'figures/cond_heatmap.png',
        ],
    }

    ds_list = []
    for ds_name, graphs in datasets.items():
        examples = []
        for i, g in enumerate(graphs):
            input_str = json.dumps({
                'dataset': ds_name,
                'graph_idx': i,
                'num_nodes': g['num_nodes'],
                'n_eigenvalues': len(g['eigenvalues']),
                'delta_min': g['delta_min'],
                'sri_K20': g['sri']['K=20'],
            })

            output_str = json.dumps({
                'delta_min': g['delta_min'],
                'sri_K20': g['sri']['K=20'],
                'vandermonde_cond_K20': g['vandermonde_cond']['K=20'],
            })

            median_eff_rank, median_pr = _per_graph_sparsity(g)

            predict_str = json.dumps({
                'sparsity_eff_rank_1pct_median': median_eff_rank,
                'sparsity_participation_ratio_median': median_pr,
                'sri_K20': g['sri']['K=20'],
                'sri_below_1': g['sri']['K=20'] < 1.0,
                'vandermonde_cond_K20_log10': float(np.log10(max(g['vandermonde_cond']['K=20'], 1.0))),
                'resolution_diagnosis': (
                    'well_resolved' if g['sri']['K=20'] > 5.0
                    else 'marginal' if g['sri']['K=20'] > 1.0
                    else 'aliased'
                ),
            })

            eigs = np.sort(g['eigenvalues'])
            clusters = 1
            for j in range(1, len(eigs)):
                if abs(eigs[j] - eigs[j - 1]) > 0.01:
                    clusters += 1
            diffs = np.diff(eigs)
            nonzero = diffs[diffs > 1e-10]
            baseline_str = json.dumps({
                'n_eigenvalue_clusters': clusters,
                'spectral_gap': float(nonzero[0]) if len(nonzero) > 0 else 0.0,
                'max_eigenvalue_gap': float(np.max(nonzero)) if len(nonzero) > 0 else 0.0,
            })

            examples.append({
                'input': input_str,
                'output': output_str,
                'predict_spectral_diagnostics': predict_str,
                'predict_eigenvalue_clustering_baseline': baseline_str,
            })

        ds_list.append({
            'dataset': ds_name,
            'examples': examples,
        })

    return {
        'metadata': metadata,
        'datasets': ds_list,
    }


# ===================================================================
# MAIN
# ===================================================================

@logger.catch
def main():
    t_start = time.time()

    run_mode = os.environ.get('RUN_MODE', 'full')
    if run_mode == 'mini':
        logger.info("=== MINI RUN MODE ===")
        datasets = load_mini_data()
    elif run_mode.startswith('limit_'):
        limit = int(run_mode.split('_')[1])
        logger.info(f"=== LIMITED RUN MODE (max {limit} per dataset) ===")
        datasets = load_data(max_per_dataset=limit)
    else:
        logger.info("=== FULL RUN MODE ===")
        datasets = load_data()

    total_graphs = sum(len(gs) for gs in datasets.values())
    logger.info(f"Total graphs loaded: {total_graphs}")

    # Run all analyses
    sparsity_results, sparsity_cache = run_sparsity_analysis(datasets)
    sri_results, sri_by_dataset = run_sri_analysis(datasets)
    node_vs_graph_results, scatter_cache = run_node_vs_graph_analysis(datasets)
    vander_results, reconstruction_results, vander_scatter_cache = run_vandermonde_analysis(datasets)
    baseline_results = run_eigenvalue_clustering_baseline(datasets)

    # Generate all plots
    logger.info("=== Generating Plots ===")
    try:
        plot_sparsity(sparsity_cache)
    except Exception:
        logger.exception("Failed to generate sparsity plots")
    try:
        plot_sri(sri_by_dataset, datasets)
    except Exception:
        logger.exception("Failed to generate SRI plots")
    try:
        plot_node_vs_graph(scatter_cache)
    except Exception:
        logger.exception("Failed to generate node vs graph plots")
    try:
        plot_vandermonde(vander_results, vander_scatter_cache)
    except Exception:
        logger.exception("Failed to generate Vandermonde plots")

    # Build and save output
    logger.info("=== Saving Results ===")
    method_out = build_method_out(
        sparsity_results, sri_results, node_vs_graph_results,
        vander_results, reconstruction_results, baseline_results, datasets
    )
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(method_out, indent=2))
    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved method_out.json ({size_mb:.1f} MB)")

    if size_mb > 95:
        logger.warning(f"method_out.json is {size_mb:.1f} MB, may need splitting")

    elapsed = time.time() - t_start
    logger.info(f"=== COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} min) ===")

    # Print summary
    for ds in sorted(sparsity_results.keys()):
        logger.info(f"  {ds}:")
        logger.info(f"    Sparsity confirmed: {sparsity_results[ds].get('sparsity_confirmed')}")
    if 'ks_tests' in sri_results:
        for pair, res in sri_results['ks_tests'].items():
            logger.info(f"    KS {pair}: SRI p={res['sri_20']['p_value']:.2e}")
    for ds in sorted(node_vs_graph_results.keys()):
        t05 = node_vs_graph_results[ds].get('threshold_0.05', {})
        logger.info(f"    {ds} node_better: {t05.get('pct_node_better')}%")
    for ds in sorted(reconstruction_results.keys()):
        corr = reconstruction_results[ds].get('corr_cond_vs_error', {})
        logger.info(f"    {ds} cond-error corr: {corr.get('spearman_rho')}")


if __name__ == "__main__":
    main()
