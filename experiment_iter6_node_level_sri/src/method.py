#!/usr/bin/env python3
"""
Node-Level SRI vs Graph-Level SRI: Testing Walk Resolution Limit Theory
at the Correct Granularity.

This experiment tests whether node-level SRI (Spectral Resolution Index) —
computed using eigenvector localization — substantially improves the
SRI-performance correlation over graph-level SRI.

Phases:
  0) Data loading
  1) Node-level SRI computation with threshold sensitivity
  2) Node-level distinguishability (RWSE vs sign-free LapPE)
  3) Graph-level correlation comparison with bootstrap CIs
  4) SRWE benefit prediction
  5) Visualization
"""

import json
import math
import os
import sys
import time
import warnings
import resource
from pathlib import Path
from typing import Any, Optional

# ── Limit threads ──
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
from scipy import stats
from scipy.optimize import nnls
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import psutil
from loguru import logger

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Logging setup ──
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ── Resource limits ──
TOTAL_RAM_GB = psutil.virtual_memory().total / 1e9
AVAIL_RAM_GB = psutil.virtual_memory().available / 1e9
RAM_LIMIT_GB = min(50, TOTAL_RAM_GB * 0.85)
resource.setrlimit(resource.RLIMIT_AS, (int(RAM_LIMIT_GB * 1024**3), int(RAM_LIMIT_GB * 1024**3)))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))  # 1 hour CPU time

# ── Constants ──
DATA_DIR = Path("/opt/ai-inventor/aii_pipeline/runs/run__20260222_005317/3_invention_loop/iter_1/gen_art/data_id2_it1__opus")
MINI_DATA = DATA_DIR / "mini_data_out.json"
DATA_PARTS = sorted((DATA_DIR / "data_out").glob("full_data_out_*.json"))

OUTPUT_DIR = SCRIPT_DIR
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

K_WALK = 20  # Walk length for SRI computation
SEED = 42
N_BOOTSTRAP = 5000
EPSILON = 1e-4  # Distinguishability threshold
THRESHOLD_FACTORS = [0.5, 1.0, 2.0]
SAMPLE_PER_DATASET = 500  # Graphs to sample per dataset for Phase 2
MAX_NODE_PAIRS = 500  # Max node pairs to sample per graph

# ── Hardware detection ──
NUM_CPUS = os.cpu_count() or 1
logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, {AVAIL_RAM_GB:.1f}GB available")


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 0: DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_all_data(max_per_dataset: Optional[int] = None) -> dict[str, list[dict]]:
    """Load all data parts, organize by dataset name. Efficiently truncate during load."""
    logger.info("Loading data from all parts...")
    datasets: dict[str, list[dict]] = {}
    ds_counts: dict[str, int] = {}

    for part_path in DATA_PARTS:
        logger.info(f"  Loading {part_path.name} ({part_path.stat().st_size / 1e6:.1f} MB)")
        try:
            raw = json.loads(part_path.read_text())
        except Exception:
            logger.exception(f"Failed to load {part_path.name}")
            continue

        for ds in raw.get("datasets", []):
            ds_name = ds["dataset"]
            examples = ds.get("examples", [])
            if ds_name not in datasets:
                datasets[ds_name] = []
                ds_counts[ds_name] = 0

            # Efficiently take only what we need
            if max_per_dataset is not None:
                remaining = max_per_dataset - ds_counts[ds_name]
                if remaining <= 0:
                    continue
                examples = examples[:remaining]

            datasets[ds_name].extend(examples)
            ds_counts[ds_name] = len(datasets[ds_name])
            logger.info(f"    {ds_name}: +{len(examples)} examples (total: {ds_counts[ds_name]})")

        # Free memory
        del raw

    total = sum(len(v) for v in datasets.values())
    logger.info(f"Total loaded: {total} examples across {len(datasets)} datasets")
    return datasets


def load_mini_data() -> dict[str, list[dict]]:
    """Load mini dataset for testing."""
    logger.info(f"Loading mini data from {MINI_DATA}")
    raw = json.loads(MINI_DATA.read_text())
    datasets: dict[str, list[dict]] = {}
    for ds in raw.get("datasets", []):
        datasets[ds["dataset"]] = ds.get("examples", [])
    return datasets


def parse_input(example: dict) -> dict:
    """Parse the JSON-encoded input field of an example."""
    return json.loads(example["input"])


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 1: NODE-LEVEL SRI COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_node_level_sri(
    spectral: dict,
    num_nodes: int,
    threshold_factor: float = 1.0,
    k: int = K_WALK,
) -> dict:
    """
    Compute node-level SRI from local_spectral data.

    For each node u:
      1. Get the eigenvalues & squared eigenvector components from local_spectral
      2. Filter by threshold: v_i(u)^2 > threshold_factor / n
      3. Compute min gap among relevant eigenvalues
      4. node_sri = min_gap * K

    Returns dict with per-node SRIs and graph-level aggregates.
    """
    eigenvalues = np.array(spectral.get("eigenvalues", []))
    local_spectral = spectral.get("local_spectral", [])
    n = num_nodes
    n_nodes_with_spectral = len(local_spectral)

    if n_nodes_with_spectral == 0 or len(eigenvalues) == 0:
        return {
            "node_sris": [],
            "effective_ranks": [],
            "local_sparsities": [],
            "mean_node_sri": float("inf"),
            "min_node_sri": float("inf"),
            "median_node_sri": float("inf"),
            "std_node_sri": 0.0,
            "p10_node_sri": float("inf"),
            "mean_effective_rank": 0.0,
            "mean_local_sparsity": 0.0,
        }

    threshold = threshold_factor / max(n, 1)
    node_sris = []
    effective_ranks = []
    local_sparsities = []

    for u in range(n_nodes_with_spectral):
        components = local_spectral[u]  # list of [eigenvalue, weight] pairs
        if not components:
            node_sris.append(float("inf"))
            effective_ranks.append(0)
            local_sparsities.append(0.0)
            continue

        # Extract eigenvalues and weights for this node
        node_eigs = []
        node_weights = []
        for comp in components:
            if len(comp) >= 2:
                node_eigs.append(comp[0])
                node_weights.append(comp[1])

        if not node_eigs:
            node_sris.append(float("inf"))
            effective_ranks.append(0)
            local_sparsities.append(0.0)
            continue

        node_eigs = np.array(node_eigs)
        node_weights = np.array(node_weights)

        # Filter by threshold
        relevant_mask = node_weights > threshold
        relevant_eigs = node_eigs[relevant_mask]
        eff_rank = int(np.sum(relevant_mask))
        effective_ranks.append(eff_rank)
        local_sparsities.append(eff_rank / max(n, 1))

        if len(relevant_eigs) <= 1:
            node_sris.append(float("inf"))
            continue

        # Sort relevant eigenvalues, compute min gap
        sorted_eigs = np.sort(relevant_eigs)
        gaps = np.diff(sorted_eigs)
        nonzero_gaps = gaps[np.abs(gaps) > 1e-12]

        if len(nonzero_gaps) == 0:
            node_sris.append(0.0)  # All eigenvalues identical -> unresolvable
        else:
            local_delta_min = float(np.min(np.abs(nonzero_gaps)))
            node_sris.append(local_delta_min * k)

    node_sris_arr = np.array(node_sris, dtype=float)
    finite_sris = node_sris_arr[np.isfinite(node_sris_arr)]

    if len(finite_sris) == 0:
        return {
            "node_sris": node_sris,
            "effective_ranks": effective_ranks,
            "local_sparsities": local_sparsities,
            "mean_node_sri": float("inf"),
            "min_node_sri": float("inf"),
            "median_node_sri": float("inf"),
            "std_node_sri": 0.0,
            "p10_node_sri": float("inf"),
            "mean_effective_rank": float(np.mean(effective_ranks)) if effective_ranks else 0.0,
            "mean_local_sparsity": float(np.mean(local_sparsities)) if local_sparsities else 0.0,
        }

    return {
        "node_sris": node_sris,
        "effective_ranks": effective_ranks,
        "local_sparsities": local_sparsities,
        "mean_node_sri": float(np.mean(finite_sris)),
        "min_node_sri": float(np.min(finite_sris)),
        "median_node_sri": float(np.median(finite_sris)),
        "std_node_sri": float(np.std(finite_sris)),
        "p10_node_sri": float(np.percentile(finite_sris, 10)),
        "mean_effective_rank": float(np.mean(effective_ranks)),
        "mean_local_sparsity": float(np.mean(local_sparsities)),
    }


def compute_graph_level_sri(spectral: dict, k: int = K_WALK) -> float:
    """Compute graph-level SRI = delta_min * K."""
    delta_min = spectral.get("delta_min", 0.0)
    return delta_min * k


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 2: NODE-LEVEL DISTINGUISHABILITY
# ═══════════════════════════════════════════════════════════════════════════

def compute_lappe_features(local_spectral: list, eigenvalues: list, num_dims: int = 10) -> np.ndarray:
    """
    Compute sign-free LapPE features for each node:
      lappe(u) = [v_1(u)^2, ..., v_d(u)^2]

    Uses the local_spectral data which already contains (eigenvalue, weight=v_i(u)^2) pairs.
    We need to map to a fixed-dimensional representation based on eigenvalue indices.
    """
    n_nodes = len(local_spectral)
    n_eigs = len(eigenvalues)
    d = min(num_dims, n_eigs)

    if n_nodes == 0 or n_eigs == 0:
        return np.zeros((0, d))

    # Create eigenvalue-to-index mapping
    eig_array = np.array(eigenvalues)

    features = np.zeros((n_nodes, d), dtype=np.float64)
    for u in range(n_nodes):
        components = local_spectral[u]
        for comp in components:
            if len(comp) < 2:
                continue
            eig_val, weight = comp[0], comp[1]
            # Find the closest eigenvalue index
            diffs = np.abs(eig_array - eig_val)
            idx = int(np.argmin(diffs))
            if idx < d:
                features[u, idx] = weight

    return features


def compute_node_distinguishability(
    spectral: dict,
    num_nodes: int,
    rng: np.random.RandomState,
    max_pairs: int = MAX_NODE_PAIRS,
    epsilon: float = EPSILON,
) -> dict:
    """
    Compute node-level distinguishability comparing RWSE vs sign-free LapPE.

    Returns per-node scores and the gap (LapPE_score - RWSE_score).
    """
    rwse = spectral.get("rwse", [])
    local_spectral = spectral.get("local_spectral", [])
    eigenvalues = spectral.get("eigenvalues", [])

    # Determine how many nodes we can analyze (limited by local_spectral coverage)
    n_analyzable = min(len(rwse), len(local_spectral))
    if n_analyzable < 2:
        return {"node_rwse_scores": [], "node_lappe_scores": [], "node_gaps": [], "graph_gap": 0.0}

    # Get RWSE features (already stored)
    rwse_feats = np.array(rwse[:n_analyzable], dtype=np.float64)

    # Compute LapPE features
    lappe_feats = compute_lappe_features(local_spectral[:n_analyzable], eigenvalues, num_dims=min(10, len(eigenvalues)))

    # Normalize both feature sets per-graph (zero mean, unit variance per dimension)
    def normalize(X: np.ndarray) -> np.ndarray:
        if X.shape[0] < 2:
            return X
        m = X.mean(axis=0)
        s = X.std(axis=0)
        s[s < 1e-10] = 1.0
        return (X - m) / s

    rwse_norm = normalize(rwse_feats)
    lappe_norm = normalize(lappe_feats)

    # Sample node pairs
    n = n_analyzable
    total_pairs = n * (n - 1) // 2
    n_pairs = min(max_pairs, total_pairs)

    if total_pairs <= max_pairs:
        # Use all pairs
        pairs_i, pairs_j = [], []
        for i in range(n):
            for j in range(i + 1, n):
                pairs_i.append(i)
                pairs_j.append(j)
        pairs_i = np.array(pairs_i)
        pairs_j = np.array(pairs_j)
    else:
        # Random sampling
        all_pairs = rng.choice(total_pairs, size=n_pairs, replace=False)
        pairs_i = np.zeros(n_pairs, dtype=int)
        pairs_j = np.zeros(n_pairs, dtype=int)
        for idx, p in enumerate(all_pairs):
            # Convert linear index to (i, j) pair
            i = int(n - 2 - math.floor(math.sqrt(-8 * p + 4 * n * (n - 1) - 7) / 2 - 0.5))
            j = int(p + i + 1 - n * (n - 1) // 2 + (n - i) * ((n - i) - 1) // 2)
            if i >= n or j >= n or i < 0 or j < 0 or i >= j:
                # Fallback: random pair
                i = rng.randint(0, n - 1)
                j = rng.randint(i + 1, n)
            pairs_i[idx] = i
            pairs_j[idx] = j

    # Compute pairwise distances
    rwse_dists = np.linalg.norm(rwse_norm[pairs_i] - rwse_norm[pairs_j], axis=1)
    lappe_dists = np.linalg.norm(lappe_norm[pairs_i] - lappe_norm[pairs_j], axis=1)

    # Distinguished if dist > epsilon
    rwse_distinguished = rwse_dists > epsilon
    lappe_distinguished = lappe_dists > epsilon

    # Per-node scores: fraction of pairs involving this node that are distinguished
    node_rwse_scores = np.zeros(n)
    node_lappe_scores = np.zeros(n)
    node_pair_counts = np.zeros(n)

    for idx in range(len(pairs_i)):
        i, j = pairs_i[idx], pairs_j[idx]
        node_pair_counts[i] += 1
        node_pair_counts[j] += 1
        if rwse_distinguished[idx]:
            node_rwse_scores[i] += 1
            node_rwse_scores[j] += 1
        if lappe_distinguished[idx]:
            node_lappe_scores[i] += 1
            node_lappe_scores[j] += 1

    # Normalize by pair counts
    mask = node_pair_counts > 0
    node_rwse_scores[mask] /= node_pair_counts[mask]
    node_lappe_scores[mask] /= node_pair_counts[mask]

    node_gaps = node_lappe_scores - node_rwse_scores

    # Graph-level gap: fraction of pairs where LapPE distinguishes but RWSE doesn't
    graph_gap = float(np.mean(lappe_distinguished & ~rwse_distinguished))

    return {
        "node_rwse_scores": node_rwse_scores.tolist(),
        "node_lappe_scores": node_lappe_scores.tolist(),
        "node_gaps": node_gaps.tolist(),
        "graph_gap": graph_gap,
        "n_analyzable": n_analyzable,
        "n_pairs": int(len(pairs_i)),
        "frac_rwse_distinguished": float(np.mean(rwse_distinguished)),
        "frac_lappe_distinguished": float(np.mean(lappe_distinguished)),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 4: SRWE (Super-Resolution Walk Encoding) BENEFIT PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

def compute_srwe_tikhonov(
    eigenvalues: np.ndarray,
    k: int = K_WALK,
    alpha: float = 1e-3,
) -> np.ndarray:
    """
    Compute Tikhonov-regularized SRWE weights.

    V[k,i] = eigenvalues[i]^(k+1), k=0..K-1
    Solve (V^T V + alpha*I) w = V^T m for each node
    where m = RWSE features = sum_i lambda_i^k * v_i(u)^2

    Returns the weight matrix (n_eigs,).
    """
    n_eigs = len(eigenvalues)
    if n_eigs == 0 or k == 0:
        return np.zeros(0)

    # Build Vandermonde-like matrix
    V = np.zeros((k, n_eigs), dtype=np.float64)
    for ki in range(k):
        V[ki, :] = eigenvalues ** (ki + 1)

    # Check condition number
    cond = np.linalg.cond(V)
    if not np.isfinite(cond) or cond > 1e15:
        alpha = max(alpha, 0.1)  # Increase regularization

    # Regularized solution: (V^T V + alpha I) w = V^T m
    VtV = V.T @ V + alpha * np.eye(n_eigs)
    return V, VtV


def compute_srwe_features(
    spectral: dict,
    num_nodes: int,
    k: int = K_WALK,
    alpha: float = 1e-3,
) -> np.ndarray:
    """
    Compute SRWE features for each node using Tikhonov regularization.

    For each node, solve for weights w such that:
      V @ w ≈ rwse_features
    where V is the Vandermonde matrix of eigenvalues.

    Tries multiple alpha values and picks the one giving best reconstruction.
    """
    eigenvalues = np.array(spectral.get("eigenvalues", []))
    rwse = spectral.get("rwse", [])
    local_spectral = spectral.get("local_spectral", [])
    n_eigs = len(eigenvalues)
    n_nodes_available = min(len(rwse), len(local_spectral))

    if n_nodes_available == 0 or n_eigs == 0:
        return np.zeros((0, 0))

    k_use = min(k, n_eigs)

    # Build Vandermonde matrix
    V = np.zeros((k_use, n_eigs), dtype=np.float64)
    for ki in range(k_use):
        V[ki, :] = eigenvalues ** (ki + 1)

    # Try multiple alpha values, pick best per-graph
    alphas = [1e-6, 1e-4, 1e-2, 0.1]
    best_alpha = alpha

    # Quick test with first node to pick best alpha
    if n_nodes_available > 0:
        test_rwse = np.array(rwse[0][:k_use], dtype=np.float64)
        if len(test_rwse) < k_use:
            test_rwse = np.pad(test_rwse, (0, k_use - len(test_rwse)))
        best_resid = float("inf")
        for a in alphas:
            try:
                VtV_test = V.T @ V + a * np.eye(n_eigs)
                w_test = np.linalg.solve(VtV_test, V.T @ test_rwse)
                w_test = np.maximum(w_test, 0)
                resid = np.linalg.norm(V @ w_test - test_rwse)
                if resid < best_resid:
                    best_resid = resid
                    best_alpha = a
            except np.linalg.LinAlgError:
                continue

    # Regularized matrix with best alpha
    VtV = V.T @ V + best_alpha * np.eye(n_eigs)

    # For each node, solve for SRWE weights
    srwe_features = np.zeros((n_nodes_available, n_eigs), dtype=np.float64)

    for u in range(n_nodes_available):
        rwse_vec = np.array(rwse[u][:k_use], dtype=np.float64)
        if len(rwse_vec) < k_use:
            rwse_vec = np.pad(rwse_vec, (0, k_use - len(rwse_vec)))

        try:
            # Solve Tikhonov problem
            Vt_m = V.T @ rwse_vec
            w = np.linalg.solve(VtV, Vt_m)
            w = np.maximum(w, 0)  # Non-negative weights
            srwe_features[u, :] = w
        except np.linalg.LinAlgError:
            # Fallback: use NNLS
            try:
                w, _ = nnls(V.T, rwse_vec[:min(len(rwse_vec), V.shape[0])])
                if len(w) < n_eigs:
                    w = np.pad(w, (0, n_eigs - len(w)))
                srwe_features[u, :] = w[:n_eigs]
            except Exception:
                pass  # Leave as zeros

    return srwe_features


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 3: CORRELATIONS & BOOTSTRAP
# ═══════════════════════════════════════════════════════════════════════════

def spearman_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Compute Spearman correlation, handling edge cases."""
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    if len(x_clean) < 3:
        return 0.0, 1.0
    if np.std(x_clean) < 1e-10 or np.std(y_clean) < 1e-10:
        return 0.0, 1.0
    rho, p = stats.spearmanr(x_clean, y_clean)
    if not np.isfinite(rho):
        return 0.0, 1.0
    return float(rho), float(p)


def bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    n_resamples: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> dict:
    """Bootstrap 95% CI for Spearman correlation."""
    rng = np.random.RandomState(seed)
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    n = len(x_clean)

    if n < 5:
        return {"rho": 0.0, "p_value": 1.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": n}

    rho, p = spearman_corr(x_clean, y_clean)

    boot_rhos = np.zeros(n_resamples)
    for i in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        r, _ = spearman_corr(x_clean[idx], y_clean[idx])
        boot_rhos[i] = r

    ci_lower = float(np.percentile(boot_rhos, 2.5))
    ci_upper = float(np.percentile(boot_rhos, 97.5))

    return {"rho": rho, "p_value": p, "ci_lower": ci_lower, "ci_upper": ci_upper, "n": n}


def partial_spearman(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[float, float]:
    """
    Partial Spearman correlation between x and y, controlling for z.
    Uses rank residualization.
    """
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x_c, y_c, z_c = x[mask], y[mask], z[mask]
    if len(x_c) < 5:
        return 0.0, 1.0

    # Rank transform
    x_r = stats.rankdata(x_c)
    y_r = stats.rankdata(y_c)
    z_r = stats.rankdata(z_c)

    # Residualize x on z
    slope_xz = np.polyfit(z_r, x_r, 1)
    x_resid = x_r - np.polyval(slope_xz, z_r)

    # Residualize y on z
    slope_yz = np.polyfit(z_r, y_r, 1)
    y_resid = y_r - np.polyval(slope_yz, z_r)

    return spearman_corr(x_resid, y_resid)


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 5: VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def plot_correlation_comparison(results: dict, output_path: Path) -> None:
    """
    KEY FIGURE: Bar chart comparing |rho| for all SRI variants.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    fig.suptitle("SRI-Performance Correlation: Graph-Level vs Node-Level Aggregations", fontsize=14, y=1.02)

    datasets = list(results.keys())
    if not datasets:
        plt.close(fig)
        return

    # Add "Pooled" if we have multiple datasets
    all_keys = datasets + (["Pooled"] if "Pooled" in results else [])

    for ax_idx, ds_key in enumerate(all_keys[:4]):
        if ds_key not in results:
            continue
        ax = axes[ax_idx] if ax_idx < 4 else axes[-1]
        ds_results = results[ds_key]

        labels = []
        rhos = []
        ci_lowers = []
        ci_uppers = []

        for metric_name, metric_data in ds_results.items():
            if isinstance(metric_data, dict) and "rho" in metric_data:
                labels.append(metric_name.replace("_", "\n"))
                rhos.append(abs(metric_data["rho"]))
                ci_lowers.append(abs(metric_data.get("ci_lower", 0)))
                ci_uppers.append(abs(metric_data.get("ci_upper", 0)))

        if not labels:
            continue

        x = np.arange(len(labels))
        rhos = np.array(rhos)
        errors_lower = rhos - np.array(ci_lowers)
        errors_upper = np.array(ci_uppers) - rhos
        errors = np.array([np.maximum(errors_lower, 0), np.maximum(errors_upper, 0)])

        colors = ["#2196F3" if "graph" in labels[i].lower() else "#FF9800" for i in range(len(labels))]
        ax.bar(x, rhos, yerr=errors, color=colors, alpha=0.8, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        ax.set_title(ds_key, fontsize=11)
        ax.set_ylabel("|Spearman ρ|" if ax_idx == 0 else "")
        ax.axhline(y=0.3, color="red", linestyle="--", alpha=0.5, label="ρ=0.3 threshold")
        ax.set_ylim(0, 1.0)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved correlation comparison figure: {output_path}")


def plot_scatter_sri_comparison(
    graph_sris: np.ndarray,
    mean_node_sris: np.ndarray,
    graph_gaps: np.ndarray,
    dataset_names: list[str],
    output_path: Path,
) -> None:
    """Scatter: graph_SRI vs mean_node_SRI, colored by gap."""
    unique_ds = sorted(set(dataset_names))
    n_panels = min(len(unique_ds), 4)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    for ax_idx, ds_name in enumerate(unique_ds[:n_panels]):
        mask = np.array([d == ds_name for d in dataset_names])
        ax = axes[ax_idx]

        x = graph_sris[mask]
        y = mean_node_sris[mask]
        c = graph_gaps[mask]

        # Filter finite values
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
        if np.sum(finite) == 0:
            ax.set_title(f"{ds_name}\n(no valid data)")
            continue

        sc = ax.scatter(x[finite], y[finite], c=c[finite], cmap="RdYlBu_r",
                       s=10, alpha=0.5, vmin=0, vmax=max(0.5, np.percentile(c[finite], 95)))
        ax.set_xlabel("Graph-Level SRI")
        ax.set_ylabel("Mean Node-Level SRI")
        ax.set_title(ds_name)
        plt.colorbar(sc, ax=ax, label="LapPE-RWSE Gap")

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved scatter comparison: {output_path}")


def plot_srwe_benefit(
    node_sris_by_category: dict,
    output_path: Path,
) -> None:
    """Violin plot: node_SRI distribution by SRWE resolution category."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    categories = ["already_resolved", "newly_resolved", "still_unresolved"]
    cat_labels = ["Already Resolved\n(RWSE)", "Newly Resolved\n(SRWE only)", "Still Unresolved"]
    data_to_plot = []
    positions = []
    colors = ["#4CAF50", "#FF9800", "#F44336"]

    for i, cat in enumerate(categories):
        vals = node_sris_by_category.get(cat, [])
        if vals:
            finite_vals = [v for v in vals if np.isfinite(v)]
            if finite_vals:
                data_to_plot.append(finite_vals)
                positions.append(i)

    if data_to_plot:
        parts = ax.violinplot(data_to_plot, positions=positions, showmeans=True, showmedians=True)
        for i, pc in enumerate(parts.get("bodies", [])):
            pc.set_facecolor(colors[positions[i]])
            pc.set_alpha(0.6)

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(cat_labels, fontsize=10)
    ax.set_ylabel("Node-Level SRI")
    ax.set_title("Node SRI Distribution by SRWE Resolution Category")

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved SRWE benefit figure: {output_path}")


def plot_within_graph_correlations(
    within_graph_rhos: list[float],
    output_path: Path,
) -> None:
    """Histogram of within-graph Spearman correlations."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    valid_rhos = [r for r in within_graph_rhos if np.isfinite(r)]
    if valid_rhos:
        ax.hist(valid_rhos, bins=30, color="#2196F3", alpha=0.7, edgecolor="black")
        ax.axvline(np.mean(valid_rhos), color="red", linestyle="--",
                  label=f"Mean = {np.mean(valid_rhos):.3f}")
        ax.axvline(0, color="black", linestyle="-", alpha=0.3)
        ax.legend()

    ax.set_xlabel("Within-Graph Spearman ρ (node_SRI vs node_gap)")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Within-Graph Correlations (n={len(valid_rhos)} graphs)")

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved within-graph correlations: {output_path}")


def plot_node_sri_heatmaps(
    example_graphs: list[dict],
    output_path: Path,
) -> None:
    """Node SRI bar charts for example graphs (2 per dataset, high-gap and low-gap)."""
    n_examples = len(example_graphs)
    if n_examples == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes = axes.flatten()

    for i, graph_info in enumerate(example_graphs[:6]):
        ax = axes[i]
        node_sris = graph_info.get("node_sris", [])
        ds_name = graph_info.get("dataset", "")
        gap_type = graph_info.get("gap_type", "")

        finite_sris = [s if np.isfinite(s) else 0 for s in node_sris[:50]]  # Cap at 50 nodes
        if finite_sris:
            colors = plt.cm.RdYlBu_r(np.array(finite_sris) / max(max(finite_sris), 1e-10))
            ax.bar(range(len(finite_sris)), finite_sris, color=colors, alpha=0.8)

        ax.set_xlabel("Node Index")
        ax.set_ylabel("Node SRI")
        ax.set_title(f"{ds_name} ({gap_type})\nn={graph_info.get('n_nodes', '?')}")

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved node SRI heatmaps: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_analysis(
    datasets: dict[str, list[dict]],
    max_sample_per_ds: int = SAMPLE_PER_DATASET,
    run_srwe: bool = True,
) -> dict:
    """
    Run the full analysis pipeline.

    Returns a comprehensive results dictionary.
    """
    rng = np.random.RandomState(SEED)
    start_time = time.time()

    results = {
        "metadata": {
            "method_name": "Node-Level SRI vs Graph-Level SRI Analysis",
            "description": "Testing whether node-level SRI improves SRI-performance correlation over graph-level SRI",
            "parameters": {
                "K": K_WALK,
                "threshold_factors": THRESHOLD_FACTORS,
                "n_bootstrap": N_BOOTSTRAP,
                "epsilon": EPSILON,
                "max_sample_per_ds": max_sample_per_ds,
                "seed": SEED,
            },
        },
        "phase1_node_sri": {},
        "phase2_distinguishability": {},
        "phase3_correlations": {},
        "phase4_srwe": {},
    }

    # ── PHASE 1: Node-Level SRI Computation ──
    logger.info("=" * 60)
    logger.info("PHASE 1: Node-Level SRI Computation")
    logger.info("=" * 60)

    # Store per-graph metrics for all datasets
    all_graph_metrics: dict[str, list[dict]] = {}

    for ds_name, examples in datasets.items():
        logger.info(f"Processing {ds_name} ({len(examples)} examples)")
        graph_metrics = []

        for ex_idx, example in enumerate(examples):
            try:
                inp = parse_input(example)
                spectral = inp.get("spectral", {})
                num_nodes = inp.get("num_nodes", 0)

                # Graph-level SRI
                graph_sri = compute_graph_level_sri(spectral)

                # Node-level SRI for each threshold factor
                node_results = {}
                for tf in THRESHOLD_FACTORS:
                    node_res = compute_node_level_sri(spectral, num_nodes, threshold_factor=tf)
                    node_results[f"tf_{tf}"] = node_res

                # Use threshold_factor=1.0 as primary
                primary = node_results["tf_1.0"]

                metric = {
                    "idx": ex_idx,
                    "num_nodes": num_nodes,
                    "graph_sri": graph_sri,
                    "delta_min": spectral.get("delta_min", 0.0),
                    "mean_node_sri": primary["mean_node_sri"],
                    "min_node_sri": primary["min_node_sri"],
                    "median_node_sri": primary["median_node_sri"],
                    "std_node_sri": primary["std_node_sri"],
                    "p10_node_sri": primary["p10_node_sri"],
                    "mean_effective_rank": primary["mean_effective_rank"],
                    "mean_local_sparsity": primary["mean_local_sparsity"],
                    "node_sris": primary["node_sris"],  # Keep for Phase 2
                    "node_results_by_tf": {
                        tf_key: {
                            "mean_node_sri": nr["mean_node_sri"],
                            "min_node_sri": nr["min_node_sri"],
                            "median_node_sri": nr["median_node_sri"],
                        }
                        for tf_key, nr in node_results.items()
                    },
                }
                graph_metrics.append(metric)

            except Exception:
                logger.exception(f"Failed on example {ex_idx} in {ds_name}")
                continue

            if (ex_idx + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                logger.info(f"  Phase 1: {ds_name} {ex_idx + 1}/{len(examples)} ({elapsed:.1f}s)")

        all_graph_metrics[ds_name] = graph_metrics
        logger.info(f"  {ds_name}: {len(graph_metrics)} graphs processed")

        # Phase 1 summary stats
        finite_mean_sris = [m["mean_node_sri"] for m in graph_metrics if np.isfinite(m["mean_node_sri"])]
        finite_graph_sris = [m["graph_sri"] for m in graph_metrics if np.isfinite(m["graph_sri"])]
        results["phase1_node_sri"][ds_name] = {
            "n_graphs": len(graph_metrics),
            "mean_graph_sri": float(np.mean(finite_graph_sris)) if finite_graph_sris else 0.0,
            "mean_node_sri_avg": float(np.mean(finite_mean_sris)) if finite_mean_sris else 0.0,
            "std_within_graph": float(np.mean([m["std_node_sri"] for m in graph_metrics if np.isfinite(m["std_node_sri"])])) if graph_metrics else 0.0,
        }

    # ── PHASE 2: Node-Level Distinguishability ──
    logger.info("=" * 60)
    logger.info("PHASE 2: Node-Level Distinguishability")
    logger.info("=" * 60)

    all_sampled_metrics: dict[str, list[dict]] = {}
    within_graph_rhos: list[float] = []
    example_graphs: list[dict] = []  # For visualization

    for ds_name, examples in datasets.items():
        n_sample = min(max_sample_per_ds, len(examples))
        if ds_name == "Synthetic-aliased-pairs":
            n_sample = len(examples)  # Use all synthetic

        sample_indices = rng.choice(len(examples), size=n_sample, replace=False) if n_sample < len(examples) else np.arange(len(examples))
        logger.info(f"Phase 2: {ds_name} — sampling {n_sample} graphs")

        sampled_metrics = []
        ds_high_gap = None
        ds_low_gap = None
        ds_high_gap_val = -1.0
        ds_low_gap_val = 2.0

        for s_idx, orig_idx in enumerate(sample_indices):
            try:
                example = examples[orig_idx]
                inp = parse_input(example)
                spectral = inp.get("spectral", {})
                num_nodes = inp.get("num_nodes", 0)

                # Get graph metrics (if we have them from Phase 1)
                gm = all_graph_metrics[ds_name][orig_idx] if orig_idx < len(all_graph_metrics[ds_name]) else None

                # Compute distinguishability
                dist_result = compute_node_distinguishability(
                    spectral, num_nodes, rng, max_pairs=MAX_NODE_PAIRS, epsilon=EPSILON
                )

                node_sris = gm["node_sris"] if gm else []
                n_analyzable = dist_result.get("n_analyzable", 0)

                sampled_metric = {
                    "orig_idx": int(orig_idx),
                    "num_nodes": num_nodes,
                    "graph_sri": gm["graph_sri"] if gm else 0.0,
                    "mean_node_sri": gm["mean_node_sri"] if gm else 0.0,
                    "min_node_sri": gm["min_node_sri"] if gm else 0.0,
                    "median_node_sri": gm["median_node_sri"] if gm else 0.0,
                    "p10_node_sri": gm["p10_node_sri"] if gm else 0.0,
                    "graph_gap": dist_result["graph_gap"],
                    "frac_rwse": dist_result.get("frac_rwse_distinguished", 0.0),
                    "frac_lappe": dist_result.get("frac_lappe_distinguished", 0.0),
                    "node_sris": node_sris[:n_analyzable],
                    "node_gaps": dist_result["node_gaps"],
                }
                sampled_metrics.append(sampled_metric)

                # Track high/low gap examples for visualization
                gg = dist_result["graph_gap"]
                if gg > ds_high_gap_val and n_analyzable >= 5:
                    ds_high_gap_val = gg
                    ds_high_gap = {
                        "node_sris": node_sris[:n_analyzable],
                        "dataset": ds_name,
                        "gap_type": "high-gap",
                        "n_nodes": num_nodes,
                        "graph_gap": gg,
                    }
                if gg < ds_low_gap_val and n_analyzable >= 5:
                    ds_low_gap_val = gg
                    ds_low_gap = {
                        "node_sris": node_sris[:n_analyzable],
                        "dataset": ds_name,
                        "gap_type": "low-gap",
                        "n_nodes": num_nodes,
                        "graph_gap": gg,
                    }

                # Within-graph correlation (if enough nodes)
                if n_analyzable >= 10 and len(node_sris) >= n_analyzable:
                    ns = np.array(node_sris[:n_analyzable], dtype=float)
                    ng = np.array(dist_result["node_gaps"], dtype=float)
                    finite_mask = np.isfinite(ns) & np.isfinite(ng)
                    if np.sum(finite_mask) >= 5:
                        rho_within, _ = spearman_corr(ns[finite_mask], ng[finite_mask])
                        within_graph_rhos.append(rho_within)

            except Exception:
                logger.exception(f"Phase 2: Failed on {ds_name} idx {orig_idx}")
                continue

            if (s_idx + 1) % 200 == 0:
                elapsed = time.time() - start_time
                logger.info(f"  Phase 2: {ds_name} {s_idx + 1}/{n_sample} ({elapsed:.1f}s)")

        all_sampled_metrics[ds_name] = sampled_metrics
        if ds_high_gap:
            example_graphs.append(ds_high_gap)
        if ds_low_gap:
            example_graphs.append(ds_low_gap)

        # Node-level correlation: node_sri vs node_gap across ALL nodes in sampled graphs
        all_node_sris_flat = []
        all_node_gaps_flat = []
        for sm in sampled_metrics:
            ns = sm.get("node_sris", [])
            ng = sm.get("node_gaps", [])
            n_use = min(len(ns), len(ng))
            for i in range(n_use):
                if np.isfinite(ns[i]) and np.isfinite(ng[i]):
                    all_node_sris_flat.append(ns[i])
                    all_node_gaps_flat.append(ng[i])

        if all_node_sris_flat:
            node_rho, node_p = spearman_corr(
                np.array(all_node_sris_flat), np.array(all_node_gaps_flat)
            )
            results["phase2_distinguishability"][ds_name] = {
                "n_sampled_graphs": len(sampled_metrics),
                "n_total_nodes": len(all_node_sris_flat),
                "node_level_rho": node_rho,
                "node_level_p": node_p,
                "mean_graph_gap": float(np.mean([sm["graph_gap"] for sm in sampled_metrics])),
                "n_within_graph_rhos": len(within_graph_rhos),
                "mean_within_graph_rho": float(np.mean(within_graph_rhos)) if within_graph_rhos else 0.0,
            }
        else:
            results["phase2_distinguishability"][ds_name] = {
                "n_sampled_graphs": len(sampled_metrics),
                "n_total_nodes": 0,
                "node_level_rho": 0.0,
                "node_level_p": 1.0,
            }

        logger.info(f"  {ds_name}: {len(sampled_metrics)} graphs, node-level rho={results['phase2_distinguishability'][ds_name].get('node_level_rho', 0):.4f}")

    # ── PHASE 3: Graph-Level Correlations and Comparison ──
    logger.info("=" * 60)
    logger.info("PHASE 3: Graph-Level Correlations and Comparison")
    logger.info("=" * 60)

    correlation_results: dict[str, dict] = {}

    for ds_name, sampled in all_sampled_metrics.items():
        if len(sampled) < 5:
            logger.warning(f"Skipping {ds_name} correlations: only {len(sampled)} samples")
            continue

        graph_gap = np.array([s["graph_gap"] for s in sampled], dtype=float)
        graph_sri = np.array([s["graph_sri"] for s in sampled], dtype=float)
        mean_node_sri = np.array([s["mean_node_sri"] for s in sampled], dtype=float)
        min_node_sri = np.array([s["min_node_sri"] for s in sampled], dtype=float)
        median_node_sri = np.array([s["median_node_sri"] for s in sampled], dtype=float)
        p10_node_sri = np.array([s["p10_node_sri"] for s in sampled], dtype=float)
        log_n_nodes = np.log(np.array([s["num_nodes"] for s in sampled], dtype=float) + 1)

        # Compute correlations for all metrics
        metrics = {
            "graph_sri": graph_sri,
            "mean_node_sri": mean_node_sri,
            "min_node_sri": min_node_sri,
            "median_node_sri": median_node_sri,
            "p10_node_sri": p10_node_sri,
        }

        ds_corr = {}
        for metric_name, metric_vals in metrics.items():
            # Regular Spearman with bootstrap CI
            boot = bootstrap_spearman(metric_vals, graph_gap)
            ds_corr[metric_name] = boot

            # Partial correlation controlling for log(n_nodes)
            partial_rho, partial_p = partial_spearman(metric_vals, graph_gap, log_n_nodes)
            ds_corr[f"{metric_name}_partial"] = {
                "rho": partial_rho,
                "p_value": partial_p,
            }

        # Additional aggregations from fallback plan
        # Weighted by degree centrality (approximate using effective_rank)
        graph_metrics_for_ds = all_graph_metrics.get(ds_name, [])
        if graph_metrics_for_ds:
            # Harmonic mean of node SRIs
            harmonic_sris = []
            entropy_sris = []
            frac_low_sri = []
            for sm in sampled:
                ns = sm.get("node_sris", [])
                finite_ns = [v for v in ns if np.isfinite(v) and v > 0]
                if finite_ns:
                    harmonic_sris.append(float(len(finite_ns) / sum(1.0 / v for v in finite_ns)))
                    # Entropy of node SRI distribution
                    ns_arr = np.array(finite_ns)
                    ns_norm = ns_arr / (ns_arr.sum() + 1e-10)
                    ns_norm = ns_norm[ns_norm > 0]
                    entropy_sris.append(float(-np.sum(ns_norm * np.log(ns_norm + 1e-10))))
                    # Fraction of nodes with SRI < 1
                    frac_low_sri.append(float(np.mean(np.array(finite_ns) < 1.0)))
                else:
                    harmonic_sris.append(float("inf"))
                    entropy_sris.append(0.0)
                    frac_low_sri.append(0.0)

            for name, vals in [("harmonic_node_sri", harmonic_sris),
                              ("entropy_node_sri", entropy_sris),
                              ("frac_low_sri", frac_low_sri)]:
                arr = np.array(vals, dtype=float)
                boot = bootstrap_spearman(arr, graph_gap)
                ds_corr[name] = boot

        correlation_results[ds_name] = ds_corr
        logger.info(f"  {ds_name}: graph_sri rho={ds_corr['graph_sri']['rho']:.4f}, "
                    f"mean_node_sri rho={ds_corr['mean_node_sri']['rho']:.4f}, "
                    f"min_node_sri rho={ds_corr['min_node_sri']['rho']:.4f}")

    # Pooled correlation across all datasets
    all_graph_gap = []
    all_graph_sri = []
    all_mean_node_sri = []
    all_min_node_sri = []
    all_median_node_sri = []
    all_p10_node_sri = []
    all_log_n = []
    all_ds_names = []

    for ds_name, sampled in all_sampled_metrics.items():
        for s in sampled:
            all_graph_gap.append(s["graph_gap"])
            all_graph_sri.append(s["graph_sri"])
            all_mean_node_sri.append(s["mean_node_sri"])
            all_min_node_sri.append(s["min_node_sri"])
            all_median_node_sri.append(s["median_node_sri"])
            all_p10_node_sri.append(s["p10_node_sri"])
            all_log_n.append(math.log(s["num_nodes"] + 1))
            all_ds_names.append(ds_name)

    if len(all_graph_gap) >= 5:
        pooled_corr = {}
        gap_arr = np.array(all_graph_gap, dtype=float)
        for name, arr in [("graph_sri", all_graph_sri), ("mean_node_sri", all_mean_node_sri),
                          ("min_node_sri", all_min_node_sri), ("median_node_sri", all_median_node_sri),
                          ("p10_node_sri", all_p10_node_sri)]:
            boot = bootstrap_spearman(np.array(arr, dtype=float), gap_arr)
            pooled_corr[name] = boot

            partial_rho, partial_p = partial_spearman(
                np.array(arr, dtype=float), gap_arr, np.array(all_log_n, dtype=float)
            )
            pooled_corr[f"{name}_partial"] = {"rho": partial_rho, "p_value": partial_p}

        correlation_results["Pooled"] = pooled_corr
        logger.info(f"  Pooled: graph_sri rho={pooled_corr['graph_sri']['rho']:.4f}, "
                    f"mean_node_sri rho={pooled_corr['mean_node_sri']['rho']:.4f}")

    results["phase3_correlations"] = correlation_results

    # Bootstrap difference test
    if "Pooled" in correlation_results:
        best_node_metric = max(
            ["mean_node_sri", "min_node_sri", "median_node_sri", "p10_node_sri"],
            key=lambda m: abs(correlation_results["Pooled"].get(m, {}).get("rho", 0))
        )
        graph_rho = abs(correlation_results["Pooled"]["graph_sri"]["rho"])
        node_rho = abs(correlation_results["Pooled"][best_node_metric]["rho"])
        results["phase3_correlations"]["improvement"] = {
            "best_node_metric": best_node_metric,
            "graph_rho": graph_rho,
            "node_rho": node_rho,
            "improvement": node_rho - graph_rho,
            "significant": node_rho > graph_rho + 0.05,
        }
        logger.info(f"  Best node metric: {best_node_metric}, improvement: {node_rho - graph_rho:.4f}")

    # ── PHASE 4: SRWE Benefit Prediction ──
    logger.info("=" * 60)
    logger.info("PHASE 4: SRWE Benefit Prediction")
    logger.info("=" * 60)

    srwe_node_sris_by_category = {"already_resolved": [], "newly_resolved": [], "still_unresolved": []}
    n_srwe_processed = 0
    # Distribute SRWE budget equally across datasets
    n_datasets = len(all_sampled_metrics)
    srwe_per_ds = max(50, 200 // max(n_datasets, 1))

    if run_srwe:
        for ds_name, sampled in all_sampled_metrics.items():
            n_use = min(len(sampled), srwe_per_ds)
            logger.info(f"Phase 4: Processing {n_use} graphs from {ds_name}")

            for s_idx in range(n_use):
                try:
                    sm = sampled[s_idx]
                    example = datasets[ds_name][sm["orig_idx"]]
                    inp = parse_input(example)
                    spectral = inp.get("spectral", {})
                    num_nodes = inp.get("num_nodes", 0)

                    # Compute SRWE features
                    srwe_feats = compute_srwe_features(spectral, num_nodes, k=K_WALK, alpha=1e-3)

                    if srwe_feats.shape[0] < 2:
                        n_srwe_processed += 1
                        continue

                    # Get RWSE and LapPE features for comparison
                    rwse = np.array(spectral.get("rwse", [])[:srwe_feats.shape[0]], dtype=np.float64)
                    local_spectral = spectral.get("local_spectral", [])
                    eigenvalues = spectral.get("eigenvalues", [])

                    n_compare = min(rwse.shape[0], srwe_feats.shape[0], len(local_spectral))
                    if n_compare < 2:
                        n_srwe_processed += 1
                        continue

                    # Normalize features
                    def normalize(X: np.ndarray) -> np.ndarray:
                        if X.shape[0] < 2:
                            return X
                        m = X.mean(axis=0)
                        s = X.std(axis=0)
                        s[s < 1e-10] = 1.0
                        return (X - m) / s

                    rwse_norm = normalize(rwse[:n_compare])
                    srwe_norm = normalize(srwe_feats[:n_compare])

                    node_sris = sm.get("node_sris", [])

                    # Also compute LapPE for a 3-way comparison
                    lappe_feats = compute_lappe_features(
                        local_spectral[:n_compare], eigenvalues, num_dims=min(10, len(eigenvalues))
                    )
                    lappe_norm = normalize(lappe_feats[:n_compare])

                    # Sample pairs and categorize
                    n_pairs_check = min(200, n_compare * (n_compare - 1) // 2)
                    for _ in range(n_pairs_check):
                        u = rng.randint(0, n_compare - 1)
                        w = rng.randint(u + 1, n_compare)

                        rwse_dist = np.linalg.norm(rwse_norm[u] - rwse_norm[w])
                        srwe_dist = np.linalg.norm(srwe_norm[u] - srwe_norm[w])
                        lappe_dist = np.linalg.norm(lappe_norm[u] - lappe_norm[w]) if lappe_norm.shape[0] > w else 0.0

                        # Use max of SRWE and LapPE for "super-resolution" comparison
                        super_dist = max(srwe_dist, lappe_dist)

                        pair_sri = min(
                            node_sris[u] if u < len(node_sris) and np.isfinite(node_sris[u]) else float("inf"),
                            node_sris[w] if w < len(node_sris) and np.isfinite(node_sris[w]) else float("inf"),
                        )

                        if rwse_dist > EPSILON:
                            srwe_node_sris_by_category["already_resolved"].append(pair_sri)
                        elif super_dist > EPSILON:
                            srwe_node_sris_by_category["newly_resolved"].append(pair_sri)
                        else:
                            srwe_node_sris_by_category["still_unresolved"].append(pair_sri)

                    n_srwe_processed += 1

                except Exception:
                    logger.exception(f"Phase 4: Failed on {ds_name} idx {s_idx}")
                    n_srwe_processed += 1
                    continue

                if (s_idx + 1) % 50 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"  Phase 4: {ds_name} {s_idx + 1}/{n_use} ({elapsed:.1f}s)")

        # Mann-Whitney U test
        newly = [v for v in srwe_node_sris_by_category["newly_resolved"] if np.isfinite(v)]
        already = [v for v in srwe_node_sris_by_category["already_resolved"] if np.isfinite(v)]
        unresolved = [v for v in srwe_node_sris_by_category["still_unresolved"] if np.isfinite(v)]

        results["phase4_srwe"] = {
            "n_already_resolved": len(already),
            "n_newly_resolved": len(newly),
            "n_still_unresolved": len(unresolved),
            "mean_sri_already": float(np.mean(already)) if already else 0.0,
            "mean_sri_newly": float(np.mean(newly)) if newly else 0.0,
            "mean_sri_unresolved": float(np.mean(unresolved)) if unresolved else 0.0,
        }

        if len(newly) >= 3 and len(already) >= 3:
            try:
                u_stat, u_p = stats.mannwhitneyu(newly, already, alternative="less")
                results["phase4_srwe"]["mann_whitney_newly_vs_already"] = {
                    "U_statistic": float(u_stat),
                    "p_value": float(u_p),
                    "significant": float(u_p) < 0.05,
                }
                logger.info(f"Mann-Whitney U: newly vs already: U={u_stat:.1f}, p={u_p:.4f}")
            except Exception:
                logger.exception("Mann-Whitney U test failed")
        else:
            logger.info(f"Insufficient data for Mann-Whitney: newly={len(newly)}, already={len(already)}")

        logger.info(f"  SRWE categories: already={len(already)}, newly={len(newly)}, unresolved={len(unresolved)}")

    # ── PHASE 5: Visualization ──
    logger.info("=" * 60)
    logger.info("PHASE 5: Visualization")
    logger.info("=" * 60)

    try:
        # Figure 1: Correlation comparison bar chart (KEY FIGURE)
        plot_correlation_comparison(correlation_results, FIGURES_DIR / "fig1_correlation_comparison.png")
    except Exception:
        logger.exception("Failed to generate Fig 1")

    try:
        # Figure 2: Scatter plot graph_SRI vs mean_node_SRI
        plot_scatter_sri_comparison(
            np.array(all_graph_sri, dtype=float),
            np.array(all_mean_node_sri, dtype=float),
            np.array(all_graph_gap, dtype=float),
            all_ds_names,
            FIGURES_DIR / "fig2_sri_scatter.png",
        )
    except Exception:
        logger.exception("Failed to generate Fig 2")

    try:
        # Figure 3: Within-graph correlations
        plot_within_graph_correlations(within_graph_rhos, FIGURES_DIR / "fig3_within_graph_rhos.png")
    except Exception:
        logger.exception("Failed to generate Fig 3")

    try:
        # Figure 4: Node SRI heatmaps
        plot_node_sri_heatmaps(example_graphs, FIGURES_DIR / "fig4_node_sri_heatmaps.png")
    except Exception:
        logger.exception("Failed to generate Fig 4")

    try:
        # Figure 5: SRWE benefit violin plot
        if srwe_node_sris_by_category.get("newly_resolved") or srwe_node_sris_by_category.get("already_resolved"):
            plot_srwe_benefit(srwe_node_sris_by_category, FIGURES_DIR / "fig5_srwe_benefit.png")
    except Exception:
        logger.exception("Failed to generate Fig 5")

    # Summary
    elapsed = time.time() - start_time
    results["metadata"]["total_runtime_seconds"] = elapsed
    results["metadata"]["figure_paths"] = [
        str(p) for p in sorted(FIGURES_DIR.glob("*.png"))
    ]

    logger.info(f"Total runtime: {elapsed:.1f}s")

    return results


def format_output_for_schema(
    datasets: dict[str, list[dict]],
    results: dict,
) -> dict:
    """
    Format results into the exp_gen_sol_out.json schema:
    {
        "metadata": {...},
        "datasets": [
            {"dataset": "...", "examples": [
                {"input": "...", "output": "...", "predict_baseline": "...", "predict_our_method": "..."}
            ]}
        ]
    }
    """
    output = {
        "metadata": results.get("metadata", {}),
        "datasets": [],
    }

    for ds_name, examples in datasets.items():
        ds_examples = []
        phase1 = results.get("phase1_node_sri", {}).get(ds_name, {})

        for ex_idx, example in enumerate(examples):
            try:
                inp = parse_input(example)
                spectral = inp.get("spectral", {})
                num_nodes = inp.get("num_nodes", 0)

                # Baseline: graph-level SRI
                graph_sri = compute_graph_level_sri(spectral)

                # Our method: node-level SRI
                node_res = compute_node_level_sri(spectral, num_nodes, threshold_factor=1.0)

                # Format predictions as strings (schema requirement)
                baseline_str = json.dumps({
                    "graph_sri": round(graph_sri, 6),
                    "delta_min": round(spectral.get("delta_min", 0.0), 6),
                })

                our_method_str = json.dumps({
                    "mean_node_sri": round(node_res["mean_node_sri"], 6) if np.isfinite(node_res["mean_node_sri"]) else "inf",
                    "min_node_sri": round(node_res["min_node_sri"], 6) if np.isfinite(node_res["min_node_sri"]) else "inf",
                    "median_node_sri": round(node_res["median_node_sri"], 6) if np.isfinite(node_res["median_node_sri"]) else "inf",
                    "std_node_sri": round(node_res["std_node_sri"], 6) if np.isfinite(node_res["std_node_sri"]) else 0.0,
                    "p10_node_sri": round(node_res["p10_node_sri"], 6) if np.isfinite(node_res["p10_node_sri"]) else "inf",
                    "mean_effective_rank": round(node_res["mean_effective_rank"], 4),
                    "mean_local_sparsity": round(node_res["mean_local_sparsity"], 6),
                })

                ds_examples.append({
                    "input": example["input"],
                    "output": example["output"],
                    "predict_baseline": baseline_str,
                    "predict_our_method": our_method_str,
                })

            except Exception:
                logger.exception(f"Failed formatting example {ex_idx} in {ds_name}")
                continue

        output["datasets"].append({
            "dataset": ds_name,
            "examples": ds_examples,
        })

    return output


@logger.catch
def main() -> None:
    logger.info("=" * 60)
    logger.info("Node-Level SRI vs Graph-Level SRI Analysis")
    logger.info("=" * 60)

    # Determine run mode from command line
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mini", "10", "50", "100", "200", "full"], default="mini")
    parser.add_argument("--no-srwe", action="store_true")
    args = parser.parse_args()

    mode = args.mode
    run_srwe = not args.no_srwe

    logger.info(f"Run mode: {mode}, SRWE: {'enabled' if run_srwe else 'disabled'}")

    # Load data
    if mode == "mini":
        datasets = load_mini_data()
        max_sample = 3
    elif mode == "10":
        datasets = load_all_data(max_per_dataset=10)
        max_sample = 10
    elif mode == "50":
        datasets = load_all_data(max_per_dataset=50)
        max_sample = 50
    elif mode == "100":
        datasets = load_all_data(max_per_dataset=100)
        max_sample = 100
    elif mode == "200":
        datasets = load_all_data(max_per_dataset=200)
        max_sample = 200
    else:  # full
        datasets = load_all_data()
        max_sample = SAMPLE_PER_DATASET

    # Run analysis
    results = run_analysis(datasets, max_sample_per_ds=max_sample, run_srwe=run_srwe)

    # Save analysis results
    analysis_path = OUTPUT_DIR / "analysis_results.json"
    # Clean non-serializable values
    def clean_for_json(obj: Any) -> Any:
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return str(obj)
            return obj
        if isinstance(obj, np.floating):
            val = float(obj)
            if math.isnan(val) or math.isinf(val):
                return str(val)
            return val
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        return obj

    analysis_clean = clean_for_json(results)
    analysis_path.write_text(json.dumps(analysis_clean, indent=2))
    logger.info(f"Saved analysis results: {analysis_path}")

    # Format output for schema
    logger.info("Formatting output for exp_gen_sol_out schema...")
    output = format_output_for_schema(datasets, results)
    output_clean = clean_for_json(output)

    output_path = OUTPUT_DIR / "method_out.json"
    output_path.write_text(json.dumps(output_clean, indent=2))
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved method_out.json ({file_size_mb:.1f} MB)")

    # Split if over 100 MB
    FILE_SIZE_LIMIT_MB = 100
    if file_size_mb > FILE_SIZE_LIMIT_MB:
        logger.info(f"Output exceeds {FILE_SIZE_LIMIT_MB} MB, splitting...")
        split_dir = OUTPUT_DIR / "method_out"
        split_dir.mkdir(exist_ok=True)

        part_num = 0
        for ds in output_clean["datasets"]:
            # Check if this dataset needs sub-splitting
            ds_json = json.dumps({"metadata": output_clean.get("metadata", {}), "datasets": [ds]}, indent=2)
            ds_size_mb = len(ds_json.encode()) / (1024 * 1024)

            if ds_size_mb <= FILE_SIZE_LIMIT_MB * 0.95:
                part_num += 1
                part_path = split_dir / f"method_out_{part_num}.json"
                part_path.write_text(ds_json)
                logger.info(f"  Part {part_num}: {ds['dataset']} ({ds_size_mb:.1f} MB)")
            else:
                # Sub-split this dataset into chunks under the limit
                examples = ds["examples"]
                n = len(examples)
                chunk_size = max(1, int(n * (FILE_SIZE_LIMIT_MB * 0.85) / ds_size_mb))
                for start in range(0, n, chunk_size):
                    chunk_ex = examples[start:start + chunk_size]
                    part_num += 1
                    chunk_ds = {"dataset": ds["dataset"], "examples": chunk_ex}
                    chunk_data = {"metadata": output_clean.get("metadata", {}), "datasets": [chunk_ds]}
                    part_path = split_dir / f"method_out_{part_num}.json"
                    part_path.write_text(json.dumps(chunk_data, indent=2))
                    chunk_mb = part_path.stat().st_size / (1024 * 1024)
                    logger.info(f"  Part {part_num}: {ds['dataset']} [{start}:{start+len(chunk_ex)}] ({chunk_mb:.1f} MB)")

        output_path.unlink()
        logger.info(f"Split into {part_num} parts in method_out/")

    logger.info("=" * 60)
    logger.info("Done!")


if __name__ == "__main__":
    main()
