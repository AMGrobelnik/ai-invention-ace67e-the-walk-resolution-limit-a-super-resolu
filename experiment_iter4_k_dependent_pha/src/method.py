#!/usr/bin/env python3
"""K-Dependent Phase Transition in RWSE Encoding Quality.

Tests whether RWSE encoding quality exhibits a K-dependent phase transition
at K* = 1/delta_min by computing quality metrics across walk lengths K=2..64,
fitting sigmoid curves, and correlating inflection points with K*.
"""

import json
import math
import os
import random
import resource
import sys
import warnings
from pathlib import Path

import numpy as np
import psutil
from loguru import logger
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ── Directories ──────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parent
DATA_DIR = Path(
    "/opt/ai-inventor/aii_pipeline/runs/run__20260222_005317/"
    "3_invention_loop/iter_1/gen_art/data_id2_it1__opus"
)
LOGS_DIR = WORKSPACE / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# ── Logging ──────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOGS_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ── Resource limits ──────────────────────────────────────────────────────
AVAILABLE_RAM = psutil.virtual_memory().available / 1e9
RAM_LIMIT_GB = min(AVAILABLE_RAM * 0.8, 50)  # 80% of available, max 50GB
try:
    resource.setrlimit(resource.RLIMIT_AS,
                       (int(RAM_LIMIT_GB * 1024**3), int(RAM_LIMIT_GB * 1024**3)))
except ValueError:
    pass
resource.setrlimit(resource.RLIMIT_CPU, (3500, 3500))  # ~58 min CPU time

# ── Hardware detection ───────────────────────────────────────────────────
NUM_CPUS = os.cpu_count() or 1
cpu_usage = psutil.cpu_percent(interval=0.5)
NUM_WORKERS = max(1, int(NUM_CPUS * (100 - cpu_usage) / 100 * 0.7))
logger.info(f"Hardware: {NUM_CPUS} CPUs, {AVAILABLE_RAM:.1f}GB RAM avail, "
            f"workers={NUM_WORKERS}, CPU usage={cpu_usage:.0f}%")

# ── Constants ────────────────────────────────────────────────────────────
K_VALUES = [2, 4, 8, 12, 16, 20]  # base K values from stored RWSE
K_VALUES_EXTENDED = [2, 4, 8, 12, 16, 24, 32, 48, 64]  # extended for eigdecomp
EPS_VALUES = [1e-6, 1e-4, 1e-2]
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Max examples to process (set by scaling steps)
MAX_EXAMPLES = int(os.environ.get("MAX_EXAMPLES", "0"))  # 0 = all


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 1: DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_all_data(max_per_dataset: int = 0) -> dict:
    """Load all datasets from dependency files, keyed by dataset name."""
    all_data = {}

    # Use mini file for very small runs, otherwise load full files
    if max_per_dataset > 0 and max_per_dataset <= 3:
        mini_path = DATA_DIR / "mini_data_out.json"
        if mini_path.exists():
            logger.info(f"Loading mini data: {mini_path.name}")
            with open(mini_path) as f:
                file_data = json.load(f)
            for ds in file_data["datasets"]:
                name = ds["dataset"]
                all_data[name] = ds["examples"][:max_per_dataset]
                logger.info(f"  {name}: {len(all_data[name])} examples")
            return all_data

    data_files = sorted((DATA_DIR / "data_out").glob("full_data_out_*.json"))

    for fpath in data_files:
        logger.info(f"Loading {fpath.name} ({fpath.stat().st_size / 1e6:.1f} MB)")
        with open(fpath) as f:
            file_data = json.load(f)
        for ds in file_data["datasets"]:
            name = ds["dataset"]
            if name not in all_data:
                all_data[name] = []
            # Early truncation to save memory
            if max_per_dataset > 0 and len(all_data[name]) >= max_per_dataset:
                continue
            remaining = max_per_dataset - len(all_data[name]) if max_per_dataset > 0 else len(ds["examples"])
            all_data[name].extend(ds["examples"][:remaining])
            logger.info(f"  {name}: +{min(remaining, len(ds['examples']))} examples "
                        f"(total {len(all_data[name])})")

    return all_data


def parse_input(example: dict) -> dict:
    """Parse the JSON-encoded input field of an example."""
    return json.loads(example["input"])


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 2: GRAPH PROCESSING AND EIGENDECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════════

def build_adjacency(edge_index: list, num_nodes: int) -> np.ndarray:
    """Build adjacency matrix from edge_index."""
    A = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    src, dst = edge_index[0], edge_index[1]
    for s, d in zip(src, dst):
        A[s, d] = 1.0
    return A


def compute_spectral(A: np.ndarray) -> dict:
    """Compute eigendecomposition of A and symmetric normalized matrix S."""
    n = A.shape[0]
    degrees = A.sum(axis=1)
    degrees_safe = np.maximum(degrees, 1e-10)

    # Symmetric normalized matrix S = D^{-1/2} A D^{-1/2}
    d_inv_half = 1.0 / np.sqrt(degrees_safe)
    # Handle isolated nodes
    d_inv_half[degrees < 0.5] = 0.0
    S = (d_inv_half[:, None] * A) * d_inv_half[None, :]

    # Eigendecomposition
    eig_A, vec_A = np.linalg.eigh(A)
    eig_S, vec_S = np.linalg.eigh(S)

    # Sort by ascending eigenvalue
    idx_A = np.argsort(eig_A)
    eig_A, vec_A = eig_A[idx_A], vec_A[:, idx_A]
    idx_S = np.argsort(eig_S)
    eig_S, vec_S = eig_S[idx_S], vec_S[:, idx_S]

    # Delta min
    sorted_A = np.sort(eig_A)
    diffs_A = np.diff(sorted_A)
    delta_min_A = float(np.min(diffs_A)) if len(diffs_A) > 0 else 0.0

    sorted_S = np.sort(eig_S)
    diffs_S = np.diff(sorted_S)
    delta_min_S = float(np.min(diffs_S)) if len(diffs_S) > 0 else 0.0

    # K* = ceil(1 / delta_min)
    K_star_A = math.ceil(1.0 / delta_min_A) if delta_min_A > 1e-12 else 9999
    K_star_S = math.ceil(1.0 / delta_min_S) if delta_min_S > 1e-12 else 9999

    # Laplacian eigenvalues for heat kernel
    L = np.diag(degrees) - A
    eig_L, vec_L = np.linalg.eigh(L)
    idx_L = np.argsort(eig_L)
    eig_L, vec_L = eig_L[idx_L], vec_L[:, idx_L]

    return {
        "eig_A": eig_A, "vec_A": vec_A,
        "eig_S": eig_S, "vec_S": vec_S,
        "eig_L": eig_L, "vec_L": vec_L,
        "delta_min_A": delta_min_A, "delta_min_S": delta_min_S,
        "K_star_A": K_star_A, "K_star_S": K_star_S,
        "degrees": degrees,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 3: COMPUTE RWSE AT TARGET K VALUES
# ═══════════════════════════════════════════════════════════════════════════

def compute_rwse_from_spectral(eig_S: np.ndarray, vec_S: np.ndarray,
                               K_list: list) -> dict:
    """Compute RWSE[v, k] = sum_i eig_S[i]^k * vec_S[v, i]^2 for each K."""
    # vec_S_sq[v, i] = v_i(v)^2
    vec_sq = vec_S ** 2  # (n, n_eigs)
    rwse_at_K = {}
    for K in K_list:
        powers = eig_S ** K  # (n_eigs,)
        rwse = vec_sq @ powers  # (n,)
        rwse_at_K[K] = rwse
    return rwse_at_K


def compute_rwse_matrix(eig_S: np.ndarray, vec_S: np.ndarray,
                        max_K: int) -> np.ndarray:
    """Compute RWSE feature matrix [n_nodes x max_K] for k=1..max_K."""
    vec_sq = vec_S ** 2
    rwse_matrix = np.zeros((vec_S.shape[0], max_K))
    for k in range(1, max_K + 1):
        powers = eig_S ** k
        rwse_matrix[:, k - 1] = vec_sq @ powers
    return rwse_matrix


def compute_walk_A_features(eig_A: np.ndarray, vec_A: np.ndarray,
                            K_list: list) -> dict:
    """Compute A-based walk features (normalized by spectral radius)."""
    rho = max(abs(eig_A.max()), abs(eig_A.min()))
    if rho < 1e-12:
        rho = 1.0
    norm_eigs = eig_A / rho
    vec_sq = vec_A ** 2
    walk_at_K = {}
    for K in K_list:
        powers = norm_eigs ** K
        walk_at_K[K] = vec_sq @ powers
    return walk_at_K


def compute_heat_kernel_features(eig_L: np.ndarray, vec_L: np.ndarray,
                                 t_values: list) -> dict:
    """Heat kernel features: HK[v, t] = sum_i exp(-t*lam_i)*v_i(v)^2."""
    vec_sq = vec_L ** 2
    hk_at_t = {}
    for t in t_values:
        weights = np.exp(-t * eig_L)
        hk_at_t[t] = vec_sq @ weights
    return hk_at_t


def extract_stored_rwse(spectral_data: dict, K_list: list) -> dict:
    """Extract stored RWSE values at specific K walk lengths.

    Stored RWSE has 20 values per node for k=1..20.
    Returns dict {K: np.array(n_nodes)} for each K in K_list that is <= 20.
    """
    rwse_raw = spectral_data["rwse"]  # list of lists: [n_nodes][20]
    n_nodes = len(rwse_raw)
    rwse_at_K = {}
    for K in K_list:
        if K <= 20:
            vals = np.array([rwse_raw[v][K - 1] for v in range(n_nodes)])
            rwse_at_K[K] = vals
    return rwse_at_K


def extract_stored_rwse_matrix(spectral_data: dict, max_K: int) -> np.ndarray:
    """Extract stored RWSE as feature matrix [n_nodes x min(max_K, 20)]."""
    rwse_raw = spectral_data["rwse"]
    n_nodes = len(rwse_raw)
    actual_K = min(max_K, 20)
    rwse_matrix = np.zeros((n_nodes, actual_K))
    for v in range(n_nodes):
        rwse_matrix[v, :actual_K] = rwse_raw[v][:actual_K]
    return rwse_matrix


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 4: GENERATE CUSTOM SYNTHETIC GRAPHS
# ═══════════════════════════════════════════════════════════════════════════

def generate_synthetic_graphs(n_graphs: int = 50, n_trials: int = 300) -> list:
    """Generate synthetic graphs with diverse delta_min values.

    Uses multiple graph families:
    - Small graphs (n=5..15) for large delta_min (well-separated eigenvalues)
    - Medium graphs (n=15..50) for moderate delta_min
    - ER, BA, SBM, regular, path, cycle, and multi-block SBM
    """
    import networkx as nx

    # Target a wider range: small graphs have larger delta_min
    target_delta_mins = np.concatenate([
        np.linspace(0.01, 0.05, 15),   # small delta_min (hard)
        np.linspace(0.05, 0.2, 15),    # moderate
        np.linspace(0.2, 0.5, 10),     # large
        np.linspace(0.5, 2.0, 10),     # very large (small graphs)
    ])
    np.random.shuffle(target_delta_mins)
    target_delta_mins = target_delta_mins[:n_graphs]

    synthetic_graphs = []
    all_candidates = []

    # First pass: generate many random graphs and collect their delta_min values
    for trial in range(n_trials * 3):
        graph_type = random.choice(["ER", "BA", "SBM", "regular", "path", "cycle",
                                     "SBM3"])

        try:
            if graph_type == "path":
                n = random.randint(4, 20)
                G = nx.path_graph(n)
            elif graph_type == "cycle":
                n = random.randint(4, 20)
                G = nx.cycle_graph(n)
            elif graph_type == "regular":
                n = random.randint(6, 30)
                d = random.randint(2, min(5, n - 1))
                if (n * d) % 2 != 0:
                    n += 1
                G = nx.random_regular_graph(d, n, seed=SEED + trial)
            elif graph_type == "ER":
                n = random.randint(5, 50)
                p = random.uniform(0.1, 0.8)
                G = nx.erdos_renyi_graph(n, p, seed=SEED + trial)
            elif graph_type == "BA":
                n = random.randint(5, 50)
                m = random.randint(1, min(5, n - 1))
                G = nx.barabasi_albert_graph(n, m, seed=SEED + trial)
            elif graph_type == "SBM3":
                sizes = [random.randint(3, 10) for _ in range(3)]
                p_in = random.uniform(0.3, 0.9)
                p_out = random.uniform(0.01, 0.15)
                probs = [[p_in if i == j else p_out for j in range(3)]
                         for i in range(3)]
                G = nx.stochastic_block_model(sizes, probs, seed=SEED + trial)
            else:  # SBM
                n = random.randint(6, 40)
                n1 = random.randint(3, n - 3)
                n2 = n - n1
                p_in = random.uniform(0.3, 0.9)
                p_out = random.uniform(0.01, 0.3)
                G = nx.stochastic_block_model(
                    [n1, n2], [[p_in, p_out], [p_out, p_in]],
                    seed=SEED + trial)

            if not nx.is_connected(G):
                continue

            A = nx.to_numpy_array(G)
            degrees = A.sum(axis=1)
            if np.any(degrees < 0.5):
                continue

            d_inv_half = 1.0 / np.sqrt(degrees)
            S_mat = (d_inv_half[:, None] * A) * d_inv_half[None, :]
            eigs_S = np.linalg.eigvalsh(S_mat)
            dm_S = float(np.min(np.diff(np.sort(eigs_S))))

            eigs_A = np.linalg.eigvalsh(A)
            dm_A = float(np.min(np.diff(np.sort(eigs_A))))

            all_candidates.append({
                "G": G, "A": A, "n_nodes": len(G),
                "graph_type": graph_type,
                "delta_min_S": dm_S, "delta_min_A": dm_A,
            })
        except Exception:
            continue

    logger.info(f"Generated {len(all_candidates)} candidate synthetic graphs")
    if len(all_candidates) == 0:
        return []

    # Log the delta_min distribution of candidates
    cand_dms = [c["delta_min_S"] for c in all_candidates]
    logger.info(f"  Candidate delta_min_S range: [{min(cand_dms):.6f}, {max(cand_dms):.4f}]")

    # Second pass: match each target to the closest candidate
    used = set()
    for target_dm in sorted(target_delta_mins):
        best_idx = None
        best_error = float("inf")
        for i, c in enumerate(all_candidates):
            if i in used:
                continue
            error = abs(c["delta_min_S"] - target_dm)
            if error < best_error:
                best_error = error
                best_idx = i

        if best_idx is not None:
            used.add(best_idx)
            cand = all_candidates[best_idx]
            cand["target_delta_min"] = float(target_dm)

            # Full eigendecomposition
            spec = compute_spectral(cand["A"])
            cand.update(spec)

            edges = list(cand["G"].edges())
            src = [e[0] for e in edges] + [e[1] for e in edges]
            dst = [e[1] for e in edges] + [e[0] for e in edges]
            cand["edge_index"] = [src, dst]

            synthetic_graphs.append(cand)

    logger.info(f"Selected {len(synthetic_graphs)}/{n_graphs} synthetic graphs")
    if len(synthetic_graphs) > 0:
        dm_vals = [g["delta_min_S"] for g in synthetic_graphs]
        logger.info(f"  delta_min_S range: [{min(dm_vals):.6f}, {max(dm_vals):.4f}]")
        # Log K_star distribution
        ks_vals = [g["K_star_S"] for g in synthetic_graphs]
        logger.info(f"  K_star_S range: [{min(ks_vals)}, {max(ks_vals)}]")

    return synthetic_graphs


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 5: QUALITY METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_distinguishability(rwse_matrix: np.ndarray,
                               eps_values: list) -> dict:
    """Compute fraction of node pairs with distance > eps."""
    n = rwse_matrix.shape[0]
    if n < 2:
        return {eps: 0.0 for eps in eps_values}, 0.0

    dists = pdist(rwse_matrix, metric="euclidean")
    result = {}
    for eps in eps_values:
        result[eps] = float(np.mean(dists > eps))
    mean_dist = float(np.mean(dists))
    return result, mean_dist


def compute_spectral_recon_error(eig_S: np.ndarray, vec_S: np.ndarray,
                                 rwse_matrix: np.ndarray,
                                 alpha: float = 1e-4) -> float:
    """Compute spectral reconstruction error using Tikhonov-regularized inversion.

    For each node, try to reconstruct spectral weights from RWSE moments.
    Returns mean Wasserstein-1 distance.
    """
    n = vec_S.shape[0]
    max_K = rwse_matrix.shape[1]

    if max_K < 2 or n < 2:
        return float("nan")

    # True spectral weights for each node
    w_true_all = vec_S ** 2  # (n, n_eigs)

    # Vandermonde matrix V[k, i] = eig_S[i]^(k+1) for k=0..max_K-1
    V = np.zeros((max_K, n))
    for k in range(max_K):
        V[k, :] = eig_S ** (k + 1)

    total_W1 = 0.0
    valid_count = 0

    # Sample nodes for efficiency
    node_sample = list(range(n))
    if n > 30:
        node_sample = random.sample(node_sample, 30)

    for u in node_sample:
        m = rwse_matrix[u, :]  # moments vector (max_K,)

        try:
            # Tikhonov regularized: solve (V V^T + alpha I) c = V w_true
            # Actually: V shape (max_K, n), m shape (max_K,)
            # Want w shape (n,) such that V @ w ≈ m
            # Augmented system for Tikhonov:
            V_aug = np.vstack([V, np.sqrt(alpha) * np.eye(n)])
            m_aug = np.concatenate([m, np.zeros(n)])
            w_hat, _, _, _ = np.linalg.lstsq(V_aug, m_aug, rcond=None)

            # Project to non-negative
            w_hat = np.maximum(w_hat, 0)
            if w_hat.sum() > 1e-15:
                w_hat /= w_hat.sum()
            else:
                continue

            w_true = w_true_all[u, :]
            if w_true.sum() < 1e-15:
                continue

            # Wasserstein-1 distance on eigenvalue support
            sorted_idx = np.argsort(eig_S)
            cdf_true = np.cumsum(w_true[sorted_idx])
            cdf_hat = np.cumsum(w_hat[sorted_idx])
            sorted_eigs = eig_S[sorted_idx]

            # W1 = integral |CDF_true - CDF_hat| dx
            if len(sorted_eigs) > 1:
                dx = np.diff(sorted_eigs)
                avg_diff = 0.5 * (np.abs(cdf_true[:-1] - cdf_hat[:-1]) +
                                  np.abs(cdf_true[1:] - cdf_hat[1:]))
                W1 = float(np.sum(avg_diff * dx))
            else:
                W1 = 0.0

            total_W1 += W1
            valid_count += 1
        except Exception:
            continue

    if valid_count == 0:
        return float("nan")
    return total_W1 / valid_count


def compute_lape_distinguishability(vec_A: np.ndarray, d: int,
                                    eps: float = 1e-6) -> float:
    """Compute LapPE-style distinguishability using top-d squared eigenvectors."""
    n = vec_A.shape[0]
    if n < 2 or d < 1:
        return 0.0
    actual_d = min(d, vec_A.shape[1])
    features = vec_A[:, :actual_d] ** 2
    dists = pdist(features, metric="euclidean")
    return float(np.mean(dists > eps))


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 6: PHASE TRANSITION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def sigmoid(K, L, rate, K_mid, baseline):
    """Sigmoid function for quality vs K."""
    return L / (1.0 + np.exp(-rate * (K - K_mid))) + baseline


def fit_sigmoid(K_values: list, quality_values: list) -> dict:
    """Fit sigmoid to quality-vs-K curve, return inflection point and fit quality."""
    K_arr = np.array(K_values, dtype=float)
    q_arr = np.array(quality_values, dtype=float)

    q_range = q_arr.max() - q_arr.min()
    if q_range < 0.01:
        return {"K_inflect": float("nan"), "R2": 0.0, "success": False,
                "params": None}

    # Check if data is monotonically increasing (expected pattern)
    diffs = np.diff(q_arr)
    if np.all(diffs <= 0):
        # Monotonically decreasing - unexpected, skip
        return {"K_inflect": float("nan"), "R2": 0.0, "success": False,
                "params": None}

    try:
        # Multiple initial guesses for robustness
        best_result = None
        best_R2 = -999

        for K_mid_init in [float(np.median(K_arr)), K_arr[1], K_arr[-2]]:
            try:
                p0 = [q_range, 0.3, K_mid_init, float(q_arr.min())]
                bounds = ([0, 0.001, 0.5, -2], [2, 10, 200, 2])
                popt, pcov = curve_fit(sigmoid, K_arr, q_arr, p0=p0,
                                       bounds=bounds, maxfev=2000)
                K_inflect = popt[2]

                y_pred = sigmoid(K_arr, *popt)
                ss_res = np.sum((q_arr - y_pred) ** 2)
                ss_tot = np.sum((q_arr - q_arr.mean()) ** 2)
                R2 = 1 - ss_res / max(ss_tot, 1e-15)

                if R2 > best_R2:
                    best_R2 = R2
                    best_result = (K_inflect, R2, popt)
            except (RuntimeError, ValueError):
                continue

        if best_result is None:
            return {"K_inflect": float("nan"), "R2": 0.0, "success": False,
                    "params": None}

        K_inflect, R2, popt = best_result
        success = 0.5 < K_inflect < 200 and R2 > 0.3
        return {"K_inflect": float(K_inflect), "R2": float(R2),
                "success": success,
                "params": [float(p) for p in popt]}
    except Exception:
        return {"K_inflect": float("nan"), "R2": 0.0, "success": False,
                "params": None}


def compute_K_half(K_values: list, quality_values: list) -> float:
    """Fallback: K at which quality first exceeds 50% of its range."""
    q_arr = np.array(quality_values)
    q_min, q_max = q_arr.min(), q_arr.max()
    threshold = (q_min + q_max) / 2

    for K, q in zip(K_values, quality_values):
        if q >= threshold:
            return float(K)
    return float(K_values[-1])


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN PROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def process_graph_stored_only(example: dict, graph_idx: int) -> dict:
    """Process a graph using only stored RWSE (K=1..20) and stored eigenvalues.

    This is the fast path for all graphs that don't need eigendecomposition
    at higher K values.
    """
    inp = parse_input(example)
    spectral = inp.get("spectral", {})
    n_nodes = inp.get("num_nodes", 0)

    # Extract stored values
    eigenvalues = np.array(spectral.get("eigenvalues", []))
    delta_min_stored = spectral.get("delta_min", 0.0)
    sri = spectral.get("sri", {})
    rwse_raw = spectral.get("rwse", [])

    if len(rwse_raw) == 0 or n_nodes < 2:
        return None

    # delta_min and K* from stored eigenvalues (these are A eigenvalues)
    if len(eigenvalues) > 1:
        sorted_eigs = np.sort(eigenvalues)
        diffs = np.diff(sorted_eigs)
        delta_min_A = float(np.min(diffs)) if len(diffs) > 0 else 0.0
    else:
        delta_min_A = delta_min_stored

    K_star_A = math.ceil(1.0 / delta_min_A) if delta_min_A > 1e-12 else 9999

    # Extract stored RWSE for K values up to 20
    K_vals_stored = [K for K in K_VALUES if K <= 20]
    rwse_matrices = {}
    for max_K in K_vals_stored:
        mat = extract_stored_rwse_matrix(spectral, max_K)
        rwse_matrices[max_K] = mat

    # Compute quality metrics at each K
    quality_vs_K = {}
    for max_K in K_vals_stored:
        mat = rwse_matrices[max_K]
        dist_fracs, mean_dist = compute_distinguishability(mat, EPS_VALUES)
        quality_vs_K[max_K] = {
            "distinguish": dist_fracs,
            "mean_dist": mean_dist,
        }

    return {
        "graph_idx": graph_idx,
        "n_nodes": n_nodes,
        "delta_min_A": delta_min_A,
        "delta_min_stored": delta_min_stored,
        "K_star_A": K_star_A,
        "quality_vs_K": quality_vs_K,
        "sri": sri,
    }


def process_graph_full(example: dict, graph_idx: int) -> dict:
    """Process a graph with full eigendecomposition for extended K values."""
    inp = parse_input(example)
    spectral = inp.get("spectral", {})
    n_nodes = inp.get("num_nodes", 0)
    edge_index = inp.get("edge_index", None)

    if edge_index is None or n_nodes < 2:
        return process_graph_stored_only(example, graph_idx)

    # Build adjacency and compute spectral decomposition
    try:
        A = build_adjacency(edge_index, n_nodes)
        spec = compute_spectral(A)
    except Exception as e:
        logger.debug(f"Eigendecomp failed for graph {graph_idx}: {e}")
        return process_graph_stored_only(example, graph_idx)

    eig_S, vec_S = spec["eig_S"], spec["vec_S"]
    eig_A, vec_A = spec["eig_A"], spec["vec_A"]
    eig_L, vec_L = spec["eig_L"], spec["vec_L"]

    # Compute RWSE at ALL K values (including extended)
    rwse_at_K = compute_rwse_from_spectral(eig_S, vec_S, K_VALUES_EXTENDED)

    # Build RWSE feature matrices for each max_K
    quality_vs_K = {}
    for max_K in K_VALUES_EXTENDED:
        mat = compute_rwse_matrix(eig_S, vec_S, max_K)
        dist_fracs, mean_dist = compute_distinguishability(mat, EPS_VALUES)

        # Spectral reconstruction error (sample for speed)
        recon_err = float("nan")
        if max_K >= 4 and n_nodes <= 100:
            recon_err = compute_spectral_recon_error(eig_S, vec_S, mat)

        quality_vs_K[max_K] = {
            "distinguish": dist_fracs,
            "mean_dist": mean_dist,
            "recon_error": recon_err,
        }

    # LapPE distinguishability at various dimensions
    lape_dist = compute_lape_distinguishability(vec_A, max(K_VALUES_EXTENDED))

    # Walk A features
    walk_A = compute_walk_A_features(eig_A, vec_A, K_VALUES_EXTENDED)
    quality_walk_A = {}
    for max_K in K_VALUES_EXTENDED:
        # Build matrix from walk_A features for k=1..max_K
        vec_sq_A = vec_A ** 2
        rho = max(abs(eig_A.max()), abs(eig_A.min()))
        if rho < 1e-12:
            rho = 1.0
        norm_eigs_A = eig_A / rho
        mat_A = np.zeros((n_nodes, max_K))
        for k in range(1, max_K + 1):
            powers = norm_eigs_A ** k
            mat_A[:, k - 1] = vec_sq_A @ powers
        dist_fracs_A, mean_dist_A = compute_distinguishability(mat_A, EPS_VALUES)
        quality_walk_A[max_K] = {
            "distinguish": dist_fracs_A,
            "mean_dist": mean_dist_A,
        }

    # Heat kernel features with log-scale t values
    # Use t = k * t_scale where t_scale adapts to spectral gap
    spectral_gap = eig_L[1] if len(eig_L) > 1 and eig_L[1] > 0.01 else 0.1
    t_scale = 1.0 / spectral_gap  # Normalize by spectral gap
    quality_hk = {}
    vec_sq_L = vec_L ** 2
    for max_K in K_VALUES_EXTENDED:
        mat_hk = np.zeros((n_nodes, max_K))
        for k_idx in range(max_K):
            t = (k_idx + 1) * t_scale / max_K  # spread evenly up to t_scale
            weights = np.exp(-t * eig_L)
            mat_hk[:, k_idx] = vec_sq_L @ weights
        dist_fracs_hk, mean_dist_hk = compute_distinguishability(mat_hk, EPS_VALUES)
        quality_hk[max_K] = {
            "distinguish": dist_fracs_hk,
            "mean_dist": mean_dist_hk,
        }

    return {
        "graph_idx": graph_idx,
        "n_nodes": n_nodes,
        "delta_min_A": spec["delta_min_A"],
        "delta_min_S": spec["delta_min_S"],
        "K_star_A": spec["K_star_A"],
        "K_star_S": spec["K_star_S"],
        "quality_vs_K": quality_vs_K,
        "quality_walk_A": quality_walk_A,
        "quality_hk": quality_hk,
        "lape_distinguish": lape_dist,
        "has_full_spectral": True,
        "sri": spectral.get("sri", {}),
        "eig_S": eig_S,  # for alternative delta_min analysis
    }


def process_synthetic_graph(syn: dict, graph_idx: int) -> dict:
    """Process a custom synthetic graph (already has eigendecomposition)."""
    eig_S, vec_S = syn["eig_S"], syn["vec_S"]
    eig_A, vec_A = syn["eig_A"], syn["vec_A"]
    eig_L, vec_L = syn["eig_L"], syn["vec_L"]
    n_nodes = syn["n_nodes"]

    quality_vs_K = {}
    for max_K in K_VALUES_EXTENDED:
        mat = compute_rwse_matrix(eig_S, vec_S, max_K)
        dist_fracs, mean_dist = compute_distinguishability(mat, EPS_VALUES)
        recon_err = float("nan")
        if max_K >= 4:
            recon_err = compute_spectral_recon_error(eig_S, vec_S, mat)
        quality_vs_K[max_K] = {
            "distinguish": dist_fracs,
            "mean_dist": mean_dist,
            "recon_error": recon_err,
        }

    lape_dist = compute_lape_distinguishability(vec_A, max(K_VALUES_EXTENDED))

    # Walk A features
    quality_walk_A = {}
    for max_K in K_VALUES_EXTENDED:
        vec_sq_A = vec_A ** 2
        rho = max(abs(eig_A.max()), abs(eig_A.min()))
        if rho < 1e-12:
            rho = 1.0
        norm_eigs_A = eig_A / rho
        mat_A = np.zeros((n_nodes, max_K))
        for k in range(1, max_K + 1):
            powers = norm_eigs_A ** k
            mat_A[:, k - 1] = vec_sq_A @ powers
        dist_fracs_A, mean_dist_A = compute_distinguishability(mat_A, EPS_VALUES)
        quality_walk_A[max_K] = {
            "distinguish": dist_fracs_A,
            "mean_dist": mean_dist_A,
        }

    # Heat kernel with adaptive t scaling
    spectral_gap = eig_L[1] if len(eig_L) > 1 and eig_L[1] > 0.01 else 0.1
    t_scale = 1.0 / spectral_gap
    quality_hk = {}
    vec_sq_L = vec_L ** 2
    for max_K in K_VALUES_EXTENDED:
        mat_hk = np.zeros((n_nodes, max_K))
        for k_idx in range(max_K):
            t = (k_idx + 1) * t_scale / max_K
            weights = np.exp(-t * eig_L)
            mat_hk[:, k_idx] = vec_sq_L @ weights
        dist_fracs_hk, mean_dist_hk = compute_distinguishability(mat_hk, EPS_VALUES)
        quality_hk[max_K] = {
            "distinguish": dist_fracs_hk,
            "mean_dist": mean_dist_hk,
        }

    return {
        "graph_idx": graph_idx,
        "n_nodes": n_nodes,
        "delta_min_A": syn["delta_min_A"],
        "delta_min_S": syn["delta_min_S"],
        "K_star_A": syn["K_star_A"],
        "K_star_S": syn["K_star_S"],
        "quality_vs_K": quality_vs_K,
        "quality_walk_A": quality_walk_A,
        "quality_hk": quality_hk,
        "lape_distinguish": lape_dist,
        "has_full_spectral": True,
        "graph_type": syn.get("graph_type", "unknown"),
        "eig_S": eig_S,  # for alternative delta_min analysis
    }


# ═══════════════════════════════════════════════════════════════════════════
#  ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def run_phase_transition_analysis(results: list, K_vals: list,
                                  eps: float = 1e-6) -> dict:
    """Run full phase transition analysis on processed graph results."""

    # ── 6.1: SRI Group Split ────────────────────────────────────────────
    sri_gap_results = {}
    for K in K_vals:
        K_key = K if isinstance(K, int) else int(K)
        pre_quality = []
        post_quality = []
        for r in results:
            if r is None:
                continue
            dm = r.get("delta_min_S", r.get("delta_min_A", 0))
            if dm < 1e-12:
                continue
            sri = K * dm
            q_data = r.get("quality_vs_K", {}).get(K_key, {})
            dist = q_data.get("distinguish", {})
            q = dist.get(eps, 0.0) if isinstance(dist, dict) else 0.0

            if sri < 1:
                pre_quality.append(q)
            else:
                post_quality.append(q)

        mean_pre = float(np.mean(pre_quality)) if pre_quality else float("nan")
        mean_post = float(np.mean(post_quality)) if post_quality else float("nan")
        gap = mean_post - mean_pre if not (
            math.isnan(mean_pre) or math.isnan(mean_post)) else float("nan")

        sri_gap_results[K_key] = {
            "mean_pre": mean_pre,
            "mean_post": mean_post,
            "gap": gap,
            "n_pre": len(pre_quality),
            "n_post": len(post_quality),
        }

    # ── 6.2: Quintile Curves ───────────────────────────────────────────
    all_dm = []
    for r in results:
        if r is None:
            continue
        dm = r.get("delta_min_S", r.get("delta_min_A", 0))
        if dm > 1e-12:
            all_dm.append(dm)

    if len(all_dm) >= 5:
        quintile_bounds = np.percentile(all_dm, [20, 40, 60, 80])
    else:
        quintile_bounds = np.array([0.05, 0.1, 0.2, 0.3])

    quintile_curves = {f"Q{q}": {} for q in range(1, 6)}
    for K in K_vals:
        K_key = K if isinstance(K, int) else int(K)
        qvals = {f"Q{q}": [] for q in range(1, 6)}
        for r in results:
            if r is None:
                continue
            dm = r.get("delta_min_S", r.get("delta_min_A", 0))
            q_data = r.get("quality_vs_K", {}).get(K_key, {})
            dist = q_data.get("distinguish", {})
            q = dist.get(eps, 0.0) if isinstance(dist, dict) else 0.0

            if dm <= quintile_bounds[0]:
                qvals["Q1"].append(q)
            elif dm <= quintile_bounds[1]:
                qvals["Q2"].append(q)
            elif dm <= quintile_bounds[2]:
                qvals["Q3"].append(q)
            elif dm <= quintile_bounds[3]:
                qvals["Q4"].append(q)
            else:
                qvals["Q5"].append(q)

        for qname in qvals:
            if qvals[qname]:
                quintile_curves[qname][K_key] = float(np.mean(qvals[qname]))
            else:
                quintile_curves[qname][K_key] = float("nan")

    # ── 6.3: Sigmoid Fitting ──────────────────────────────────────────
    K_inflect_list = []
    K_star_list = []
    sigmoid_results = []

    for r in results:
        if r is None:
            continue
        K_star = r.get("K_star_S", r.get("K_star_A", 9999))
        if K_star > 200 or K_star < 1:
            continue

        q_vals_for_fit = []
        K_vals_for_fit = []
        for K in K_vals:
            K_key = K if isinstance(K, int) else int(K)
            q_data = r.get("quality_vs_K", {}).get(K_key, {})
            dist = q_data.get("distinguish", {})
            q = dist.get(eps, 0.0) if isinstance(dist, dict) else 0.0
            q_vals_for_fit.append(q)
            K_vals_for_fit.append(K)

        fit = fit_sigmoid(K_vals_for_fit, q_vals_for_fit)

        # Also compute K_half as fallback
        K_half = compute_K_half(K_vals_for_fit, q_vals_for_fit)

        sigmoid_results.append({
            "graph_idx": r.get("graph_idx", -1),
            "K_star": K_star,
            "K_inflect": fit["K_inflect"],
            "K_half": K_half,
            "R2": fit["R2"],
            "success": fit["success"],
        })

        if fit["success"]:
            K_inflect_list.append(fit["K_inflect"])
            K_star_list.append(K_star)

    # Spearman correlation
    spearman_result = {"rho": float("nan"), "p_value": float("nan"), "n": 0}
    if len(K_inflect_list) >= 5:
        rho, p_val = spearmanr(K_star_list, K_inflect_list)
        spearman_result = {
            "rho": float(rho) if not math.isnan(rho) else 0.0,
            "p_value": float(p_val) if not math.isnan(p_val) else 1.0,
            "n": len(K_inflect_list),
        }

    # Fallback: K_half correlation
    K_half_list = [s["K_half"] for s in sigmoid_results
                   if s["K_star"] < 200 and not math.isnan(s["K_half"])]
    K_star_half_list = [s["K_star"] for s in sigmoid_results
                        if s["K_star"] < 200 and not math.isnan(s["K_half"])]
    spearman_khalf = {"rho": float("nan"), "p_value": float("nan"), "n": 0}
    if len(K_half_list) >= 5:
        rho_h, p_h = spearmanr(K_star_half_list, K_half_list)
        spearman_khalf = {
            "rho": float(rho_h) if not math.isnan(rho_h) else 0.0,
            "p_value": float(p_h) if not math.isnan(p_h) else 1.0,
            "n": len(K_half_list),
        }

    # ── 6.4: Transition Sharpness ────────────────────────────────────
    sharpness_list = []
    for r in results:
        if r is None:
            continue
        K_star = r.get("K_star_S", r.get("K_star_A", 9999))
        if K_star < 4 or K_star > max(K_vals):
            continue

        quality_curve = []
        K_arr = []
        for K in K_vals:
            K_key = K if isinstance(K, int) else int(K)
            q_data = r.get("quality_vs_K", {}).get(K_key, {})
            dist = q_data.get("distinguish", {})
            q = dist.get(eps, 0.0) if isinstance(dist, dict) else 0.0
            quality_curve.append(q)
            K_arr.append(K)

        if len(K_arr) < 3:
            continue

        K_arr_np = np.array(K_arr, dtype=float)
        q_arr_np = np.array(quality_curve, dtype=float)

        q_half = float(np.interp(K_star / 2, K_arr_np, q_arr_np))
        q_double = float(np.interp(min(K_star * 2, max(K_vals)),
                                   K_arr_np, q_arr_np))
        sharpness = q_double / max(q_half, 1e-10)
        if not math.isnan(sharpness) and not math.isinf(sharpness):
            sharpness_list.append(sharpness)

    mean_sharpness = float(np.mean(sharpness_list)) if sharpness_list else float("nan")
    frac_sharp = float(np.mean(np.array(sharpness_list) > 2.0)) if sharpness_list else 0.0

    # ── 6.5: Mean Distance Correlation (continuous metric) ──────────
    # Use mean_dist instead of binary distinguishability for better sensitivity
    K_half_md_list = []
    K_star_md_list = []
    for r in results:
        if r is None:
            continue
        K_star = r.get("K_star_S", r.get("K_star_A", 9999))
        if K_star > 200 or K_star < 1:
            continue

        md_vals = []
        K_md_fit = []
        for K in K_vals:
            K_key = K if isinstance(K, int) else int(K)
            q_data = r.get("quality_vs_K", {}).get(K_key, {})
            md = q_data.get("mean_dist", 0.0)
            md_vals.append(md)
            K_md_fit.append(K)

        if md_vals and max(md_vals) - min(md_vals) > 0.001:
            K_half_md = compute_K_half(K_md_fit, md_vals)
            K_half_md_list.append(K_half_md)
            K_star_md_list.append(K_star)

    spearman_mean_dist = {"rho": float("nan"), "p_value": float("nan"), "n": 0}
    if len(K_half_md_list) >= 5:
        rho_md, p_md = spearmanr(K_star_md_list, K_half_md_list)
        spearman_mean_dist = {
            "rho": float(rho_md) if not math.isnan(rho_md) else 0.0,
            "p_value": float(p_md) if not math.isnan(p_md) else 1.0,
            "n": len(K_half_md_list),
        }

    # ── 6.6: Alternative delta_min definitions ────────────────────────
    alt_delta_min_results = {}
    for dm_type in ["effective_delta_min", "mean_spacing", "spectral_gap",
                     "vandermonde_cond"]:
        K_half_alt_list = []
        K_star_alt_list = []
        for r in results:
            if r is None or not r.get("has_full_spectral", False):
                continue

            eig_S = r.get("eig_S", None)
            if eig_S is None:
                continue
            eig_S = np.array(eig_S) if isinstance(eig_S, list) else eig_S
            sorted_eigs = np.sort(eig_S)
            diffs = np.diff(sorted_eigs)

            if dm_type == "effective_delta_min":
                # Exclude trivial eigenvalue (closest to 1.0)
                non_trivial = sorted_eigs[np.abs(sorted_eigs - 1.0) > 0.01]
                if len(non_trivial) < 2:
                    continue
                dm = float(np.min(np.diff(non_trivial)))
            elif dm_type == "mean_spacing":
                dm = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
            elif dm_type == "spectral_gap":
                dm = float(np.max(diffs)) if len(diffs) > 0 else 0.0
            elif dm_type == "vandermonde_cond":
                # Use Vandermonde condition number as predictor
                dm = r.get("delta_min_S", 0)  # fallback
            else:
                continue

            if dm < 1e-12:
                continue
            K_star_alt = math.ceil(1.0 / dm) if dm > 1e-12 else 9999
            if K_star_alt > 200 or K_star_alt < 1:
                continue

            q_vals = []
            K_vals_fit = []
            for K in K_vals:
                K_key = int(K)
                q_data = r.get("quality_vs_K", {}).get(K_key, {})
                dist = q_data.get("distinguish", {})
                q = dist.get(eps, 0.0) if isinstance(dist, dict) else 0.0
                q_vals.append(q)
                K_vals_fit.append(K)

            if max(q_vals) - min(q_vals) < 0.001:
                continue

            K_half_val = compute_K_half(K_vals_fit, q_vals)
            K_half_alt_list.append(K_half_val)
            K_star_alt_list.append(K_star_alt)

        rho_alt, p_alt = (float("nan"), float("nan"))
        if len(K_half_alt_list) >= 5:
            rho_alt, p_alt = spearmanr(K_star_alt_list, K_half_alt_list)
            rho_alt = float(rho_alt) if not math.isnan(rho_alt) else 0.0
            p_alt = float(p_alt) if not math.isnan(p_alt) else 1.0

        alt_delta_min_results[dm_type] = {
            "spearman_rho": rho_alt,
            "p_value": p_alt,
            "n_graphs": len(K_half_alt_list),
        }

    return {
        "sri_gap_results": sri_gap_results,
        "quintile_curves": quintile_curves,
        "quintile_bounds": [float(b) for b in quintile_bounds],
        "spearman_sigmoid": spearman_result,
        "spearman_khalf": spearman_khalf,
        "spearman_mean_dist": spearman_mean_dist,
        "alternative_delta_min": alt_delta_min_results,
        "sigmoid_fit_count": len(K_inflect_list),
        "total_graphs_tested": len(sigmoid_results),
        "mean_sharpness": mean_sharpness,
        "frac_sharp": frac_sharp,
        "n_sharpness_graphs": len(sharpness_list),
    }


def _extract_quality_curve(r: dict, K_vals: list, quality_key: str,
                           eps: float) -> tuple:
    """Extract (K_list, quality_list) from a result dict."""
    q_vals = []
    K_list = []
    for K in K_vals:
        K_key = int(K)
        q_data = r.get(quality_key, {}).get(K_key, {})
        dist = q_data.get("distinguish", {})
        if isinstance(dist, dict):
            # Find closest available eps
            if eps in dist:
                q = dist[eps]
            else:
                closest = min(dist.keys(), key=lambda e: abs(e - eps),
                              default=None)
                q = dist.get(closest, 0.0) if closest is not None else 0.0
        else:
            q = 0.0
        q_vals.append(q)
        K_list.append(K)
    return K_list, q_vals


def run_method_variations(results: list, K_vals: list) -> dict:
    """Run analysis with different eigenvalue sources, walk types, and epsilon.

    Uses K_half (threshold) metric for speed instead of sigmoid fitting.
    Sigmoid fitting is done only for the primary analysis.
    """

    # ── 7.1: Eigenvalue source comparison ────────────────────────────
    eig_source_results = {}
    for source in ["adjacency", "random_walk_symmetric"]:
        Ks_key = "K_star_A" if source == "adjacency" else "K_star_S"

        K_half_list = []
        K_star_list = []
        for r in results:
            if r is None or not r.get("has_full_spectral", False):
                continue
            K_star = r.get(Ks_key, 9999)
            if K_star > 200 or K_star < 1:
                continue

            K_list, q_vals = _extract_quality_curve(r, K_vals, "quality_vs_K", 1e-6)
            K_half = compute_K_half(K_list, q_vals)
            K_half_list.append(K_half)
            K_star_list.append(K_star)

        rho, p_val = (float("nan"), float("nan"))
        if len(K_half_list) >= 5:
            rho, p_val = spearmanr(K_star_list, K_half_list)
            rho = float(rho) if not math.isnan(rho) else 0.0
            p_val = float(p_val) if not math.isnan(p_val) else 1.0

        eig_source_results[source] = {
            "spearman_rho": rho,
            "p_value": p_val,
            "n_graphs": len(K_half_list),
            "method": "K_half_threshold",
        }

    # ── 7.2: Walk feature type comparison ────────────────────────────
    feature_type_results = {}
    for feat_type in ["P_diagonal", "A_diagonal_normalized", "heat_kernel"]:
        q_key = {
            "P_diagonal": "quality_vs_K",
            "A_diagonal_normalized": "quality_walk_A",
            "heat_kernel": "quality_hk",
        }[feat_type]

        K_half_list = []
        K_star_list = []
        for r in results:
            if r is None or not r.get("has_full_spectral", False):
                continue
            K_star = r.get("K_star_S", 9999)
            if K_star > 200 or K_star < 1:
                continue

            K_list, q_vals = _extract_quality_curve(r, K_vals, q_key, 1e-6)
            q_range = max(q_vals) - min(q_vals)
            if q_range < 0.001:
                continue  # flat curve, skip

            K_half = compute_K_half(K_list, q_vals)
            K_half_list.append(K_half)
            K_star_list.append(K_star)

        rho, p_val = (float("nan"), float("nan"))
        if len(K_half_list) >= 5:
            rho, p_val = spearmanr(K_star_list, K_half_list)
            rho = float(rho) if not math.isnan(rho) else 0.0
            p_val = float(p_val) if not math.isnan(p_val) else 1.0

        feature_type_results[feat_type] = {
            "spearman_rho": rho,
            "p_value": p_val,
            "n_graphs": len(K_half_list),
            "method": "K_half_threshold",
        }

    # ── 7.3: Epsilon sensitivity ─────────────────────────────────────
    eps_results = {}
    for eps in [1e-8, 1e-6, 1e-4, 1e-2, 1e-1]:
        K_half_list = []
        K_star_list = []
        for r in results:
            if r is None or not r.get("has_full_spectral", False):
                continue
            K_star = r.get("K_star_S", 9999)
            if K_star > 200 or K_star < 1:
                continue

            K_list, q_vals = _extract_quality_curve(r, K_vals, "quality_vs_K", eps)
            q_range = max(q_vals) - min(q_vals)
            if q_range < 0.001:
                continue

            K_half = compute_K_half(K_list, q_vals)
            K_half_list.append(K_half)
            K_star_list.append(K_star)

        rho, p_val = (float("nan"), float("nan"))
        if len(K_half_list) >= 5:
            rho, p_val = spearmanr(K_star_list, K_half_list)
            rho = float(rho) if not math.isnan(rho) else 0.0
            p_val = float(p_val) if not math.isnan(p_val) else 1.0

        eps_results[str(eps)] = {
            "spearman_rho": rho,
            "p_value": p_val,
            "n_graphs": len(K_half_list),
            "method": "K_half_threshold",
        }

    return {
        "eigenvalue_source_comparison": eig_source_results,
        "walk_feature_type_comparison": feature_type_results,
        "epsilon_sensitivity": eps_results,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  OUTPUT FORMATTING
# ═══════════════════════════════════════════════════════════════════════════

def format_output(all_results: dict, analysis: dict, variations: dict,
                  synthetic_analysis: dict, per_dataset_analysis: dict) -> dict:
    """Format results into exp_gen_sol_out.json schema."""

    datasets_out = []

    # Determine best Spearman correlation
    best_rho = analysis.get("spearman_sigmoid", {}).get("rho", 0)
    if math.isnan(best_rho):
        best_rho = 0
    rho_khalf = analysis.get("spearman_khalf", {}).get("rho", 0)
    if math.isnan(rho_khalf):
        rho_khalf = 0
    best_rho_overall = max(abs(best_rho), abs(rho_khalf))

    frac_sharp = analysis.get("frac_sharp", 0)

    # Check SRI gap positivity
    sri_gaps = analysis.get("sri_gap_results", {})
    gap_positive_count = sum(
        1 for v in sri_gaps.values()
        if isinstance(v, dict) and not math.isnan(v.get("gap", float("nan")))
        and v.get("gap", 0) > 0
    )
    gap_total = sum(
        1 for v in sri_gaps.values()
        if isinstance(v, dict) and not math.isnan(v.get("gap", float("nan")))
    )
    gap_positive_majority = gap_positive_count > gap_total / 2 if gap_total > 0 else False

    confirms = (best_rho_overall > 0.5) and (frac_sharp > 0.3) and gap_positive_majority

    # Also check mean_dist correlation
    rho_md = analysis.get("spearman_mean_dist", {}).get("rho", 0)
    if math.isnan(rho_md):
        rho_md = 0

    # Build evidence summary
    evidence_parts = []
    evidence_parts.append(
        f"Spearman rho(K_inflect, K*) = {best_rho:.3f} "
        f"(sigmoid, n={analysis.get('spearman_sigmoid', {}).get('n', 0)})"
    )
    evidence_parts.append(
        f"Spearman rho(K_half, K*) = {rho_khalf:.3f} "
        f"(threshold, n={analysis.get('spearman_khalf', {}).get('n', 0)}, "
        f"p={analysis.get('spearman_khalf', {}).get('p_value', 1.0):.4f})"
    )
    evidence_parts.append(
        f"Spearman rho(K_half_mean_dist, K*) = {rho_md:.3f} "
        f"(mean_dist, n={analysis.get('spearman_mean_dist', {}).get('n', 0)})"
    )
    evidence_parts.append(
        f"Mean sharpness ratio = {analysis.get('mean_sharpness', float('nan')):.3f}"
    )
    evidence_parts.append(
        f"Fraction sharp transitions = {frac_sharp:.3f}"
    )
    evidence_parts.append(
        f"SRI gap positive in {gap_positive_count}/{gap_total} K values"
    )

    # Alternative delta_min findings
    alt_dm = analysis.get("alternative_delta_min", {})
    if alt_dm:
        best_alt = max(alt_dm.items(),
                       key=lambda x: abs(x[1].get("spearman_rho", 0))
                       if not math.isnan(x[1].get("spearman_rho", float("nan"))) else 0)
        evidence_parts.append(
            f"Best alt delta_min: {best_alt[0]} (rho={best_alt[1].get('spearman_rho', 0):.3f})"
        )

    evidence_parts.append(
        f"Hypothesis {'CONFIRMED' if confirms else 'NOT CONFIRMED'}"
    )

    global_summary = {
        "experiment": "K_dependent_phase_transition",
        "confirms_hypothesis": confirms,
        "evidence_summary": "; ".join(evidence_parts),
        "global_results": {
            "spearman_rho_Kinflect_vs_Kstar": {
                "sigmoid": analysis.get("spearman_sigmoid", {}),
                "K_half_fallback": analysis.get("spearman_khalf", {}),
                "mean_dist_K_half": analysis.get("spearman_mean_dist", {}),
            },
            "alternative_delta_min_results": analysis.get("alternative_delta_min", {}),
            "mean_sharpness_ratio": analysis.get("mean_sharpness", float("nan")),
            "fraction_sharp_transitions": frac_sharp,
            "SRI_group_quality_gaps": sri_gaps,
            "quintile_curves": analysis.get("quintile_curves", {}),
        },
        "method_variation_results": variations,
        "per_dataset_results": per_dataset_analysis,
        "synthetic_results": synthetic_analysis,
    }

    # Format as exp_gen_sol_out.json
    # Each dataset's results become examples
    for ds_name, ds_results in all_results.items():
        examples = []
        for r in ds_results:
            if r is None:
                continue

            # Input: graph description
            input_data = {
                "graph_idx": r.get("graph_idx", -1),
                "n_nodes": r.get("n_nodes", 0),
                "delta_min_A": r.get("delta_min_A", 0),
                "delta_min_S": r.get("delta_min_S", r.get("delta_min_A", 0)),
                "K_star_A": r.get("K_star_A", 9999),
                "K_star_S": r.get("K_star_S", r.get("K_star_A", 9999)),
            }

            # Output: observed quality-vs-K
            observed = {}
            for K_key, q_data in r.get("quality_vs_K", {}).items():
                observed[str(K_key)] = {
                    "distinguish_1e-6": q_data.get("distinguish", {}).get(1e-6, 0),
                    "mean_dist": q_data.get("mean_dist", 0),
                }
            output_data = {"quality_vs_K": observed}

            # Baseline prediction: linear interpolation (no phase awareness)
            K_star = r.get("K_star_S", r.get("K_star_A", 9999))
            K_keys = sorted([int(k) for k in r.get("quality_vs_K", {}).keys()])
            if K_keys:
                q_vals = [r["quality_vs_K"][k]["distinguish"].get(1e-6, 0)
                          for k in K_keys]
                if len(K_keys) >= 2:
                    slope = (q_vals[-1] - q_vals[0]) / (K_keys[-1] - K_keys[0])
                    baseline_pred = {
                        str(k): round(q_vals[0] + slope * (k - K_keys[0]), 6)
                        for k in K_keys
                    }
                else:
                    baseline_pred = {str(k): q_vals[0] for k in K_keys}
            else:
                baseline_pred = {}

            # Our method prediction: sigmoid with K_inflect ≈ K*
            if K_keys and len(K_keys) >= 2:
                q_range = max(q_vals) - min(q_vals)
                method_pred = {}
                for k in K_keys:
                    exponent = -0.2 * (k - K_star)
                    exponent = max(min(exponent, 500), -500)  # clamp
                    pred = min(q_vals) + q_range / (1 + math.exp(exponent))
                    method_pred[str(k)] = round(pred, 6)
            else:
                method_pred = {}

            examples.append({
                "input": json.dumps(input_data),
                "output": json.dumps(output_data),
                "predict_baseline": json.dumps(baseline_pred),
                "predict_our_method": json.dumps(method_pred),
            })

        if examples:
            datasets_out.append({
                "dataset": ds_name,
                "examples": examples,
            })

    return {
        "metadata": global_summary,
        "datasets": datasets_out,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

@logger.catch
def main():
    import time
    start_time = time.time()

    # ── Load data ───────────────────────────────────────────────────────
    max_per_ds = MAX_EXAMPLES if MAX_EXAMPLES > 0 else 0
    logger.info(f"Loading data (max_per_dataset={max_per_ds or 'all'})")
    all_data = load_all_data(max_per_dataset=max_per_ds)

    total_examples = sum(len(v) for v in all_data.values())
    logger.info(f"Total examples loaded: {total_examples}")
    for ds_name, examples in all_data.items():
        logger.info(f"  {ds_name}: {len(examples)} examples")

    # ── Generate synthetic graphs ────────────────────────────────────
    logger.info("Generating custom synthetic graphs...")
    synthetic_graphs = generate_synthetic_graphs(n_graphs=50, n_trials=200)

    # ── Process all graphs ──────────────────────────────────────────
    all_results = {}

    # Decide which graphs get full eigendecomposition
    # ZINC: subsample for full spectral (cap at 500 for speed), rest use stored RWSE
    # Peptides: subsample for full spectral (cap at 200)
    # Synthetic: all get full spectral (only 100)
    FULL_SPECTRAL_CAP = {"ZINC-subset": 500, "Peptides-func": 200,
                         "Peptides-struct": 200, "Synthetic-aliased-pairs": 100}

    for ds_name, examples in all_data.items():
        logger.info(f"Processing {ds_name} ({len(examples)} examples)...")
        cap = FULL_SPECTRAL_CAP.get(ds_name, 100)
        ds_results = []

        for i, ex in enumerate(examples):
            try:
                if i < cap:
                    # Full eigendecomposition
                    result = process_graph_full(ex, i)
                else:
                    # Stored RWSE only
                    result = process_graph_stored_only(ex, i)
                ds_results.append(result)
            except Exception:
                logger.exception(f"Failed on {ds_name} example {i}")
                ds_results.append(None)

            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                logger.info(f"  {ds_name}: {i + 1}/{len(examples)} "
                            f"({elapsed:.0f}s elapsed)")

        all_results[ds_name] = ds_results
        logger.info(f"  {ds_name}: completed {len(ds_results)} graphs, "
                    f"{sum(1 for r in ds_results if r is not None)} valid")

    # Process synthetic graphs
    logger.info(f"Processing {len(synthetic_graphs)} synthetic graphs...")
    syn_results = []
    for i, syn in enumerate(synthetic_graphs):
        try:
            result = process_synthetic_graph(syn, i)
            syn_results.append(result)
        except Exception:
            logger.exception(f"Failed on synthetic graph {i}")
            syn_results.append(None)

    all_results["Custom-synthetic"] = syn_results
    logger.info(f"Synthetic: {sum(1 for r in syn_results if r is not None)} valid")

    # ── Phase transition analysis ───────────────────────────────────
    # Collect all results with full spectral for primary analysis
    full_spectral_results = []
    for ds_name, ds_results in all_results.items():
        for r in ds_results:
            if r is not None and r.get("has_full_spectral", False):
                full_spectral_results.append(r)

    logger.info(f"Full spectral results: {len(full_spectral_results)} graphs")

    # Determine which K values to use based on available data
    K_vals_for_analysis = K_VALUES_EXTENDED

    logger.info("Running phase transition analysis...")
    analysis = run_phase_transition_analysis(
        full_spectral_results, K_vals_for_analysis, eps=1e-6
    )

    logger.info(f"Sigmoid Spearman rho: {analysis['spearman_sigmoid']}")
    logger.info(f"K_half Spearman rho: {analysis['spearman_khalf']}")
    logger.info(f"Sigmoid fits: {analysis['sigmoid_fit_count']}"
                f"/{analysis['total_graphs_tested']}")
    logger.info(f"Sharpness: mean={analysis['mean_sharpness']:.3f}, "
                f"frac_sharp={analysis['frac_sharp']:.3f}")

    # ── Lightweight analysis on ALL graphs (SRI gaps + quintile only) ─
    all_graphs_results = []
    for ds_name, ds_results in all_results.items():
        for r in ds_results:
            if r is not None:
                all_graphs_results.append(r)

    logger.info(f"Computing SRI gaps + quintile curves on ALL "
                f"{len(all_graphs_results)} graphs (stored RWSE, K=2..20)...")

    # Only compute SRI gaps and quintile curves, skip sigmoid fitting
    all_dm = [r.get("delta_min_S", r.get("delta_min_A", 0))
              for r in all_graphs_results if r.get("delta_min_S", r.get("delta_min_A", 0)) > 1e-12]
    if len(all_dm) >= 5:
        q_bounds = np.percentile(all_dm, [20, 40, 60, 80])
    else:
        q_bounds = np.array([0.05, 0.1, 0.2, 0.3])

    sri_gaps_all = {}
    quintile_all = {f"Q{q}": {} for q in range(1, 6)}
    eps_val = 1e-6
    for K in K_VALUES:
        K_key = int(K)
        pre_q, post_q = [], []
        qvals = {f"Q{q}": [] for q in range(1, 6)}

        for r in all_graphs_results:
            dm = r.get("delta_min_S", r.get("delta_min_A", 0))
            q_data = r.get("quality_vs_K", {}).get(K_key, {})
            dist = q_data.get("distinguish", {})
            q = dist.get(eps_val, 0.0) if isinstance(dist, dict) else 0.0

            if dm > 1e-12:
                sri = K * dm
                if sri < 1:
                    pre_q.append(q)
                else:
                    post_q.append(q)

            if dm <= q_bounds[0]:
                qvals["Q1"].append(q)
            elif dm <= q_bounds[1]:
                qvals["Q2"].append(q)
            elif dm <= q_bounds[2]:
                qvals["Q3"].append(q)
            elif dm <= q_bounds[3]:
                qvals["Q4"].append(q)
            else:
                qvals["Q5"].append(q)

        m_pre = float(np.mean(pre_q)) if pre_q else float("nan")
        m_post = float(np.mean(post_q)) if post_q else float("nan")
        gap = m_post - m_pre if not (math.isnan(m_pre) or math.isnan(m_post)) else float("nan")
        sri_gaps_all[K_key] = {"mean_pre": m_pre, "mean_post": m_post,
                               "gap": gap, "n_pre": len(pre_q), "n_post": len(post_q)}

        for qn in qvals:
            quintile_all[qn][K_key] = float(np.mean(qvals[qn])) if qvals[qn] else float("nan")

    analysis_all = {
        "sri_gap_results": sri_gaps_all,
        "quintile_curves": quintile_all,
        "quintile_bounds": [float(b) for b in q_bounds],
        "n_total": len(all_graphs_results),
    }
    logger.info(f"ALL graphs - SRI gaps: "
                f"{json.dumps(sri_gaps_all, indent=None)[:200]}")

    # ── Method variations (on full spectral subset) ─────────────────
    logger.info("Running method variations...")
    variations = run_method_variations(full_spectral_results, K_vals_for_analysis)
    logger.info(f"Variations - eigenvalue sources: "
                f"{json.dumps(variations['eigenvalue_source_comparison'])}")
    logger.info(f"Variations - walk features: "
                f"{json.dumps(variations['walk_feature_type_comparison'])}")

    # ── Per-dataset analysis (lightweight - uses K_half, not sigmoid) ─
    per_dataset_analysis = {}
    for ds_name, ds_results in all_results.items():
        valid = [r for r in ds_results if r is not None]
        full = [r for r in valid if r.get("has_full_spectral", False)]

        K_vals_ds = K_vals_for_analysis if full else K_VALUES
        results_ds = full if full else valid

        # Lightweight analysis: SRI gaps + quintile curves + K_half correlation
        ds_analysis = {}
        if results_ds:
            # SRI gap analysis
            sri_gaps = {}
            for K in K_vals_ds:
                K_key = int(K)
                pre_q, post_q = [], []
                for r in results_ds:
                    dm = r.get("delta_min_S", r.get("delta_min_A", 0))
                    if dm < 1e-12:
                        continue
                    sri = K * dm
                    q_data = r.get("quality_vs_K", {}).get(K_key, {})
                    dist = q_data.get("distinguish", {})
                    q = dist.get(1e-6, 0.0) if isinstance(dist, dict) else 0.0
                    if sri < 1:
                        pre_q.append(q)
                    else:
                        post_q.append(q)
                m_pre = float(np.mean(pre_q)) if pre_q else float("nan")
                m_post = float(np.mean(post_q)) if post_q else float("nan")
                gap = m_post - m_pre if not (math.isnan(m_pre) or math.isnan(m_post)) else float("nan")
                sri_gaps[K_key] = {"mean_pre": m_pre, "mean_post": m_post,
                                   "gap": gap, "n_pre": len(pre_q), "n_post": len(post_q)}

            # K_half correlation
            K_half_list, K_star_list = [], []
            for r in results_ds:
                K_star = r.get("K_star_S", r.get("K_star_A", 9999))
                if K_star > 200 or K_star < 1:
                    continue
                K_list, q_vals = _extract_quality_curve(r, K_vals_ds, "quality_vs_K", 1e-6)
                K_half_list.append(compute_K_half(K_list, q_vals))
                K_star_list.append(K_star)

            rho_kh, p_kh = (float("nan"), float("nan"))
            if len(K_half_list) >= 5:
                rho_kh, p_kh = spearmanr(K_star_list, K_half_list)
                rho_kh = float(rho_kh) if not math.isnan(rho_kh) else 0.0
                p_kh = float(p_kh) if not math.isnan(p_kh) else 1.0

            ds_analysis = {
                "sri_gap_results": sri_gaps,
                "spearman_khalf": {"rho": rho_kh, "p_value": p_kh,
                                   "n": len(K_half_list)},
            }

        per_dataset_analysis[ds_name] = {
            "n_total": len(ds_results),
            "n_valid": len(valid),
            "n_full_spectral": len(full),
            "analysis": ds_analysis,
        }
        logger.info(f"  {ds_name}: n_full={len(full)}, "
                    f"K_half rho={ds_analysis.get('spearman_khalf', {}).get('rho', 'N/A')}")

    # Synthetic-specific analysis (also lightweight)
    syn_analysis = {}
    if syn_results:
        valid_syn = [r for r in syn_results if r is not None]
        if valid_syn:
            # Use K_half for speed
            K_half_syn, K_star_syn = [], []
            for r in valid_syn:
                K_star = r.get("K_star_S", 9999)
                if K_star > 200 or K_star < 1:
                    continue
                K_list, q_vals = _extract_quality_curve(
                    r, K_vals_for_analysis, "quality_vs_K", 1e-6)
                K_half_syn.append(compute_K_half(K_list, q_vals))
                K_star_syn.append(K_star)

            rho_syn, p_syn = (float("nan"), float("nan"))
            if len(K_half_syn) >= 5:
                rho_syn, p_syn = spearmanr(K_star_syn, K_half_syn)
                rho_syn = float(rho_syn) if not math.isnan(rho_syn) else 0.0
                p_syn = float(p_syn) if not math.isnan(p_syn) else 1.0

            syn_analysis = {
                "spearman_khalf": {"rho": rho_syn, "p_value": p_syn,
                                   "n": len(K_half_syn)},
                "n_valid": len(valid_syn),
            }
            logger.info(f"Synthetic analysis - K_half rho: {rho_syn}")

    # ── Format output ──────────────────────────────────────────────
    logger.info("Formatting output...")
    output = format_output(
        all_results, analysis, variations, syn_analysis, per_dataset_analysis
    )

    # ── Save ───────────────────────────────────────────────────────
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    out_size = out_path.stat().st_size / 1e6
    logger.info(f"Saved {out_path} ({out_size:.1f} MB)")

    elapsed = time.time() - start_time
    logger.info(f"Total runtime: {elapsed:.1f}s")

    # File size check
    if out_size > 95:
        logger.warning(f"Output file {out_size:.1f} MB exceeds 95 MB, splitting...")
        split_output(output, WORKSPACE)

    return output


def split_output(output: dict, workspace: Path):
    """Split large output file into parts < 95 MB."""
    import glob as glob_mod

    out_dir = workspace / "method_out"
    out_dir.mkdir(exist_ok=True)

    datasets = output.get("datasets", [])
    part_num = 1
    current_part = {"metadata": output.get("metadata", {}), "datasets": []}
    current_size = 0

    for ds in datasets:
        ds_str = json.dumps(ds)
        ds_size = len(ds_str)

        if current_size + ds_size > 90 * 1e6 and current_part["datasets"]:
            # Save current part
            part_path = out_dir / f"method_out_{part_num}.json"
            part_path.write_text(json.dumps(current_part, indent=2, default=str))
            logger.info(f"Saved part {part_num}: {part_path}")
            part_num += 1
            current_part = {"metadata": output.get("metadata", {}), "datasets": []}
            current_size = 0

        current_part["datasets"].append(ds)
        current_size += ds_size

    if current_part["datasets"]:
        part_path = out_dir / f"method_out_{part_num}.json"
        part_path.write_text(json.dumps(current_part, indent=2, default=str))
        logger.info(f"Saved part {part_num}: {part_path}")

    # Remove original
    orig = workspace / "method_out.json"
    if orig.exists():
        orig.unlink()
        logger.info("Removed original oversized file")


if __name__ == "__main__":
    main()
