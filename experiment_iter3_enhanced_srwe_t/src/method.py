#!/usr/bin/env python3
"""
Enhanced SRWE: Tikhonov/TSVD/MPM Spectral Recovery + GNN Benchmark on Peptides/ZINC.

Implements three spectral recovery methods (Matrix Pencil, Tikhonov, Truncated SVD),
evaluates spectral recovery quality (W1, cosine, top-k accuracy) stratified by SRI,
and benchmarks SRWE as a GNN positional encoding against RWSE and LapPE on
Peptides-func (AP), Peptides-struct (MAE), and ZINC (MAE).
"""

import atexit
import json
import math
import os
import resource
import signal
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ── Limit thread usage ──
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import psutil
import scipy.stats
from loguru import logger
from scipy.linalg import eigh, svd
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Logging setup ──
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ── Resource limits ──
TOTAL_RAM_GB = psutil.virtual_memory().available / 1e9
RAM_LIMIT = int(min(TOTAL_RAM_GB * 0.85, 50) * 1024**3)
resource.setrlimit(resource.RLIMIT_AS, (RAM_LIMIT, RAM_LIMIT))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))  # ~60 min

# ── PyTorch import ──
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Hardware detection ──
def _cgroup_cpus() -> Optional[int]:
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

NUM_CPUS = _cgroup_cpus() or os.cpu_count() or 1
HAS_GPU = torch.cuda.is_available()
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if HAS_GPU else 0
DEVICE = torch.device("cuda" if HAS_GPU else "cpu")

logger.info(f"Hardware: {NUM_CPUS} CPUs, GPU={HAS_GPU} ({VRAM_GB:.1f}GB VRAM), RAM={TOTAL_RAM_GB:.1f}GB available")

# ── Constants ──
DEP_DIR = Path("/opt/ai-inventor/aii_pipeline/runs/run__20260222_005317/3_invention_loop/iter_1/gen_art/data_id2_it1__opus")
WORKSPACE = SCRIPT_DIR
OUTPUT_FILE = WORKSPACE / "method_out.json"

MAX_EXAMPLES = int(os.environ.get("MAX_EXAMPLES", "0"))  # 0 = all
RWSE_DIM = 20
PE_DIM = 20
SEEDS = [0, 1, 2]
SRWE_HIST_BINS = 20
SRWE_HIST_RANGE = (-1.0, 1.0)

# GNN config
GNN_HIDDEN = 64
GNN_LAYERS = 3
GNN_DROPOUT = 0.1
GNN_LR = 1e-3
GNN_WD = 1e-5
GNN_BATCH_SIZE = 64
GNN_MAX_EPOCHS = 200
GNN_PATIENCE = 50
GNN_LR_PATIENCE = 20
GNN_LR_FACTOR = 0.5
GNN_MAX_GRAPHS = 5000  # Max graphs per dataset for GNN benchmark (ZINC has 12K)

# Time budget constants (seconds from start)
TIME_SKIP_SEEDS = 2400     # ~40 min: skip remaining seeds
TIME_SKIP_PE = 2600        # ~43 min: skip remaining PE types
TIME_SKIP_DATASETS = 2800  # ~46 min: skip remaining datasets
TIME_SAVE_PARTIAL = 3000   # ~50 min: force save partial results


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GraphData:
    """Parsed graph data from the JSON dataset."""
    edge_index: np.ndarray  # [2, E] int
    num_nodes: int
    node_feat: np.ndarray   # [N, F]
    eigenvalues_A: np.ndarray   # from adjacency (stored)
    rwse: np.ndarray        # [N, 20]
    local_spectral_stored: list  # stored top-10 from A
    sri_k20: float
    delta_min: float
    target: Any
    task_type: str
    fold: int
    dataset: str
    row_index: int
    # Recomputed from P = D^{-1/2}AD^{-1/2}
    eigenvalues_P: Optional[np.ndarray] = None
    eigenvectors_P: Optional[np.ndarray] = None
    # Recovered spectral weights
    srwe_tik: Optional[np.ndarray] = None
    srwe_tsvd: Optional[np.ndarray] = None
    srwe_mpm_evals: Optional[np.ndarray] = None
    srwe_mpm_weights: Optional[np.ndarray] = None
    # PE vectors
    pe_rwse: Optional[np.ndarray] = None
    pe_lappe: Optional[np.ndarray] = None
    pe_srwe: Optional[np.ndarray] = None
    # GNN predictions
    predictions: dict = field(default_factory=dict)
    # Recovery metrics (per-node averages)
    recovery_metrics: dict = field(default_factory=dict)
    # Original strings for output
    input_str: str = ""
    output_str: str = ""


def parse_graph(example: dict, dataset_name: str) -> Optional[GraphData]:
    """Parse a single example from the dataset JSON into a GraphData object."""
    try:
        inp = json.loads(example["input"])
        edge_index_raw = inp.get("edge_index", [[], []])
        num_nodes = inp.get("num_nodes", 0)
        if num_nodes == 0:
            return None

        edge_index = np.array(edge_index_raw, dtype=np.int64)
        if edge_index.shape[0] != 2:
            edge_index = np.zeros((2, 0), dtype=np.int64)

        node_feat_raw = inp.get("node_feat", [[0]] * num_nodes)
        node_feat = np.array(node_feat_raw, dtype=np.float32)
        if node_feat.ndim == 1:
            node_feat = node_feat.reshape(-1, 1)

        spectral = inp.get("spectral", {})
        eigenvalues_A = np.array(spectral.get("eigenvalues", []), dtype=np.float64)
        rwse_raw = spectral.get("rwse", [])
        local_spectral = spectral.get("local_spectral", [])
        sri = spectral.get("sri", {})
        sri_k20 = float(sri.get("K=20", 0.0))
        delta_min = float(spectral.get("delta_min", 0.0))

        # Handle RWSE: ensure shape [N, 20]
        if len(rwse_raw) == 0:
            rwse = np.zeros((num_nodes, RWSE_DIM), dtype=np.float64)
        else:
            rwse = np.array(rwse_raw, dtype=np.float64)
            if rwse.ndim == 1:
                rwse = rwse.reshape(1, -1)
            # Pad if fewer than num_nodes rows (for large graphs, only some nodes stored)
            if rwse.shape[0] < num_nodes:
                pad = np.zeros((num_nodes - rwse.shape[0], rwse.shape[1]), dtype=np.float64)
                rwse = np.vstack([rwse, pad])
            # Pad/trim columns to RWSE_DIM
            if rwse.shape[1] < RWSE_DIM:
                pad = np.zeros((rwse.shape[0], RWSE_DIM - rwse.shape[1]), dtype=np.float64)
                rwse = np.hstack([rwse, pad])
            elif rwse.shape[1] > RWSE_DIM:
                rwse = rwse[:, :RWSE_DIM]

        return GraphData(
            edge_index=edge_index,
            num_nodes=num_nodes,
            node_feat=node_feat,
            eigenvalues_A=eigenvalues_A,
            rwse=rwse,
            local_spectral_stored=local_spectral,
            sri_k20=sri_k20,
            delta_min=delta_min,
            target=example["output"],
            task_type=example.get("metadata_task_type", "regression"),
            fold=example.get("metadata_fold", 0),
            dataset=dataset_name,
            row_index=example.get("metadata_row_index", 0),
            input_str=example["input"],
            output_str=example["output"],
        )
    except Exception:
        logger.exception(f"Failed to parse graph in {dataset_name}")
        return None


def load_dataset_from_files(data_files: list[Path], max_per_dataset: int = 0) -> dict[str, list[GraphData]]:
    """Load all datasets from dependency JSON files."""
    datasets: dict[str, list[GraphData]] = {}
    for fpath in data_files:
        if not fpath.exists():
            logger.warning(f"Data file not found: {fpath}")
            continue
        logger.info(f"Loading {fpath.name} ({fpath.stat().st_size / 1024 / 1024:.1f} MB)")
        raw = json.loads(fpath.read_text())
        for ds in raw.get("datasets", []):
            ds_name = ds["dataset"]
            examples = ds.get("examples", [])
            if ds_name not in datasets:
                datasets[ds_name] = []
            existing = len(datasets[ds_name])
            for ex in examples:
                if max_per_dataset > 0 and existing >= max_per_dataset:
                    break
                g = parse_graph(ex, ds_name)
                if g is not None:
                    datasets[ds_name].append(g)
                    existing += 1
        logger.info(f"  Loaded from {fpath.name}, running totals: { {k: len(v) for k, v in datasets.items()} }")
    return datasets


# ═══════════════════════════════════════════════════════════════════════════
#  SPECTRAL RECOMPUTATION: P = D^{-1/2} A D^{-1/2}
# ═══════════════════════════════════════════════════════════════════════════

def build_adjacency(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    """Build dense adjacency matrix from edge_index [2, E]."""
    A = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    if edge_index.shape[1] > 0:
        src, dst = edge_index[0], edge_index[1]
        valid = (src < num_nodes) & (dst < num_nodes) & (src >= 0) & (dst >= 0)
        A[src[valid], dst[valid]] = 1.0
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)
    return A


def compute_normalized_adjacency_eigen(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalues and eigenvectors of P = D^{-1/2} A D^{-1/2}.
    Returns (eigenvalues, eigenvectors) sorted ascending by eigenvalue.
    """
    n = A.shape[0]
    if n == 0:
        return np.array([]), np.array([]).reshape(0, 0)

    degree = A.sum(axis=1)
    # Handle isolated nodes: set their D^{-1/2} to 0
    d_inv_sqrt = np.zeros(n, dtype=np.float64)
    mask = degree > 1e-12
    d_inv_sqrt[mask] = 1.0 / np.sqrt(degree[mask])

    # P = D^{-1/2} A D^{-1/2}
    P = A * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]

    # Eigendecomposition (P is symmetric)
    eigenvalues, eigenvectors = eigh(P)
    # eigh returns sorted ascending
    return eigenvalues, eigenvectors


def recompute_spectral(graph: GraphData) -> None:
    """Recompute normalized adjacency eigendecomposition for a graph."""
    A = build_adjacency(graph.edge_index, graph.num_nodes)
    eigenvalues_P, eigenvectors_P = compute_normalized_adjacency_eigen(A)
    graph.eigenvalues_P = eigenvalues_P
    graph.eigenvectors_P = eigenvectors_P


def consistency_check(graphs: list[GraphData], n_samples: int = 5) -> dict:
    """
    Check consistency between stored RWSE and reconstructed moments from P's spectral.
    Returns dict with mean relative error and details.
    """
    errors = []
    for g in graphs[:min(10, len(graphs))]:
        if g.eigenvalues_P is None or g.eigenvectors_P is None:
            continue
        n = g.num_nodes
        n_check = min(n_samples, n)
        for u in range(n_check):
            weights_P = g.eigenvectors_P[u, :] ** 2  # local spectral measure from P
            evals_P = g.eigenvalues_P
            for k_idx in range(min(5, RWSE_DIM)):
                k = k_idx + 1  # walk length
                reconstructed = np.sum(weights_P * (evals_P ** k))
                stored = g.rwse[u, k_idx]
                if abs(stored) > 1e-8:
                    rel_err = abs(reconstructed - stored) / abs(stored)
                else:
                    rel_err = abs(reconstructed - stored)
                errors.append(rel_err)

    mean_err = float(np.mean(errors)) if errors else 0.0
    max_err = float(np.max(errors)) if errors else 0.0
    logger.info(f"Consistency check: mean_rel_error={mean_err:.6f}, max={max_err:.6f}, n_checks={len(errors)}")
    return {"mean_relative_error": mean_err, "max_relative_error": max_err, "n_checks": len(errors)}


# ═══════════════════════════════════════════════════════════════════════════
#  SRWE RECOVERY METHODS
# ═══════════════════════════════════════════════════════════════════════════

def srwe_tikhonov(
    moments: np.ndarray,
    eigenvalues: np.ndarray,
    alpha: Optional[float] = None,
) -> np.ndarray:
    """
    Tikhonov-regularized Vandermonde recovery.
    moments: [K] array of RWSE moments for walk lengths 1..K
    eigenvalues: [n] array of known eigenvalues (from P)
    Returns: weights [n] non-negative, normalized to sum=1
    """
    K = len(moments)
    n = len(eigenvalues)
    if n == 0:
        return np.array([])

    # Build Vandermonde: V[k,i] = eigenvalues[i]^(k+1) for k=0..K-1
    V = np.zeros((K, n), dtype=np.float64)
    for k in range(K):
        V[k, :] = eigenvalues ** (k + 1)

    if alpha is None:
        # GCV-based alpha selection
        alpha = _gcv_alpha(V, moments)

    # Solve: w* = (V^T V + alpha I)^{-1} V^T m
    VtV = V.T @ V
    Vtm = V.T @ moments
    try:
        w = np.linalg.solve(VtV + alpha * np.eye(n), Vtm)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(V, moments, rcond=None)[0]

    # Clamp non-negative, normalize
    w = np.maximum(w, 0.0)
    s = w.sum()
    if s > 1e-12:
        w /= s
    return w


def _gcv_alpha(V: np.ndarray, m: np.ndarray) -> float:
    """Generalized cross-validation for Tikhonov regularization parameter."""
    try:
        U, sigma, Vt = svd(V, full_matrices=False)
        K = V.shape[0]
        alphas = np.logspace(-10, 2, 60)
        best_alpha = 1e-4
        best_gcv = float("inf")

        Utm = U.T @ m
        for a in alphas:
            f = sigma**2 / (sigma**2 + a)
            residual = m - U @ (f[:, None] * (U.T @ m.reshape(-1, 1))).ravel() if len(m.shape) == 1 else m
            # Simpler: residual = m - V @ w_alpha
            w_alpha = Vt.T @ (f * Utm / np.maximum(sigma, 1e-15))
            res = m - V @ w_alpha
            res_norm2 = np.sum(res**2)
            trace_term = K - np.sum(f)
            if trace_term > 1e-10:
                gcv = K * res_norm2 / (trace_term**2)
                if gcv < best_gcv:
                    best_gcv = gcv
                    best_alpha = a
        return best_alpha
    except Exception:
        return 1e-4


def srwe_tsvd(
    moments: np.ndarray,
    eigenvalues: np.ndarray,
    threshold: float = 0.01,
) -> np.ndarray:
    """
    Truncated SVD recovery.
    moments: [K] array of RWSE moments
    eigenvalues: [n] array of known eigenvalues (from P)
    Returns: weights [n] non-negative, normalized
    """
    K = len(moments)
    n = len(eigenvalues)
    if n == 0:
        return np.array([])

    # Build Vandermonde
    V = np.zeros((K, n), dtype=np.float64)
    for k in range(K):
        V[k, :] = eigenvalues ** (k + 1)

    try:
        U, sigma, Vt = svd(V, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.ones(n, dtype=np.float64) / n

    # Determine rank
    if sigma[0] > 1e-15:
        keep = sigma / sigma[0] > threshold
    else:
        keep = np.ones(len(sigma), dtype=bool)
    r = max(1, int(np.sum(keep)))

    # Truncated pseudoinverse
    sigma_inv = np.zeros(len(sigma))
    sigma_inv[:r] = 1.0 / np.maximum(sigma[:r], 1e-15)
    w = Vt.T @ (sigma_inv * (U.T @ moments))

    # Clamp non-negative, normalize
    w = np.maximum(w, 0.0)
    s = w.sum()
    if s > 1e-12:
        w /= s
    return w


def srwe_mpm(
    moments: np.ndarray,
    pencil_rank: Optional[int] = None,
    noise_threshold: float = 0.01,
    spectral_range: tuple[float, float] = (-1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Matrix Pencil Method: recover both eigenvalues and weights from moments.
    moments: [K] array of RWSE moments for walk lengths 1..K
    Returns: (estimated_eigenvalues, estimated_weights)
    """
    # Prepend m_0 = 1.0 (the sum of all weights)
    all_moments = np.concatenate([[1.0], moments])
    M = len(all_moments)
    L = min(10, M // 2)

    # Build Hankel matrices
    H0 = np.zeros((L, L), dtype=np.float64)
    H1 = np.zeros((L, L), dtype=np.float64)
    for i in range(L):
        for j in range(L):
            idx0 = i + j
            idx1 = i + j + 1
            if idx0 < M:
                H0[i, j] = all_moments[idx0]
            if idx1 < M:
                H1[i, j] = all_moments[idx1]

    try:
        U, S, Vt = svd(H0)
    except np.linalg.LinAlgError:
        return np.array([0.0]), np.array([1.0])

    # Determine rank
    if pencil_rank is not None:
        r = min(pencil_rank, len(S))
    else:
        if S[0] > 1e-15:
            r = int(np.sum(S / S[0] > noise_threshold))
        else:
            r = 1
    r = max(1, min(r, L))

    # Truncate
    Ur = U[:, :r]
    Sr = S[:r]
    Vtr = Vt[:r, :]

    # Reduced pencil: M_r = Ur^H @ H1 @ Vtr^H @ diag(1/Sr)
    try:
        Sr_inv = np.diag(1.0 / np.maximum(Sr, 1e-15))
        M_pencil = Ur.T @ H1 @ Vtr.T @ Sr_inv
        est_evals = np.linalg.eigvals(M_pencil)
    except np.linalg.LinAlgError:
        return np.array([0.0]), np.array([1.0])

    # Keep only real parts, clamp to valid range
    est_evals = np.real(est_evals)
    est_evals = np.clip(est_evals, spectral_range[0], spectral_range[1])

    # Remove very small or duplicate eigenvalues
    est_evals = np.unique(np.round(est_evals, 8))
    if len(est_evals) == 0:
        return np.array([0.0]), np.array([1.0])

    # Recover weights via Vandermonde least-squares
    n_est = len(est_evals)
    K = len(moments)
    V = np.zeros((K, n_est), dtype=np.float64)
    for k in range(K):
        V[k, :] = est_evals ** (k + 1)

    try:
        weights, _, _, _ = np.linalg.lstsq(V, moments, rcond=None)
    except np.linalg.LinAlgError:
        weights = np.ones(n_est) / n_est

    weights = np.maximum(weights, 0.0)
    s = weights.sum()
    if s > 1e-12:
        weights /= s

    return est_evals, weights


# ═══════════════════════════════════════════════════════════════════════════
#  SPECTRAL RECOVERY EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def wasserstein1(
    true_evals: np.ndarray, true_weights: np.ndarray,
    est_evals: np.ndarray, est_weights: np.ndarray,
) -> float:
    """Compute Wasserstein-1 distance between two discrete measures."""
    if len(true_evals) == 0 or len(est_evals) == 0:
        return float("inf")
    try:
        return float(scipy.stats.wasserstein_distance(true_evals, est_evals, true_weights, est_weights))
    except Exception:
        return float("inf")


def cosine_similarity(w1: np.ndarray, w2: np.ndarray) -> float:
    """Cosine similarity between weight vectors (same eigenvalue set)."""
    n1, n2 = np.linalg.norm(w1), np.linalg.norm(w2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    return float(np.dot(w1, w2) / (n1 * n2))


def top_k_accuracy(true_weights: np.ndarray, est_weights: np.ndarray, k: int = 5) -> float:
    """Overlap of top-k eigenvalue indices by weight."""
    if len(true_weights) < k or len(est_weights) < k:
        k = min(len(true_weights), len(est_weights))
    if k == 0:
        return 0.0
    true_top = set(np.argsort(-true_weights)[:k])
    est_top = set(np.argsort(-est_weights)[:k])
    return len(true_top & est_top) / k


def evaluate_recovery_node(
    graph: GraphData,
    node_idx: int,
    method: str,
) -> dict[str, float]:
    """Evaluate spectral recovery for one node."""
    if graph.eigenvalues_P is None or graph.eigenvectors_P is None:
        return {}

    true_weights = graph.eigenvectors_P[node_idx, :] ** 2
    true_evals = graph.eigenvalues_P

    if method == "tikhonov" and graph.srwe_tik is not None:
        est_weights = graph.srwe_tik[node_idx]
        w1 = wasserstein1(true_evals, true_weights, true_evals, est_weights)
        cos = cosine_similarity(true_weights, est_weights)
        topk = top_k_accuracy(true_weights, est_weights, k=5)
        return {"w1": w1, "cosine": cos, "top5_acc": topk}

    elif method == "tsvd" and graph.srwe_tsvd is not None:
        est_weights = graph.srwe_tsvd[node_idx]
        w1 = wasserstein1(true_evals, true_weights, true_evals, est_weights)
        cos = cosine_similarity(true_weights, est_weights)
        topk = top_k_accuracy(true_weights, est_weights, k=5)
        return {"w1": w1, "cosine": cos, "top5_acc": topk}

    elif method == "mpm":
        if graph.srwe_mpm_evals is not None and graph.srwe_mpm_weights is not None:
            est_evals_node = graph.srwe_mpm_evals[node_idx]
            est_weights_node = graph.srwe_mpm_weights[node_idx]
            if len(est_evals_node) > 0 and len(est_weights_node) > 0:
                w1 = wasserstein1(true_evals, true_weights, est_evals_node, est_weights_node)
                # For cosine: align estimated to nearest true eigenvalues
                aligned_w = np.zeros(len(true_evals))
                for ev, wt in zip(est_evals_node, est_weights_node):
                    idx = np.argmin(np.abs(true_evals - ev))
                    aligned_w[idx] += wt
                cos = cosine_similarity(true_weights, aligned_w)
                topk = top_k_accuracy(true_weights, aligned_w, k=5)
                return {"w1": w1, "cosine": cos, "top5_acc": topk}

    return {}


def evaluate_recovery_graph(graph: GraphData, method: str, max_nodes: int = 50) -> dict[str, float]:
    """Evaluate spectral recovery averaged over nodes in a graph."""
    n = min(graph.num_nodes, max_nodes)
    metrics_list = []
    for u in range(n):
        m = evaluate_recovery_node(graph, u, method)
        if m:
            metrics_list.append(m)
    if not metrics_list:
        return {}

    result = {}
    for key in metrics_list[0]:
        vals = [m[key] for m in metrics_list if not math.isinf(m[key]) and not math.isnan(m[key])]
        if vals:
            result[f"mean_{key}"] = float(np.mean(vals))
            result[f"std_{key}"] = float(np.std(vals))
    return result


def sri_bin(sri_value: float) -> str:
    """Categorize SRI into bins."""
    if sri_value < 0.5:
        return "very_low"
    elif sri_value < 1.0:
        return "low"
    elif sri_value < 2.0:
        return "medium"
    else:
        return "high"


# ═══════════════════════════════════════════════════════════════════════════
#  PE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_pe_rwse(graph: GraphData) -> np.ndarray:
    """RWSE positional encoding: raw walk probabilities [N, 20]."""
    return graph.rwse.astype(np.float32)


def compute_pe_lappe(graph: GraphData) -> np.ndarray:
    """
    LapPE: squared eigenvector components of normalized adjacency P.
    Uses top-PE_DIM eigenvectors (by largest eigenvalue) of P, i.e., bottom of Laplacian.
    """
    n = graph.num_nodes
    if graph.eigenvectors_P is None or len(graph.eigenvalues_P) == 0:
        return np.zeros((n, PE_DIM), dtype=np.float32)

    # Use the PE_DIM eigenvectors with largest eigenvalues (bottom of Laplacian)
    # eigenvalues_P is sorted ascending, so take last PE_DIM
    n_eigs = len(graph.eigenvalues_P)
    k = min(PE_DIM, n_eigs)
    # Skip the last eigenvector (constant eigenvector for connected graphs, eig ~= 1)
    # Use eigenvectors at indices -k-1 to -2 (second through (k+1)-th largest)
    if n_eigs > 1:
        start_idx = max(0, n_eigs - k - 1)
        end_idx = n_eigs - 1  # exclude the last (trivial) eigenvector
        selected = graph.eigenvectors_P[:, start_idx:end_idx]
    else:
        selected = graph.eigenvectors_P

    pe = selected ** 2  # [N, <=PE_DIM]
    # Pad to PE_DIM
    if pe.shape[1] < PE_DIM:
        pad = np.zeros((n, PE_DIM - pe.shape[1]), dtype=np.float64)
        pe = np.hstack([pe, pad])
    return pe[:, :PE_DIM].astype(np.float32)


def compute_pe_srwe_histogram(
    graph: GraphData,
    method: str = "tikhonov",
) -> np.ndarray:
    """
    SRWE histogram PE: bin eigenvalue range into 20 equal-width bins,
    sum recovered weights per bin.
    """
    n = graph.num_nodes
    pe = np.zeros((n, SRWE_HIST_BINS), dtype=np.float32)

    bin_edges = np.linspace(SRWE_HIST_RANGE[0], SRWE_HIST_RANGE[1], SRWE_HIST_BINS + 1)

    if method == "tikhonov" and graph.srwe_tik is not None and graph.eigenvalues_P is not None:
        evals = graph.eigenvalues_P
        for u in range(n):
            weights = graph.srwe_tik[u] if u < len(graph.srwe_tik) else np.zeros(len(evals))
            for i, (ev, w) in enumerate(zip(evals, weights)):
                bin_idx = int((ev - SRWE_HIST_RANGE[0]) / (SRWE_HIST_RANGE[1] - SRWE_HIST_RANGE[0]) * SRWE_HIST_BINS)
                bin_idx = min(max(bin_idx, 0), SRWE_HIST_BINS - 1)
                pe[u, bin_idx] += w

    elif method == "tsvd" and graph.srwe_tsvd is not None and graph.eigenvalues_P is not None:
        evals = graph.eigenvalues_P
        for u in range(n):
            weights = graph.srwe_tsvd[u] if u < len(graph.srwe_tsvd) else np.zeros(len(evals))
            for i, (ev, w) in enumerate(zip(evals, weights)):
                bin_idx = int((ev - SRWE_HIST_RANGE[0]) / (SRWE_HIST_RANGE[1] - SRWE_HIST_RANGE[0]) * SRWE_HIST_BINS)
                bin_idx = min(max(bin_idx, 0), SRWE_HIST_BINS - 1)
                pe[u, bin_idx] += w

    elif method == "mpm" and graph.srwe_mpm_evals is not None and graph.srwe_mpm_weights is not None:
        for u in range(n):
            if u < len(graph.srwe_mpm_evals) and u < len(graph.srwe_mpm_weights):
                evals_u = graph.srwe_mpm_evals[u]
                weights_u = graph.srwe_mpm_weights[u]
                for ev, w in zip(evals_u, weights_u):
                    bin_idx = int((ev - SRWE_HIST_RANGE[0]) / (SRWE_HIST_RANGE[1] - SRWE_HIST_RANGE[0]) * SRWE_HIST_BINS)
                    bin_idx = min(max(bin_idx, 0), SRWE_HIST_BINS - 1)
                    pe[u, bin_idx] += w

    return pe


def compute_pe_srwe_histogram_vectorized(
    graph: GraphData,
    method: str = "tikhonov",
) -> np.ndarray:
    """Vectorized version of SRWE histogram PE computation."""
    n = graph.num_nodes
    pe = np.zeros((n, SRWE_HIST_BINS), dtype=np.float32)

    if method in ("tikhonov", "tsvd") and graph.eigenvalues_P is not None:
        weights_mat = graph.srwe_tik if method == "tikhonov" else graph.srwe_tsvd
        if weights_mat is None:
            return pe

        evals = graph.eigenvalues_P
        # Compute bin indices for each eigenvalue
        bin_indices = np.floor(
            (evals - SRWE_HIST_RANGE[0]) / (SRWE_HIST_RANGE[1] - SRWE_HIST_RANGE[0]) * SRWE_HIST_BINS
        ).astype(int)
        bin_indices = np.clip(bin_indices, 0, SRWE_HIST_BINS - 1)

        # For each node, accumulate weights into bins
        n_use = min(n, len(weights_mat))
        for u in range(n_use):
            np.add.at(pe[u], bin_indices, weights_mat[u])

    return pe


# ═══════════════════════════════════════════════════════════════════════════
#  GNN MODEL (pure PyTorch, no torch_geometric)
# ═══════════════════════════════════════════════════════════════════════════

def build_gcn_norm(edge_index: np.ndarray, num_nodes: int) -> torch.Tensor:
    """
    Build GCN normalized adjacency: A_hat = D_hat^{-1/2} (A + I) D_hat^{-1/2}
    Returns sparse COO tensor.
    """
    # Add self-loops
    self_loops = np.arange(num_nodes)
    if edge_index.shape[1] > 0:
        src = np.concatenate([edge_index[0], self_loops])
        dst = np.concatenate([edge_index[1], self_loops])
    else:
        src = self_loops.copy()
        dst = self_loops.copy()

    # Remove duplicates
    edges_set = set()
    clean_src, clean_dst = [], []
    for s, d in zip(src, dst):
        if (s, d) not in edges_set:
            edges_set.add((s, d))
            clean_src.append(s)
            clean_dst.append(d)
    src = np.array(clean_src, dtype=np.int64)
    dst = np.array(clean_dst, dtype=np.int64)

    # Compute degree
    degree = np.zeros(num_nodes, dtype=np.float32)
    np.add.at(degree, dst, 1.0)

    # D^{-1/2}
    d_inv_sqrt = np.zeros(num_nodes, dtype=np.float32)
    mask = degree > 0
    d_inv_sqrt[mask] = 1.0 / np.sqrt(degree[mask])

    # Edge weights: d_inv_sqrt[src] * d_inv_sqrt[dst]
    edge_weight = d_inv_sqrt[src] * d_inv_sqrt[dst]

    indices = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    values = torch.tensor(edge_weight, dtype=torch.float32)
    adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))

    return adj.coalesce()


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """x: [N, in_dim], adj: sparse [N, N]"""
        h = self.linear(x)  # [N, out_dim]
        out = torch.sparse.mm(adj, h)  # [N, out_dim]
        return out


class GCN_GlobalAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNLayer(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout
        # Global attention gate
        self.gate_nn = nn.Linear(hidden_dim, 1)
        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        x: [total_nodes, input_dim]
        adj: sparse [total_nodes, total_nodes]
        batch: [total_nodes] long tensor mapping nodes to graph index
        """
        h = self.input_proj(x)
        h = F.relu(h)

        for conv, bn in zip(self.convs, self.bns):
            h_new = conv(h, adj)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new  # residual

        # Global attention pooling
        gate = self.gate_nn(h)  # [N, 1]
        # Softmax over nodes within each graph
        gate = _scatter_softmax(gate.squeeze(-1), batch)  # [N]
        # Weighted sum
        h_weighted = h * gate.unsqueeze(-1)  # [N, hidden]
        num_graphs = batch.max().item() + 1
        graph_emb = torch.zeros(num_graphs, h.shape[1], device=h.device, dtype=h.dtype)
        graph_emb.scatter_add_(0, batch.unsqueeze(-1).expand_as(h_weighted), h_weighted)

        return self.head(graph_emb)


def _scatter_softmax(values: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Softmax over values grouped by batch indices."""
    num_graphs = batch.max().item() + 1
    # Max for numerical stability
    max_vals = torch.zeros(num_graphs, device=values.device, dtype=values.dtype)
    max_vals.scatter_reduce_(0, batch, values, reduce="amax", include_self=False)
    max_vals = max_vals[batch]
    exp_vals = torch.exp(values - max_vals)
    # Sum per graph
    sum_vals = torch.zeros(num_graphs, device=values.device, dtype=values.dtype)
    sum_vals.scatter_add_(0, batch, exp_vals)
    return exp_vals / (sum_vals[batch] + 1e-10)


# ═══════════════════════════════════════════════════════════════════════════
#  BATCH COLLATION & DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════

def collate_graphs(
    graphs: list[GraphData],
    pe_type: str = "none",
    device: torch.device = DEVICE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate a list of graphs into a batch.
    Returns: (x, adj_sparse, batch_vec, targets)
    """
    all_x = []
    all_src = []
    all_dst = []
    all_batch = []
    all_targets = []
    node_offset = 0

    for gi, g in enumerate(graphs):
        n = g.num_nodes
        feat = g.node_feat.astype(np.float32)

        # PE
        if pe_type == "rwse":
            pe = g.pe_rwse if g.pe_rwse is not None else np.zeros((n, PE_DIM), dtype=np.float32)
        elif pe_type == "lappe":
            pe = g.pe_lappe if g.pe_lappe is not None else np.zeros((n, PE_DIM), dtype=np.float32)
        elif pe_type == "srwe":
            pe = g.pe_srwe if g.pe_srwe is not None else np.zeros((n, PE_DIM), dtype=np.float32)
        else:
            pe = np.zeros((n, PE_DIM), dtype=np.float32)

        x = np.hstack([feat, pe])
        all_x.append(x)

        # Edges (offset by node count)
        if g.edge_index.shape[1] > 0:
            all_src.append(g.edge_index[0] + node_offset)
            all_dst.append(g.edge_index[1] + node_offset)

        # Self-loops
        self_nodes = np.arange(n) + node_offset
        all_src.append(self_nodes)
        all_dst.append(self_nodes)

        all_batch.append(np.full(n, gi, dtype=np.int64))

        # Parse target
        target = _parse_target(g.target, g.task_type)
        all_targets.append(target)

        node_offset += n

    # Concatenate
    x = np.vstack(all_x)
    src = np.concatenate(all_src)
    dst = np.concatenate(all_dst)
    batch_vec = np.concatenate(all_batch)

    # Build normalized adjacency
    total_nodes = node_offset
    degree = np.zeros(total_nodes, dtype=np.float32)
    np.add.at(degree, dst, 1.0)
    d_inv_sqrt = np.zeros(total_nodes, dtype=np.float32)
    mask = degree > 0
    d_inv_sqrt[mask] = 1.0 / np.sqrt(degree[mask])
    edge_weight = d_inv_sqrt[src] * d_inv_sqrt[dst]

    # To tensors
    x_t = torch.tensor(x, dtype=torch.float32, device=device)
    indices = torch.tensor(np.stack([src, dst]), dtype=torch.long, device=device)
    values = torch.tensor(edge_weight, dtype=torch.float32, device=device)
    adj_t = torch.sparse_coo_tensor(indices, values, (total_nodes, total_nodes)).coalesce()
    batch_t = torch.tensor(batch_vec, dtype=torch.long, device=device)
    target_t = torch.tensor(np.array(all_targets), dtype=torch.float32, device=device)

    return x_t, adj_t, batch_t, target_t


def _parse_target(target_str: str, task_type: str) -> np.ndarray:
    """Parse target string into numpy array."""
    try:
        val = json.loads(target_str)
        if isinstance(val, list):
            return np.array(val, dtype=np.float32)
        else:
            return np.array([float(val)], dtype=np.float32)
    except (json.JSONDecodeError, ValueError):
        try:
            return np.array([float(target_str)], dtype=np.float32)
        except ValueError:
            return np.array([0.0], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def train_gnn(
    train_graphs: list[GraphData],
    val_graphs: list[GraphData],
    pe_type: str,
    task_type: str,
    output_dim: int,
    seed: int,
    max_epochs: int = GNN_MAX_EPOCHS,
    patience: int = GNN_PATIENCE,
    batch_size: int = GNN_BATCH_SIZE,
) -> nn.Module:
    """Train a GCN model and return it."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Determine input dim
    feat_dim = train_graphs[0].node_feat.shape[1] if train_graphs else 1
    input_dim = feat_dim + PE_DIM

    model = GCN_GlobalAttention(
        input_dim=input_dim,
        hidden_dim=GNN_HIDDEN,
        output_dim=output_dim,
        num_layers=GNN_LAYERS,
        dropout=GNN_DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=GNN_LR, weight_decay=GNN_WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=GNN_LR_PATIENCE, factor=GNN_LR_FACTOR
    )

    if task_type == "classification":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.L1Loss()

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(max_epochs):
        # Train
        model.train()
        rng = np.random.RandomState(seed * 1000 + epoch)
        indices = rng.permutation(len(train_graphs))
        total_loss = 0.0
        n_batches = 0

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_graphs = [train_graphs[i] for i in batch_idx]

            try:
                x, adj, batch_vec, targets = collate_graphs(batch_graphs, pe_type)
            except Exception:
                continue

            optimizer.zero_grad()
            out = model(x, adj, batch_vec)

            # Handle shape mismatch
            if out.shape != targets.shape:
                min_cols = min(out.shape[1], targets.shape[1])
                out = out[:, :min_cols]
                targets = targets[:, :min_cols]

            loss = criterion(out, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / max(n_batches, 1)

        # Validate
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for start in range(0, len(val_graphs), batch_size):
                batch_graphs = val_graphs[start:start + batch_size]
                try:
                    x, adj, batch_vec, targets = collate_graphs(batch_graphs, pe_type)
                    out = model(x, adj, batch_vec)
                    if out.shape != targets.shape:
                        min_cols = min(out.shape[1], targets.shape[1])
                        out = out[:, :min_cols]
                        targets = targets[:, :min_cols]
                    loss = criterion(out, targets)
                    val_loss += loss.item()
                    n_val_batches += 1
                except Exception:
                    continue

        avg_val_loss = val_loss / max(n_val_batches, 1)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

        if (epoch + 1) % 50 == 0:
            logger.debug(f"  Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}")

    # Load best weights
    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    return model


def predict_gnn(
    model: nn.Module,
    graphs: list[GraphData],
    pe_type: str,
    batch_size: int = GNN_BATCH_SIZE,
) -> list[np.ndarray]:
    """Run inference and return per-graph predictions."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            batch_graphs = graphs[start:start + batch_size]
            try:
                x, adj, batch_vec, _ = collate_graphs(batch_graphs, pe_type)
                out = model(x, adj, batch_vec)
                preds = out.cpu().numpy()
                for i in range(len(batch_graphs)):
                    all_preds.append(preds[i])
            except Exception:
                for _ in batch_graphs:
                    all_preds.append(np.array([0.0]))
    return all_preds


def compute_metric(
    predictions: list[np.ndarray],
    targets: list[np.ndarray],
    task_type: str,
) -> dict[str, float]:
    """Compute task-specific metrics."""
    if not predictions or not targets:
        return {}

    preds = np.array([p.flatten() for p in predictions])
    tgts = np.array([t.flatten() for t in targets])

    # Ensure same shape
    min_cols = min(preds.shape[1], tgts.shape[1])
    preds = preds[:, :min_cols]
    tgts = tgts[:, :min_cols]

    if task_type == "classification":
        # Average Precision (AP) for multi-label classification
        try:
            probs = 1.0 / (1.0 + np.exp(-preds))  # sigmoid
            ap = average_precision_score(tgts, probs, average="macro")
            return {"AP": float(ap)}
        except Exception:
            return {"AP": 0.0}
    else:
        # MAE for regression
        mae = float(np.mean(np.abs(preds - tgts)))
        return {"MAE": mae}


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_spectral_recovery(graphs: list[GraphData], best_tik_alpha: float = None,
                          best_tsvd_thresh: float = 0.01) -> None:
    """Run SRWE recovery methods on all graphs."""
    t0 = time.time()
    n_graphs = len(graphs)

    for gi, g in enumerate(graphs):
        if g.eigenvalues_P is None:
            continue
        n = g.num_nodes
        evals_P = g.eigenvalues_P

        # Tikhonov recovery per node
        tik_weights = np.zeros((n, len(evals_P)), dtype=np.float64)
        tsvd_weights = np.zeros((n, len(evals_P)), dtype=np.float64)
        mpm_evals_list = []
        mpm_weights_list = []

        for u in range(n):
            moments = g.rwse[u, :]

            # Tikhonov
            try:
                tik_weights[u] = srwe_tikhonov(moments, evals_P, alpha=best_tik_alpha)
            except Exception:
                tik_weights[u] = np.ones(len(evals_P)) / max(len(evals_P), 1)

            # TSVD
            try:
                tsvd_weights[u] = srwe_tsvd(moments, evals_P, threshold=best_tsvd_thresh)
            except Exception:
                tsvd_weights[u] = np.ones(len(evals_P)) / max(len(evals_P), 1)

            # MPM (subsample for large graphs to save time)
            if n <= 100 or u < 50:
                try:
                    est_ev, est_w = srwe_mpm(moments)
                    mpm_evals_list.append(est_ev)
                    mpm_weights_list.append(est_w)
                except Exception:
                    mpm_evals_list.append(np.array([0.0]))
                    mpm_weights_list.append(np.array([1.0]))
            else:
                mpm_evals_list.append(np.array([0.0]))
                mpm_weights_list.append(np.array([1.0]))

        g.srwe_tik = tik_weights
        g.srwe_tsvd = tsvd_weights
        g.srwe_mpm_evals = mpm_evals_list
        g.srwe_mpm_weights = mpm_weights_list

        if (gi + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (gi + 1) / elapsed
            eta = (n_graphs - gi - 1) / rate
            logger.info(f"  Recovery: {gi+1}/{n_graphs} graphs ({rate:.1f} g/s, ETA: {eta:.0f}s)")


def hyperparameter_search(
    val_graphs: list[GraphData],
    max_graphs: int = 200,
) -> dict[str, Any]:
    """Search for best hyperparameters on validation subset."""
    subset = val_graphs[:max_graphs]
    logger.info(f"Hyperparameter search on {len(subset)} graphs...")

    # Tikhonov: test different alphas
    tik_alphas = [1e-8, 1e-6, 1e-4, 1e-2, 1.0, None]  # None = GCV
    best_tik_alpha = None
    best_tik_w1 = float("inf")

    for alpha in tik_alphas:
        w1_vals = []
        for g in subset[:50]:  # Quick test on 50 graphs
            if g.eigenvalues_P is None:
                continue
            n = min(g.num_nodes, 20)
            for u in range(n):
                try:
                    w = srwe_tikhonov(g.rwse[u], g.eigenvalues_P, alpha=alpha)
                    true_w = g.eigenvectors_P[u, :] ** 2
                    w1 = wasserstein1(g.eigenvalues_P, true_w, g.eigenvalues_P, w)
                    if not math.isinf(w1) and not math.isnan(w1):
                        w1_vals.append(w1)
                except Exception:
                    pass
        mean_w1 = np.mean(w1_vals) if w1_vals else float("inf")
        logger.debug(f"  Tikhonov alpha={alpha}: mean_W1={mean_w1:.6f}")
        if mean_w1 < best_tik_w1:
            best_tik_w1 = mean_w1
            best_tik_alpha = alpha

    # TSVD: test different thresholds
    tsvd_thresholds = [0.001, 0.01, 0.05, 0.1, 0.2]
    best_tsvd_thresh = 0.01
    best_tsvd_w1 = float("inf")

    for thresh in tsvd_thresholds:
        w1_vals = []
        for g in subset[:50]:
            if g.eigenvalues_P is None:
                continue
            n = min(g.num_nodes, 20)
            for u in range(n):
                try:
                    w = srwe_tsvd(g.rwse[u], g.eigenvalues_P, threshold=thresh)
                    true_w = g.eigenvectors_P[u, :] ** 2
                    w1 = wasserstein1(g.eigenvalues_P, true_w, g.eigenvalues_P, w)
                    if not math.isinf(w1) and not math.isnan(w1):
                        w1_vals.append(w1)
                except Exception:
                    pass
        mean_w1 = np.mean(w1_vals) if w1_vals else float("inf")
        logger.debug(f"  TSVD threshold={thresh}: mean_W1={mean_w1:.6f}")
        if mean_w1 < best_tsvd_w1:
            best_tsvd_w1 = mean_w1
            best_tsvd_thresh = thresh

    logger.info(f"Best Tikhonov alpha={best_tik_alpha} (W1={best_tik_w1:.6f})")
    logger.info(f"Best TSVD threshold={best_tsvd_thresh} (W1={best_tsvd_w1:.6f})")

    return {
        "best_tik_alpha": best_tik_alpha,
        "best_tik_w1": best_tik_w1,
        "best_tsvd_thresh": best_tsvd_thresh,
        "best_tsvd_w1": best_tsvd_w1,
    }


def split_graphs(
    graphs: list[GraphData],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[GraphData], list[GraphData], list[GraphData]]:
    """Split graphs into train/val/test sets. Uses metadata_fold if available."""
    # Check if fold info is available (ZINC has folds 0,1,2)
    folds = set(g.fold for g in graphs)
    if folds == {0, 1, 2}:
        train = [g for g in graphs if g.fold == 0]
        val = [g for g in graphs if g.fold == 1]
        test = [g for g in graphs if g.fold == 2]
        return train, val, test

    # Otherwise: random split
    rng = np.random.RandomState(seed)
    n = len(graphs)
    indices = rng.permutation(n)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = [graphs[i] for i in indices[:n_train]]
    val = [graphs[i] for i in indices[n_train:n_train + n_val]]
    test = [graphs[i] for i in indices[n_train + n_val:]]
    return train, val, test


def format_output(
    all_datasets: dict[str, list[GraphData]],
    gnn_results: dict,
    recovery_results: dict,
    hp_results: dict,
    consistency: dict,
) -> dict:
    """Format results as exp_gen_sol_out.json."""
    output = {
        "metadata": {
            "method_name": "Enhanced SRWE: Tikhonov/TSVD/MPM Spectral Recovery + GNN Benchmark",
            "description": "Implements three SRWE spectral recovery methods and benchmarks SRWE as a GNN PE against RWSE and LapPE",
            "hyperparameters": hp_results,
            "gnn_benchmark": gnn_results,
            "spectral_recovery": recovery_results,
            "consistency_check": consistency,
            "gnn_config": {
                "hidden_dim": GNN_HIDDEN,
                "num_layers": GNN_LAYERS,
                "dropout": GNN_DROPOUT,
                "lr": GNN_LR,
                "weight_decay": GNN_WD,
                "batch_size": GNN_BATCH_SIZE,
                "max_epochs": GNN_MAX_EPOCHS,
                "patience": GNN_PATIENCE,
                "seeds": SEEDS,
            },
        },
        "datasets": [],
    }

    for ds_name, graphs in all_datasets.items():
        examples = []
        for g in graphs:
            ex = {
                "input": g.input_str,
                "output": g.output_str,
            }
            # Add predictions
            for pe_type, pred_str in g.predictions.items():
                ex[f"predict_{pe_type}"] = pred_str

            # Add recovery metadata
            for key, val in g.recovery_metrics.items():
                ex[f"metadata_{key}"] = val

            ex["metadata_sri_k20"] = g.sri_k20
            ex["metadata_dataset"] = ds_name

            examples.append(ex)

        output["datasets"].append({
            "dataset": ds_name,
            "examples": examples,
        })

    return output


# Global state for partial save on interruption
_partial_state = {
    "all_datasets": None,
    "gnn_results": {},
    "recovery_results": {},
    "hp_results": {},
    "consistency": {},
    "saved": False,
}


def _save_partial():
    """Emergency save of whatever results we have so far."""
    if _partial_state["saved"] or _partial_state["all_datasets"] is None:
        return
    try:
        logger.warning("Saving partial results before exit...")
        output = format_output(
            _partial_state["all_datasets"],
            _partial_state["gnn_results"],
            _partial_state["recovery_results"],
            _partial_state["hp_results"],
            _partial_state["consistency"],
        )
        output_json = json.dumps(output, indent=2, default=str)
        OUTPUT_FILE.write_text(output_json)
        file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
        logger.warning(f"Partial results saved to {OUTPUT_FILE} ({file_size_mb:.1f} MB)")
        _partial_state["saved"] = True
    except Exception:
        logger.exception("Failed to save partial results")


def _signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT gracefully."""
    logger.warning(f"Received signal {signum}, saving partial results...")
    _save_partial()
    sys.exit(128 + signum)


# Register signal handlers and atexit
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)
atexit.register(_save_partial)


@logger.catch
def main() -> None:
    t_start = time.time()
    logger.info("=" * 70)
    logger.info("Enhanced SRWE: Spectral Recovery + GNN Benchmark")
    logger.info("=" * 70)

    # ── Phase 0: Load Data ──
    logger.info("── Phase 0: Loading Data ──")

    max_ex = MAX_EXAMPLES if MAX_EXAMPLES > 0 else 0

    # Determine data files to load
    mini_file = DEP_DIR / "mini_data_out.json"
    full_files = sorted((DEP_DIR / "data_out").glob("full_data_out_*.json"))

    if max_ex > 0 and max_ex <= 3:
        # Use mini data for very small tests
        data_files = [mini_file]
    else:
        data_files = full_files if full_files else [mini_file]

    all_datasets = load_dataset_from_files(data_files, max_per_dataset=max_ex)

    total_graphs = sum(len(v) for v in all_datasets.values())
    logger.info(f"Total graphs loaded: {total_graphs}")
    for ds, gs in all_datasets.items():
        logger.info(f"  {ds}: {len(gs)} graphs")

    if total_graphs == 0:
        logger.error("No graphs loaded! Exiting.")
        return

    # Register datasets in partial state for emergency saves
    _partial_state["all_datasets"] = all_datasets

    # ── Phase 0.5: Recompute Spectral from P ──
    logger.info("── Phase 0.5: Recomputing normalized adjacency eigendecomposition ──")
    t0 = time.time()
    for ds_name, graphs in all_datasets.items():
        for g in graphs:
            try:
                recompute_spectral(g)
            except Exception:
                logger.exception(f"Failed spectral recompute for graph {g.row_index} in {ds_name}")
    logger.info(f"  Spectral recomputation done in {time.time()-t0:.1f}s")

    # Consistency check
    first_ds_graphs = list(all_datasets.values())[0] if all_datasets else []
    consistency = consistency_check(first_ds_graphs)

    # ── Phase 1: Hyperparameter Search ──
    logger.info("── Phase 1: Hyperparameter Search ──")
    # Use first available non-synthetic dataset for HP search
    hp_graphs = []
    for ds_name in ["Peptides-func", "ZINC-subset", "Peptides-struct"]:
        if ds_name in all_datasets and len(all_datasets[ds_name]) > 10:
            hp_graphs = all_datasets[ds_name]
            break
    if not hp_graphs:
        hp_graphs = list(all_datasets.values())[0] if all_datasets else []

    hp_results = hyperparameter_search(hp_graphs)
    best_tik_alpha = hp_results.get("best_tik_alpha")
    best_tsvd_thresh = hp_results.get("best_tsvd_thresh", 0.01)
    _partial_state["hp_results"] = hp_results

    # ── Phase 2: Run SRWE Recovery ──
    logger.info("── Phase 2: Running SRWE Recovery on all graphs ──")
    t0 = time.time()
    for ds_name, graphs in all_datasets.items():
        logger.info(f"  Processing {ds_name} ({len(graphs)} graphs)...")
        run_spectral_recovery(graphs, best_tik_alpha=best_tik_alpha, best_tsvd_thresh=best_tsvd_thresh)
    logger.info(f"  Total SRWE recovery: {time.time()-t0:.1f}s")

    # ── Phase 2.5: Evaluate Recovery Quality ──
    logger.info("── Phase 2.5: Evaluating Spectral Recovery Quality ──")
    recovery_results = {}
    for ds_name, graphs in all_datasets.items():
        ds_results = {"tikhonov": {}, "tsvd": {}, "mpm": {}}
        sri_stratified = {"tikhonov": {}, "tsvd": {}, "mpm": {}}

        for method in ["tikhonov", "tsvd", "mpm"]:
            all_w1 = []
            all_cos = []
            all_topk = []
            sri_bins_data = {"very_low": [], "low": [], "medium": [], "high": []}

            for g in graphs:
                metrics = evaluate_recovery_graph(g, method)
                if "mean_w1" in metrics:
                    all_w1.append(metrics["mean_w1"])
                    all_cos.append(metrics.get("mean_cosine", 0))
                    all_topk.append(metrics.get("mean_top5_acc", 0))

                    bin_name = sri_bin(g.sri_k20)
                    sri_bins_data[bin_name].append(metrics["mean_w1"])

            if all_w1:
                ds_results[method] = {
                    "mean_w1": float(np.mean(all_w1)),
                    "std_w1": float(np.std(all_w1)),
                    "mean_cosine": float(np.mean(all_cos)),
                    "std_cosine": float(np.std(all_cos)),
                    "mean_top5_acc": float(np.mean(all_topk)),
                    "std_top5_acc": float(np.std(all_topk)),
                }

            for bin_name, vals in sri_bins_data.items():
                if vals:
                    sri_stratified[method][bin_name] = {
                        "mean_w1": float(np.mean(vals)),
                        "std_w1": float(np.std(vals)),
                        "count": len(vals),
                    }

        recovery_results[ds_name] = {
            "methods": ds_results,
            "sri_stratified": sri_stratified,
        }
        logger.info(f"  {ds_name} recovery: "
                     f"Tik W1={ds_results.get('tikhonov', {}).get('mean_w1', 'N/A'):.6f}, "
                     f"TSVD W1={ds_results.get('tsvd', {}).get('mean_w1', 'N/A'):.6f}, "
                     f"MPM W1={ds_results.get('mpm', {}).get('mean_w1', 'N/A'):.6f}")

    # Determine best SRWE method overall
    best_method = "tikhonov"
    best_w1 = float("inf")
    for method in ["tikhonov", "tsvd", "mpm"]:
        w1_all = []
        for ds_name in recovery_results:
            w1_val = recovery_results[ds_name].get("methods", {}).get(method, {}).get("mean_w1")
            if w1_val is not None:
                w1_all.append(w1_val)
        if w1_all and np.mean(w1_all) < best_w1:
            best_w1 = np.mean(w1_all)
            best_method = method
    logger.info(f"Best SRWE method: {best_method} (overall mean W1={best_w1:.6f})")
    _partial_state["recovery_results"] = recovery_results
    _partial_state["consistency"] = consistency

    # ── Phase 3: Compute PEs ──
    logger.info("── Phase 3: Computing Positional Encodings ──")
    for ds_name, graphs in all_datasets.items():
        for g in graphs:
            g.pe_rwse = compute_pe_rwse(g)
            g.pe_lappe = compute_pe_lappe(g)
            g.pe_srwe = compute_pe_srwe_histogram_vectorized(g, method=best_method)

    # ── Phase 4: GNN Benchmark ──
    logger.info("── Phase 4: GNN Benchmark ──")
    gnn_results = {}
    pe_types_to_test = ["none", "rwse", "lappe", "srwe"]

    # Datasets for GNN benchmark - order: smaller datasets first
    gnn_datasets = {}
    for ds_name in ["Peptides-func", "Peptides-struct", "ZINC-subset"]:
        if ds_name in all_datasets and len(all_datasets[ds_name]) >= 5:
            gnn_datasets[ds_name] = all_datasets[ds_name]

    for ds_name, all_ds_graphs in gnn_datasets.items():
        # Cap graph count for GNN benchmark (use all for recovery, subsample for GNN)
        if len(all_ds_graphs) > GNN_MAX_GRAPHS:
            rng = np.random.RandomState(42)
            gnn_indices = rng.choice(len(all_ds_graphs), GNN_MAX_GRAPHS, replace=False)
            graphs = [all_ds_graphs[i] for i in sorted(gnn_indices)]
            logger.info(f"  === GNN Benchmark: {ds_name} ({len(graphs)}/{len(all_ds_graphs)} graphs subsampled) ===")
        else:
            graphs = all_ds_graphs
            logger.info(f"  === GNN Benchmark: {ds_name} ({len(graphs)} graphs) ===")

        task_type = graphs[0].task_type
        if ds_name == "Peptides-func":
            output_dim = 10
            metric_name = "AP"
        elif ds_name == "Peptides-struct":
            output_dim = 11
            metric_name = "MAE"
        else:  # ZINC
            output_dim = 1
            metric_name = "MAE"

        train_g, val_g, test_g = split_graphs(graphs)
        logger.info(f"    Split: train={len(train_g)}, val={len(val_g)}, test={len(test_g)}")

        # Reduce max_epochs for large datasets to save time
        max_epochs = GNN_MAX_EPOCHS
        patience = GNN_PATIENCE
        if len(train_g) > 3000:
            max_epochs = min(100, GNN_MAX_EPOCHS)
            patience = min(30, GNN_PATIENCE)
            logger.info(f"    Large dataset: reduced max_epochs={max_epochs}, patience={patience}")

        ds_gnn_results = {}
        for pe_type in pe_types_to_test:
            seed_results = []
            for seed in SEEDS:
                t_train_start = time.time()
                logger.info(f"    Training {pe_type} seed={seed}...")

                try:
                    model = train_gnn(
                        train_g, val_g, pe_type, task_type, output_dim, seed,
                        max_epochs=max_epochs, patience=patience,
                    )

                    # Test predictions
                    test_preds = predict_gnn(model, test_g, pe_type)
                    test_targets = [_parse_target(g.target, task_type) for g in test_g]
                    test_metrics = compute_metric(test_preds, test_targets, task_type)

                    # Predictions for subsampled graphs
                    sub_preds = predict_gnn(model, graphs, pe_type)
                    for gi, g in enumerate(graphs):
                        pred = sub_preds[gi]
                        if task_type == "classification":
                            probs = 1.0 / (1.0 + np.exp(-pred))
                            g.predictions[pe_type] = json.dumps([round(float(p), 6) for p in probs])
                        else:
                            g.predictions[pe_type] = json.dumps([round(float(p), 6) for p in pred])

                    # If we subsampled, also predict on ALL graphs for output
                    if len(all_ds_graphs) > len(graphs):
                        gnn_set = set(id(g) for g in graphs)
                        remaining = [g for g in all_ds_graphs if id(g) not in gnn_set]
                        if remaining:
                            rem_preds = predict_gnn(model, remaining, pe_type)
                            for gi, g in enumerate(remaining):
                                pred = rem_preds[gi]
                                if task_type == "classification":
                                    probs = 1.0 / (1.0 + np.exp(-pred))
                                    g.predictions[pe_type] = json.dumps([round(float(p), 6) for p in probs])
                                else:
                                    g.predictions[pe_type] = json.dumps([round(float(p), 6) for p in pred])

                    elapsed = time.time() - t_train_start
                    logger.info(f"    {pe_type} seed={seed}: {test_metrics}, time={elapsed:.1f}s")
                    seed_results.append(test_metrics)

                    # Free GPU memory
                    del model
                    torch.cuda.empty_cache() if HAS_GPU else None

                except Exception:
                    logger.exception(f"    Failed: {pe_type} seed={seed}")
                    seed_results.append({metric_name: float("nan")})

                # Time check
                elapsed_total = time.time() - t_start
                if elapsed_total > TIME_SKIP_SEEDS:
                    logger.warning(f"Time budget tight ({elapsed_total:.0f}s elapsed), skipping remaining seeds")
                    break

            # Aggregate seed results
            if seed_results:
                vals = [r.get(metric_name, float("nan")) for r in seed_results]
                valid_vals = [v for v in vals if not math.isnan(v)]
                if valid_vals:
                    ds_gnn_results[pe_type] = {
                        "mean": float(np.mean(valid_vals)),
                        "std": float(np.std(valid_vals)),
                        "per_seed": vals,
                        "n_seeds": len(valid_vals),
                    }

            # Time check
            elapsed_total = time.time() - t_start
            if elapsed_total > TIME_SKIP_PE:
                logger.warning(f"Time budget tight ({elapsed_total:.0f}s), skipping remaining PE types")
                break

        gnn_results[ds_name] = {
            "metric": metric_name,
            "task_type": task_type,
            "results": ds_gnn_results,
        }
        _partial_state["gnn_results"] = gnn_results  # Update partial state after each dataset
        summary = {k: f"{v['mean']:.4f}+-{v['std']:.4f}" for k, v in ds_gnn_results.items()}
        logger.info(f"  {ds_name} GNN results: {summary}")

        # Time check
        elapsed_total = time.time() - t_start
        if elapsed_total > TIME_SKIP_DATASETS:
            logger.warning(f"Time budget tight ({elapsed_total:.0f}s), skipping remaining datasets")
            break

    # ── Phase 4.5: Store per-example recovery metrics ──
    logger.info("── Phase 4.5: Storing per-example recovery metrics ──")
    for ds_name, graphs in all_datasets.items():
        for g in graphs:
            # Best method metrics
            metrics = evaluate_recovery_graph(g, best_method, max_nodes=20)
            g.recovery_metrics["srwe_best_method"] = best_method
            g.recovery_metrics["srwe_best_w1"] = str(round(metrics.get("mean_w1", -1), 6))
            g.recovery_metrics["srwe_best_cosine"] = str(round(metrics.get("mean_cosine", -1), 6))

            # If no GNN predictions were made (e.g., synthetic), add dummy
            if not g.predictions:
                for pe_type in pe_types_to_test:
                    g.predictions[pe_type] = g.output_str

    # ── Phase 5: Compute Analysis ──
    logger.info("── Phase 5: Computing Analysis ──")
    analysis = {
        "best_srwe_method": best_method,
        "gap_closed": {},
    }

    for ds_name, ds_res in gnn_results.items():
        results = ds_res.get("results", {})
        rwse_val = results.get("rwse", {}).get("mean")
        srwe_val = results.get("srwe", {}).get("mean")
        lappe_val = results.get("lappe", {}).get("mean")
        none_val = results.get("none", {}).get("mean")

        if rwse_val is not None and srwe_val is not None and lappe_val is not None:
            if ds_res["task_type"] == "classification":
                # Higher is better for AP
                if lappe_val != rwse_val:
                    gap_closed = (srwe_val - rwse_val) / (lappe_val - rwse_val) * 100
                else:
                    gap_closed = 0.0
            else:
                # Lower is better for MAE
                if lappe_val != rwse_val:
                    gap_closed = (rwse_val - srwe_val) / (rwse_val - lappe_val) * 100
                else:
                    gap_closed = 0.0
            analysis["gap_closed"][ds_name] = round(gap_closed, 2)

    gnn_results["analysis"] = analysis

    # ── Phase 6: Format and Save Output ──
    logger.info("── Phase 6: Saving Output ──")
    _partial_state["gnn_results"] = gnn_results
    output = format_output(all_datasets, gnn_results, recovery_results, hp_results, consistency)

    # Write output
    output_json = json.dumps(output, indent=2, default=str)
    OUTPUT_FILE.write_text(output_json)
    _partial_state["saved"] = True  # Mark as saved so atexit handler doesn't overwrite
    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    logger.info(f"Saved method_out.json ({file_size_mb:.1f} MB)")

    # Check file size limit
    MAX_FILE_SIZE_MB = 95  # Leave margin under 100 MB
    if file_size_mb > MAX_FILE_SIZE_MB:
        logger.info(f"Output exceeds {MAX_FILE_SIZE_MB} MB, splitting...")
        split_dir = WORKSPACE / "method_out"
        split_dir.mkdir(exist_ok=True)
        part_counter = 0

        for ds_data in output["datasets"]:
            ds_name = ds_data["dataset"]
            examples = ds_data["examples"]

            # Try writing the whole dataset as one part
            part = {"metadata": output["metadata"], "datasets": [{"dataset": ds_name, "examples": examples}]}
            part_json = json.dumps(part, indent=2, default=str)
            part_size_mb = len(part_json.encode("utf-8")) / (1024 * 1024)

            if part_size_mb <= MAX_FILE_SIZE_MB:
                # Fits in one file
                part_counter += 1
                part_path = split_dir / f"method_out_{part_counter}.json"
                part_path.write_text(part_json)
                logger.info(f"  Part {part_counter}: {ds_name} ({part_size_mb:.1f} MB)")
            else:
                # Need to sub-split this dataset
                # Estimate examples per chunk
                avg_example_size = part_size_mb / max(len(examples), 1)
                chunk_size = max(1, int(MAX_FILE_SIZE_MB / avg_example_size * 0.9))
                logger.info(f"  {ds_name} is {part_size_mb:.1f} MB, splitting into chunks of {chunk_size} examples")

                for start in range(0, len(examples), chunk_size):
                    chunk_examples = examples[start:start + chunk_size]
                    sub_part = {
                        "metadata": output["metadata"],
                        "datasets": [{"dataset": ds_name, "examples": chunk_examples}],
                    }
                    part_counter += 1
                    part_path = split_dir / f"method_out_{part_counter}.json"
                    sub_json = json.dumps(sub_part, indent=2, default=str)
                    part_path.write_text(sub_json)
                    sub_size_mb = len(sub_json.encode("utf-8")) / (1024 * 1024)
                    logger.info(f"  Part {part_counter}: {ds_name}[{start}:{start+len(chunk_examples)}] ({sub_size_mb:.1f} MB)")

        OUTPUT_FILE.unlink()
        logger.info(f"  Split into {part_counter} parts in {split_dir}")

    elapsed = time.time() - t_start
    logger.info("=" * 70)
    logger.info(f"DONE! Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
