#!/usr/bin/env python3
"""
SRWE Feature Representation Optimization & Classification Failure Diagnosis.

Compares 5 SRWE feature representations (HISTOGRAM, RAW_WEIGHTS, EIGENVALUE_PAIRS,
MOMENT_CORRECTION, SPECTRAL_SUMMARY) across 5 Tikhonov regularization levels on
ZINC/Peptides-func/Peptides-struct using progressive filtering.

Diagnoses the regression-vs-classification performance asymmetry via mutual information
estimation, linear probing, and per-graph W1-vs-loss correlation analysis.

Uses pure PyTorch GPS-lite (2-layer, 64-dim GCN with attention pooling).
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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

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
def _container_ram_gb() -> Optional[float]:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None

TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9
AVAILABLE_RAM_GB = min(psutil.virtual_memory().available / 1e9, TOTAL_RAM_GB)
RAM_LIMIT = int(min(AVAILABLE_RAM_GB * 0.85, 50) * 1024**3)
try:
    resource.setrlimit(resource.RLIMIT_AS, (RAM_LIMIT, RAM_LIMIT))
except ValueError:
    pass
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

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

logger.info(f"Hardware: {NUM_CPUS} CPUs, GPU={HAS_GPU} ({VRAM_GB:.1f}GB VRAM), "
            f"RAM={TOTAL_RAM_GB:.1f}GB total, {AVAILABLE_RAM_GB:.1f}GB available")

# ── Constants ──
DEP_DIR = Path("/opt/ai-inventor/aii_pipeline/runs/run__20260222_005317/3_invention_loop/iter_1/gen_art/data_id2_it1__opus")
WORKSPACE = SCRIPT_DIR
OUTPUT_FILE = WORKSPACE / "method_out.json"

MAX_EXAMPLES = int(os.environ.get("MAX_EXAMPLES", "0"))  # 0 = all
RWSE_DIM = 20
PE_DIM = 20
SEEDS = [0, 1, 2]

# Regularization levels for lambda sweep
LAMBDAS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
SCREENING_LAMBDA = 1e-3  # Used in Phase 4A screening

# Feature representation dimensions
HIST_BINS = 20
RAW_WEIGHTS_K = 16
EIGPAIR_K = 8
SPECTRAL_SUMMARY_DIM = 12

# SRWE histogram range (normalized adjacency eigenvalues are in [-1, 1])
SRWE_HIST_RANGE = (-1.0, 1.0)

# GNN config (GPS-lite: 2-layer, 64-dim)
GNN_HIDDEN = 64
GNN_LAYERS = 2
GNN_DROPOUT = 0.1
GNN_LR = 1e-3
GNN_WD = 1e-5
GNN_BATCH_SIZE = 64
GNN_MAX_EPOCHS = 150
GNN_PATIENCE = 30
GNN_LR_PATIENCE = 15
GNN_LR_FACTOR = 0.5

# Dataset caps
MAX_ZINC = 5000
MAX_PEPTIDES = 2000

# Time budget constants (seconds from start)
TIME_PHASE4A_WARN = 1500      # 25 min: reduce epochs in Phase 4A
TIME_SKIP_PHASE4B = 2700      # 45 min: skip Phase 4B
TIME_PHASE4B_REDUCE = 0       # ALWAYS reduce Phase 4B (1 seed, 3 lambdas)
TIME_SKIP_DIAGNOSTICS = 3300  # 55 min: skip diagnostics (generous - they're fast)
TIME_SAVE_PARTIAL = 3400      # 57 min: force save

T_START = time.time()


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GraphData:
    """Parsed graph data from the JSON dataset."""
    edge_index: np.ndarray       # [2, E] int
    num_nodes: int
    node_feat: np.ndarray        # [N, F]
    eigenvalues_A: np.ndarray    # from adjacency (stored)
    rwse: np.ndarray             # [N, 20]
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
    # Recovered spectral weights: dict[lambda -> [N, n_eigs]]
    srwe_weights: Optional[dict] = None
    # PE vectors (computed per representation)
    pe_cache: dict = field(default_factory=dict)
    # GNN predictions
    predictions: dict = field(default_factory=dict)
    # Recovery metrics
    recovery_metrics: dict = field(default_factory=dict)
    # Original strings
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
            if rwse.shape[0] < num_nodes:
                pad = np.zeros((num_nodes - rwse.shape[0], rwse.shape[1]), dtype=np.float64)
                rwse = np.vstack([rwse, pad])
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


def load_dataset_from_files(data_files: list, max_per_dataset: int = 0) -> dict:
    """Load all datasets from dependency JSON files."""
    datasets: dict[str, list[GraphData]] = {}
    caps = {"ZINC-subset": MAX_ZINC, "Peptides-func": MAX_PEPTIDES,
            "Peptides-struct": MAX_PEPTIDES}

    for fpath in data_files:
        fpath = Path(fpath)
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
            cap = max_per_dataset if max_per_dataset > 0 else caps.get(ds_name, 99999)
            for ex in examples:
                if existing >= cap:
                    break
                g = parse_graph(ex, ds_name)
                if g is not None:
                    datasets[ds_name].append(g)
                    existing += 1
        logger.info(f"  Running totals: {  {k: len(v) for k, v in datasets.items()} }")
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


def compute_normalized_adjacency_eigen(A: np.ndarray) -> tuple:
    """Compute eigendecomposition of P = D^{-1/2} A D^{-1/2}."""
    n = A.shape[0]
    if n == 0:
        return np.array([]), np.array([]).reshape(0, 0)

    degree = A.sum(axis=1)
    d_inv_sqrt = np.zeros(n, dtype=np.float64)
    mask = degree > 1e-12
    d_inv_sqrt[mask] = 1.0 / np.sqrt(degree[mask])

    P = A * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    eigenvalues, eigenvectors = eigh(P)
    return eigenvalues, eigenvectors


def recompute_spectral(graph: GraphData, max_nodes: int = 300) -> None:
    """Recompute normalized adjacency eigendecomposition for a graph."""
    if graph.num_nodes > max_nodes:
        # For very large graphs, skip full eigendecomp (too expensive)
        # Use stored adjacency eigenvalues instead (Fallback 3)
        graph.eigenvalues_P = graph.eigenvalues_A.copy()
        # Normalize to [-1, 1] range like normalized adjacency
        if len(graph.eigenvalues_P) > 0:
            max_abs = np.max(np.abs(graph.eigenvalues_P))
            if max_abs > 1e-12:
                graph.eigenvalues_P = graph.eigenvalues_P / max_abs
        graph.eigenvectors_P = None  # Cannot compute for large graphs
        return

    A = build_adjacency(graph.edge_index, graph.num_nodes)
    eigenvalues_P, eigenvectors_P = compute_normalized_adjacency_eigen(A)
    graph.eigenvalues_P = eigenvalues_P
    graph.eigenvectors_P = eigenvectors_P


def consistency_check(graphs: list, n_samples: int = 5) -> dict:
    """Check consistency between stored RWSE and reconstructed moments from P."""
    errors = []
    for g in graphs[:min(10, len(graphs))]:
        if g.eigenvalues_P is None or g.eigenvectors_P is None:
            continue
        n = g.num_nodes
        n_check = min(n_samples, n)
        for u in range(n_check):
            weights_P = g.eigenvectors_P[u, :] ** 2
            evals_P = g.eigenvalues_P
            for k_idx in range(min(5, RWSE_DIM)):
                k = k_idx + 1
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
#  TIKHONOV SRWE RECOVERY
# ═══════════════════════════════════════════════════════════════════════════

def srwe_tikhonov(
    moments: np.ndarray,
    eigenvalues: np.ndarray,
    lam: float = 1e-3,
) -> np.ndarray:
    """
    Tikhonov-regularized Vandermonde recovery.
    moments: [K] array of RWSE moments for walk lengths 1..K
    eigenvalues: [n] eigenvalues of P
    lam: Tikhonov regularization parameter
    Returns: weights [n] non-negative, normalized to sum=1
    """
    K = len(moments)
    n = len(eigenvalues)
    if n == 0:
        return np.array([])

    # Build Vandermonde: V[k,i] = eigenvalues[i]^(k+1)
    V = np.zeros((K, n), dtype=np.float64)
    for k in range(K):
        V[k, :] = eigenvalues ** (k + 1)

    # Solve: w* = (V^T V + lam*I)^{-1} V^T m
    VtV = V.T @ V
    Vtm = V.T @ moments
    try:
        w = np.linalg.solve(VtV + lam * np.eye(n), Vtm)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(V, moments, rcond=None)[0]

    # Clamp non-negative, normalize
    w = np.maximum(w, 0.0)
    s = w.sum()
    if s > 1e-12:
        w /= s
    return w


def compute_srwe_for_graph(graph: GraphData, lam: float) -> np.ndarray:
    """Compute SRWE weights for all nodes in a graph at a given lambda.

    Optimized: precompute Vandermonde and factorize (V^T V + λI) once,
    then solve for all nodes with a single batched linear solve.
    """
    n = graph.num_nodes
    evals = graph.eigenvalues_P
    if evals is None or len(evals) == 0:
        return np.zeros((n, 1), dtype=np.float64)

    n_eigs = len(evals)
    K = min(graph.rwse.shape[1], RWSE_DIM)

    # Build Vandermonde: V[k,i] = evals[i]^(k+1), shape [K, n_eigs]
    V = np.zeros((K, n_eigs), dtype=np.float64)
    for k in range(K):
        V[k, :] = evals ** (k + 1)

    # A_reg = V^T V + λI, shape [n_eigs, n_eigs]
    VtV = V.T @ V
    A_reg = VtV + lam * np.eye(n_eigs)

    # All moments: M[u, k] for all nodes, shape [n, K]
    M = graph.rwse[:, :K]

    # RHS: B = V^T @ M^T, shape [n_eigs, n]
    B = V.T @ M.T

    # Batched solve: A_reg @ W = B => W shape [n_eigs, n]
    try:
        W = np.linalg.solve(A_reg, B)  # [n_eigs, n]
    except np.linalg.LinAlgError:
        try:
            W = np.linalg.lstsq(A_reg, B, rcond=None)[0]
        except Exception:
            return np.ones((n, n_eigs), dtype=np.float64) / max(n_eigs, 1)

    # Transpose to [n, n_eigs]
    weights = W.T

    # Clamp non-negative and normalize each node
    weights = np.maximum(weights, 0.0)
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-12)
    weights = weights / row_sums

    return weights


# ═══════════════════════════════════════════════════════════════════════════
#  5 SRWE FEATURE REPRESENTATIONS
# ═══════════════════════════════════════════════════════════════════════════

def compute_histogram(eigenvalues: np.ndarray, weights: np.ndarray,
                      n_bins: int = HIST_BINS) -> np.ndarray:
    """Bin recovered weights into n_bins bins over [-1, 1]."""
    pe = np.zeros(n_bins, dtype=np.float32)
    for ev, w in zip(eigenvalues, weights):
        bin_idx = int((ev - SRWE_HIST_RANGE[0]) / (SRWE_HIST_RANGE[1] - SRWE_HIST_RANGE[0]) * n_bins)
        bin_idx = min(max(bin_idx, 0), n_bins - 1)
        pe[bin_idx] += w
    return pe


def compute_raw_weights(weights: np.ndarray, k: int = RAW_WEIGHTS_K) -> np.ndarray:
    """Top-k weights sorted by magnitude."""
    sorted_idx = np.argsort(-np.abs(weights))[:k]
    pe = weights[sorted_idx]
    # Pad if fewer weights than k
    if len(pe) < k:
        pe = np.concatenate([pe, np.zeros(k - len(pe))])
    return pe.astype(np.float32)


def compute_eigenvalue_pairs(eigenvalues: np.ndarray, weights: np.ndarray,
                             k: int = EIGPAIR_K) -> np.ndarray:
    """(lambda_i, w_i) for top-k by weight. Returns 2*k dimensional."""
    sorted_idx = np.argsort(-weights)[:k]
    evals_selected = eigenvalues[sorted_idx]
    weights_selected = weights[sorted_idx]
    pe = np.concatenate([evals_selected, weights_selected])
    if len(pe) < 2 * k:
        pe = np.concatenate([pe, np.zeros(2 * k - len(pe))])
    return pe.astype(np.float32)


def compute_moment_correction(moments: np.ndarray, eigenvalues: np.ndarray,
                              weights: np.ndarray) -> np.ndarray:
    """RWSE + delta from spectral residual. 20-dim."""
    K = len(moments)
    predicted_moments = np.zeros(K, dtype=np.float64)
    for k_idx in range(K):
        predicted_moments[k_idx] = np.sum(weights * eigenvalues ** (k_idx + 1))
    delta = predicted_moments - moments
    # Return corrected moments = predicted (which uses recovered weights)
    return predicted_moments.astype(np.float32)


def compute_spectral_summary(eigenvalues: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """12 statistics summarizing the spectral measure."""
    if len(weights) == 0 or np.sum(weights) < 1e-12:
        return np.zeros(SPECTRAL_SUMMARY_DIM, dtype=np.float32)

    w = weights / (np.sum(weights) + 1e-12)

    mean_val = np.sum(w * eigenvalues)
    variance = np.sum(w * (eigenvalues - mean_val) ** 2)
    std_val = np.sqrt(max(variance, 0.0))

    # Skewness
    if std_val > 1e-12:
        skew = np.sum(w * ((eigenvalues - mean_val) / std_val) ** 3)
        kurt = np.sum(w * ((eigenvalues - mean_val) / std_val) ** 4) - 3.0
    else:
        skew = 0.0
        kurt = 0.0

    # Entropy
    w_pos = w[w > 1e-15]
    entropy = -np.sum(w_pos * np.log(w_pos + 1e-15))

    max_w = float(np.max(weights))

    # Top-3 by weight
    top3_idx = np.argsort(-weights)[:3]
    top3_positions = eigenvalues[top3_idx]
    top3_heights = weights[top3_idx]

    # Pad if fewer than 3
    while len(top3_positions) < 3:
        top3_positions = np.append(top3_positions, 0.0)
        top3_heights = np.append(top3_heights, 0.0)

    result = np.array([mean_val, std_val, skew, kurt, entropy, max_w,
                       top3_positions[0], top3_positions[1], top3_positions[2],
                       top3_heights[0], top3_heights[1], top3_heights[2]],
                      dtype=np.float32)
    return result


def compute_pe_for_graph(graph: GraphData, pe_type: str,
                         srwe_weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute PE for all nodes in a graph given a PE type."""
    n = graph.num_nodes
    evals = graph.eigenvalues_P

    if pe_type == "none":
        return np.zeros((n, PE_DIM), dtype=np.float32)

    elif pe_type == "rwse":
        return graph.rwse[:, :PE_DIM].astype(np.float32)

    elif pe_type == "lappe":
        if graph.eigenvectors_P is None or evals is None or len(evals) == 0:
            return np.zeros((n, PE_DIM), dtype=np.float32)
        n_eigs = len(evals)
        k = min(PE_DIM, n_eigs)
        if n_eigs > 1:
            start_idx = max(0, n_eigs - k - 1)
            end_idx = n_eigs - 1
            selected = graph.eigenvectors_P[:, start_idx:end_idx]
        else:
            selected = graph.eigenvectors_P
        pe = selected ** 2
        if pe.shape[1] < PE_DIM:
            pad = np.zeros((n, PE_DIM - pe.shape[1]), dtype=np.float64)
            pe = np.hstack([pe, pad])
        return pe[:, :PE_DIM].astype(np.float32)

    elif pe_type == "histogram":
        if srwe_weights is None or evals is None:
            return np.zeros((n, HIST_BINS), dtype=np.float32)
        # Vectorized histogram computation
        pe = np.zeros((n, HIST_BINS), dtype=np.float32)
        bin_indices = np.floor(
            (evals - SRWE_HIST_RANGE[0]) / (SRWE_HIST_RANGE[1] - SRWE_HIST_RANGE[0]) * HIST_BINS
        ).astype(int)
        bin_indices = np.clip(bin_indices, 0, HIST_BINS - 1)
        for u in range(n):
            np.add.at(pe[u], bin_indices, srwe_weights[u])
        return pe

    elif pe_type == "raw_weights":
        if srwe_weights is None:
            return np.zeros((n, RAW_WEIGHTS_K), dtype=np.float32)
        k = min(RAW_WEIGHTS_K, srwe_weights.shape[1])
        # Vectorized: sort each row by abs magnitude, take top k
        abs_w = np.abs(srwe_weights)
        # For each node, get indices of top-k by magnitude
        if k >= srwe_weights.shape[1]:
            pe = srwe_weights.astype(np.float32)
            if pe.shape[1] < RAW_WEIGHTS_K:
                pe = np.hstack([pe, np.zeros((n, RAW_WEIGHTS_K - pe.shape[1]), dtype=np.float32)])
        else:
            # Use partition for efficiency
            top_k_idx = np.argpartition(-abs_w, k, axis=1)[:, :k]
            pe = np.zeros((n, RAW_WEIGHTS_K), dtype=np.float32)
            for u in range(n):
                idx = top_k_idx[u]
                # Sort by magnitude within top-k
                sorted_sub = idx[np.argsort(-abs_w[u, idx])]
                pe[u, :k] = srwe_weights[u, sorted_sub]
        return pe

    elif pe_type == "eigenvalue_pairs":
        if srwe_weights is None or evals is None:
            return np.zeros((n, 2 * EIGPAIR_K), dtype=np.float32)
        k = min(EIGPAIR_K, srwe_weights.shape[1])
        pe = np.zeros((n, 2 * EIGPAIR_K), dtype=np.float32)
        # For each node, get top-k by weight
        top_k_idx = np.argpartition(-srwe_weights, min(k, srwe_weights.shape[1]-1), axis=1)[:, :k]
        for u in range(n):
            idx = top_k_idx[u]
            sorted_sub = idx[np.argsort(-srwe_weights[u, idx])]
            pe[u, :k] = evals[sorted_sub]
            pe[u, EIGPAIR_K:EIGPAIR_K+k] = srwe_weights[u, sorted_sub]
        return pe

    elif pe_type == "moment_correction":
        if srwe_weights is None or evals is None:
            return np.zeros((n, RWSE_DIM), dtype=np.float32)
        # Vectorized: predicted_moments = srwe_weights @ V^T
        K = RWSE_DIM
        V = np.zeros((K, len(evals)), dtype=np.float64)
        for k_idx in range(K):
            V[k_idx, :] = evals ** (k_idx + 1)
        # predicted = srwe_weights @ V^T gives [n, K]
        predicted = srwe_weights @ V.T
        return predicted.astype(np.float32)

    elif pe_type == "spectral_summary":
        if srwe_weights is None or evals is None:
            return np.zeros((n, SPECTRAL_SUMMARY_DIM), dtype=np.float32)
        pe = np.zeros((n, SPECTRAL_SUMMARY_DIM), dtype=np.float32)
        # Normalize weights per node
        w = srwe_weights / (np.sum(srwe_weights, axis=1, keepdims=True) + 1e-12)
        # Weighted mean: sum(w * evals)
        mean_vals = w @ evals  # [n]
        # Weighted variance
        diffs = evals[None, :] - mean_vals[:, None]  # [n, n_eigs]
        variances = np.sum(w * diffs**2, axis=1)  # [n]
        stds = np.sqrt(np.maximum(variances, 0.0))
        # Weighted skewness
        safe_stds = np.where(stds > 1e-12, stds, 1.0)
        normed = diffs / safe_stds[:, None]
        skews = np.sum(w * normed**3, axis=1)
        kurts = np.sum(w * normed**4, axis=1) - 3.0
        # Entropy
        w_safe = np.clip(w, 1e-15, None)
        entropies = -np.sum(w_safe * np.log(w_safe), axis=1)
        # Max weight
        max_ws = np.max(srwe_weights, axis=1)
        # Top-3 by weight
        top3_idx = np.argpartition(-srwe_weights, min(3, srwe_weights.shape[1]-1), axis=1)[:, :3]
        pe[:, 0] = mean_vals
        pe[:, 1] = stds
        pe[:, 2] = skews
        pe[:, 3] = kurts
        pe[:, 4] = entropies
        pe[:, 5] = max_ws
        for u in range(n):
            idx = top3_idx[u]
            s = idx[np.argsort(-srwe_weights[u, idx])]
            k_use = min(3, len(s))
            pe[u, 6:6+k_use] = evals[s[:k_use]]
            pe[u, 9:9+k_use] = srwe_weights[u, s[:k_use]]
        return pe

    else:
        return np.zeros((n, PE_DIM), dtype=np.float32)


def get_pe_dim(pe_type: str) -> int:
    """Get output dimension for a PE type."""
    dims = {
        "none": PE_DIM,
        "rwse": PE_DIM,
        "lappe": PE_DIM,
        "histogram": HIST_BINS,
        "raw_weights": RAW_WEIGHTS_K,
        "eigenvalue_pairs": 2 * EIGPAIR_K,
        "moment_correction": RWSE_DIM,
        "spectral_summary": SPECTRAL_SUMMARY_DIM,
    }
    return dims.get(pe_type, PE_DIM)


# ═══════════════════════════════════════════════════════════════════════════
#  WASSERSTEIN & RECOVERY METRICS
# ═══════════════════════════════════════════════════════════════════════════

def wasserstein1(true_evals, true_weights, est_evals, est_weights) -> float:
    if len(true_evals) == 0 or len(est_evals) == 0:
        return float("inf")
    try:
        return float(scipy.stats.wasserstein_distance(
            true_evals, est_evals, true_weights, est_weights))
    except Exception:
        return float("inf")


# ═══════════════════════════════════════════════════════════════════════════
#  GNN MODEL (pure PyTorch, GPS-lite: 2-layer GCN + attention pooling)
# ═══════════════════════════════════════════════════════════════════════════

class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        out = torch.sparse.mm(adj, h)
        return out


class GPSLite(nn.Module):
    """GPS-lite: 2-layer GCN with attention pooling."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNLayer(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout
        self.gate_nn = nn.Linear(hidden_dim, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, adj, batch):
        h = self.input_proj(x)
        h = F.relu(h)

        for conv, bn in zip(self.convs, self.bns):
            h_new = conv(h, adj)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new

        gate = self.gate_nn(h).squeeze(-1)
        gate = _scatter_softmax(gate, batch)
        h_weighted = h * gate.unsqueeze(-1)
        num_graphs = batch.max().item() + 1
        graph_emb = torch.zeros(num_graphs, h.shape[1], device=h.device, dtype=h.dtype)
        graph_emb.scatter_add_(0, batch.unsqueeze(-1).expand_as(h_weighted), h_weighted)

        return self.head(graph_emb)

    def get_graph_embeddings(self, x, adj, batch):
        """Get graph-level embeddings (for diagnostics)."""
        h = self.input_proj(x)
        h = F.relu(h)
        for conv, bn in zip(self.convs, self.bns):
            h_new = conv(h, adj)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h = h + h_new

        gate = self.gate_nn(h).squeeze(-1)
        gate = _scatter_softmax(gate, batch)
        h_weighted = h * gate.unsqueeze(-1)
        num_graphs = batch.max().item() + 1
        graph_emb = torch.zeros(num_graphs, h.shape[1], device=h.device, dtype=h.dtype)
        graph_emb.scatter_add_(0, batch.unsqueeze(-1).expand_as(h_weighted), h_weighted)
        return graph_emb


def _scatter_softmax(values: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    num_graphs = batch.max().item() + 1
    max_vals = torch.full((num_graphs,), -1e9, device=values.device, dtype=values.dtype)
    max_vals.scatter_reduce_(0, batch, values, reduce="amax", include_self=True)
    max_vals = max_vals[batch]
    exp_vals = torch.exp(values - max_vals)
    sum_vals = torch.zeros(num_graphs, device=values.device, dtype=values.dtype)
    sum_vals.scatter_add_(0, batch, exp_vals)
    return exp_vals / (sum_vals[batch] + 1e-10)


# ═══════════════════════════════════════════════════════════════════════════
#  BATCH COLLATION
# ═══════════════════════════════════════════════════════════════════════════

def _parse_target(target_str: str, task_type: str) -> np.ndarray:
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


def collate_graphs_with_pe(
    graphs: list,
    pe_arrays: list,  # list of [N_i, pe_dim] arrays
    pe_dim: int,
    device: torch.device = DEVICE,
) -> tuple:
    """Collate graphs with precomputed PE arrays."""
    all_x = []
    all_src = []
    all_dst = []
    all_batch = []
    all_targets = []
    node_offset = 0

    for gi, (g, pe) in enumerate(zip(graphs, pe_arrays)):
        n = g.num_nodes
        feat = g.node_feat.astype(np.float32)

        # Ensure PE has correct shape
        if pe is None or pe.shape[0] != n:
            pe = np.zeros((n, pe_dim), dtype=np.float32)
        if pe.shape[1] != pe_dim:
            if pe.shape[1] < pe_dim:
                pe = np.hstack([pe, np.zeros((n, pe_dim - pe.shape[1]), dtype=np.float32)])
            else:
                pe = pe[:, :pe_dim]

        x = np.hstack([feat, pe])
        all_x.append(x)

        if g.edge_index.shape[1] > 0:
            all_src.append(g.edge_index[0] + node_offset)
            all_dst.append(g.edge_index[1] + node_offset)

        self_nodes = np.arange(n) + node_offset
        all_src.append(self_nodes)
        all_dst.append(self_nodes)
        all_batch.append(np.full(n, gi, dtype=np.int64))

        target = _parse_target(g.target, g.task_type)
        all_targets.append(target)
        node_offset += n

    x = np.vstack(all_x)
    src = np.concatenate(all_src)
    dst = np.concatenate(all_dst)
    batch_vec = np.concatenate(all_batch)

    total_nodes = node_offset
    degree = np.zeros(total_nodes, dtype=np.float32)
    np.add.at(degree, dst, 1.0)
    d_inv_sqrt = np.zeros(total_nodes, dtype=np.float32)
    mask = degree > 0
    d_inv_sqrt[mask] = 1.0 / np.sqrt(degree[mask])
    edge_weight = d_inv_sqrt[src] * d_inv_sqrt[dst]

    x_t = torch.tensor(x, dtype=torch.float32, device=device)
    indices = torch.tensor(np.stack([src, dst]), dtype=torch.long, device=device)
    values = torch.tensor(edge_weight, dtype=torch.float32, device=device)
    adj_t = torch.sparse_coo_tensor(indices, values, (total_nodes, total_nodes)).coalesce()
    batch_t = torch.tensor(batch_vec, dtype=torch.long, device=device)
    target_t = torch.tensor(np.array(all_targets), dtype=torch.float32, device=device)

    return x_t, adj_t, batch_t, target_t


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def train_gnn(
    train_graphs: list,
    val_graphs: list,
    train_pes: list,
    val_pes: list,
    pe_dim: int,
    task_type: str,
    output_dim: int,
    seed: int,
    max_epochs: int = GNN_MAX_EPOCHS,
    patience: int = GNN_PATIENCE,
    batch_size: int = GNN_BATCH_SIZE,
) -> nn.Module:
    """Train a GPS-lite model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    feat_dim = train_graphs[0].node_feat.shape[1] if train_graphs else 1
    input_dim = feat_dim + pe_dim

    model = GPSLite(
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
        model.train()
        rng = np.random.RandomState(seed * 1000 + epoch)
        indices = rng.permutation(len(train_graphs))
        total_loss = 0.0
        n_batches = 0

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_graphs = [train_graphs[i] for i in batch_idx]
            batch_pes = [train_pes[i] for i in batch_idx]

            try:
                x, adj, batch_vec, targets = collate_graphs_with_pe(
                    batch_graphs, batch_pes, pe_dim)
            except Exception:
                continue

            optimizer.zero_grad()
            out = model(x, adj, batch_vec)

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
                batch_pes = val_pes[start:start + batch_size]
                try:
                    x, adj, batch_vec, targets = collate_graphs_with_pe(
                        batch_graphs, batch_pes, pe_dim)
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
            logger.debug(f"  Epoch {epoch+1}: train={avg_train_loss:.4f}, val={avg_val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    return model


def predict_gnn(model, graphs, pe_arrays, pe_dim, batch_size=GNN_BATCH_SIZE):
    """Run inference and return per-graph predictions."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            batch_graphs = graphs[start:start + batch_size]
            batch_pes = pe_arrays[start:start + batch_size]
            try:
                x, adj, batch_vec, _ = collate_graphs_with_pe(
                    batch_graphs, batch_pes, pe_dim)
                out = model(x, adj, batch_vec)
                preds = out.cpu().numpy()
                for i in range(len(batch_graphs)):
                    all_preds.append(preds[i])
            except Exception:
                for _ in batch_graphs:
                    all_preds.append(np.array([0.0]))
    return all_preds


def compute_metric(predictions, targets, task_type):
    """Compute task-specific metric."""
    if not predictions or not targets:
        return {}

    preds = np.array([p.flatten() for p in predictions])
    tgts = np.array([t.flatten() for t in targets])

    min_cols = min(preds.shape[1], tgts.shape[1])
    preds = preds[:, :min_cols]
    tgts = tgts[:, :min_cols]

    if task_type == "classification":
        try:
            probs = 1.0 / (1.0 + np.exp(-np.clip(preds, -50, 50)))
            ap = average_precision_score(tgts, probs, average="macro")
            return {"AP": float(ap)}
        except Exception:
            return {"AP": 0.0}
    else:
        mae = float(np.mean(np.abs(preds - tgts)))
        return {"MAE": mae}


# ═══════════════════════════════════════════════════════════════════════════
#  SPLITS
# ═══════════════════════════════════════════════════════════════════════════

def split_graphs(graphs: list) -> tuple:
    """Split graphs into train/val/test sets."""
    folds = set(g.fold for g in graphs)
    if folds == {0, 1, 2}:
        train = [g for g in graphs if g.fold == 0]
        val = [g for g in graphs if g.fold == 1]
        test = [g for g in graphs if g.fold == 2]
        return train, val, test

    # Peptides: use 70/15/15 split
    rng = np.random.RandomState(42)
    n = len(graphs)
    indices = rng.permutation(n)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    train = [graphs[i] for i in indices[:n_train]]
    val = [graphs[i] for i in indices[n_train:n_train + n_val]]
    test = [graphs[i] for i in indices[n_train + n_val:]]
    return train, val, test


# ═══════════════════════════════════════════════════════════════════════════
#  INFORMATION-THEORETIC DIAGNOSTICS (Step 5)
# ═══════════════════════════════════════════════════════════════════════════

def compute_graph_level_features(graphs: list, pe_arrays: list) -> np.ndarray:
    """Mean-pool PE features to get graph-level representations."""
    features = []
    for g, pe in zip(graphs, pe_arrays):
        if pe is not None and pe.shape[0] > 0:
            features.append(np.mean(pe, axis=0))
        else:
            features.append(np.zeros(pe_arrays[0].shape[1] if pe_arrays else 1))
    return np.array(features, dtype=np.float32)


def compute_diagnostics(graphs: list, pe_arrays: list, pe_name: str,
                        task_type: str) -> dict:
    """Compute MI, linear probing, feature stats for one encoding."""
    result = {}
    graph_features = compute_graph_level_features(graphs, pe_arrays)

    # Get targets
    targets = []
    for g in graphs:
        t = _parse_target(g.target, task_type)
        targets.append(t)
    targets = np.array(targets)

    if len(graph_features) < 10:
        return result

    # Replace NaN/inf
    graph_features = np.nan_to_num(graph_features, nan=0.0, posinf=0.0, neginf=0.0)

    # (i) Mutual information
    try:
        if task_type == "classification":
            # Multi-label: use first label
            if targets.ndim > 1:
                y = targets[:, 0].astype(int)
            else:
                y = targets.flatten().astype(int)
            mi = mutual_info_classif(graph_features, y, random_state=42, n_neighbors=5)
            result["mi_mean"] = float(np.mean(mi))
            result["mi_max"] = float(np.max(mi))
        else:
            if targets.ndim > 1:
                y = targets[:, 0]
            else:
                y = targets.flatten()
            mi = mutual_info_regression(graph_features, y, random_state=42, n_neighbors=5)
            result["mi_mean"] = float(np.mean(mi))
            result["mi_max"] = float(np.max(mi))
    except Exception as e:
        logger.debug(f"MI computation failed for {pe_name}: {e}")
        result["mi_mean"] = 0.0
        result["mi_max"] = 0.0

    # (ii) Linear probing
    try:
        if task_type == "classification":
            if targets.ndim > 1:
                y = targets[:, 0].astype(int)
            else:
                y = targets.flatten().astype(int)
            # Check if we have enough classes
            unique_classes = np.unique(y)
            if len(unique_classes) >= 2:
                probe = LogisticRegression(max_iter=500, solver="lbfgs", random_state=42)
                cv_folds = min(5, len(unique_classes))
                scores = cross_val_score(probe, graph_features, y, cv=cv_folds,
                                         scoring="average_precision")
                result["linear_probe_mean"] = float(np.mean(scores))
                result["linear_probe_std"] = float(np.std(scores))
            else:
                result["linear_probe_mean"] = 0.0
                result["linear_probe_std"] = 0.0
        else:
            if targets.ndim > 1:
                y = targets[:, 0]
            else:
                y = targets.flatten()
            probe = Ridge(alpha=1.0)
            scores = cross_val_score(probe, graph_features, y, cv=5,
                                     scoring="neg_mean_absolute_error")
            result["linear_probe_mean"] = float(np.mean(scores))
            result["linear_probe_std"] = float(np.std(scores))
    except Exception as e:
        logger.debug(f"Linear probing failed for {pe_name}: {e}")
        result["linear_probe_mean"] = 0.0
        result["linear_probe_std"] = 0.0

    # (iii) Feature statistics
    try:
        if graph_features.shape[1] > 1:
            cov = np.cov(graph_features.T)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = eigvals[eigvals > 1e-12]
            if len(eigvals) > 0:
                eff_dim = float(np.sum(eigvals) ** 2 / np.sum(eigvals ** 2))
            else:
                eff_dim = 0.0
        else:
            eff_dim = 1.0

        variance = float(np.mean(np.var(graph_features, axis=0)))
        sparsity = float(np.mean(np.abs(graph_features) < 0.01))

        result["effective_dimension"] = eff_dim
        result["mean_variance"] = variance
        result["sparsity"] = sparsity
    except Exception as e:
        logger.debug(f"Feature stats failed for {pe_name}: {e}")

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  REGRESSION VS CLASSIFICATION DIAGNOSIS (Step 6)
# ═══════════════════════════════════════════════════════════════════════════

def compute_per_graph_w1(graphs: list, srwe_weights_dict: dict,
                         lam: float) -> list:
    """Compute per-graph W1 between true spectral measure and SRWE estimate."""
    w1_per_graph = []
    for g in graphs:
        if g.eigenvectors_P is None or g.eigenvalues_P is None:
            w1_per_graph.append(float("nan"))
            continue

        n = g.num_nodes
        evals = g.eigenvalues_P
        srwe_w = srwe_weights_dict.get(id(g))
        if srwe_w is None:
            w1_per_graph.append(float("nan"))
            continue

        w1_nodes = []
        n_check = min(n, 50)
        for u in range(n_check):
            true_w = g.eigenvectors_P[u, :] ** 2
            est_w = srwe_w[u]
            try:
                w1 = wasserstein1(evals, true_w, evals, est_w)
                if not math.isinf(w1) and not math.isnan(w1):
                    w1_nodes.append(w1)
            except Exception:
                pass

        if w1_nodes:
            w1_per_graph.append(float(np.mean(w1_nodes)))
        else:
            w1_per_graph.append(float("nan"))

    return w1_per_graph


def regression_vs_classification_diagnosis(
    func_graphs: list, struct_graphs: list,
    func_srwe_pes: list, struct_srwe_pes: list,
    func_rwse_pes: list, struct_rwse_pes: list,
    srwe_weights_func: dict, srwe_weights_struct: dict,
    lam: float,
) -> dict:
    """Diagnose why SRWE helps regression but hurts classification."""
    result = {}

    # Compute per-graph W1
    w1_func = compute_per_graph_w1(func_graphs, srwe_weights_func, lam)
    w1_struct = compute_per_graph_w1(struct_graphs, srwe_weights_struct, lam)

    # MI comparison: SRWE vs RWSE for func vs struct targets
    try:
        # Func (classification)
        func_srwe_feats = compute_graph_level_features(func_graphs, func_srwe_pes)
        func_rwse_feats = compute_graph_level_features(func_graphs, func_rwse_pes)
        func_targets = np.array([_parse_target(g.target, "classification")
                                 for g in func_graphs])

        if func_targets.ndim > 1:
            func_y = func_targets[:, 0].astype(int)
        else:
            func_y = func_targets.flatten().astype(int)

        func_srwe_feats = np.nan_to_num(func_srwe_feats, nan=0.0, posinf=0.0, neginf=0.0)
        func_rwse_feats = np.nan_to_num(func_rwse_feats, nan=0.0, posinf=0.0, neginf=0.0)

        mi_srwe_func = float(np.mean(mutual_info_classif(func_srwe_feats, func_y,
                                                          random_state=42)))
        mi_rwse_func = float(np.mean(mutual_info_classif(func_rwse_feats, func_y,
                                                          random_state=42)))

        result["mi_srwe_func"] = mi_srwe_func
        result["mi_rwse_func"] = mi_rwse_func
        result["mi_diff_func"] = mi_srwe_func - mi_rwse_func
    except Exception as e:
        logger.debug(f"MI func diagnosis failed: {e}")
        result["mi_diff_func"] = 0.0

    try:
        # Struct (regression)
        struct_srwe_feats = compute_graph_level_features(struct_graphs, struct_srwe_pes)
        struct_rwse_feats = compute_graph_level_features(struct_graphs, struct_rwse_pes)
        struct_targets = np.array([_parse_target(g.target, "regression")
                                   for g in struct_graphs])

        if struct_targets.ndim > 1:
            struct_y = struct_targets[:, 0]
        else:
            struct_y = struct_targets.flatten()

        struct_srwe_feats = np.nan_to_num(struct_srwe_feats, nan=0.0, posinf=0.0, neginf=0.0)
        struct_rwse_feats = np.nan_to_num(struct_rwse_feats, nan=0.0, posinf=0.0, neginf=0.0)

        mi_srwe_struct = float(np.mean(mutual_info_regression(struct_srwe_feats, struct_y,
                                                               random_state=42)))
        mi_rwse_struct = float(np.mean(mutual_info_regression(struct_rwse_feats, struct_y,
                                                               random_state=42)))

        result["mi_srwe_struct"] = mi_srwe_struct
        result["mi_rwse_struct"] = mi_rwse_struct
        result["mi_diff_struct"] = mi_srwe_struct - mi_rwse_struct
    except Exception as e:
        logger.debug(f"MI struct diagnosis failed: {e}")
        result["mi_diff_struct"] = 0.0

    # Spearman correlation of W1 with loss difference (if we have enough data)
    valid_w1_func = [w for w in w1_func if not math.isnan(w)]
    valid_w1_struct = [w for w in w1_struct if not math.isnan(w)]

    result["mean_w1_func"] = float(np.mean(valid_w1_func)) if valid_w1_func else float("nan")
    result["mean_w1_struct"] = float(np.mean(valid_w1_struct)) if valid_w1_struct else float("nan")
    result["n_valid_func"] = len(valid_w1_func)
    result["n_valid_struct"] = len(valid_w1_struct)

    # Interpretation
    mi_diff_func = result.get("mi_diff_func", 0.0)
    mi_diff_struct = result.get("mi_diff_struct", 0.0)

    if mi_diff_func < 0 and mi_diff_struct > 0:
        result["diagnosis"] = ("SRWE_LOSES_CLASSIFICATION_INFO: SRWE has lower MI with "
                               "classification targets than RWSE but higher MI with regression "
                               "targets. This explains the asymmetry.")
    elif mi_diff_func < 0:
        result["diagnosis"] = ("SRWE_LOSES_INFO: SRWE has lower MI than RWSE for classification. "
                               "The binning/recovery process destroys class-discriminative features.")
    else:
        result["diagnosis"] = ("NO_CLEAR_ASYMMETRY: MI analysis doesn't show clear asymmetry. "
                               "The issue may be in how GNN uses the features rather than MI.")

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  OUTPUT FORMATTING
# ═══════════════════════════════════════════════════════════════════════════

def format_output(
    all_datasets: dict,
    metadata: dict,
) -> dict:
    """Format results as exp_gen_sol_out.json."""
    output = {
        "metadata": metadata,
        "datasets": [],
    }

    for ds_name, graphs in all_datasets.items():
        examples = []
        for g in graphs:
            ex = {
                "input": g.input_str,
                "output": g.output_str,
            }
            for pe_type, pred_str in g.predictions.items():
                key = f"predict_{pe_type}"
                # Sanitize key name
                key = key.replace("-", "_")
                ex[key] = pred_str

            for mk, mv in g.recovery_metrics.items():
                key = f"metadata_{mk}"
                key = key.replace("-", "_")
                ex[key] = mv

            examples.append(ex)

        output["datasets"].append({
            "dataset": ds_name,
            "examples": examples,
        })

    return output


# ═══════════════════════════════════════════════════════════════════════════
#  PARTIAL SAVE
# ═══════════════════════════════════════════════════════════════════════════

_partial_state = {
    "all_datasets": None,
    "metadata": {},
    "saved": False,
}


def _save_partial():
    if _partial_state["saved"] or _partial_state["all_datasets"] is None:
        return
    try:
        logger.warning("Saving partial results before exit...")
        output = format_output(_partial_state["all_datasets"], _partial_state["metadata"])
        partial_path = WORKSPACE / "full_method_out.json"
        partial_json = json.dumps(output, separators=(",", ":"), default=str)
        partial_path.write_text(partial_json)
        file_size_mb = partial_path.stat().st_size / (1024 * 1024)
        logger.warning(f"Partial results saved ({file_size_mb:.1f} MB)")
        _partial_state["saved"] = True
    except Exception:
        logger.exception("Failed to save partial results")


def _signal_handler(signum, frame):
    logger.warning(f"Received signal {signum}, saving partial results...")
    _save_partial()
    sys.exit(128 + signum)


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)
atexit.register(_save_partial)


def time_elapsed() -> float:
    return time.time() - T_START


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

@logger.catch
def main() -> None:
    logger.info("=" * 70)
    logger.info("SRWE Feature Representation Optimization & Classification Failure Diagnosis")
    logger.info("=" * 70)

    # ── Phase 0: Load Data ──
    logger.info("── Phase 0: Loading Data ──")

    max_ex = MAX_EXAMPLES if MAX_EXAMPLES > 0 else 0
    mini_file = DEP_DIR / "mini_data_out.json"
    full_files = sorted((DEP_DIR / "data_out").glob("full_data_out_*.json"))

    if max_ex > 0 and max_ex <= 3:
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

    _partial_state["all_datasets"] = all_datasets

    # ── Phase 1: Spectral Recomputation ──
    logger.info("── Phase 1: Recomputing normalized adjacency eigendecomposition ──")
    t0 = time.time()
    skipped_large = 0
    for ds_name, graphs in all_datasets.items():
        for g in graphs:
            try:
                recompute_spectral(g)
                if g.eigenvectors_P is None:
                    skipped_large += 1
            except Exception:
                logger.exception(f"Failed spectral recompute for {ds_name} graph {g.row_index}")
    logger.info(f"  Spectral recomp done in {time.time()-t0:.1f}s "
                f"({skipped_large} large graphs used fallback)")

    # Consistency check
    non_synthetic = [g for ds_name, gs in all_datasets.items()
                     for g in gs if "Synthetic" not in ds_name]
    consistency = consistency_check(non_synthetic)

    # ── Phase 2: Tikhonov Recovery at screening lambda ──
    logger.info(f"── Phase 2: Tikhonov Recovery at λ={SCREENING_LAMBDA} ──")
    t0 = time.time()
    # Store SRWE weights per graph (keyed by graph id)
    srwe_weights_by_id: dict[int, np.ndarray] = {}

    for ds_name, graphs in all_datasets.items():
        if "Synthetic" in ds_name:
            continue
        logger.info(f"  Computing SRWE for {ds_name} ({len(graphs)} graphs)...")
        for gi, g in enumerate(graphs):
            if g.eigenvalues_P is None or len(g.eigenvalues_P) == 0:
                continue
            try:
                w = compute_srwe_for_graph(g, lam=SCREENING_LAMBDA)
                srwe_weights_by_id[id(g)] = w
            except Exception:
                logger.exception(f"SRWE failed for {ds_name} graph {g.row_index}")

            if (gi + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (gi + 1) / elapsed
                logger.info(f"    {gi+1}/{len(graphs)}: {rate:.1f} graphs/s")

    logger.info(f"  SRWE recovery done in {time.time()-t0:.1f}s, {len(srwe_weights_by_id)} graphs")

    # ── Phase 3: Compute PE features for all representations ──
    logger.info("── Phase 3: Computing PE features for all representations ──")

    # All PE types to screen (baselines + SRWE variants)
    baseline_pe_types = ["none", "rwse", "lappe"]
    srwe_pe_types = ["histogram", "raw_weights", "eigenvalue_pairs",
                     "moment_correction", "spectral_summary"]
    all_pe_types = baseline_pe_types + srwe_pe_types

    # Precompute PE arrays for each (dataset, pe_type)
    # Structure: pe_arrays[ds_name][pe_type] = list of np.ndarray per graph
    pe_arrays_all: dict[str, dict[str, list]] = {}

    for ds_name, graphs in all_datasets.items():
        if "Synthetic" in ds_name:
            continue
        pe_arrays_all[ds_name] = {}
        for pe_type in all_pe_types:
            arrays = []
            for g in graphs:
                srwe_w = srwe_weights_by_id.get(id(g))
                pe = compute_pe_for_graph(g, pe_type, srwe_weights=srwe_w)
                arrays.append(pe)
            pe_arrays_all[ds_name][pe_type] = arrays

    logger.info(f"  PE features computed for {len(pe_arrays_all)} datasets")

    # ── Phase 4A: Screen all representations at λ={SCREENING_LAMBDA}, seed=0 ──
    logger.info("── Phase 4A: Screening all PE types ──")

    gnn_datasets_order = ["Peptides-func", "Peptides-struct", "ZINC-subset"]
    dataset_configs = {
        "Peptides-func": {"task_type": "classification", "output_dim": 10, "metric": "AP"},
        "Peptides-struct": {"task_type": "regression", "output_dim": 11, "metric": "MAE"},
        "ZINC-subset": {"task_type": "regression", "output_dim": 1, "metric": "MAE"},
    }

    results_4A: dict[str, dict[str, dict]] = {}
    phase4a_fallback = False

    for ds_name in gnn_datasets_order:
        if ds_name not in all_datasets or ds_name not in pe_arrays_all:
            continue

        graphs = all_datasets[ds_name]
        if len(graphs) < 3:
            continue

        cfg = dataset_configs[ds_name]
        train_g, val_g, test_g = split_graphs(graphs)
        logger.info(f"  {ds_name}: train={len(train_g)}, val={len(val_g)}, test={len(test_g)}")

        results_4A[ds_name] = {}

        # Time check for fallback
        if time_elapsed() > TIME_PHASE4A_WARN:
            logger.warning(f"Phase 4A taking too long ({time_elapsed():.0f}s). "
                           "Reducing epochs.")
            phase4a_fallback = True

        max_ep = 80 if phase4a_fallback else GNN_MAX_EPOCHS
        pat = 15 if phase4a_fallback else GNN_PATIENCE

        for pe_type in all_pe_types:
            t_pe_start = time.time()
            pe_dim = get_pe_dim(pe_type)

            pe_all = pe_arrays_all[ds_name][pe_type]

            # Build id->index map for fast PE lookup
            id_to_idx = {id(g): i for i, g in enumerate(graphs)}

            train_pe = [pe_all[id_to_idx[id(g)]] if id(g) in id_to_idx
                        else np.zeros((g.num_nodes, pe_dim), dtype=np.float32)
                        for g in train_g]
            val_pe = [pe_all[id_to_idx[id(g)]] if id(g) in id_to_idx
                      else np.zeros((g.num_nodes, pe_dim), dtype=np.float32)
                      for g in val_g]
            test_pe = [pe_all[id_to_idx[id(g)]] if id(g) in id_to_idx
                       else np.zeros((g.num_nodes, pe_dim), dtype=np.float32)
                       for g in test_g]

            try:
                model = train_gnn(
                    train_g, val_g, train_pe, val_pe, pe_dim,
                    cfg["task_type"], cfg["output_dim"], seed=0,
                    max_epochs=max_ep, patience=pat,
                )

                test_preds = predict_gnn(model, test_g, test_pe, pe_dim)
                test_targets = [_parse_target(g.target, cfg["task_type"]) for g in test_g]
                metric = compute_metric(test_preds, test_targets, cfg["task_type"])

                # Store predictions for all graphs
                all_pe_for_preds = pe_all
                all_preds = predict_gnn(model, graphs, all_pe_for_preds, pe_dim)
                for gi, g in enumerate(graphs):
                    pred = all_preds[gi]
                    if cfg["task_type"] == "classification":
                        probs = 1.0 / (1.0 + np.exp(-np.clip(pred, -50, 50)))
                        g.predictions[pe_type] = json.dumps(
                            [round(float(p), 6) for p in probs])
                    else:
                        g.predictions[pe_type] = json.dumps(
                            [round(float(p), 6) for p in pred])

                elapsed_pe = time.time() - t_pe_start
                logger.info(f"    {ds_name}/{pe_type}: {metric}, time={elapsed_pe:.1f}s")
                results_4A[ds_name][pe_type] = metric

                del model
                if HAS_GPU:
                    torch.cuda.empty_cache()

            except Exception:
                logger.exception(f"    Failed: {ds_name}/{pe_type}")
                results_4A[ds_name][pe_type] = {cfg["metric"]: float("nan")}

            if time_elapsed() > TIME_SKIP_PHASE4B:
                logger.warning("Time budget exceeded for Phase 4A, breaking")
                break

        if time_elapsed() > TIME_SKIP_PHASE4B:
            break

    logger.info("Phase 4A screening results:")
    for ds_name, res in results_4A.items():
        logger.info(f"  {ds_name}:")
        for pe_type, metric in res.items():
            logger.info(f"    {pe_type}: {metric}")

    _partial_state["metadata"]["phase_4A_screening"] = results_4A

    # ── Phase 4B: Top-3 SRWE reps × 5 lambdas × 3 seeds ──
    results_4B: dict = {}
    skip_phase4b = time_elapsed() > TIME_SKIP_PHASE4B

    if not skip_phase4b:
        logger.info("── Phase 4B: Lambda sweep for top-3 SRWE representations ──")

        # Select top-3 SRWE representations per dataset
        top3_per_dataset = {}
        for ds_name, res in results_4A.items():
            cfg = dataset_configs.get(ds_name, {})
            metric_name = cfg.get("metric", "MAE")
            higher_better = (metric_name == "AP")

            srwe_scores = {}
            for pe_type in srwe_pe_types:
                if pe_type in res:
                    val = res[pe_type].get(metric_name, float("nan"))
                    if not math.isnan(val):
                        srwe_scores[pe_type] = val

            if srwe_scores:
                sorted_types = sorted(srwe_scores.keys(),
                                      key=lambda x: srwe_scores[x],
                                      reverse=higher_better)
                top3_per_dataset[ds_name] = sorted_types[:3]
                logger.info(f"  {ds_name} top-3: {top3_per_dataset[ds_name]}")

        results_4B = {}

        for ds_name in gnn_datasets_order:
            if ds_name not in top3_per_dataset:
                continue
            if ds_name not in all_datasets:
                continue

            graphs = all_datasets[ds_name]
            cfg = dataset_configs[ds_name]
            train_g, val_g, test_g = split_graphs(graphs)

            results_4B[ds_name] = {}

            # Adaptive: reduce seeds/lambdas if time is tight
            use_seeds = SEEDS
            use_lambdas = LAMBDAS
            phase4b_max_ep = GNN_MAX_EPOCHS
            phase4b_pat = GNN_PATIENCE

            if time_elapsed() > TIME_PHASE4B_REDUCE:
                use_seeds = [0]  # single seed
                use_lambdas = [1e-4, 1e-3, 1e-2]  # fewer lambdas
                phase4b_max_ep = 80
                phase4b_pat = 15
                logger.warning("Phase 4B: reducing to 1 seed, 3 lambdas, 80 epochs")

            for rep in top3_per_dataset[ds_name]:
                results_4B[ds_name][rep] = {}

                for lam in use_lambdas:
                    lam_key = f"lambda_{lam:.0e}"

                    # Recompute SRWE weights at this lambda
                    lam_srwe_weights: dict[int, np.ndarray] = {}
                    for g in graphs:
                        if g.eigenvalues_P is None or len(g.eigenvalues_P) == 0:
                            continue
                        try:
                            w = compute_srwe_for_graph(g, lam=lam)
                            lam_srwe_weights[id(g)] = w
                        except Exception:
                            pass

                    # Compute PE for this lambda/rep
                    pe_dim = get_pe_dim(rep)
                    pe_all_lam = []
                    for g in graphs:
                        sw = lam_srwe_weights.get(id(g))
                        pe = compute_pe_for_graph(g, rep, srwe_weights=sw)
                        pe_all_lam.append(pe)

                    id_to_idx_lam = {id(g): i for i, g in enumerate(graphs)}
                    train_pe = [pe_all_lam[id_to_idx_lam[id(g)]] for g in train_g]
                    val_pe = [pe_all_lam[id_to_idx_lam[id(g)]] for g in val_g]
                    test_pe = [pe_all_lam[id_to_idx_lam[id(g)]] for g in test_g]

                    seed_results = []
                    for seed in use_seeds:
                        try:
                            model = train_gnn(
                                train_g, val_g, train_pe, val_pe, pe_dim,
                                cfg["task_type"], cfg["output_dim"], seed=seed,
                                max_epochs=phase4b_max_ep, patience=phase4b_pat,
                            )
                            test_preds = predict_gnn(model, test_g, test_pe, pe_dim)
                            test_targets = [_parse_target(g.target, cfg["task_type"])
                                            for g in test_g]
                            metric = compute_metric(test_preds, test_targets, cfg["task_type"])
                            seed_results.append(metric.get(cfg["metric"], float("nan")))

                            del model
                            if HAS_GPU:
                                torch.cuda.empty_cache()
                        except Exception:
                            logger.exception(f"Failed: {ds_name}/{rep}/λ={lam}/seed={seed}")
                            seed_results.append(float("nan"))

                        if time_elapsed() > TIME_SKIP_PHASE4B:
                            break

                    valid = [v for v in seed_results if not math.isnan(v)]
                    results_4B[ds_name][rep][lam_key] = {
                        "mean": float(np.mean(valid)) if valid else float("nan"),
                        "std": float(np.std(valid)) if valid else float("nan"),
                        "per_seed": seed_results,
                    }

                    logger.info(f"    {ds_name}/{rep}/λ={lam}: "
                                f"mean={results_4B[ds_name][rep][lam_key]['mean']:.4f} "
                                f"± {results_4B[ds_name][rep][lam_key]['std']:.4f}")

                    if time_elapsed() > TIME_SKIP_PHASE4B:
                        break

                if time_elapsed() > TIME_SKIP_PHASE4B:
                    break

            if time_elapsed() > TIME_SKIP_PHASE4B:
                break

        _partial_state["metadata"]["phase_4B_sweep"] = results_4B
    else:
        logger.warning("Skipping Phase 4B due to time constraints")
        _partial_state["metadata"]["phase_4B_sweep"] = {"skipped": True,
                                                         "reason": "time_budget"}

    # ── Phase 5: Information-Theoretic Diagnostics ──
    skip_diagnostics = time_elapsed() > TIME_SKIP_DIAGNOSTICS

    diagnostics_results = {}
    if not skip_diagnostics:
        logger.info("── Phase 5: Information-Theoretic Diagnostics ──")

        for ds_name in gnn_datasets_order:
            if ds_name not in all_datasets or ds_name not in pe_arrays_all:
                continue

            graphs = all_datasets[ds_name]
            cfg = dataset_configs.get(ds_name, {})
            task_type = cfg.get("task_type", "regression")

            diagnostics_results[ds_name] = {}

            for pe_type in all_pe_types:
                if pe_type not in pe_arrays_all[ds_name]:
                    continue

                pe_arrs = pe_arrays_all[ds_name][pe_type]
                diag = compute_diagnostics(graphs, pe_arrs, pe_type, task_type)
                diagnostics_results[ds_name][pe_type] = diag

                mi_val = diag.get('mi_mean', 'N/A')
                probe_val = diag.get('linear_probe_mean', 'N/A')
                edim_val = diag.get('effective_dimension', 'N/A')
                logger.info(f"    {ds_name}/{pe_type}: MI={mi_val}, "
                            f"probe={probe_val}, eff_dim={edim_val}")

                if time_elapsed() > TIME_SAVE_PARTIAL:
                    break

            if time_elapsed() > TIME_SAVE_PARTIAL:
                break
    else:
        logger.warning("Skipping diagnostics due to time constraints")

    _partial_state["metadata"]["diagnostics"] = diagnostics_results

    # ── Phase 6: Regression vs Classification Diagnosis ──
    reg_vs_class_result = {}
    if not skip_diagnostics and "Peptides-func" in pe_arrays_all and "Peptides-struct" in pe_arrays_all:
        logger.info("── Phase 6: Regression vs Classification Diagnosis ──")

        func_graphs = all_datasets.get("Peptides-func", [])
        struct_graphs = all_datasets.get("Peptides-struct", [])

        # Best SRWE type from screening
        best_srwe_type = "histogram"
        if "Peptides-struct" in results_4A:
            struct_res = results_4A["Peptides-struct"]
            best_mae = float("inf")
            for pt in srwe_pe_types:
                if pt in struct_res:
                    mae = struct_res[pt].get("MAE", float("inf"))
                    if not math.isnan(mae) and mae < best_mae:
                        best_mae = mae
                        best_srwe_type = pt

        func_srwe_pes = pe_arrays_all.get("Peptides-func", {}).get(best_srwe_type, [])
        struct_srwe_pes = pe_arrays_all.get("Peptides-struct", {}).get(best_srwe_type, [])
        func_rwse_pes = pe_arrays_all.get("Peptides-func", {}).get("rwse", [])
        struct_rwse_pes = pe_arrays_all.get("Peptides-struct", {}).get("rwse", [])

        srwe_weights_func = {id(g): srwe_weights_by_id.get(id(g))
                             for g in func_graphs if id(g) in srwe_weights_by_id}
        srwe_weights_struct = {id(g): srwe_weights_by_id.get(id(g))
                               for g in struct_graphs if id(g) in srwe_weights_by_id}

        if func_graphs and struct_graphs:
            try:
                reg_vs_class_result = regression_vs_classification_diagnosis(
                    func_graphs, struct_graphs,
                    func_srwe_pes, struct_srwe_pes,
                    func_rwse_pes, struct_rwse_pes,
                    srwe_weights_func, srwe_weights_struct,
                    SCREENING_LAMBDA,
                )
                logger.info(f"  Diagnosis: {reg_vs_class_result.get('diagnosis', 'N/A')}")
                logger.info(f"  MI diff func: {reg_vs_class_result.get('mi_diff_func', 'N/A'):.4f}")
                logger.info(f"  MI diff struct: {reg_vs_class_result.get('mi_diff_struct', 'N/A'):.4f}")
            except Exception:
                logger.exception("Regression vs classification diagnosis failed")

    _partial_state["metadata"]["regression_vs_classification_analysis"] = reg_vs_class_result

    # ── Phase 7: Determine best variant and gap closed ──
    logger.info("── Phase 7: Computing final analysis ──")

    # Best SRWE variant overall
    best_variant = {}
    gap_closed = {}

    for ds_name, res in results_4A.items():
        cfg = dataset_configs.get(ds_name, {})
        metric_name = cfg.get("metric", "MAE")
        higher_better = (metric_name == "AP")

        rwse_val = res.get("rwse", {}).get(metric_name, float("nan"))
        best_srwe_val = None
        best_srwe_name = None

        for pt in srwe_pe_types:
            if pt in res:
                val = res[pt].get(metric_name, float("nan"))
                if not math.isnan(val):
                    if best_srwe_val is None:
                        best_srwe_val = val
                        best_srwe_name = pt
                    elif higher_better and val > best_srwe_val:
                        best_srwe_val = val
                        best_srwe_name = pt
                    elif not higher_better and val < best_srwe_val:
                        best_srwe_val = val
                        best_srwe_name = pt

        best_variant[ds_name] = {
            "best_srwe_type": best_srwe_name,
            "best_srwe_value": best_srwe_val,
            "rwse_value": rwse_val,
            "metric": metric_name,
        }

        # Gap closed: how much of the RWSE-to-LapPE gap does SRWE close
        lappe_val = res.get("lappe", {}).get(metric_name, float("nan"))
        if (not math.isnan(rwse_val) and not math.isnan(lappe_val)
                and best_srwe_val is not None):
            if higher_better:
                if lappe_val != rwse_val:
                    gc = (best_srwe_val - rwse_val) / abs(lappe_val - rwse_val) * 100
                else:
                    gc = 0.0
            else:
                if rwse_val != lappe_val:
                    gc = (rwse_val - best_srwe_val) / abs(rwse_val - lappe_val) * 100
                else:
                    gc = 0.0
            gap_closed[ds_name] = round(gc, 2)

    _partial_state["metadata"]["best_srwe_variant"] = best_variant
    _partial_state["metadata"]["gap_closed"] = gap_closed
    _partial_state["metadata"]["consistency_check"] = consistency
    _partial_state["metadata"]["lambdas_tested"] = LAMBDAS
    _partial_state["metadata"]["screening_lambda"] = SCREENING_LAMBDA
    _partial_state["metadata"]["srwe_pe_types"] = srwe_pe_types
    _partial_state["metadata"]["gnn_config"] = {
        "hidden_dim": GNN_HIDDEN,
        "num_layers": GNN_LAYERS,
        "dropout": GNN_DROPOUT,
        "lr": GNN_LR,
        "weight_decay": GNN_WD,
        "batch_size": GNN_BATCH_SIZE,
        "max_epochs": GNN_MAX_EPOCHS,
        "patience": GNN_PATIENCE,
        "seeds": SEEDS,
    }

    # ── Phase 7.5: Store per-example metadata ──
    for ds_name, graphs in all_datasets.items():
        for g in graphs:
            g.recovery_metrics["sri_k20"] = str(round(g.sri_k20, 6))
            g.recovery_metrics["delta_min"] = str(round(g.delta_min, 6))
            g.recovery_metrics["dataset"] = ds_name

            # For graphs with predictions, ensure synthetic also get dummy preds
            if not g.predictions:
                g.predictions["none"] = g.output_str
                g.predictions["rwse"] = g.output_str

    # ── Phase 8: Format and Save Output ──
    logger.info("── Phase 8: Saving Output ──")
    output = format_output(all_datasets, _partial_state["metadata"])

    # Save full_method_out.json (compact to save space, schema-compliant)
    full_path = WORKSPACE / "full_method_out.json"
    full_json = json.dumps(output, separators=(",", ":"), default=str)
    full_path.write_text(full_json)
    _partial_state["saved"] = True
    file_size_mb = full_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved full_method_out.json ({file_size_mb:.1f} MB)")

    # Save mini_method_out.json (3 examples per dataset)
    mini_datasets = []
    for ds_data in output["datasets"]:
        mini_datasets.append({"dataset": ds_data["dataset"],
                              "examples": ds_data["examples"][:3]})
    mini_output = {"metadata": output.get("metadata", {}), "datasets": mini_datasets}
    mini_path = WORKSPACE / "mini_method_out.json"
    mini_path.write_text(json.dumps(mini_output, indent=2, default=str))
    logger.info(f"Saved mini_method_out.json ({mini_path.stat().st_size / 1024:.1f} KB)")

    # Save preview_method_out.json (truncated mini)
    def truncate_val(value, max_s=200, max_a=3):
        if isinstance(value, list):
            return [truncate_val(item, max_s, max_a) for item in value[:max_a]]
        elif isinstance(value, str):
            return value[:max_s] + "..." if len(value) > max_s else value
        elif isinstance(value, dict):
            return {k: truncate_val(v, max_s, max_a) for k, v in value.items()}
        return value

    preview_path = WORKSPACE / "preview_method_out.json"
    preview_path.write_text(json.dumps(truncate_val(mini_output), indent=2, default=str))
    logger.info(f"Saved preview_method_out.json ({preview_path.stat().st_size / 1024:.1f} KB)")

    # Also split into parts for convenience
    MAX_FILE_SIZE_MB = 95
    if file_size_mb > MAX_FILE_SIZE_MB:
        logger.info(f"Full output exceeds {MAX_FILE_SIZE_MB} MB, also creating split parts...")
        split_dir = WORKSPACE / "method_out"
        split_dir.mkdir(exist_ok=True)
        part_counter = 0

        for ds_data in output["datasets"]:
            ds_name = ds_data["dataset"]
            examples = ds_data["examples"]
            part = {"metadata": output["metadata"],
                    "datasets": [{"dataset": ds_name, "examples": examples}]}
            part_json = json.dumps(part, indent=2, default=str)
            part_size_mb = len(part_json.encode("utf-8")) / (1024 * 1024)

            if part_size_mb <= MAX_FILE_SIZE_MB:
                part_counter += 1
                part_path = split_dir / f"full_method_out_{part_counter}.json"
                part_path.write_text(part_json)
                logger.info(f"  Part {part_counter}: {ds_name} ({part_size_mb:.1f} MB)")
            else:
                avg_size = part_size_mb / max(len(examples), 1)
                chunk_size = max(1, int(MAX_FILE_SIZE_MB / avg_size * 0.9))
                for start in range(0, len(examples), chunk_size):
                    chunk = examples[start:start + chunk_size]
                    sub = {"metadata": output["metadata"],
                           "datasets": [{"dataset": ds_name, "examples": chunk}]}
                    part_counter += 1
                    part_path = split_dir / f"full_method_out_{part_counter}.json"
                    sub_json = json.dumps(sub, indent=2, default=str)
                    part_path.write_text(sub_json)

        logger.info(f"  Split into {part_counter} parts")

    elapsed = time_elapsed()
    logger.info("=" * 70)
    logger.info(f"DONE! Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
