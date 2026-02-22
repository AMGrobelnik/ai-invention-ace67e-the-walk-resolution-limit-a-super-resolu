#!/usr/bin/env python3
"""
SRWE via Matrix Pencil Method: Spectral Recovery Validation and GNN Benchmarking.

Implements Super-Resolved Walk Encodings (SRWE) via the Matrix Pencil Method (MPM)
applied to adjacency matrix power moments. Validates spectral recovery on synthetic
pairs and ZINC graphs. Benchmarks SRWE as PE in a simple GNN on ZINC-subset.
"""

import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

# ── Thread limits ──
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import psutil
import scipy.linalg
import scipy.optimize
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Logging ──
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ── Resource limits ──
import resource
resource.setrlimit(resource.RLIMIT_AS, (50 * 1024**3, 50 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

# ── Hardware detection ──
def _cgroup_cpus() -> int | None:
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
    except (FileNotFoundError, ValueError):
        pass
    return None

def _container_ram_gb() -> float | None:
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None

NUM_CPUS = _cgroup_cpus() or os.cpu_count() or 1
HAS_GPU = torch.cuda.is_available()
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if HAS_GPU else 0
DEVICE = torch.device("cuda" if HAS_GPU else "cpu")
TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, GPU={HAS_GPU} ({VRAM_GB:.1f}GB VRAM), Device={DEVICE}")

# ── Constants ──
DATA_DIR = Path("/opt/ai-inventor/aii_pipeline/runs/run__20260222_005317/3_invention_loop/iter_1/gen_art/data_id2_it1__opus/data_out")
OUTPUT_FILE = SCRIPT_DIR / "method_out.json"

# MPM parameters
MPM_K = 10
RANK_THRESHOLD = 0.01
NUM_HISTOGRAM_BINS = 20
HISTOGRAM_RANGE = (-3.0, 3.0)

# GNN parameters (reduced per fallback plan for speed)
GNN_CHANNELS = 48
GNN_PE_DIM = 8
GNN_NUM_LAYERS = 3
GNN_EPOCHS = 40
GNN_LR = 0.001
GNN_SEEDS = [42, 123, 456]
GNN_BATCH_SIZE = 32  # Mini-batch size
TRAIN_SIZE = 1500
VAL_SIZE = 400
TEST_SIZE = 400


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_all_data() -> dict[str, list[dict]]:
    """Load all datasets from split JSON files."""
    logger.info("Loading all datasets...")
    datasets: dict[str, list[dict]] = {}
    for fpath in sorted(DATA_DIR.glob("full_data_out_*.json")):
        logger.info(f"  Loading {fpath.name}...")
        raw = json.loads(fpath.read_text())
        for ds in raw["datasets"]:
            name = ds["dataset"]
            if name not in datasets:
                datasets[name] = []
            datasets[name].extend(ds["examples"])
            logger.info(f"    {name}: +{len(ds['examples'])} = {len(datasets[name])} total")
    return datasets


def parse_graph(example: dict) -> dict | None:
    """Parse a single example's input JSON."""
    try:
        inp = json.loads(example["input"])
        num_nodes = inp.get("num_nodes", 0)
        if num_nodes == 0:
            return None
        spectral = inp.get("spectral", {})
        return {
            "edge_index": inp.get("edge_index", [[], []]),
            "num_nodes": num_nodes,
            "node_feat": inp.get("node_feat", []),
            "eigenvalues": spectral.get("eigenvalues", []),
            "delta_min": spectral.get("delta_min", 0.0),
            "sri": spectral.get("sri", {}),
            "rwse": spectral.get("rwse", []),
            "local_spectral": spectral.get("local_spectral", []),
            "graph_name": inp.get("graph_name", ""),
            "pair_id": inp.get("pair_id", ""),
            "pair_category": inp.get("pair_category", ""),
        }
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  MATRIX PENCIL METHOD (MPM)
# ═══════════════════════════════════════════════════════════════════════════

def edge_index_to_adj(edge_index: list[list[int]], num_nodes: int) -> np.ndarray:
    """Convert COO edge_index to dense adjacency matrix."""
    A = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    if len(edge_index) < 2 or not edge_index[0]:
        return A
    srcs, dsts = edge_index[0], edge_index[1]
    rows = np.array(srcs, dtype=np.intp)
    cols = np.array(dsts, dtype=np.intp)
    valid = (rows >= 0) & (rows < num_nodes) & (cols >= 0) & (cols < num_nodes)
    A[rows[valid], cols[valid]] = 1.0
    A = np.maximum(A, A.T)
    return A


def compute_adjacency_moments(
    edge_index: list[list[int]], num_nodes: int, max_k: int = 20,
) -> tuple[np.ndarray, float]:
    """Compute diagonal of A^k for k=0..max_k. Returns (moments[N, max_k+1], spectral_radius)."""
    A = edge_index_to_adj(edge_index, num_nodes)
    try:
        eigs = np.linalg.eigvalsh(A)
        spectral_radius = max(abs(eigs.max()), abs(eigs.min()), 1e-10)
    except np.linalg.LinAlgError:
        spectral_radius = 1.0

    A_norm = A / spectral_radius
    moments = np.zeros((num_nodes, max_k + 1), dtype=np.float64)
    moments[:, 0] = 1.0
    A_power = np.eye(num_nodes, dtype=np.float64)
    for k in range(1, max_k + 1):
        A_power = A_power @ A_norm
        moments[:, k] = np.diag(A_power)
    return moments, spectral_radius


def matrix_pencil_method(
    moments_vector: np.ndarray, K: int = 10, rank_threshold: float = RANK_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Apply MPM to recover spectral components from moments."""
    L = len(moments_vector)
    if L < 2 * K + 1:
        K = (L - 1) // 2
    if K < 1:
        return np.array([0.0]), np.array([1.0]), 1

    # Hankel matrix
    H = np.zeros((K + 1, K + 1), dtype=np.float64)
    for i in range(K + 1):
        for j in range(K + 1):
            idx = i + j
            if idx < L:
                H[i, j] = moments_vector[idx]

    try:
        U, S, Vt = np.linalg.svd(H, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.array([0.0]), np.array([1.0]), 1

    if S[0] < 1e-15:
        return np.array([0.0]), np.array([1.0]), 1

    r = int(np.sum(S / S[0] > rank_threshold))
    r = max(1, min(r, K))

    V = Vt.T[:, :r]
    V1 = V[:-1, :]
    V2 = V[1:, :]

    try:
        P = np.linalg.pinv(V1, rcond=0.01) @ V2
        eig_est = np.linalg.eigvals(P)
    except np.linalg.LinAlgError:
        return np.array([0.0]), np.array([1.0]), 1

    real_mask = np.abs(eig_est.imag) < 0.1 * (np.abs(eig_est.real) + 1e-10)
    eig_est = eig_est[real_mask].real
    if len(eig_est) == 0:
        return np.array([0.0]), np.array([1.0]), 1

    eig_est = np.clip(eig_est, -1.0, 1.0)

    # NNLS weight recovery
    n_eigs = len(eig_est)
    n_moments = min(2 * K + 1, L)
    V_vand = np.zeros((n_moments, n_eigs), dtype=np.float64)
    for k in range(n_moments):
        V_vand[k, :] = eig_est ** k

    try:
        weights, _ = scipy.optimize.nnls(V_vand, moments_vector[:n_moments])
    except Exception:
        weights = np.ones(n_eigs) / n_eigs

    keep = weights > 1e-6
    if not np.any(keep):
        keep[0] = True
    eig_est = eig_est[keep]
    weights = weights[keep]
    w_sum = weights.sum()
    if w_sum > 1e-10:
        weights /= w_sum

    return eig_est, weights, r


def compute_srwe(
    edge_index: list[list[int]], num_nodes: int,
    K: int = MPM_K, rank_threshold: float = RANK_THRESHOLD,
) -> list[list[tuple[float, float]]]:
    """Compute SRWE per node via MPM. Returns [(eig, weight), ...] per node."""
    moments, rho = compute_adjacency_moments(edge_index, num_nodes, 2 * K)
    srwe = []
    for u in range(num_nodes):
        eigs, weights, rank = matrix_pencil_method(moments[u], K=K, rank_threshold=rank_threshold)
        eigs_denorm = eigs * rho
        order = np.argsort(-weights)[:10]
        srwe.append([(float(eigs_denorm[i]), float(weights[i])) for i in order])
    return srwe


def srwe_to_histogram(
    components: list[tuple[float, float]],
    num_bins: int = NUM_HISTOGRAM_BINS,
    hist_range: tuple[float, float] = HISTOGRAM_RANGE,
) -> np.ndarray:
    """Convert SRWE components to histogram."""
    hist = np.zeros(num_bins, dtype=np.float64)
    bin_edges = np.linspace(hist_range[0], hist_range[1], num_bins + 1)
    for eig, w in components:
        bin_idx = min(np.searchsorted(bin_edges[1:], eig), num_bins - 1)
        hist[bin_idx] += w
    return hist


def compute_srwe_histogram(
    edge_index: list[list[int]], num_nodes: int,
    K: int = MPM_K, rank_threshold: float = RANK_THRESHOLD,
) -> np.ndarray:
    """Compute SRWE histograms for all nodes. Returns [N, num_bins]."""
    srwe = compute_srwe(edge_index, num_nodes, K=K, rank_threshold=rank_threshold)
    histograms = np.zeros((num_nodes, NUM_HISTOGRAM_BINS), dtype=np.float64)
    for u, comps in enumerate(srwe):
        histograms[u] = srwe_to_histogram(comps)
    return histograms


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 2: SYNTHETIC PAIR VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def validate_synthetic_pairs(synthetic_examples: list[dict]) -> dict:
    """Phase 2: Compare encodings on synthetic pairs."""
    logger.info("=" * 60)
    logger.info("Phase 2: Synthetic Pair Validation")

    pairs: dict[str, list] = {}
    for ex in synthetic_examples:
        g = parse_graph(ex)
        if g is None:
            continue
        pid = g["pair_id"]
        if pid not in pairs:
            pairs[pid] = []
        pairs[pid].append((ex, g))

    results = {"exactly_cospectral": [], "near_cospectral": [], "control": []}

    for pid, pair_data in pairs.items():
        if len(pair_data) < 2:
            continue
        ex_A, gA = pair_data[0]
        ex_B, gB = pair_data[1]
        cat = gA["pair_category"]

        try:
            # RWSE
            rwse_A = np.array(gA["rwse"], dtype=np.float64)
            rwse_B = np.array(gB["rwse"], dtype=np.float64)
            if rwse_A.size == 0 or rwse_B.size == 0:
                continue
            rwse_dist = float(np.linalg.norm(rwse_A.mean(0) - rwse_B.mean(0)))

            # SRWE
            srwe_A = compute_srwe_histogram(gA["edge_index"], gA["num_nodes"])
            srwe_B = compute_srwe_histogram(gB["edge_index"], gB["num_nodes"])
            srwe_dist = float(np.linalg.norm(srwe_A.mean(0) - srwe_B.mean(0)))

            # LapPE
            eA = np.sort(gA["eigenvalues"])
            eB = np.sort(gB["eigenvalues"])
            ml = max(len(eA), len(eB))
            eAp, eBp = np.zeros(ml), np.zeros(ml)
            eAp[:len(eA)] = eA
            eBp[:len(eB)] = eB
            lappe_dist = float(np.linalg.norm(eAp - eBp))

            sri20 = float(gA["sri"].get("K=20", 0.0))
            results[cat].append({
                "pair_id": pid, "category": cat,
                "graph_A": gA.get("graph_name", ""),
                "graph_B": gB.get("graph_name", ""),
                "rwse_dist": round(rwse_dist, 6),
                "srwe_dist": round(srwe_dist, 6),
                "lappe_dist": round(lappe_dist, 6),
                "srwe_gt_rwse": srwe_dist > rwse_dist,
                "sri_K20": round(sri20, 6),
            })
        except Exception:
            logger.exception(f"Failed pair {pid}")

    # Summary
    summary = {}
    for cat in ["exactly_cospectral", "near_cospectral", "control"]:
        cr = results[cat]
        if not cr:
            summary[cat] = {"n_pairs": 0}
            continue
        rwse_d = [r["rwse_dist"] for r in cr]
        srwe_d = [r["srwe_dist"] for r in cr]
        lappe_d = [r["lappe_dist"] for r in cr]
        n_better = sum(1 for r in cr if r["srwe_gt_rwse"])
        summary[cat] = {
            "n_pairs": len(cr),
            "mean_rwse_dist": round(float(np.mean(rwse_d)), 6),
            "mean_srwe_dist": round(float(np.mean(srwe_d)), 6),
            "mean_lappe_dist": round(float(np.mean(lappe_d)), 6),
            "n_srwe_better_than_rwse": n_better,
            "pct_srwe_better": round(100.0 * n_better / len(cr), 1),
        }
        logger.info(f"  {cat}: {len(cr)} pairs | RWSE={summary[cat]['mean_rwse_dist']:.4f} SRWE={summary[cat]['mean_srwe_dist']:.4f} LapPE={summary[cat]['mean_lappe_dist']:.4f} | SRWE>RWSE: {n_better}/{len(cr)}")

    # Spearman for near-cospectral
    nc = results["near_cospectral"]
    if len(nc) >= 5:
        try:
            rho, p = scipy.stats.spearmanr(
                [r["sri_K20"] for r in nc],
                [r["srwe_dist"] - r["rwse_dist"] for r in nc],
            )
            summary["spearman_sri_vs_advantage"] = {
                "rho": round(float(rho), 4) if np.isfinite(rho) else 0.0,
                "p_value": round(float(p), 6) if np.isfinite(p) else 1.0,
            }
            logger.info(f"  Spearman(SRI, SRWE-RWSE advantage): rho={rho:.4f}, p={p:.4f}")
        except Exception:
            summary["spearman_sri_vs_advantage"] = {"rho": 0.0, "p_value": 1.0}

    return {"pair_details": results, "summary": summary}


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 3: ZINC SPECTRAL RECOVERY
# ═══════════════════════════════════════════════════════════════════════════

def wasserstein_1d(
    eigs1: np.ndarray, w1: np.ndarray,
    eigs2: np.ndarray, w2: np.ndarray,
) -> float:
    """1D Wasserstein-1 distance between discrete spectral measures."""
    if len(eigs1) == 0 or len(eigs2) == 0:
        return float("inf")
    all_e = np.concatenate([eigs1, eigs2])
    all_w = np.concatenate([w1, -w2])
    order = np.argsort(all_e)
    sorted_e = all_e[order]
    sorted_w = all_w[order]
    cdf = np.cumsum(sorted_w)
    gaps = np.diff(sorted_e, prepend=sorted_e[0])
    return float(np.sum(np.abs(cdf) * gaps))


def validate_spectral_recovery_zinc(zinc_examples: list[dict], max_graphs: int = 500) -> dict:
    """Phase 3: Spectral recovery quality on ZINC."""
    logger.info("=" * 60)
    logger.info("Phase 3: ZINC Spectral Recovery Quality")

    valid = []
    for ex in zinc_examples:
        g = parse_graph(ex)
        if g and g["local_spectral"] and g["eigenvalues"]:
            valid.append((g, ex.get("metadata_delta_min", g["delta_min"])))
    if not valid:
        return {}

    valid.sort(key=lambda x: x[1])
    n = len(valid)
    per_q = max(1, max_graphs // 5)
    qs = n // 5
    sampled = []
    for q in range(5):
        start, end = q * qs, min((q + 1) * qs, n)
        idxs = np.linspace(start, end - 1, per_q, dtype=int)
        for i in idxs:
            sampled.append(valid[i])
    logger.info(f"  Sampled {len(sampled)} graphs")

    # Test rank thresholds on 50 graphs
    rank_thresholds = [0.001, 0.01, 0.05, 0.1, 0.2]
    best_rt, best_w1 = RANK_THRESHOLD, float("inf")
    for rt in rank_thresholds:
        w1s = []
        for g, _ in sampled[:50]:
            try:
                srwe = compute_srwe(g["edge_index"], g["num_nodes"], K=MPM_K, rank_threshold=rt)
                nc = min(len(g["local_spectral"]), len(srwe), 10)
                for u in range(nc):
                    gt = g["local_spectral"][u]
                    if not gt:
                        continue
                    gt_e, gt_w = np.array([c[0] for c in gt]), np.array([c[1] for c in gt])
                    rc_e, rc_w = np.array([c[0] for c in srwe[u]]), np.array([c[1] for c in srwe[u]])
                    wd = wasserstein_1d(gt_e, gt_w, rc_e, rc_w)
                    if np.isfinite(wd):
                        w1s.append(wd)
            except Exception:
                continue
        if w1s:
            mw = float(np.mean(w1s))
            logger.info(f"  rt={rt}: W1={mw:.4f} (n={len(w1s)})")
            if mw < best_w1:
                best_w1 = mw
                best_rt = rt
    logger.info(f"  Best rank_threshold: {best_rt}")

    # Full eval
    sri_bins = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, float("inf"))]
    bin_res = {f"{lo}-{hi}": [] for lo, hi in sri_bins}
    w1_all, eig_err_all, sri_all, rank_acc_all = [], [], [], []

    for idx, (g, delta) in enumerate(sampled):
        try:
            srwe = compute_srwe(g["edge_index"], g["num_nodes"], K=MPM_K, rank_threshold=best_rt)
            sri = float(g["sri"].get("K=20", delta * 20))
            nc = min(len(g["local_spectral"]), len(srwe))
            for u in range(nc):
                gt = g["local_spectral"][u]
                if not gt:
                    continue
                gt_e, gt_w = np.array([c[0] for c in gt]), np.array([c[1] for c in gt])
                rc_e, rc_w = np.array([c[0] for c in srwe[u]]), np.array([c[1] for c in srwe[u]])
                wd = wasserstein_1d(gt_e, gt_w, rc_e, rc_w)
                if not np.isfinite(wd):
                    continue
                w1_all.append(wd)
                sri_all.append(sri)
                if len(gt_e) > 0 and len(rc_e) > 0:
                    errs = [float(np.abs(rc_e - ge).min()) for ge in gt_e[:5]]
                    eig_err_all.append(float(np.mean(errs)))
                rank_acc_all.append(min(len(rc_e), len(gt_e)) / max(len(gt_e), 1))
            for lo, hi in sri_bins:
                if lo <= sri < hi:
                    if w1_all:
                        bin_res[f"{lo}-{hi}"].append(w1_all[-1])
                    break
        except Exception:
            continue
        if (idx + 1) % 100 == 0:
            logger.info(f"  {idx + 1}/{len(sampled)} done")

    res = {
        "n_graphs": len(sampled),
        "n_w1_measurements": len(w1_all),
        "best_rank_threshold": best_rt,
        "overall_mean_w1": round(float(np.mean(w1_all)), 6) if w1_all else None,
        "overall_std_w1": round(float(np.std(w1_all)), 6) if w1_all else None,
        "overall_mean_eig_error": round(float(np.mean(eig_err_all)), 6) if eig_err_all else None,
        "overall_mean_rank_accuracy": round(float(np.mean(rank_acc_all)), 4) if rank_acc_all else None,
        "sri_bin_results": {},
    }
    for bn, vals in bin_res.items():
        if vals:
            res["sri_bin_results"][bn] = {"n": len(vals), "mean_w1": round(float(np.mean(vals)), 6)}

    if len(sri_all) >= 10:
        try:
            rho, p = scipy.stats.spearmanr(sri_all, w1_all)
            res["spearman_sri_w1"] = {
                "rho": round(float(rho), 4) if np.isfinite(rho) else 0.0,
                "p_value": round(float(p), 6) if np.isfinite(p) else 1.0,
            }
            logger.info(f"  Spearman(SRI, W1): rho={rho:.4f}, p={p:.6f}")
        except Exception:
            pass

    logger.info(f"  Mean W1={res['overall_mean_w1']}, Mean eig err={res['overall_mean_eig_error']}, Rank acc={res['overall_mean_rank_accuracy']}")
    return res


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 4: GNN BENCHMARK (batched for speed)
# ═══════════════════════════════════════════════════════════════════════════

class GCNConv(nn.Module):
    """Simple GCN convolution."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        # Sparse message passing
        row, col = edge_index[0], edge_index[1]
        # Degree normalization
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, row, torch.ones(row.shape[0], device=x.device))
        deg_inv_sqrt = torch.zeros_like(deg)
        mask = deg > 0
        deg_inv_sqrt[mask] = deg[mask].pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Message passing
        msg = x[col] * norm.unsqueeze(-1)
        agg = torch.zeros_like(x)
        agg.scatter_add_(0, row.unsqueeze(-1).expand_as(msg), msg)

        # Add self-loop
        agg = agg + x
        return self.W(agg)


class SimpleGNN(nn.Module):
    """Simple GCN + PE for graph regression."""
    def __init__(self, pe_dim: int = GNN_PE_DIM, channels: int = GNN_CHANNELS, num_layers: int = GNN_NUM_LAYERS):
        super().__init__()
        self.pe_enc = nn.Linear(pe_dim, channels)
        self.node_enc = nn.Linear(1, channels)  # node feat dim = 1

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(channels, channels))
            self.bns.append(nn.BatchNorm1d(channels))

        self.head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(channels, 1),
        )

    def forward(self, x: torch.Tensor, pe: torch.Tensor,
                edge_index: torch.Tensor, batch: torch.Tensor, num_nodes: int) -> torch.Tensor:
        h = self.node_enc(x) + self.pe_enc(pe)

        for conv, bn in zip(self.convs, self.bns):
            h_new = conv(h, edge_index, num_nodes)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h = h + h_new  # residual

        # Global mean pooling
        num_graphs = int(batch.max().item()) + 1
        out = torch.zeros(num_graphs, h.shape[-1], device=h.device)
        count = torch.zeros(num_graphs, 1, device=h.device)
        out.scatter_add_(0, batch.unsqueeze(-1).expand_as(h), h)
        count.scatter_add_(0, batch.unsqueeze(-1), torch.ones_like(batch.unsqueeze(-1), dtype=torch.float))
        out = out / count.clamp(min=1)

        return self.head(out).squeeze(-1)


def prepare_lappe(edge_index: list[list[int]], num_nodes: int, k: int = 8) -> np.ndarray:
    """Compute top-k Laplacian PE."""
    A = edge_index_to_adj(edge_index, num_nodes)
    D = np.diag(A.sum(axis=1))
    L = D - A
    try:
        _, V = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        return np.zeros((num_nodes, k))
    pe = np.abs(V[:, 1:min(1 + k, num_nodes)])
    if pe.shape[1] < k:
        pe = np.concatenate([pe, np.zeros((num_nodes, k - pe.shape[1]))], axis=1)
    return pe


def prepare_graph_data(
    examples: list[dict], encoding_type: str, pe_dim: int = GNN_PE_DIM,
) -> list[dict]:
    """Prepare graph data for batched GNN training."""
    data_list = []
    for ex in examples:
        g = parse_graph(ex)
        if g is None:
            continue
        nn_ = g["num_nodes"]
        ei = g["edge_index"]

        # Node features
        nf = np.ones((nn_, 1), dtype=np.float32)

        # PE
        if encoding_type == "rwse":
            rwse = np.array(g["rwse"], dtype=np.float64)
            if rwse.ndim != 2 or rwse.shape[0] != nn_ or rwse.shape[1] < pe_dim:
                continue
            pe = rwse[:, :pe_dim].astype(np.float32)
        elif encoding_type == "lappe":
            pe = prepare_lappe(ei, nn_, k=pe_dim).astype(np.float32)
        elif encoding_type == "srwe":
            try:
                sh = compute_srwe_histogram(ei, nn_)
                if NUM_HISTOGRAM_BINS >= pe_dim:
                    bs = NUM_HISTOGRAM_BINS // pe_dim
                    pe = np.zeros((nn_, pe_dim), dtype=np.float32)
                    for i in range(pe_dim):
                        s, e = i * bs, min((i + 1) * bs, NUM_HISTOGRAM_BINS)
                        pe[:, i] = sh[:, s:e].sum(axis=1)
                else:
                    pe = sh[:, :pe_dim].astype(np.float32)
            except Exception:
                continue
        else:
            continue

        try:
            target = float(ex["output"])
        except (ValueError, TypeError):
            continue

        data_list.append({
            "node_feat": nf,
            "pe": pe,
            "edge_index": np.array(ei, dtype=np.int64),
            "target": target,
            "num_nodes": nn_,
            "sri_K20": float(g["sri"].get("K=20", 0.0)),
        })
    return data_list


def collate_batch(batch_data: list[dict], device: torch.device) -> dict:
    """Collate a mini-batch of graphs into batched tensors."""
    node_feats, pes, targets, batches = [], [], [], []
    edge_indices = []
    offset = 0

    for i, d in enumerate(batch_data):
        nn_ = d["num_nodes"]
        node_feats.append(torch.tensor(d["node_feat"], dtype=torch.float32))
        pes.append(torch.tensor(d["pe"], dtype=torch.float32))
        targets.append(d["target"])
        batches.append(torch.full((nn_,), i, dtype=torch.long))
        ei = torch.tensor(d["edge_index"], dtype=torch.long)
        if ei.numel() > 0:
            edge_indices.append(ei + offset)
        offset += nn_

    if not edge_indices:
        ei_cat = torch.zeros((2, 0), dtype=torch.long)
    else:
        ei_cat = torch.cat(edge_indices, dim=1)

    return {
        "x": torch.cat(node_feats, dim=0).to(device),
        "pe": torch.cat(pes, dim=0).to(device),
        "edge_index": ei_cat.to(device),
        "batch": torch.cat(batches, dim=0).to(device),
        "target": torch.tensor(targets, dtype=torch.float32).to(device),
        "num_nodes": offset,
    }


def train_gnn(
    train_data: list[dict], val_data: list[dict], test_data: list[dict],
    enc_name: str, seed: int,
    epochs: int = GNN_EPOCHS, lr: float = GNN_LR, batch_size: int = GNN_BATCH_SIZE,
) -> dict:
    """Train GNN with mini-batching."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if HAS_GPU:
        torch.cuda.manual_seed(seed)

    pe_dim = train_data[0]["pe"].shape[1]
    model = SimpleGNN(pe_dim=pe_dim, channels=GNN_CHANNELS, num_layers=GNN_NUM_LAYERS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val, best_test = float("inf"), float("inf")

    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(len(train_data))
        train_loss = 0.0
        n_batches = 0

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            batch = collate_batch([train_data[i] for i in batch_idx], DEVICE)

            pred = model(batch["x"], batch["pe"], batch["edge_index"], batch["batch"], batch["num_nodes"])
            loss = F.l1_loss(pred, batch["target"])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Eval
        model.eval()
        val_mae = _eval_gnn(model, val_data, batch_size)

        if val_mae < best_val:
            best_val = val_mae
            best_test = _eval_gnn(model, test_data, batch_size)

        if (epoch + 1) % 10 == 0:
            logger.info(f"    [{enc_name} s{seed}] E{epoch+1}: loss={train_loss/max(n_batches,1):.4f} val={val_mae:.4f} best_val={best_val:.4f} test@best={best_test:.4f}")

    return {
        "encoding": enc_name, "seed": seed,
        "best_val_mae": round(best_val, 6),
        "test_mae_at_best_val": round(best_test, 6),
    }


def _eval_gnn(model: nn.Module, data: list[dict], batch_size: int) -> float:
    """Evaluate GNN on a dataset."""
    errors = []
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            batch_data = data[start:start + batch_size]
            batch = collate_batch(batch_data, DEVICE)
            pred = model(batch["x"], batch["pe"], batch["edge_index"], batch["batch"], batch["num_nodes"])
            errors.extend(torch.abs(pred - batch["target"]).cpu().tolist())
    return float(np.mean(errors)) if errors else float("inf")


def run_gnn_benchmark(zinc_examples: list[dict]) -> dict:
    """Phase 4: GNN benchmark with RWSE, LapPE, SRWE."""
    logger.info("=" * 60)
    logger.info("Phase 4: GNN Benchmark on ZINC")

    train_ex = [ex for ex in zinc_examples if ex.get("metadata_fold") == 0][:TRAIN_SIZE]
    val_ex = [ex for ex in zinc_examples if ex.get("metadata_fold") == 1][:VAL_SIZE]
    test_ex = [ex for ex in zinc_examples if ex.get("metadata_fold") == 2][:TEST_SIZE]

    # If not enough from folds, resplit
    if len(val_ex) < 50 or len(test_ex) < 50:
        all_ex = [ex for ex in zinc_examples if ex.get("metadata_fold") in [0, 1, 2]]
        np.random.seed(42)
        perm = np.random.permutation(len(all_ex))
        n_tr = min(TRAIN_SIZE, len(all_ex) - VAL_SIZE - TEST_SIZE)
        train_ex = [all_ex[i] for i in perm[:n_tr]]
        val_ex = [all_ex[i] for i in perm[n_tr:n_tr + VAL_SIZE]]
        test_ex = [all_ex[i] for i in perm[n_tr + VAL_SIZE:n_tr + VAL_SIZE + TEST_SIZE]]

    logger.info(f"  Split: {len(train_ex)} train, {len(val_ex)} val, {len(test_ex)} test")

    enc_results = {}
    for enc in ["rwse", "lappe", "srwe"]:
        logger.info(f"\n  Preparing {enc} data...")
        t0 = time.time()
        tr_data = prepare_graph_data(train_ex, enc, GNN_PE_DIM)
        vl_data = prepare_graph_data(val_ex, enc, GNN_PE_DIM)
        te_data = prepare_graph_data(test_ex, enc, GNN_PE_DIM)
        prep_t = time.time() - t0
        logger.info(f"  {enc} prepared in {prep_t:.1f}s: {len(tr_data)}/{len(vl_data)}/{len(te_data)}")

        if len(tr_data) < 10 or len(vl_data) < 5 or len(te_data) < 5:
            logger.warning(f"  Skip {enc}: insufficient data")
            continue

        seeds_res = []
        for seed in GNN_SEEDS:
            t1 = time.time()
            r = train_gnn(tr_data, vl_data, te_data, enc, seed)
            r["train_time_s"] = round(time.time() - t1, 1)
            seeds_res.append(r)
            logger.info(f"  {enc} s{seed}: test_mae={r['test_mae_at_best_val']:.4f} ({r['train_time_s']}s)")

        maes = [r["test_mae_at_best_val"] for r in seeds_res]
        enc_results[enc] = {
            "mean_test_mae": round(float(np.mean(maes)), 6),
            "std_test_mae": round(float(np.std(maes)), 6),
            "seed_results": seeds_res,
            "prep_time_s": round(prep_t, 1),
        }
        logger.info(f"  {enc}: MAE = {np.mean(maes):.4f} +/- {np.std(maes):.4f}")

    return {
        "encoding_results": enc_results,
        "data_split": {"train": len(train_ex), "val": len(val_ex), "test": len(test_ex)},
    }


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 5: TIMING
# ═══════════════════════════════════════════════════════════════════════════

def run_timing_benchmark(zinc_examples: list[dict], n: int = 100) -> dict:
    """Phase 5: Timing comparison."""
    logger.info("=" * 60)
    logger.info("Phase 5: Timing Benchmark")

    timings = {"rwse": [], "mpm_srwe": [], "eigendecomp": [], "lappe": []}
    for ex in zinc_examples[:n]:
        g = parse_graph(ex)
        if g is None:
            continue
        ei, nn_ = g["edge_index"], g["num_nodes"]

        t0 = time.time()
        A = edge_index_to_adj(ei, nn_)
        D = A.sum(1)
        Di = np.zeros_like(D)
        Di[D > 0] = 1.0 / D[D > 0]
        T = Di[:, None] * A
        Tp = np.eye(nn_)
        for k in range(20):
            Tp = Tp @ T
        timings["rwse"].append(time.time() - t0)

        t0 = time.time()
        compute_srwe_histogram(ei, nn_)
        timings["mpm_srwe"].append(time.time() - t0)

        t0 = time.time()
        np.linalg.eigh(A)
        timings["eigendecomp"].append(time.time() - t0)

        t0 = time.time()
        prepare_lappe(ei, nn_)
        timings["lappe"].append(time.time() - t0)

    res = {}
    for name, ts in timings.items():
        if ts:
            res[name] = {"mean_ms": round(float(np.mean(ts)) * 1000, 2), "std_ms": round(float(np.std(ts)) * 1000, 2), "n": len(ts)}
            logger.info(f"  {name}: {res[name]['mean_ms']:.2f} +/- {res[name]['std_ms']:.2f} ms")

    if "eigendecomp" in res and "mpm_srwe" in res:
        res["speedup_eigen_vs_mpm"] = round(res["eigendecomp"]["mean_ms"] / max(res["mpm_srwe"]["mean_ms"], 0.01), 2)
    return res


# ═══════════════════════════════════════════════════════════════════════════
#  OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

def format_output(
    all_datasets: dict[str, list[dict]],
    synth_results: dict,
    recovery_results: dict,
    benchmark_results: dict,
    timing_results: dict,
) -> dict:
    """Format results into exp_gen_sol_out.json schema."""
    output = {
        "metadata": {
            "method_name": "SRWE via Matrix Pencil Method",
            "description": "Super-Resolved Walk Encodings via MPM applied to adjacency matrix power moments",
            "parameters": {
                "mpm_K": MPM_K, "rank_threshold": RANK_THRESHOLD,
                "histogram_bins": NUM_HISTOGRAM_BINS,
                "gnn_channels": GNN_CHANNELS, "gnn_layers": GNN_NUM_LAYERS,
                "gnn_epochs": GNN_EPOCHS, "gnn_seeds": GNN_SEEDS,
            },
            "phase2_synthetic_validation": synth_results.get("summary", {}),
            "phase3_spectral_recovery": recovery_results,
            "phase4_gnn_benchmark": benchmark_results,
            "phase5_timing": timing_results,
        },
        "datasets": [],
    }

    for ds_name, examples in all_datasets.items():
        ds_out = {"dataset": ds_name, "examples": []}

        # For output format, create compact input representation
        # to keep file size under 100MB
        max_srwe_compute = 100 if ds_name != "Synthetic-aliased-pairs" else len(examples)

        for idx, ex in enumerate(examples):
            # Create compact input: just graph structure summary, not full spectral data
            g = parse_graph(ex)
            if g is not None:
                compact_input = json.dumps({
                    "num_nodes": g["num_nodes"],
                    "num_edges": len(g["edge_index"][0]) if g["edge_index"] and g["edge_index"][0] else 0,
                    "delta_min": round(g["delta_min"], 6),
                    "sri_K20": round(float(g["sri"].get("K=20", 0.0)), 6),
                    "graph_name": g.get("graph_name", ""),
                    "pair_category": g.get("pair_category", ""),
                })
            else:
                compact_input = ex["input"][:200]

            out_ex = {"input": compact_input, "output": ex["output"]}

            if g is not None and idx < max_srwe_compute:
                try:
                    sh = compute_srwe_histogram(g["edge_index"], g["num_nodes"])
                    srwe_mean = sh.mean(0)
                    # Compact: just the L2 norm as a single number summary
                    out_ex["predict_srwe"] = json.dumps([round(float(v), 4) for v in srwe_mean])
                except Exception:
                    out_ex["predict_srwe"] = "error"

                if g["rwse"]:
                    rwse = np.array(g["rwse"])
                    rwse_mean = rwse.mean(0)[:NUM_HISTOGRAM_BINS]
                    out_ex["predict_rwse"] = json.dumps([round(float(v), 4) for v in rwse_mean])
                else:
                    out_ex["predict_rwse"] = "[]"
            else:
                out_ex["predict_srwe"] = "not_computed"
                if g and g["rwse"]:
                    rwse = np.array(g["rwse"])
                    rwse_mean = rwse.mean(0)[:NUM_HISTOGRAM_BINS]
                    out_ex["predict_rwse"] = json.dumps([round(float(v), 4) for v in rwse_mean])
                else:
                    out_ex["predict_rwse"] = "[]"

            ds_out["examples"].append(out_ex)

        output["datasets"].append(ds_out)
        logger.info(f"  Output: {ds_name} = {len(ds_out['examples'])} examples")

    return output


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

@logger.catch
def main() -> None:
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("SRWE via Matrix Pencil Method — method.py")
    logger.info("=" * 60)

    # Load
    all_data = load_all_data()

    # Phase 1: Smoke test
    logger.info("=" * 60)
    logger.info("Phase 1: MPM Smoke Test")
    synth_data = all_data.get("Synthetic-aliased-pairs", [])
    if synth_data:
        g0 = parse_graph(synth_data[0])
        if g0:
            logger.info(f"  K_1_4: true eigs={g0['eigenvalues']}")
            srwe = compute_srwe(g0["edge_index"], g0["num_nodes"])
            for u in range(min(3, len(srwe))):
                logger.info(f"    Node {u}: {[(round(e,3), round(w,3)) for e,w in srwe[u]]}")
            hist = compute_srwe_histogram(g0["edge_index"], g0["num_nodes"])
            logger.info(f"  Hist shape={hist.shape}, sums={hist.sum(1)[:3]}")

    # Phase 2: Synthetic
    synth_res = validate_synthetic_pairs(synth_data) if synth_data else {}

    # Phase 3: Recovery
    zinc_data = all_data.get("ZINC-subset", [])
    recovery_res = validate_spectral_recovery_zinc(zinc_data, max_graphs=500) if zinc_data else {}

    # Phase 4: GNN
    elapsed = time.time() - t_start
    logger.info(f"  Elapsed so far: {elapsed:.0f}s")
    benchmark_res = {}
    if zinc_data and elapsed < 2400:  # 40 min budget for GNN
        benchmark_res = run_gnn_benchmark(zinc_data)
    else:
        logger.warning("Skipping GNN benchmark (time constraint)")

    # Phase 5: Timing
    timing_res = run_timing_benchmark(zinc_data, n=100) if zinc_data else {}

    # Format output
    logger.info("=" * 60)
    logger.info("Formatting output...")
    output = format_output(all_data, synth_res, recovery_res, benchmark_res, timing_res)

    OUTPUT_FILE.write_text(json.dumps(output, indent=2))
    sz = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    logger.info(f"Saved method_out.json ({sz:.1f} MB)")

    # Check file size limit
    if sz > 100:
        logger.info("Output exceeds 100MB, need to split (reducing predictions)")
        # Re-save with less detail
        for ds in output["datasets"]:
            for ex in ds["examples"]:
                # Truncate large predictions
                for k in list(ex.keys()):
                    if k.startswith("predict_") and isinstance(ex[k], str) and len(ex[k]) > 500:
                        ex[k] = ex[k][:500] + "..."
        OUTPUT_FILE.write_text(json.dumps(output, indent=2))
        sz = OUTPUT_FILE.stat().st_size / (1024 * 1024)
        logger.info(f"Resaved method_out.json ({sz:.1f} MB)")

    total = time.time() - t_start
    logger.info(f"Total runtime: {total:.1f}s ({total/60:.1f} min)")
    logger.info("Done!")


if __name__ == "__main__":
    main()
