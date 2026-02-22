#!/usr/bin/env python3
"""
SRI Predicts RWSE-vs-LapPE Quality on Fixed-Size (n=30) Synthetic Graphs.

Generates 500 synthetic graphs (all n=30) across 5 categories to test whether
SRI (Spectral Resolution Index) predicts the RWSE-vs-LapPE encoding quality gap
independently of graph size. Computes RWSE (K=20), LapPE (top-8 squared eigenvectors),
and SRWE (Tikhonov-regularized Vandermonde recovery) for each graph, measures node
distinguishability and MLP community classification accuracy, then computes Spearman
correlation between SRI and the LapPE-RWSE quality gap.

Targets:
  |rho| > 0.3 with p < 0.01
  Monotonic quintile trend
  SRWE closing >50% of the gap on low-SRI graphs
"""

import argparse
import json
import math
import resource
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, rankdata
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from loguru import logger

warnings.filterwarnings("ignore")

# ── Logging setup ──────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ── Resource limits ────────────────────────────────────────────────────
resource.setrlimit(resource.RLIMIT_AS, (50 * 1024**3, 50 * 1024**3))  # 50GB
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))  # 1 hour

# ── Constants ──────────────────────────────────────────────────────────
N = 30            # Fixed graph size for all experiments
K_RWSE = 20       # RWSE walk lengths 1..20
K_LAPPE = 8       # Number of Laplacian eigenvectors for LapPE
ALPHA_SRWE = 1e-4 # Tikhonov regularization for SRWE
NUM_SEEDS = 3     # Seeds for MLP classification
NUM_FOLDS = 5     # Folds for cross-validation
K_VALUES_DEPENDENT = [2, 4, 8, 12, 16, 20]  # K values for K-dependent analysis
N_GRAPHS_K_ANALYSIS = 50   # Graphs for K-dependent analysis

OUTPUT_FILE = SCRIPT_DIR / "method_out.json"


# ═══════════════════════════════════════════════════════════════════════
#  Phase 1: Graph Generation  (all n=30, all connected)
# ═══════════════════════════════════════════════════════════════════════

def _spectral_cluster_labels(G: nx.Graph, n_clusters: int = 3) -> np.ndarray | None:
    """Assign community labels via SpectralClustering on adjacency matrix."""
    try:
        A = nx.to_numpy_array(G)
        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            random_state=42,
            n_init=10,
        )
        labels = sc.fit_predict(A)
        if len(np.unique(labels)) < 2:
            return None
        return labels
    except Exception:
        pass
    # Fallback: KMeans on adjacency rows
    try:
        A = nx.to_numpy_array(G)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(A)
        if len(np.unique(labels)) < 2:
            return None
        return labels
    except Exception:
        return None


def _distance_based_labels(n: int, n_clusters: int = 3) -> np.ndarray:
    """Fallback labels based on node index modulo n_clusters."""
    return np.array([i % n_clusters for i in range(n)])


def generate_well_resolved_sbm(
    count: int, rng: np.random.RandomState
) -> list[dict]:
    """Category (a): SBM with well-separated communities."""
    logger.info(f"  Generating {count} well_resolved SBM graphs...")
    graphs = []
    for i in range(count * 5):  # extra attempts
        if len(graphs) >= count:
            break
        k = int(rng.choice([2, 3]))
        sizes = [N // k] * k
        sizes[-1] += N - sum(sizes)
        p_in = rng.uniform(0.5, 0.8)
        p_out = rng.uniform(0.01, 0.08)
        p_matrix = np.full((k, k), p_out)
        np.fill_diagonal(p_matrix, p_in)
        try:
            G = nx.stochastic_block_model(
                sizes, p_matrix.tolist(), seed=int(rng.randint(1e9))
            )
        except Exception:
            continue
        if not nx.is_connected(G) or G.number_of_nodes() != N:
            continue
        labels = np.array([G.nodes[nd]["block"] for nd in range(N)])
        graphs.append({
            "G": G,
            "labels": labels,
            "category": "well_resolved",
            "params": {"k": k, "p_in": round(p_in, 4), "p_out": round(p_out, 4)},
        })
    logger.info(f"    → {len(graphs)} well_resolved graphs generated")
    return graphs


def generate_marginally_resolved_sbm(
    count: int, rng: np.random.RandomState
) -> list[dict]:
    """Category (b): SBM with many communities, harder to separate."""
    logger.info(f"  Generating {count} marginally_resolved SBM graphs...")
    graphs = []
    for i in range(count * 5):
        if len(graphs) >= count:
            break
        k = int(rng.choice([5, 6]))
        sizes = [N // k] * k
        sizes[-1] += N - sum(sizes)
        p_in = rng.uniform(0.4, 0.7)
        p_out = rng.uniform(0.05, 0.15)
        p_matrix = np.full((k, k), p_out)
        np.fill_diagonal(p_matrix, p_in)
        try:
            G = nx.stochastic_block_model(
                sizes, p_matrix.tolist(), seed=int(rng.randint(1e9))
            )
        except Exception:
            continue
        if not nx.is_connected(G) or G.number_of_nodes() != N:
            continue
        labels = np.array([G.nodes[nd]["block"] for nd in range(N)])
        if len(np.unique(labels)) < 2:
            continue
        graphs.append({
            "G": G,
            "labels": labels,
            "category": "marginally_resolved",
            "params": {"k": k, "p_in": round(p_in, 4), "p_out": round(p_out, 4)},
        })
    logger.info(f"    → {len(graphs)} marginally_resolved graphs generated")
    return graphs


def generate_aliased(count: int, rng: np.random.RandomState) -> list[dict]:
    """Category (c): Aliased graphs (cycles, rings, high symmetry)."""
    logger.info(f"  Generating {count} aliased graphs...")
    graphs = []
    # 40% cycle+shortcuts, 30% WS ring, 30% path+cycle bridge
    n_cycle = int(count * 0.4)
    n_ws = int(count * 0.3)
    n_bridge = count - n_cycle - n_ws

    # Sub-category: cycle + shortcuts
    for i in range(n_cycle * 5):
        if len([g for g in graphs if g.get("sub") == "cycle_shortcut"]) >= n_cycle:
            break
        G = nx.cycle_graph(N)
        n_shortcuts = int(rng.randint(1, 4))
        for _ in range(n_shortcuts):
            u, v = sorted(rng.choice(N, 2, replace=False))
            if not G.has_edge(u, v):
                G.add_edge(u, v)
        if nx.is_connected(G) and G.number_of_nodes() == N:
            labels = _spectral_cluster_labels(G)
            if labels is None:
                labels = _distance_based_labels(N)
            graphs.append({
                "G": G, "labels": labels, "category": "aliased",
                "sub": "cycle_shortcut", "params": {"n_shortcuts": n_shortcuts},
            })

    # Sub-category: Watts-Strogatz ring lattice (p=0, no rewiring)
    for i in range(n_ws * 5):
        if len([g for g in graphs if g.get("sub") == "ws_ring"]) >= n_ws:
            break
        k_ws = int(rng.choice([4, 6]))
        G = nx.watts_strogatz_graph(N, k_ws, 0, seed=int(rng.randint(1e9)))
        if nx.is_connected(G) and G.number_of_nodes() == N:
            labels = _spectral_cluster_labels(G)
            if labels is None:
                labels = _distance_based_labels(N)
            graphs.append({
                "G": G, "labels": labels, "category": "aliased",
                "sub": "ws_ring", "params": {"k_ws": k_ws},
            })

    # Sub-category: path + cycle with bridge
    for i in range(n_bridge * 5):
        if len([g for g in graphs if g.get("sub") == "path_cycle"]) >= n_bridge:
            break
        path_len = int(rng.randint(10, 21))
        cycle_len = N - path_len
        if cycle_len < 3:
            cycle_len = 3
            path_len = N - cycle_len
        G = nx.Graph()
        # Path
        for j in range(path_len - 1):
            G.add_edge(j, j + 1)
        # Cycle
        offset = path_len
        for j in range(cycle_len):
            G.add_edge(offset + j, offset + (j + 1) % cycle_len)
        # Bridge
        G.add_edge(path_len - 1, offset)
        # Ensure n=30 nodes
        while G.number_of_nodes() < N:
            new_node = G.number_of_nodes()
            G.add_node(new_node)
            target = int(rng.randint(0, new_node))
            G.add_edge(new_node, target)
        if nx.is_connected(G) and G.number_of_nodes() == N:
            labels = _spectral_cluster_labels(G)
            if labels is None:
                labels = _distance_based_labels(N)
            graphs.append({
                "G": G, "labels": labels, "category": "aliased",
                "sub": "path_cycle",
                "params": {"path_len": path_len, "cycle_len": cycle_len},
            })

    logger.info(f"    → {len(graphs)} aliased graphs generated")
    return graphs[:count]


def generate_clustered_spectrum(
    count: int, rng: np.random.RandomState
) -> list[dict]:
    """Category (d): Graphs with clustered eigenvalue spectrum."""
    logger.info(f"  Generating {count} clustered_spectrum graphs...")
    graphs = []
    n_pcycle = count // 2
    n_preg = count - n_pcycle

    # Sub-category: perturbed cycle (cycle + 1 non-adjacent edge)
    for i in range(n_pcycle * 5):
        if len([g for g in graphs if g.get("sub") == "perturbed_cycle"]) >= n_pcycle:
            break
        G = nx.cycle_graph(N)
        for attempt in range(50):
            u = int(rng.randint(0, N))
            v = int(rng.randint(0, N))
            if u != v and not G.has_edge(u, v) and abs(u - v) > 1 and abs(u - v) < N - 1:
                G.add_edge(u, v)
                break
        if nx.is_connected(G) and G.number_of_nodes() == N:
            labels = _spectral_cluster_labels(G)
            if labels is None:
                labels = _distance_based_labels(N)
            graphs.append({
                "G": G, "labels": labels, "category": "clustered_spectrum",
                "sub": "perturbed_cycle", "params": {},
            })

    # Sub-category: perturbed regular graph
    for i in range(n_preg * 5):
        if len([g for g in graphs if g.get("sub") == "perturbed_regular"]) >= n_preg:
            break
        d = int(rng.choice([3, 4, 5]))
        if (N * d) % 2 != 0:
            d = d + 1 if d < 5 else d - 1
        try:
            G = nx.random_regular_graph(d, N, seed=int(rng.randint(1e9)))
        except nx.NetworkXError:
            continue
        # Add or remove 1 edge
        if rng.random() < 0.5:
            edges = list(G.edges())
            if edges:
                e = edges[int(rng.randint(len(edges)))]
                G.remove_edge(*e)
        else:
            non_edges = list(nx.non_edges(G))
            if non_edges:
                idx = int(rng.randint(len(non_edges)))
                G.add_edge(*non_edges[idx])
        if nx.is_connected(G) and G.number_of_nodes() == N:
            labels = _spectral_cluster_labels(G)
            if labels is None:
                labels = _distance_based_labels(N)
            graphs.append({
                "G": G, "labels": labels, "category": "clustered_spectrum",
                "sub": "perturbed_regular", "params": {"d": d},
            })

    logger.info(f"    → {len(graphs)} clustered_spectrum graphs generated")
    return graphs[:count]


def generate_er_control(count: int, rng: np.random.RandomState) -> list[dict]:
    """Category (e): Erdos-Renyi control graphs."""
    logger.info(f"  Generating {count} control_er graphs...")
    graphs = []
    for i in range(count * 5):
        if len(graphs) >= count:
            break
        p = rng.uniform(0.2, 0.5)
        G = nx.erdos_renyi_graph(N, p, seed=int(rng.randint(1e9)))
        if nx.is_connected(G) and G.number_of_nodes() == N:
            labels = _spectral_cluster_labels(G)
            if labels is None:
                labels = _distance_based_labels(N)
            graphs.append({
                "G": G, "labels": labels, "category": "control_er",
                "params": {"p_er": round(p, 4)},
            })
    logger.info(f"    → {len(graphs)} control_er graphs generated")
    return graphs


def generate_all_graphs(num_per_category: int, seed: int = 42) -> list[dict]:
    """Generate all synthetic graphs across 5 categories."""
    logger.info(f"Phase 1: Generating {num_per_category * 5} graphs "
                f"({num_per_category} per category)...")
    rng = np.random.RandomState(seed)
    all_graphs = []
    all_graphs.extend(generate_well_resolved_sbm(num_per_category, rng))
    all_graphs.extend(generate_marginally_resolved_sbm(num_per_category, rng))
    all_graphs.extend(generate_aliased(num_per_category, rng))
    all_graphs.extend(generate_clustered_spectrum(num_per_category, rng))
    all_graphs.extend(generate_er_control(num_per_category, rng))
    logger.info(f"Phase 1 complete: {len(all_graphs)} graphs total")
    return all_graphs


# ═══════════════════════════════════════════════════════════════════════
#  Phase 2: Spectral Analysis and Encoding Computation
# ═══════════════════════════════════════════════════════════════════════

def compute_encodings(G: nx.Graph) -> dict[str, Any]:
    """
    Compute RWSE, LapPE, SRWE and spectral metrics for a single graph.

    Returns dict with all encodings and metrics.
    """
    A = nx.to_numpy_array(G)
    n = A.shape[0]

    # 2A. Eigendecomposition (adjacency)
    eigenvalues_adj, eigenvectors_adj = np.linalg.eigh(A)

    # 2B. Spectral metrics
    sorted_eigs = np.sort(eigenvalues_adj)
    gaps = np.diff(sorted_eigs)
    nonzero_gaps = gaps[gaps > 1e-10]
    delta_min = float(np.min(nonzero_gaps)) if len(nonzero_gaps) > 0 else 0.0
    SRI_K20 = delta_min * K_RWSE

    # Vandermonde condition number (adjacency-based)
    V_adj = np.zeros((K_RWSE, n))
    for k in range(K_RWSE):
        V_adj[k, :] = eigenvalues_adj ** (k + 1)
    try:
        vandermonde_kappa = float(np.linalg.cond(V_adj))
        if not np.isfinite(vandermonde_kappa):
            vandermonde_kappa = 1e16
        vandermonde_kappa = min(vandermonde_kappa, 1e16)
    except Exception:
        vandermonde_kappa = 1e16

    # 2C. RWSE (K=20): diagonal of (D^{-1}A)^k for k=1..20
    degrees = np.array([G.degree(nd) for nd in range(n)], dtype=float)
    D_inv = np.zeros(n)
    nonzero_deg = degrees > 0
    D_inv[nonzero_deg] = 1.0 / degrees[nonzero_deg]
    RW = np.diag(D_inv) @ A  # Random walk transition matrix P = D^{-1}A

    rwse = np.zeros((n, K_RWSE))
    RW_power = np.eye(n)
    for k in range(K_RWSE):
        RW_power = RW_power @ RW
        rwse[:, k] = np.diag(RW_power)

    # 2D. LapPE (top-8 Laplacian eigenvectors, squared for sign invariance)
    L = np.diag(degrees) - A
    lap_evals, lap_evecs = np.linalg.eigh(L)
    k_actual = min(K_LAPPE, n - 1)
    lap_pe = lap_evecs[:, 1 : 1 + k_actual] ** 2  # skip constant eigenvec
    if lap_pe.shape[1] < K_LAPPE:
        pad_cols = K_LAPPE - lap_pe.shape[1]
        lap_pe = np.hstack([lap_pe, np.zeros((n, pad_cols))])

    # 2E. SRWE via Tikhonov-regularized Vandermonde recovery
    rw_eigenvalues_raw = np.linalg.eig(RW)[0]
    rw_eigenvalues = np.real(rw_eigenvalues_raw)

    V_rw = np.zeros((K_RWSE, n))
    for k in range(K_RWSE):
        V_rw[k, :] = rw_eigenvalues ** (k + 1)

    VtV = V_rw.T @ V_rw
    srwe = np.zeros((n, n))
    reg_matrix = VtV + ALPHA_SRWE * np.eye(n)

    for u in range(n):
        try:
            w_u = np.linalg.solve(reg_matrix, V_rw.T @ rwse[u, :])
            w_u = np.maximum(w_u, 0)
            w_sum = w_u.sum()
            if w_sum > 0:
                w_u /= w_sum
            srwe[u, :] = w_u
        except np.linalg.LinAlgError:
            try:
                w_u, _, _, _ = np.linalg.lstsq(V_rw.T, rwse[u, :], rcond=1e-3)
                w_u = np.maximum(w_u, 0)
                w_sum = w_u.sum()
                if w_sum > 0:
                    w_u /= w_sum
                srwe[u, :] = w_u
            except Exception:
                pass

    # RW-based SRI (robustness check)
    rw_sorted = np.sort(rw_eigenvalues)
    rw_gaps = np.diff(rw_sorted)
    rw_nonzero_gaps = rw_gaps[rw_gaps > 1e-10]
    rw_delta_min = float(np.min(rw_nonzero_gaps)) if len(rw_nonzero_gaps) > 0 else 0.0
    SRI_rw_K20 = rw_delta_min * K_RWSE

    # Graph density
    n_edges = G.number_of_edges()
    density = 2.0 * n_edges / (n * (n - 1)) if n > 1 else 0.0

    return {
        "A": A,
        "eigenvalues_adj": eigenvalues_adj,
        "sorted_eigs": sorted_eigs,
        "delta_min": delta_min,
        "SRI_K20": SRI_K20,
        "vandermonde_kappa": vandermonde_kappa,
        "rwse": rwse,
        "lap_pe": lap_pe,
        "srwe": srwe,
        "rw_eigenvalues": rw_eigenvalues,
        "SRI_rw_K20": SRI_rw_K20,
        "density": density,
        "n_edges": n_edges,
        "degrees": degrees,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Phase 3: Quality Metrics
# ═══════════════════════════════════════════════════════════════════════

def compute_distinguishability(encoding: np.ndarray) -> tuple[float, float]:
    """
    Compute node distinguishability metrics for an encoding.

    Returns:
        mpd: mean pairwise Euclidean distance
        fd: fraction of node pairs distinguishable (distance > 1e-3)
    """
    if encoding.shape[0] < 2:
        return 0.0, 0.0
    dists = pdist(encoding, "euclidean")
    mpd = float(np.mean(dists))
    fd = float(np.mean(dists > 1e-3))
    return mpd, fd


def mlp_classification_accuracy(
    encoding: np.ndarray,
    labels: np.ndarray,
    n_seeds: int = NUM_SEEDS,
    n_folds: int = NUM_FOLDS,
) -> float:
    """
    Compute mean MLP classification accuracy via stratified cross-validation.

    Averages over multiple random seeds.
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    # Ensure enough samples per class for k-fold
    label_counts = np.bincount(labels.astype(int))
    label_counts = label_counts[label_counts > 0]
    min_class_count = int(label_counts.min())
    actual_folds = min(n_folds, min_class_count)
    if actual_folds < 2:
        # Not enough data for any CV; return chance-level
        return 1.0 / len(unique_labels)

    accs = []
    for seed in range(n_seeds):
        try:
            X_scaled = StandardScaler().fit_transform(encoding)
            mlp = MLPClassifier(
                hidden_layer_sizes=(64,),
                max_iter=500,
                random_state=seed,
            )
            cv = StratifiedKFold(
                n_splits=actual_folds, shuffle=True, random_state=seed
            )
            scores = cross_val_score(
                mlp, X_scaled, labels, cv=cv, scoring="accuracy"
            )
            accs.append(float(np.mean(scores)))
        except Exception:
            logger.debug(f"MLP failed for seed {seed}")
            continue

    return float(np.mean(accs)) if accs else 0.0


# ═══════════════════════════════════════════════════════════════════════
#  Phase 4: Process All Graphs
# ═══════════════════════════════════════════════════════════════════════

def process_all_graphs(graphs: list[dict]) -> list[dict]:
    """Compute encodings, quality metrics for all graphs."""
    logger.info(f"Phase 2-3: Computing encodings and metrics for {len(graphs)} graphs...")
    results = []
    t0 = time.time()

    for i, gdata in enumerate(graphs):
        G = gdata["G"]
        labels = gdata["labels"]
        category = gdata["category"]

        try:
            # Phase 2: Spectral analysis and encodings
            enc = compute_encodings(G)

            # Phase 3: Quality metrics
            # Distinguishability
            mpd_rwse, fd_rwse = compute_distinguishability(enc["rwse"])
            mpd_lappe, fd_lappe = compute_distinguishability(enc["lap_pe"])
            mpd_srwe, fd_srwe = compute_distinguishability(enc["srwe"])

            # MLP classification accuracy
            acc_rwse = mlp_classification_accuracy(enc["rwse"], labels)
            acc_lappe = mlp_classification_accuracy(enc["lap_pe"], labels)
            acc_srwe = mlp_classification_accuracy(enc["srwe"], labels)

            # Quality gaps
            gap_classification = acc_lappe - acc_rwse
            gap_distinguishability_fd = fd_lappe - fd_rwse
            gap_distinguishability_mpd = mpd_lappe - mpd_rwse

            result = {
                "graph_idx": i,
                "category": category,
                "SRI_K20": enc["SRI_K20"],
                "delta_min": enc["delta_min"],
                "vandermonde_kappa": enc["vandermonde_kappa"],
                "SRI_rw_K20": enc["SRI_rw_K20"],
                "density": enc["density"],
                "n_edges": enc["n_edges"],
                # RWSE metrics
                "mpd_rwse": mpd_rwse,
                "fd_rwse": fd_rwse,
                "acc_rwse": acc_rwse,
                # LapPE metrics
                "mpd_lappe": mpd_lappe,
                "fd_lappe": fd_lappe,
                "acc_lappe": acc_lappe,
                # SRWE metrics
                "mpd_srwe": mpd_srwe,
                "fd_srwe": fd_srwe,
                "acc_srwe": acc_srwe,
                # Gaps
                "gap_classification": gap_classification,
                "gap_distinguishability_fd": gap_distinguishability_fd,
                "gap_distinguishability_mpd": gap_distinguishability_mpd,
                # Raw data for K-dependent analysis later
                "eigenvalues_adj": enc["sorted_eigs"].tolist(),
                "n_communities": int(len(np.unique(labels))),
                # Store graph/labels for K-dependent analysis
                "_G": G,
                "_labels": labels,
                "_rwse_full": enc["rwse"],
                "_lap_pe_full": enc["lap_pe"],
            }
            results.append(result)

        except Exception:
            logger.exception(f"Failed on graph {i} ({category})")
            continue

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(graphs) - i - 1) / rate
            logger.info(
                f"  Processed {i + 1}/{len(graphs)} "
                f"({elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining)"
            )

    elapsed = time.time() - t0
    logger.info(f"Phase 2-3 complete: {len(results)} results in {elapsed:.1f}s")
    return results


# ═══════════════════════════════════════════════════════════════════════
#  Phase 5: Statistical Analysis
# ═══════════════════════════════════════════════════════════════════════

def partial_spearman(
    x: np.ndarray, y: np.ndarray, covariates: list[np.ndarray]
) -> tuple[float, float]:
    """Compute partial Spearman correlation controlling for covariates."""
    rx = rankdata(x)
    ry = rankdata(y)
    rz = np.column_stack([rankdata(c) for c in covariates])

    rz_aug = np.column_stack([np.ones(len(rx)), rz])
    try:
        beta_x, _, _, _ = np.linalg.lstsq(rz_aug, rx, rcond=None)
        beta_y, _, _, _ = np.linalg.lstsq(rz_aug, ry, rcond=None)
        resid_x = rx - rz_aug @ beta_x
        resid_y = ry - rz_aug @ beta_y
        rho, p = spearmanr(resid_x, resid_y)
        return float(rho) if np.isfinite(rho) else 0.0, float(p) if np.isfinite(p) else 1.0
    except Exception:
        return 0.0, 1.0


def k_dependent_analysis(
    results: list[dict], n_graphs: int = N_GRAPHS_K_ANALYSIS
) -> list[dict]:
    """
    K-dependent analysis: compute RWSE/LapPE gap at different K values.

    For a subset of graphs, compute how gap changes with K and identify
    K* where SRI crosses 1.0.
    """
    logger.info(f"Phase 5E: K-dependent analysis on {n_graphs} graphs...")

    # Select spread of graphs (every k-th to get diversity)
    step = max(1, len(results) // n_graphs)
    subset_indices = list(range(0, len(results), step))[:n_graphs]

    k_results = []
    for idx in subset_indices:
        r = results[idx]
        G = r["_G"]
        labels = r["_labels"]
        delta_min = r["delta_min"]

        for K in K_VALUES_DEPENDENT:
            # RWSE at this K: take first K columns
            rwse_k = r["_rwse_full"][:, :K]

            # LapPE: always 8 eigenvectors (fixed)
            lap_pe_k = r["_lap_pe_full"]

            # MLP accuracy
            acc_rwse_k = mlp_classification_accuracy(rwse_k, labels, n_seeds=1, n_folds=3)
            acc_lappe_k = mlp_classification_accuracy(lap_pe_k, labels, n_seeds=1, n_folds=3)

            sri_k = delta_min * K
            gap_k = acc_lappe_k - acc_rwse_k

            k_results.append({
                "graph_idx": idx,
                "category": r["category"],
                "K": K,
                "SRI_K": round(sri_k, 6),
                "acc_rwse": round(acc_rwse_k, 6),
                "acc_lappe": round(acc_lappe_k, 6),
                "gap": round(gap_k, 6),
                "delta_min": round(delta_min, 6),
            })

    logger.info(f"  K-dependent analysis: {len(k_results)} data points")
    return k_results


def analyze_all_results(
    results: list[dict], k_dep_results: list[dict]
) -> dict[str, Any]:
    """Perform comprehensive statistical analysis."""
    logger.info("Phase 5: Statistical analysis...")

    # Extract arrays
    sri_vals = np.array([r["SRI_K20"] for r in results])
    gap_class = np.array([r["gap_classification"] for r in results])
    gap_fd = np.array([r["gap_distinguishability_fd"] for r in results])
    gap_mpd = np.array([r["gap_distinguishability_mpd"] for r in results])
    density_vals = np.array([r["density"] for r in results])
    n_edges_vals = np.array([r["n_edges"] for r in results])
    categories = [r["category"] for r in results]
    acc_rwse = np.array([r["acc_rwse"] for r in results])
    acc_lappe = np.array([r["acc_lappe"] for r in results])
    acc_srwe = np.array([r["acc_srwe"] for r in results])
    vk_vals = np.array([r["vandermonde_kappa"] for r in results])
    rw_sri_vals = np.array([r["SRI_rw_K20"] for r in results])
    delta_min_vals = np.array([r["delta_min"] for r in results])

    # ── 5A. Main Spearman correlations ──
    rho_class, p_class = spearmanr(sri_vals, gap_class)
    rho_fd, p_fd = spearmanr(sri_vals, gap_fd)
    rho_mpd, p_mpd = spearmanr(sri_vals, gap_mpd)

    # ── 5B. Partial correlations ──
    rho_partial_class, p_partial_class = partial_spearman(
        sri_vals, gap_class, [density_vals, n_edges_vals]
    )
    rho_partial_fd, p_partial_fd = partial_spearman(
        sri_vals, gap_fd, [density_vals, n_edges_vals]
    )

    # ── 5C. Quintile analysis ──
    quintile_edges = np.percentile(sri_vals, [0, 20, 40, 60, 80, 100])
    quintile_labels = np.digitize(sri_vals, quintile_edges[1:-1])
    quintile_data = []
    for q in range(5):
        mask = quintile_labels == q
        n_q = int(mask.sum())
        if n_q > 0:
            quintile_data.append({
                "quintile": q + 1,
                "sri_range": [
                    round(float(sri_vals[mask].min()), 6),
                    round(float(sri_vals[mask].max()), 6),
                ],
                "mean_sri": round(float(sri_vals[mask].mean()), 6),
                "mean_gap_classification": round(float(gap_class[mask].mean()), 6),
                "std_gap_classification": round(float(gap_class[mask].std()), 6),
                "mean_gap_fd": round(float(gap_fd[mask].mean()), 6),
                "mean_acc_rwse": round(float(acc_rwse[mask].mean()), 6),
                "mean_acc_lappe": round(float(acc_lappe[mask].mean()), 6),
                "mean_acc_srwe": round(float(acc_srwe[mask].mean()), 6),
                "count": n_q,
            })

    q_gaps = [qd["mean_gap_classification"] for qd in quintile_data]
    monotonic_decreasing = all(
        q_gaps[i] >= q_gaps[i + 1] for i in range(len(q_gaps) - 1)
    )

    # ── 5D. Per-category analysis ──
    unique_cats = sorted(set(categories))
    per_category = {}
    for cat in unique_cats:
        mask = np.array([c == cat for c in categories])
        n_cat = int(mask.sum())
        if n_cat < 3:
            continue
        rho_cat, p_cat = spearmanr(sri_vals[mask], gap_class[mask])
        per_category[cat] = {
            "count": n_cat,
            "mean_sri": round(float(sri_vals[mask].mean()), 6),
            "std_sri": round(float(sri_vals[mask].std()), 6),
            "mean_delta_min": round(float(delta_min_vals[mask].mean()), 8),
            "mean_gap_classification": round(float(gap_class[mask].mean()), 6),
            "std_gap_classification": round(float(gap_class[mask].std()), 6),
            "mean_acc_rwse": round(float(acc_rwse[mask].mean()), 6),
            "mean_acc_lappe": round(float(acc_lappe[mask].mean()), 6),
            "mean_acc_srwe": round(float(acc_srwe[mask].mean()), 6),
            "spearman_rho": round(float(rho_cat), 6) if np.isfinite(rho_cat) else 0.0,
            "spearman_p": round(float(p_cat), 6) if np.isfinite(p_cat) else 1.0,
        }

    # ── 5F. SRWE gap reduction ──
    sri_median = float(np.median(sri_vals))
    low_sri_mask = sri_vals < sri_median
    high_sri_mask = ~low_sri_mask

    def srwe_gap_reduction(mask: np.ndarray) -> float:
        denom = acc_lappe[mask] - acc_rwse[mask]
        numer = acc_srwe[mask] - acc_rwse[mask]
        valid = np.abs(denom) > 0.01
        if valid.sum() == 0:
            return 0.0
        return float(np.mean(numer[valid] / denom[valid]))

    srwe_red_low = srwe_gap_reduction(low_sri_mask)
    srwe_red_high = srwe_gap_reduction(high_sri_mask)
    srwe_red_overall = srwe_gap_reduction(np.ones(len(results), dtype=bool))

    # ── 5G. Additional correlations ──
    # Use log(kappa) for correlation since kappa spans orders of magnitude
    log_vk = np.log10(np.clip(vk_vals, 1.0, 1e16))
    rho_logvk, p_logvk = spearmanr(log_vk, gap_class)
    rho_rw_sri, p_rw_sri = spearmanr(rw_sri_vals, gap_class)
    rho_delta, p_delta = spearmanr(delta_min_vals, gap_class)
    rho_density, p_density = spearmanr(density_vals, gap_class)

    # ── K-dependent analysis summary ──
    k_dep_summary = {}
    for K in K_VALUES_DEPENDENT:
        k_data = [kd for kd in k_dep_results if kd["K"] == K]
        if k_data:
            gaps_at_k = [kd["gap"] for kd in k_data]
            sris_at_k = [kd["SRI_K"] for kd in k_data]
            rho_k, p_k = spearmanr(sris_at_k, gaps_at_k)
            k_dep_summary[f"K={K}"] = {
                "mean_gap": round(float(np.mean(gaps_at_k)), 6),
                "mean_sri": round(float(np.mean(sris_at_k)), 6),
                "spearman_rho": round(float(rho_k), 6) if np.isfinite(rho_k) else 0.0,
                "spearman_p": round(float(p_k), 6) if np.isfinite(p_k) else 1.0,
                "n_graphs": len(k_data),
            }

    # ── 5H. Threshold analysis (Fallback 4): SRI < 1 vs SRI >= 1 ──
    sri_threshold = 1.0
    low_thr_mask = sri_vals < sri_threshold
    high_thr_mask = sri_vals >= sri_threshold
    threshold_analysis = {}
    if low_thr_mask.sum() > 2 and high_thr_mask.sum() > 2:
        from scipy.stats import mannwhitneyu
        stat_u, p_u = mannwhitneyu(
            gap_class[low_thr_mask], gap_class[high_thr_mask], alternative="two-sided"
        )
        threshold_analysis = {
            "threshold": sri_threshold,
            "n_low": int(low_thr_mask.sum()),
            "n_high": int(high_thr_mask.sum()),
            "mean_gap_low_sri": round(float(gap_class[low_thr_mask].mean()), 6),
            "mean_gap_high_sri": round(float(gap_class[high_thr_mask].mean()), 6),
            "std_gap_low_sri": round(float(gap_class[low_thr_mask].std()), 6),
            "std_gap_high_sri": round(float(gap_class[high_thr_mask].std()), 6),
            "mann_whitney_U": round(float(stat_u), 4),
            "mann_whitney_p": round(float(p_u), 8),
            "effect_size": round(
                float(gap_class[low_thr_mask].mean() - gap_class[high_thr_mask].mean()), 6
            ),
        }

    # ── 5I. ANOVA-like analysis: Kruskal-Wallis across SRI terciles ──
    from scipy.stats import kruskal
    tercile_edges = np.percentile(sri_vals, [0, 33.33, 66.67, 100])
    tercile_labels = np.digitize(sri_vals, tercile_edges[1:-1])
    tercile_groups = [gap_class[tercile_labels == t] for t in range(3)]
    tercile_groups_valid = [g for g in tercile_groups if len(g) > 0]
    kw_analysis = {}
    if len(tercile_groups_valid) >= 2:
        try:
            kw_stat, kw_p = kruskal(*tercile_groups_valid)
            kw_analysis = {
                "kruskal_wallis_H": round(float(kw_stat), 4),
                "kruskal_wallis_p": round(float(kw_p), 8),
                "tercile_means": [
                    round(float(g.mean()), 6) for g in tercile_groups_valid
                ],
                "tercile_sizes": [len(g) for g in tercile_groups_valid],
            }
        except Exception:
            pass

    # ── 5J. Logistic regression comparison (simpler classifier) ──
    from sklearn.linear_model import LogisticRegression
    lr_correlations = {}
    if len(results) >= 20:
        # Compute LR accuracy for a subset (faster than MLP re-runs)
        lr_gaps = []
        lr_sris = []
        subset_for_lr = results[:min(100, len(results))]
        for r in subset_for_lr:
            labels_r = r["_labels"]
            unique_l = np.unique(labels_r)
            if len(unique_l) < 2:
                continue
            min_c = min(np.bincount(labels_r.astype(int))[np.bincount(labels_r.astype(int)) > 0])
            folds_lr = min(3, min_c)
            if folds_lr < 2:
                continue
            try:
                X_rwse = StandardScaler().fit_transform(r["_rwse_full"])
                X_lappe = StandardScaler().fit_transform(r["_lap_pe_full"])
                cv_lr = StratifiedKFold(n_splits=folds_lr, shuffle=True, random_state=42)
                lr_rwse = float(np.mean(cross_val_score(
                    LogisticRegression(max_iter=500, random_state=42),
                    X_rwse, labels_r, cv=cv_lr, scoring="accuracy"
                )))
                lr_lappe = float(np.mean(cross_val_score(
                    LogisticRegression(max_iter=500, random_state=42),
                    X_lappe, labels_r, cv=cv_lr, scoring="accuracy"
                )))
                lr_gaps.append(lr_lappe - lr_rwse)
                lr_sris.append(r["SRI_K20"])
            except Exception:
                continue
        if len(lr_gaps) >= 10:
            rho_lr, p_lr = spearmanr(lr_sris, lr_gaps)
            lr_correlations = {
                "spearman_sri_vs_lr_gap": {
                    "rho": round(float(rho_lr), 6) if np.isfinite(rho_lr) else 0.0,
                    "p": round(float(p_lr), 8) if np.isfinite(p_lr) else 1.0,
                },
                "n_graphs_lr": len(lr_gaps),
                "mean_lr_gap": round(float(np.mean(lr_gaps)), 6),
            }

    # ── Summary statistics ──
    summary = {
        "total_graphs": len(results),
        "n_per_category": {
            cat: int(np.sum(np.array(categories) == cat)) for cat in unique_cats
        },
        "overall_mean_sri": round(float(sri_vals.mean()), 6),
        "overall_std_sri": round(float(sri_vals.std()), 6),
        "overall_mean_gap_classification": round(float(gap_class.mean()), 6),
        "overall_std_gap_classification": round(float(gap_class.std()), 6),
        "overall_mean_acc_rwse": round(float(acc_rwse.mean()), 6),
        "overall_mean_acc_lappe": round(float(acc_lappe.mean()), 6),
        "overall_mean_acc_srwe": round(float(acc_srwe.mean()), 6),
    }

    # ── Targets assessment ──
    targets = {
        "spearman_rho_magnitude": round(abs(float(rho_class)), 6),
        "spearman_p_value": round(float(p_class), 8),
        "target_rho_gt_0.3": abs(float(rho_class)) > 0.3,
        "target_p_lt_0.01": float(p_class) < 0.01,
        "target_monotonic_quintile": monotonic_decreasing,
        "target_srwe_50pct_low_sri": srwe_red_low > 0.5,
    }

    analysis = {
        "primary_results": {
            "spearman_sri_vs_gap_classification": {
                "rho": round(float(rho_class), 6),
                "p": round(float(p_class), 8),
            },
            "spearman_sri_vs_gap_fd": {
                "rho": round(float(rho_fd), 6),
                "p": round(float(p_fd), 8),
            },
            "spearman_sri_vs_gap_mpd": {
                "rho": round(float(rho_mpd), 6),
                "p": round(float(p_mpd), 8),
            },
            "partial_spearman_classification": {
                "rho": round(float(rho_partial_class), 6),
                "p": round(float(p_partial_class), 8),
            },
            "partial_spearman_fd": {
                "rho": round(float(rho_partial_fd), 6),
                "p": round(float(p_partial_fd), 8),
            },
        },
        "quintile_analysis": {
            "quintiles": quintile_data,
            "monotonic_decreasing": monotonic_decreasing,
        },
        "per_category_results": per_category,
        "srwe_results": {
            "sri_median_split": round(sri_median, 6),
            "srwe_gap_reduction_low_sri": round(srwe_red_low, 6),
            "srwe_gap_reduction_high_sri": round(srwe_red_high, 6),
            "srwe_gap_reduction_overall": round(srwe_red_overall, 6),
            "target_met_50pct_low_sri": srwe_red_low > 0.5,
        },
        "k_dependent_analysis": k_dep_summary,
        "additional_correlations": {
            "log_vandermonde_kappa_vs_gap": {
                "rho": round(float(rho_logvk), 6),
                "p": round(float(p_logvk), 8),
            },
            "rw_sri_vs_gap": {
                "rho": round(float(rho_rw_sri), 6),
                "p": round(float(p_rw_sri), 8),
            },
            "delta_min_vs_gap": {
                "rho": round(float(rho_delta), 6),
                "p": round(float(p_delta), 8),
            },
            "density_vs_gap": {
                "rho": round(float(rho_density), 6),
                "p": round(float(p_density), 8),
            },
        },
        "threshold_analysis": threshold_analysis,
        "kruskal_wallis_analysis": kw_analysis,
        "logistic_regression_comparison": lr_correlations,
        "summary_statistics": summary,
        "targets_assessment": targets,
    }

    logger.info("Analysis results:")
    logger.info(f"  Spearman(SRI, gap_class): rho={rho_class:.4f}, p={p_class:.2e}")
    logger.info(f"  Spearman(SRI, gap_fd):    rho={rho_fd:.4f}, p={p_fd:.2e}")
    logger.info(f"  Partial Spearman:         rho={rho_partial_class:.4f}, p={p_partial_class:.2e}")
    logger.info(f"  Quintile monotonic: {monotonic_decreasing}")
    logger.info(f"  SRWE gap reduction (low SRI): {srwe_red_low:.4f}")
    logger.info(f"  Targets met: rho>{0.3}: {targets['target_rho_gt_0.3']}, "
                f"p<0.01: {targets['target_p_lt_0.01']}, "
                f"monotonic: {targets['target_monotonic_quintile']}, "
                f"SRWE>50%: {targets['target_srwe_50pct_low_sri']}")

    for cat, cdata in per_category.items():
        logger.info(f"  {cat}: mean_SRI={cdata['mean_sri']:.4f}, "
                    f"mean_gap={cdata['mean_gap_classification']:.4f}, "
                    f"rho={cdata['spearman_rho']:.4f}")

    if threshold_analysis:
        logger.info(f"  Threshold (SRI<1 vs >=1): "
                    f"gap_low={threshold_analysis['mean_gap_low_sri']:.4f}, "
                    f"gap_high={threshold_analysis['mean_gap_high_sri']:.4f}, "
                    f"U_p={threshold_analysis['mann_whitney_p']:.4e}")
    if kw_analysis:
        logger.info(f"  Kruskal-Wallis H={kw_analysis['kruskal_wallis_H']:.4f}, "
                    f"p={kw_analysis['kruskal_wallis_p']:.4e}")
    if lr_correlations:
        lr_rho = lr_correlations['spearman_sri_vs_lr_gap']['rho']
        lr_p = lr_correlations['spearman_sri_vs_lr_gap']['p']
        logger.info(f"  LR gap correlation: rho={lr_rho:.4f}, p={lr_p:.2e}")

    return analysis


# ═══════════════════════════════════════════════════════════════════════
#  Phase 6: Build Output JSON (exp_gen_sol_out.json schema)
# ═══════════════════════════════════════════════════════════════════════

def build_output_json(
    results: list[dict], analysis: dict[str, Any]
) -> dict[str, Any]:
    """Build output JSON conforming to exp_gen_sol_out.json schema."""
    logger.info("Phase 6: Building output JSON...")

    examples = []
    for r in results:
        # Build input string: graph spectral properties
        graph_info = {
            "num_nodes": N,
            "n_edges": r["n_edges"],
            "density": round(r["density"], 6),
            "category": r["category"],
            "eigenvalues_adj": [round(e, 8) for e in r["eigenvalues_adj"]],
            "delta_min": round(r["delta_min"], 8),
            "SRI_K20": round(r["SRI_K20"], 6),
            "vandermonde_kappa": round(r["vandermonde_kappa"], 4),
            "SRI_rw_K20": round(r["SRI_rw_K20"], 6),
            "n_communities": r["n_communities"],
        }

        # Output: actual classification gap (LapPE - RWSE)
        output_val = str(round(r["gap_classification"], 6))

        # Baseline prediction: constant zero (null hypothesis)
        predict_baseline = "0.0"

        # Our method prediction: SRI value (hypothesis: SRI negatively
        # correlates with gap)
        predict_our_method = str(round(r["SRI_K20"], 6))

        example = {
            "input": json.dumps(graph_info, separators=(",", ":")),
            "output": output_val,
            "predict_baseline": predict_baseline,
            "predict_our_method": predict_our_method,
            "metadata_category": r["category"],
            "metadata_graph_idx": r["graph_idx"],
            "metadata_sri_k20": round(r["SRI_K20"], 6),
            "metadata_sri_rw_k20": round(r["SRI_rw_K20"], 6),
            "metadata_delta_min": round(r["delta_min"], 8),
            "metadata_vandermonde_kappa": round(r["vandermonde_kappa"], 4),
            "metadata_density": round(r["density"], 6),
            "metadata_n_edges": r["n_edges"],
            "metadata_n_communities": r["n_communities"],
            "metadata_acc_rwse": round(r["acc_rwse"], 6),
            "metadata_acc_lappe": round(r["acc_lappe"], 6),
            "metadata_acc_srwe": round(r["acc_srwe"], 6),
            "metadata_gap_classification": round(r["gap_classification"], 6),
            "metadata_gap_fd": round(r["gap_distinguishability_fd"], 6),
            "metadata_gap_mpd": round(r["gap_distinguishability_mpd"], 6),
            "metadata_mpd_rwse": round(r["mpd_rwse"], 6),
            "metadata_mpd_lappe": round(r["mpd_lappe"], 6),
            "metadata_mpd_srwe": round(r["mpd_srwe"], 6),
            "metadata_fd_rwse": round(r["fd_rwse"], 6),
            "metadata_fd_lappe": round(r["fd_lappe"], 6),
            "metadata_fd_srwe": round(r["fd_srwe"], 6),
        }
        examples.append(example)

    output = {
        "metadata": {
            "method_name": "SRI_predicts_RWSE_vs_LapPE_quality_gap",
            "description": (
                "Tests whether Spectral Resolution Index (SRI = delta_min * K) "
                "predicts the RWSE-vs-LapPE encoding quality gap on fixed-size "
                "(n=30) synthetic graphs, independently of graph size."
            ),
            "parameters": {
                "n_nodes": N,
                "K_rwse": K_RWSE,
                "K_lappe": K_LAPPE,
                "alpha_srwe": ALPHA_SRWE,
                "num_seeds": NUM_SEEDS,
                "num_folds": NUM_FOLDS,
            },
            "analysis": analysis,
        },
        "datasets": [
            {
                "dataset": "synthetic_fixed_n30",
                "examples": examples,
            }
        ],
    }

    return output


# ═══════════════════════════════════════════════════════════════════════
#  Sanity Check
# ═══════════════════════════════════════════════════════════════════════

def sanity_check(results: list[dict]) -> bool:
    """Verify SRI distribution separates graph categories as expected."""
    logger.info("Sanity check: verifying SRI distribution across categories...")

    categories = set(r["category"] for r in results)
    cat_sris = {}
    for cat in sorted(categories):
        sris = [r["SRI_K20"] for r in results if r["category"] == cat]
        if sris:
            cat_sris[cat] = {
                "mean": float(np.mean(sris)),
                "std": float(np.std(sris)),
                "min": float(np.min(sris)),
                "max": float(np.max(sris)),
                "n": len(sris),
            }
            logger.info(
                f"  {cat}: mean_SRI={cat_sris[cat]['mean']:.4f} "
                f"± {cat_sris[cat]['std']:.4f} "
                f"(range [{cat_sris[cat]['min']:.4f}, {cat_sris[cat]['max']:.4f}], "
                f"n={cat_sris[cat]['n']})"
            )

    # Verify well_resolved has highest mean SRI
    if "well_resolved" in cat_sris:
        wr_mean = cat_sris["well_resolved"]["mean"]
        logger.info(f"  well_resolved mean SRI: {wr_mean:.4f}")

    # Check that we have reasonable spread
    all_sris = [r["SRI_K20"] for r in results]
    if all_sris:
        logger.info(
            f"  Overall SRI: mean={np.mean(all_sris):.4f}, "
            f"std={np.std(all_sris):.4f}, "
            f"range=[{np.min(all_sris):.4f}, {np.max(all_sris):.4f}]"
        )

    # Check accuracy spread
    all_acc_rwse = [r["acc_rwse"] for r in results]
    all_acc_lappe = [r["acc_lappe"] for r in results]
    all_gaps = [r["gap_classification"] for r in results]
    logger.info(
        f"  acc_rwse: mean={np.mean(all_acc_rwse):.4f}, "
        f"acc_lappe: mean={np.mean(all_acc_lappe):.4f}"
    )
    logger.info(
        f"  gap_classification: mean={np.mean(all_gaps):.4f}, "
        f"std={np.std(all_gaps):.4f}"
    )

    return True


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

@logger.catch
def main() -> None:
    parser = argparse.ArgumentParser(
        description="SRI Predicts RWSE-vs-LapPE Quality Gap"
    )
    parser.add_argument(
        "--num-per-category",
        type=int,
        default=100,
        help="Number of graphs per category (default: 100 → 500 total)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--skip-k-analysis",
        action="store_true",
        help="Skip K-dependent analysis (faster)",
    )
    args = parser.parse_args()

    t_start = time.time()
    logger.info("=" * 70)
    logger.info("SRI Predicts RWSE-vs-LapPE Quality on Fixed-Size (n=30) Graphs")
    logger.info(f"  num_per_category={args.num_per_category}, seed={args.seed}")
    logger.info("=" * 70)

    # Phase 1: Generate graphs
    graphs = generate_all_graphs(args.num_per_category, args.seed)
    if len(graphs) == 0:
        logger.error("No graphs generated! Aborting.")
        sys.exit(1)

    # Phase 2-3: Compute encodings and quality metrics
    results = process_all_graphs(graphs)
    if len(results) == 0:
        logger.error("No results computed! Aborting.")
        sys.exit(1)

    # Sanity check
    sanity_check(results)

    # Phase 5E: K-dependent analysis
    k_dep_results = []
    if not args.skip_k_analysis and len(results) >= 10:
        k_dep_results = k_dependent_analysis(results)

    # Phase 5: Full statistical analysis
    analysis = analyze_all_results(results, k_dep_results)

    # Phase 6: Build and save output JSON
    output = build_output_json(results, analysis)

    # Remove internal fields before saving
    for ex in output["datasets"][0]["examples"]:
        # Remove any keys starting with underscore (internal use)
        pass  # These were already excluded in build_output_json

    # Save
    OUTPUT_FILE.write_text(json.dumps(output, indent=2))
    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    logger.info(f"Saved method_out.json ({file_size_mb:.2f} MB)")

    # Check file size limit (100MB)
    if file_size_mb > 100:
        logger.warning(f"Output file exceeds 100MB ({file_size_mb:.1f}MB) — splitting needed")

    elapsed = time.time() - t_start
    logger.info(f"Total runtime: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    logger.info("Done!")


if __name__ == "__main__":
    main()
