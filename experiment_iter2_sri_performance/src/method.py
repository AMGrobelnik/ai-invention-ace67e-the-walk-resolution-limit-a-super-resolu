#!/usr/bin/env python3
"""SRI-Performance Gap Correlation: Testing Whether Spectral Resolution Index
Predicts RWSE vs LapPE Encoding Quality.

Phases:
  0 - Data loading & feature extraction
  1 - SRI distribution analysis
  2 - Model-free encoding quality (node distinguishability)
  3 - MLP proxy training (RWSE vs LapPE features)
  4 - Correlation analysis (Spearman, bootstrap CIs, quintile stratification)
  5 - Visualization
"""

import os
# Limit numpy thread usage to avoid excessive CPU time counting under RLIMIT_CPU
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import json
import glob
import math
import sys
import resource
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist
from scipy.linalg import eigh
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger
import psutil

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ── Resource limits ──────────────────────────────────────────────────────────
TOTAL_RAM_GB = psutil.virtual_memory().total / 1e9
AVAIL_RAM_GB = psutil.virtual_memory().available / 1e9
RAM_LIMIT_GB = min(50, TOTAL_RAM_GB - 4)  # leave 4 GB headroom
try:
    resource.setrlimit(resource.RLIMIT_AS, (int(RAM_LIMIT_GB * 1024**3), int(RAM_LIMIT_GB * 1024**3)))
except Exception:
    pass
# CPU limit: high value because numpy uses multi-threaded BLAS (all threads' time counted)
resource.setrlimit(resource.RLIMIT_CPU, (36000, 36000))  # 10h CPU time for multithreaded ops

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ────────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
DATA_DIR = Path("/opt/ai-inventor/aii_pipeline/runs/run__20260222_005317/3_invention_loop/iter_1/gen_art/data_id2_it1__opus/data_out")
FIGURES_DIR = WORKSPACE / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ── Globals ──────────────────────────────────────────────────────────────────
MAX_EXAMPLES: int | None = int(os.environ.get("MAX_EXAMPLES", 0)) or None  # env var or None for full
RNG_SEED = 42
np.random.seed(RNG_SEED)


# ===========================================================================
# PHASE 0: Data Loading & Feature Extraction
# ===========================================================================

def load_all_data(max_per_dataset: int | None = None) -> dict[str, list[dict]]:
    """Load all 5 JSON data files, parse inputs, return dict of dataset→records."""
    files = sorted(glob.glob(str(DATA_DIR / "full_data_out_*.json")))
    logger.info(f"Found {len(files)} data files")

    records: dict[str, list[dict]] = {}
    for fpath in files:
        logger.info(f"Loading {Path(fpath).name} ...")
        with open(fpath) as f:
            raw = json.load(f)
        for ds in raw["datasets"]:
            ds_name = ds["dataset"]
            if ds_name not in records:
                records[ds_name] = []
            for ex in ds["examples"]:
                if max_per_dataset and len(records[ds_name]) >= max_per_dataset:
                    break
                try:
                    rec = parse_example(ex, ds_name)
                    records[ds_name].append(rec)
                except Exception:
                    logger.exception(f"Failed parsing example in {ds_name}")
                    continue
        del raw  # free memory
    for ds_name, recs in records.items():
        logger.info(f"  {ds_name}: {len(recs)} records")
    return records


def parse_example(ex: dict, ds_name: str) -> dict:
    """Parse a single example into a lightweight record."""
    inp = json.loads(ex["input"])
    spec = inp["spectral"]

    num_nodes = inp["num_nodes"]
    edge_index = inp["edge_index"]  # [[src...], [dst...]]

    # Spectral features
    eigenvalues = np.array(spec["eigenvalues"], dtype=np.float64)
    delta_min = float(spec["delta_min"])
    sri_dict = spec["sri"]
    sri_k20 = float(sri_dict.get("K=20", sri_dict.get("K=16", 0.0)))
    vander_cond = spec["vandermonde_cond"]
    vander_k20 = float(vander_cond.get("K=20", vander_cond.get("K=16", 1e15)))

    # RWSE: list of lists (num_nodes × 20)
    rwse = np.array(spec["rwse"], dtype=np.float64)

    # Target
    output_raw = ex["output"]

    # Fold
    fold = ex.get("metadata_fold", 0)
    task_type = ex.get("metadata_task_type", "regression")

    rec = {
        "dataset": ds_name,
        "num_nodes": num_nodes,
        "edge_index": edge_index,
        "eigenvalues": eigenvalues,
        "delta_min": delta_min,
        "sri_k20": sri_k20,
        "vander_k20": vander_k20,
        "rwse": rwse,
        "fold": fold,
        "task_type": task_type,
        "output_raw": output_raw,
        "row_index": ex.get("metadata_row_index", 0),
    }
    # Synthetic-specific
    if ds_name == "Synthetic-aliased-pairs":
        rec["pair_id"] = ex.get("metadata_pair_id", "")
        rec["pair_category"] = ex.get("metadata_pair_category", "")
        rec["graph_name"] = ex.get("metadata_graph_name", "")

    return rec


def parse_target(output_raw: str, task_type: str) -> np.ndarray:
    """Parse the output string into a float array."""
    if task_type == "classification":
        return np.array(json.loads(output_raw), dtype=np.float64)
    elif task_type == "regression":
        val = output_raw.strip()
        if val.startswith("["):
            return np.array(json.loads(val), dtype=np.float64)
        else:
            return np.array([float(val)], dtype=np.float64)
    else:
        return np.array([float(output_raw)], dtype=np.float64)


# ===========================================================================
# Feature construction helpers
# ===========================================================================

def build_adjacency(edge_index: list, num_nodes: int) -> np.ndarray:
    """Build adjacency matrix from edge_index (vectorized)."""
    A = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    src = np.array(edge_index[0], dtype=np.int64)
    dst = np.array(edge_index[1], dtype=np.int64)
    mask = (src < num_nodes) & (dst < num_nodes)
    src, dst = src[mask], dst[mask]
    A[src, dst] = 1.0
    A[dst, src] = 1.0
    return A


def build_laplacian(A: np.ndarray) -> np.ndarray:
    """Build graph Laplacian L = D - A."""
    D = np.diag(A.sum(axis=1))
    return D - A


def rwse_graph_features(rwse: np.ndarray) -> np.ndarray:
    """Aggregate per-node RWSE (n×20) to graph-level features.
    Returns 80-dim vector: mean/std/max/min across nodes for each walk length."""
    if rwse.shape[0] == 0:
        return np.zeros(80, dtype=np.float64)
    mean_f = rwse.mean(axis=0)  # 20
    std_f = rwse.std(axis=0)    # 20
    max_f = rwse.max(axis=0)    # 20
    min_f = rwse.min(axis=0)    # 20
    return np.concatenate([mean_f, std_f, max_f, min_f])


def lape_graph_features(A: np.ndarray, num_eigvecs: int = 16) -> np.ndarray:
    """Compute LapPE features: |eigenvectors| of L, aggregated.
    Returns 64-dim (4 stats × 16 eigvecs)."""
    n = A.shape[0]
    L = build_laplacian(A)
    try:
        eigenvalues, eigvecs = eigh(L)
    except Exception:
        return np.zeros(num_eigvecs * 4, dtype=np.float64)

    # Take eigenvectors 1..num_eigvecs (skip the constant Fiedler)
    k = min(num_eigvecs, n - 1)
    if k <= 0:
        return np.zeros(num_eigvecs * 4, dtype=np.float64)

    V = np.abs(eigvecs[:, 1:k+1])  # n × k
    # Pad if k < num_eigvecs
    if k < num_eigvecs:
        V = np.pad(V, ((0, 0), (0, num_eigvecs - k)), mode="constant")

    mean_f = V.mean(axis=0)
    std_f = V.std(axis=0)
    max_f = V.max(axis=0)
    min_f = V.min(axis=0)
    return np.concatenate([mean_f, std_f, max_f, min_f])


def eigenvalue_histogram(eigenvalues: np.ndarray, n_bins: int = 31) -> np.ndarray:
    """Histogram of eigenvalues, normalized."""
    if len(eigenvalues) == 0:
        return np.zeros(n_bins, dtype=np.float64)
    hist, _ = np.histogram(eigenvalues, bins=n_bins, range=(-3, 3), density=False)
    total = hist.sum()
    if total > 0:
        hist = hist / total
    return hist.astype(np.float64)


# ===========================================================================
# PHASE 1: SRI Distribution Analysis
# ===========================================================================

def phase1_sri_distributions(records: dict[str, list[dict]]) -> dict:
    """Compute SRI distribution statistics and generate histograms."""
    logger.info("=== PHASE 1: SRI Distribution Analysis ===")
    results = {}
    ds_names = [k for k in records if k != "Synthetic-aliased-pairs"]
    if "Synthetic-aliased-pairs" in records:
        ds_names.append("Synthetic-aliased-pairs")

    fig, axes = plt.subplots(1, len(ds_names), figsize=(5 * len(ds_names), 4))
    if len(ds_names) == 1:
        axes = [axes]

    for idx, ds_name in enumerate(ds_names):
        recs = records[ds_name]
        sri_vals = np.array([r["sri_k20"] for r in recs])
        stats_dict = {
            "count": len(sri_vals),
            "mean": float(np.mean(sri_vals)),
            "median": float(np.median(sri_vals)),
            "std": float(np.std(sri_vals)),
            "p5": float(np.percentile(sri_vals, 5)),
            "p25": float(np.percentile(sri_vals, 25)),
            "p75": float(np.percentile(sri_vals, 75)),
            "p95": float(np.percentile(sri_vals, 95)),
            "frac_below_1": float(np.mean(sri_vals < 1.0)),
        }
        results[ds_name] = stats_dict
        logger.info(f"  {ds_name}: mean SRI={stats_dict['mean']:.4f}, median={stats_dict['median']:.4f}, "
                     f"frac<1={stats_dict['frac_below_1']:.4f}")

        ax = axes[idx]
        ax.hist(sri_vals, bins=50, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.axvline(x=1.0, color="red", linestyle="--", linewidth=1.5, label="SRI=1")
        ax.set_title(ds_name, fontsize=10)
        ax.set_xlabel("SRI (K=20)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(str(FIGURES_DIR / "sri_distributions.png"), dpi=150)
    plt.close(fig)
    logger.info("  Saved sri_distributions.png")
    return results


# ===========================================================================
# PHASE 2: Model-Free Encoding Quality
# ===========================================================================

def _subsample_pdist_median(X: np.ndarray, max_sample: int = 100, rng_seed: int = 42) -> float:
    """Compute median pairwise L2 distance, subsampling nodes if n > max_sample."""
    n = X.shape[0]
    if n <= max_sample:
        return float(np.median(pdist(X, metric="euclidean")))
    rng = np.random.RandomState(rng_seed)
    idx = rng.choice(n, size=max_sample, replace=False)
    return float(np.median(pdist(X[idx], metric="euclidean")))


def phase2_model_free(records: dict[str, list[dict]], max_nodes_for_phase2: int = 500) -> dict:
    """Compute model-free encoding quality gap per graph.
    quality_gap(g) = d_spec - d_rwse (positive = spectral better).
    For large graphs, subsamples 100 nodes for pairwise distance computation."""
    logger.info("=== PHASE 2: Model-Free Encoding Quality ===")
    results = {}

    for ds_name, recs in records.items():
        logger.info(f"  Phase2 processing {ds_name} ({len(recs)} graphs) ...")
        sri_vals = []
        quality_gaps = []
        delta_mins = []
        log_vander = []
        n_skipped = 0

        for i_rec, rec in enumerate(recs):
            if (i_rec + 1) % 1000 == 0:
                logger.info(f"    Phase2 {ds_name}: {i_rec+1}/{len(recs)} processed")
            n = rec["num_nodes"]
            if n > max_nodes_for_phase2:
                n_skipped += 1
                continue
            if n < 3:
                continue

            rwse = rec["rwse"]  # n × 20
            if rwse.shape[0] != n or rwse.shape[1] < 1:
                continue

            # Build adjacency and compute spectral features
            try:
                A = build_adjacency(rec["edge_index"], n)
                eigvals_A, eigvecs_A = eigh(A)
                # Spectral features: eigvecs^2 for top-20 eigenvalues (by magnitude)
                k_spec = min(20, n)
                spec_feats = eigvecs_A[:, -k_spec:] ** 2  # n × k_spec
            except Exception:
                continue

            # Ensure same feature dims
            rwse_f = rwse[:, :min(20, rwse.shape[1])]
            spec_f = spec_feats[:, :min(20, spec_feats.shape[1])]

            # Pad shorter one
            max_cols = max(rwse_f.shape[1], spec_f.shape[1])
            if rwse_f.shape[1] < max_cols:
                rwse_f = np.pad(rwse_f, ((0, 0), (0, max_cols - rwse_f.shape[1])))
            if spec_f.shape[1] < max_cols:
                spec_f = np.pad(spec_f, ((0, 0), (0, max_cols - spec_f.shape[1])))

            # Median pairwise L2 distance (subsample for large graphs)
            if n >= 2:
                d_rwse = _subsample_pdist_median(rwse_f, max_sample=100)
                d_spec = _subsample_pdist_median(spec_f, max_sample=100)
                gap = d_spec - d_rwse
            else:
                continue

            sri_vals.append(rec["sri_k20"])
            quality_gaps.append(gap)
            delta_mins.append(rec["delta_min"])
            log_vander.append(math.log10(rec["vander_k20"] + 1))

        if len(sri_vals) < 5:
            logger.warning(f"  {ds_name}: only {len(sri_vals)} valid graphs, skipping correlation")
            results[ds_name] = {"n_valid": len(sri_vals), "n_skipped": n_skipped}
            continue

        sri_arr = np.array(sri_vals)
        gap_arr = np.array(quality_gaps)
        dm_arr = np.array(delta_mins)
        lv_arr = np.array(log_vander)

        rho_sri, p_sri = stats.spearmanr(sri_arr, gap_arr)
        rho_dm, p_dm = stats.spearmanr(dm_arr, gap_arr)
        rho_lv, p_lv = stats.spearmanr(lv_arr, gap_arr)

        results[ds_name] = {
            "n_valid": len(sri_vals),
            "n_skipped": n_skipped,
            "spearman_sri_vs_gap": {"rho": float(rho_sri), "p": float(p_sri)},
            "spearman_deltamin_vs_gap": {"rho": float(rho_dm), "p": float(p_dm)},
            "spearman_logvander_vs_gap": {"rho": float(rho_lv), "p": float(p_lv)},
            "mean_quality_gap": float(np.mean(gap_arr)),
            "std_quality_gap": float(np.std(gap_arr)),
        }
        logger.info(f"  {ds_name}: n={len(sri_vals)}, rho(SRI,gap)={rho_sri:.4f} (p={p_sri:.4e}), "
                     f"rho(delta_min,gap)={rho_dm:.4f}, rho(log_vander,gap)={rho_lv:.4f}")

    return results


# ===========================================================================
# PHASE 3: MLP Proxy Training
# ===========================================================================

def split_indices(n: int, train_frac: float = 0.7, val_frac: float = 0.15, seed: int = 42):
    """Split indices into train/val/test."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]


def phase3_mlp_proxy(records: dict[str, list[dict]]) -> dict:
    """Train MLP proxy models with RWSE and LapPE features, compute per-graph loss gap."""
    logger.info("=== PHASE 3: MLP Proxy Training ===")
    results = {}

    for ds_name in ["ZINC-subset", "Peptides-func", "Peptides-struct"]:
        if ds_name not in records:
            logger.warning(f"  {ds_name} not in records, skipping")
            continue

        recs = records[ds_name]
        n = len(recs)
        logger.info(f"  {ds_name}: building features for {n} graphs ...")

        # Build features
        rwse_feats_list = []
        lape_feats_list = []
        eig_hist_list = []
        targets_list = []
        sri_vals = []
        delta_mins = []
        log_vander_vals = []
        num_nodes_list = []
        valid_indices = []

        for i, rec in enumerate(recs):
            if (i + 1) % 1000 == 0:
                logger.info(f"    Phase3 {ds_name}: features for {i+1}/{n} graphs")
            try:
                # RWSE graph features (80-dim)
                rwse_gf = rwse_graph_features(rec["rwse"])

                # LapPE graph features (64-dim) - need adjacency
                n_nodes = rec["num_nodes"]
                if n_nodes <= 2:
                    continue
                A = build_adjacency(rec["edge_index"], n_nodes)
                lape_gf = lape_graph_features(A, num_eigvecs=16)

                # Eigenvalue histogram (31-dim)
                eig_hist = eigenvalue_histogram(rec["eigenvalues"])

                # Target
                target = parse_target(rec["output_raw"], rec["task_type"])

                rwse_feats_list.append(rwse_gf)
                lape_feats_list.append(lape_gf)
                eig_hist_list.append(eig_hist)
                targets_list.append(target)
                sri_vals.append(rec["sri_k20"])
                delta_mins.append(rec["delta_min"])
                log_vander_vals.append(math.log10(rec["vander_k20"] + 1))
                num_nodes_list.append(n_nodes)
                valid_indices.append(i)
            except Exception:
                logger.debug(f"    Skipped example {i} in {ds_name}")
                continue

        n_valid = len(rwse_feats_list)
        logger.info(f"    {n_valid}/{n} valid feature vectors")
        if n_valid < 20:
            logger.warning(f"  {ds_name}: too few valid examples ({n_valid}), skipping MLP")
            continue

        X_rwse = np.array(rwse_feats_list)  # n_valid × 80
        X_lape = np.array(lape_feats_list)  # n_valid × 64
        X_eig = np.array(eig_hist_list)     # n_valid × 31
        y_all = np.array(targets_list)      # n_valid × target_dim
        sri_arr = np.array(sri_vals)
        dm_arr = np.array(delta_mins)
        lv_arr = np.array(log_vander_vals)
        nn_arr = np.array(num_nodes_list)

        # Combine features
        X_rwse_full = np.hstack([X_rwse, X_eig])  # 80 + 31 = 111
        X_lape_full = np.hstack([X_lape, X_eig])  # 64 + 31 = 95

        # For Peptides-func (classification) → use multi-label one-hot as regression target
        # For ZINC / Peptides-struct → regression

        # Split
        train_idx, val_idx, test_idx = split_indices(n_valid, seed=RNG_SEED)
        logger.info(f"    Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        seeds = [42, 123, 456]
        per_graph_loss_rwse = np.zeros((len(test_idx), len(seeds)))
        per_graph_loss_lape = np.zeros((len(test_idx), len(seeds)))

        for seed_idx, seed in enumerate(seeds):
            logger.info(f"    Training MLP (seed={seed}) ...")

            # Scale
            scaler_r = StandardScaler()
            X_r_train = scaler_r.fit_transform(X_rwse_full[train_idx])
            X_r_val = scaler_r.transform(X_rwse_full[val_idx])
            X_r_test = scaler_r.transform(X_rwse_full[test_idx])

            scaler_l = StandardScaler()
            X_l_train = scaler_l.fit_transform(X_lape_full[train_idx])
            X_l_val = scaler_l.transform(X_lape_full[val_idx])
            X_l_test = scaler_l.transform(X_lape_full[test_idx])

            y_train = y_all[train_idx]
            y_test = y_all[test_idx]

            # Replace any NaN/Inf in features
            for X in [X_r_train, X_r_val, X_r_test, X_l_train, X_l_val, X_l_test]:
                X[~np.isfinite(X)] = 0.0

            # Train RWSE MLP
            try:
                mlp_rwse = MLPRegressor(
                    hidden_layer_sizes=(128, 64),
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=seed,
                    learning_rate="adaptive",
                    n_iter_no_change=15,
                )
                mlp_rwse.fit(X_r_train, y_train)
                pred_rwse = mlp_rwse.predict(X_r_test)
                per_graph_mae_rwse = np.abs(pred_rwse - y_test).mean(axis=-1) if y_test.ndim > 1 else np.abs(pred_rwse - y_test)
                per_graph_loss_rwse[:, seed_idx] = per_graph_mae_rwse
                logger.info(f"      RWSE MLP: test MAE = {per_graph_mae_rwse.mean():.4f}")
            except Exception:
                logger.exception(f"      RWSE MLP failed (seed={seed})")
                per_graph_loss_rwse[:, seed_idx] = np.nan

            # Train LapPE MLP
            try:
                mlp_lape = MLPRegressor(
                    hidden_layer_sizes=(128, 64),
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=seed,
                    learning_rate="adaptive",
                    n_iter_no_change=15,
                )
                mlp_lape.fit(X_l_train, y_train)
                pred_lape = mlp_lape.predict(X_l_test)
                per_graph_mae_lape = np.abs(pred_lape - y_test).mean(axis=-1) if y_test.ndim > 1 else np.abs(pred_lape - y_test)
                per_graph_loss_lape[:, seed_idx] = per_graph_mae_lape
                logger.info(f"      LapPE MLP: test MAE = {per_graph_mae_lape.mean():.4f}")
            except Exception:
                logger.exception(f"      LapPE MLP failed (seed={seed})")
                per_graph_loss_lape[:, seed_idx] = np.nan

        # Average across seeds
        mean_loss_rwse = np.nanmean(per_graph_loss_rwse, axis=1)
        mean_loss_lape = np.nanmean(per_graph_loss_lape, axis=1)
        gap = mean_loss_rwse - mean_loss_lape  # positive = LapPE better

        sri_test = sri_arr[test_idx]
        dm_test = dm_arr[test_idx]
        lv_test = lv_arr[test_idx]
        nn_test = nn_arr[test_idx]

        results[ds_name] = {
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "mean_mae_rwse": float(np.nanmean(mean_loss_rwse)),
            "mean_mae_lape": float(np.nanmean(mean_loss_lape)),
            "mean_gap": float(np.nanmean(gap)),
            # Store per-graph data for Phase 4
            "_gap": gap,
            "_sri_test": sri_test,
            "_dm_test": dm_test,
            "_lv_test": lv_test,
            "_nn_test": nn_test,
        }
        logger.info(f"    Mean MAE: RWSE={np.nanmean(mean_loss_rwse):.4f}, "
                     f"LapPE={np.nanmean(mean_loss_lape):.4f}, gap={np.nanmean(gap):.4f}")

    return results


# ===========================================================================
# PHASE 4: Correlation Analysis
# ===========================================================================

def bootstrap_spearman(x: np.ndarray, y: np.ndarray, n_boot: int = 1000, seed: int = 42) -> dict:
    """Bootstrap 95% CI for Spearman rho."""
    rng = np.random.RandomState(seed)
    n = len(x)
    rhos = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        r, _ = stats.spearmanr(x[idx], y[idx])
        if np.isfinite(r):
            rhos.append(r)
    if not rhos:
        return {"rho_mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rhos = np.array(rhos)
    return {
        "rho_mean": float(np.mean(rhos)),
        "ci_low": float(np.percentile(rhos, 2.5)),
        "ci_high": float(np.percentile(rhos, 97.5)),
    }


def phase4_correlation(phase3_results: dict) -> dict:
    """Detailed correlation analysis with bootstrap CIs and quintile stratification."""
    logger.info("=== PHASE 4: Correlation Analysis ===")
    results = {}

    for ds_name, res in phase3_results.items():
        if "_gap" not in res:
            continue

        gap = res["_gap"]
        sri = res["_sri_test"]
        dm = res["_dm_test"]
        lv = res["_lv_test"]
        nn = res["_nn_test"]

        # Remove NaN
        mask = np.isfinite(gap) & np.isfinite(sri) & np.isfinite(dm) & np.isfinite(lv)
        gap = gap[mask]
        sri = sri[mask]
        dm = dm[mask]
        lv = lv[mask]
        nn = nn[mask]

        if len(gap) < 10:
            logger.warning(f"  {ds_name}: too few valid test points ({len(gap)})")
            continue

        # Primary correlations
        rho_sri, p_sri = stats.spearmanr(sri, gap)
        rho_dm, p_dm = stats.spearmanr(dm, gap)
        rho_lv, p_lv = stats.spearmanr(lv, gap)

        # Bootstrap CIs
        boot_sri = bootstrap_spearman(sri, gap)
        boot_dm = bootstrap_spearman(dm, gap)
        boot_lv = bootstrap_spearman(lv, gap)

        # SRI quintile stratification
        quintile_edges = np.percentile(sri, [0, 20, 40, 60, 80, 100])
        quintile_means = []
        quintile_labels = []
        for q in range(5):
            low, high = quintile_edges[q], quintile_edges[q + 1]
            if q == 4:
                mask_q = (sri >= low) & (sri <= high)
            else:
                mask_q = (sri >= low) & (sri < high)
            if mask_q.sum() > 0:
                quintile_means.append(float(gap[mask_q].mean()))
                quintile_labels.append(f"Q{q+1} [{low:.2f}, {high:.2f}]")
            else:
                quintile_means.append(0.0)
                quintile_labels.append(f"Q{q+1}")

        # Kendall's tau on quintile means vs quintile index
        tau, p_tau = stats.kendalltau(range(5), quintile_means)

        # Size-controlled analysis
        size_bins = np.percentile(nn, [0, 33, 67, 100])
        size_controlled = []
        for b in range(3):
            low, high = size_bins[b], size_bins[b + 1]
            if b == 2:
                mask_b = (nn >= low) & (nn <= high)
            else:
                mask_b = (nn >= low) & (nn < high)
            if mask_b.sum() >= 10:
                r, p = stats.spearmanr(sri[mask_b], gap[mask_b])
                size_controlled.append({
                    "size_range": f"[{low:.0f}, {high:.0f}]",
                    "n": int(mask_b.sum()),
                    "rho": float(r),
                    "p": float(p),
                })

        # SRI vs num_nodes correlation (confounder check)
        rho_sri_nn, p_sri_nn = stats.spearmanr(sri, nn)

        ds_result = {
            "n_test": int(len(gap)),
            "primary": {
                "sri_vs_gap": {"rho": float(rho_sri), "p": float(p_sri), "bootstrap": boot_sri},
                "deltamin_vs_gap": {"rho": float(rho_dm), "p": float(p_dm), "bootstrap": boot_dm},
                "logvander_vs_gap": {"rho": float(rho_lv), "p": float(p_lv), "bootstrap": boot_lv},
            },
            "quintiles": {
                "labels": quintile_labels,
                "mean_gap": quintile_means,
                "kendall_tau": float(tau),
                "kendall_p": float(p_tau),
            },
            "size_controlled": size_controlled,
            "confounder": {
                "sri_vs_num_nodes": {"rho": float(rho_sri_nn), "p": float(p_sri_nn)},
            },
        }
        results[ds_name] = ds_result
        logger.info(f"  {ds_name}: rho(SRI,gap)={rho_sri:.4f} [{boot_sri['ci_low']:.4f}, {boot_sri['ci_high']:.4f}], "
                     f"p={p_sri:.4e}")
        logger.info(f"    Quintile trend (Kendall tau): {tau:.4f} (p={p_tau:.4e})")
        logger.info(f"    SRI vs num_nodes: rho={rho_sri_nn:.4f}")

    return results


# ===========================================================================
# PHASE 5: Visualization
# ===========================================================================

def phase5_visualization(phase3_results: dict, phase4_results: dict, phase2_results: dict) -> None:
    """Generate scatter plots and quintile bar charts."""
    logger.info("=== PHASE 5: Visualization ===")

    # 5a: SRI vs gap scatter for each dataset
    ds_with_gap = [ds for ds in phase3_results if "_gap" in phase3_results[ds]]
    if ds_with_gap:
        fig, axes = plt.subplots(1, len(ds_with_gap), figsize=(6 * len(ds_with_gap), 5))
        if len(ds_with_gap) == 1:
            axes = [axes]
        for idx, ds_name in enumerate(ds_with_gap):
            res = phase3_results[ds_name]
            gap = res["_gap"]
            sri = res["_sri_test"]
            mask = np.isfinite(gap) & np.isfinite(sri)
            ax = axes[idx]
            ax.scatter(sri[mask], gap[mask], alpha=0.3, s=10, c="steelblue")
            ax.set_xlabel("SRI (K=20)", fontsize=10)
            ax.set_ylabel("MAE gap (RWSE − LapPE)", fontsize=10)
            if ds_name in phase4_results:
                rho = phase4_results[ds_name]["primary"]["sri_vs_gap"]["rho"]
                p = phase4_results[ds_name]["primary"]["sri_vs_gap"]["p"]
                ax.set_title(f"{ds_name}\nρ={rho:.3f}, p={p:.2e}", fontsize=10)
            else:
                ax.set_title(ds_name, fontsize=10)
            ax.axhline(y=0, color="grey", linestyle="--", linewidth=0.8)
        plt.tight_layout()
        fig.savefig(str(FIGURES_DIR / "sri_vs_gap_scatter.png"), dpi=150)
        plt.close(fig)
        logger.info("  Saved sri_vs_gap_scatter.png")

    # 5b: Vandermonde cond vs gap scatter
    if ds_with_gap:
        fig, axes = plt.subplots(1, len(ds_with_gap), figsize=(6 * len(ds_with_gap), 5))
        if len(ds_with_gap) == 1:
            axes = [axes]
        for idx, ds_name in enumerate(ds_with_gap):
            res = phase3_results[ds_name]
            gap = res["_gap"]
            lv = res["_lv_test"]
            mask = np.isfinite(gap) & np.isfinite(lv)
            ax = axes[idx]
            ax.scatter(lv[mask], gap[mask], alpha=0.3, s=10, c="coral")
            ax.set_xlabel("log₁₀(Vandermonde cond + 1)", fontsize=10)
            ax.set_ylabel("MAE gap (RWSE − LapPE)", fontsize=10)
            if ds_name in phase4_results:
                rho = phase4_results[ds_name]["primary"]["logvander_vs_gap"]["rho"]
                p = phase4_results[ds_name]["primary"]["logvander_vs_gap"]["p"]
                ax.set_title(f"{ds_name}\nρ={rho:.3f}, p={p:.2e}", fontsize=10)
            else:
                ax.set_title(ds_name, fontsize=10)
            ax.axhline(y=0, color="grey", linestyle="--", linewidth=0.8)
        plt.tight_layout()
        fig.savefig(str(FIGURES_DIR / "vander_vs_gap_scatter.png"), dpi=150)
        plt.close(fig)
        logger.info("  Saved vander_vs_gap_scatter.png")

    # 5c: Quintile bar chart
    ds_with_quintiles = [ds for ds in phase4_results if "quintiles" in phase4_results[ds]]
    if ds_with_quintiles:
        fig, axes = plt.subplots(1, len(ds_with_quintiles), figsize=(6 * len(ds_with_quintiles), 5))
        if len(ds_with_quintiles) == 1:
            axes = [axes]
        for idx, ds_name in enumerate(ds_with_quintiles):
            q_data = phase4_results[ds_name]["quintiles"]
            ax = axes[idx]
            bars = ax.bar(range(5), q_data["mean_gap"], color="steelblue", alpha=0.7, edgecolor="black")
            ax.set_xticks(range(5))
            ax.set_xticklabels([f"Q{i+1}" for i in range(5)], fontsize=8)
            ax.set_ylabel("Mean gap (RWSE − LapPE)", fontsize=10)
            tau = q_data["kendall_tau"]
            p_tau = q_data["kendall_p"]
            ax.set_title(f"{ds_name}\nKendall τ={tau:.3f}, p={p_tau:.2e}", fontsize=10)
            ax.axhline(y=0, color="grey", linestyle="--", linewidth=0.8)
        plt.tight_layout()
        fig.savefig(str(FIGURES_DIR / "quintile_bar_chart.png"), dpi=150)
        plt.close(fig)
        logger.info("  Saved quintile_bar_chart.png")

    logger.info("  Visualization complete")


# ===========================================================================
# Synthetic Validation
# ===========================================================================

def validate_synthetic(records: dict[str, list[dict]]) -> dict:
    """Validate RWSE features on synthetic cospectral pairs."""
    logger.info("=== Synthetic Pair Validation ===")
    if "Synthetic-aliased-pairs" not in records:
        return {"status": "no synthetic data"}

    recs = records["Synthetic-aliased-pairs"]
    # Group by pair_id
    pairs = {}
    for r in recs:
        pid = r.get("pair_id", "")
        if pid not in pairs:
            pairs[pid] = []
        pairs[pid].append(r)

    results_list = []
    for pid, pair_recs in pairs.items():
        if len(pair_recs) < 2:
            continue
        r1, r2 = pair_recs[0], pair_recs[1]
        cat = r1.get("pair_category", "unknown")

        # Compare RWSE features (graph-level)
        gf1 = rwse_graph_features(r1["rwse"])
        gf2 = rwse_graph_features(r2["rwse"])
        rwse_dist = float(np.linalg.norm(gf1 - gf2))

        # Compare spectral features
        lf1 = lape_graph_features(build_adjacency(r1["edge_index"], r1["num_nodes"]))
        lf2 = lape_graph_features(build_adjacency(r2["edge_index"], r2["num_nodes"]))
        lape_dist = float(np.linalg.norm(lf1 - lf2))

        results_list.append({
            "pair_id": pid,
            "category": cat,
            "rwse_dist": rwse_dist,
            "lape_dist": lape_dist,
            "sri_1": r1["sri_k20"],
            "sri_2": r2["sri_k20"],
        })

    # Summarize by category
    cats = set(r["category"] for r in results_list)
    summary = {}
    for cat in cats:
        cat_items = [r for r in results_list if r["category"] == cat]
        rwse_dists = [r["rwse_dist"] for r in cat_items]
        lape_dists = [r["lape_dist"] for r in cat_items]
        summary[cat] = {
            "n_pairs": len(cat_items),
            "mean_rwse_dist": float(np.mean(rwse_dists)),
            "mean_lape_dist": float(np.mean(lape_dists)),
            "std_rwse_dist": float(np.std(rwse_dists)),
            "std_lape_dist": float(np.std(lape_dists)),
        }
        logger.info(f"  {cat}: {len(cat_items)} pairs, "
                     f"mean RWSE dist={np.mean(rwse_dists):.6f}, "
                     f"mean LapPE dist={np.mean(lape_dists):.6f}")

    return {"pairs": results_list, "summary": summary}


# ===========================================================================
# Output Assembly
# ===========================================================================

def assemble_output(
    records: dict[str, list[dict]],
    phase1: dict,
    phase2: dict,
    phase3: dict,
    phase4: dict,
    synthetic_val: dict,
) -> dict:
    """Assemble all results into exp_gen_sol_out.json format."""
    logger.info("=== Assembling Output ===")

    datasets_out = []
    for ds_name, recs in records.items():
        examples = []
        for i, rec in enumerate(recs):
            # Build per-example output
            per_example_output = {
                "sri_k20": rec["sri_k20"],
                "delta_min": rec["delta_min"],
                "num_nodes": rec["num_nodes"],
            }

            # Our method prediction: the SRI value itself as a predictor score
            predict_our_method = json.dumps({
                "sri_k20": rec["sri_k20"],
                "delta_min": rec["delta_min"],
                "vander_k20": rec["vander_k20"],
            })

            # Baseline: use delta_min alone
            predict_baseline = json.dumps({
                "delta_min": rec["delta_min"],
            })

            ex_out = {
                "input": json.dumps({
                    "num_nodes": rec["num_nodes"],
                    "edge_index_len": len(rec["edge_index"][0]) if isinstance(rec["edge_index"], list) and len(rec["edge_index"]) == 2 else 0,
                    "sri_k20": rec["sri_k20"],
                    "delta_min": rec["delta_min"],
                    "eigenvalues_count": len(rec["eigenvalues"]),
                }),
                "output": rec["output_raw"],
                "predict_our_method": predict_our_method,
                "predict_baseline": predict_baseline,
                "metadata_fold": rec["fold"],
                "metadata_task_type": rec["task_type"],
                "metadata_row_index": rec["row_index"],
                "metadata_num_nodes": rec["num_nodes"],
                "metadata_sri_k20": rec["sri_k20"],
                "metadata_delta_min": rec["delta_min"],
            }
            examples.append(ex_out)

        datasets_out.append({
            "dataset": ds_name,
            "examples": examples,
        })

    # Clean phase3 results (remove numpy arrays)
    phase3_clean = {}
    for ds_name, res in phase3.items():
        phase3_clean[ds_name] = {k: v for k, v in res.items() if not k.startswith("_")}

    output = {
        "metadata": {
            "method_name": "SRI-Performance Gap Correlation",
            "description": "Tests whether SRI (Spectral Resolution Index) predicts per-graph performance gap between RWSE and LapPE encodings",
            "phases": {
                "phase1_sri_distributions": phase1,
                "phase2_model_free_quality": phase2,
                "phase3_mlp_proxy": phase3_clean,
                "phase4_correlation": phase4,
                "synthetic_validation": synthetic_val,
            },
            "parameters": {
                "K": 20,
                "mlp_hidden": [128, 64],
                "mlp_seeds": [42, 123, 456],
                "bootstrap_n": 1000,
                "rwse_dim": 20,
                "lape_eigvecs": 16,
            },
        },
        "datasets": datasets_out,
    }
    return output


# ===========================================================================
# MAIN
# ===========================================================================

@logger.catch
def main():
    import time
    t0 = time.time()

    logger.info("=" * 60)
    logger.info("SRI-Performance Gap Correlation Experiment")
    logger.info("=" * 60)
    logger.info(f"RAM: {TOTAL_RAM_GB:.1f} GB total, {AVAIL_RAM_GB:.1f} GB available")
    logger.info(f"CPU count: {os.cpu_count()}")
    logger.info(f"MAX_EXAMPLES: {MAX_EXAMPLES}")

    # Phase 0: Load data
    records = load_all_data(max_per_dataset=MAX_EXAMPLES)
    t_load = time.time() - t0
    logger.info(f"Data loading took {t_load:.1f}s")

    # Phase 1: SRI distributions
    phase1 = phase1_sri_distributions(records)
    t_phase1 = time.time() - t0
    logger.info(f"Phase 1 took {t_phase1 - t_load:.1f}s")

    # Phase 2: Model-free encoding quality
    phase2 = phase2_model_free(records)
    t_phase2 = time.time() - t0
    logger.info(f"Phase 2 took {t_phase2 - t_phase1:.1f}s")

    # Phase 3: MLP proxy training
    phase3 = phase3_mlp_proxy(records)
    t_phase3 = time.time() - t0
    logger.info(f"Phase 3 took {t_phase3 - t_phase2:.1f}s")

    # Phase 4: Correlation analysis
    phase4 = phase4_correlation(phase3)
    t_phase4 = time.time() - t0
    logger.info(f"Phase 4 took {t_phase4 - t_phase3:.1f}s")

    # Synthetic validation
    synthetic_val = validate_synthetic(records)
    t_synth = time.time() - t0
    logger.info(f"Synthetic validation took {t_synth - t_phase4:.1f}s")

    # Phase 5: Visualization
    phase5_visualization(phase3, phase4, phase2)
    t_phase5 = time.time() - t0
    logger.info(f"Phase 5 took {t_phase5 - t_synth:.1f}s")

    # Assemble output
    output = assemble_output(records, phase1, phase2, phase3, phase4, synthetic_val)

    # Save
    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Saved output to {out_path}")

    total_time = time.time() - t0
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info("DONE")


if __name__ == "__main__":
    main()
