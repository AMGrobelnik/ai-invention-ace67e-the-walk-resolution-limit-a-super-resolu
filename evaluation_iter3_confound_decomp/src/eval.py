#!/usr/bin/env python3
"""
Confound Decomposition & Formal Hypothesis Assessment of Walk Resolution Limit.

Synthesizes all iteration-2 experimental evidence (3 experiments, ~16,100 graphs)
to formally assess the walk resolution limit hypothesis through five analyses:
  1. Discrepancy Resolution: model-free vs MLP-proxy gap correlation
  2. Partial Correlations: controlling for confounds with bootstrap CIs
  3. Sparsity-Stratified Analysis: testing hypothesis within intended domain
  4. AUC-based Predictor Comparison: SRI vs alternatives
  5. Formal Hypothesis Assessment: Bayes factors and decision tree

Data strategy:
  - Per-graph quality gaps are NOT stored in experiment outputs (only aggregate statistics).
  - We use SRWE-RWSE L1 distance (from exp2) as a per-graph proxy for "encoding quality gap".
  - We use aggregate results from exp1 metadata (phase2/phase4) for correlation analyses.
  - We use resolution_diagnosis (from exp3) for binary classification AUC.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

from loguru import logger
from pathlib import Path
import json
import sys
import math
import warnings
import resource
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
import psutil

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---- Setup logging ----
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ---- Resource limits ----
TOTAL_RAM_GB = psutil.virtual_memory().total / 1e9
RAM_LIMIT_GB = min(TOTAL_RAM_GB - 2, 50)
try:
    resource.setrlimit(resource.RLIMIT_AS, (int(RAM_LIMIT_GB * 1024**3), int(RAM_LIMIT_GB * 1024**3)))
except Exception:
    pass
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

# ---- Paths ----
WORKSPACE = Path(__file__).parent
EXP1_DIR = Path("/opt/ai-inventor/aii_pipeline/runs/run__20260222_005317/3_invention_loop/iter_2/gen_art/exp_id1_it2__opus")
EXP2_DIR = Path("/opt/ai-inventor/aii_pipeline/runs/run__20260222_005317/3_invention_loop/iter_2/gen_art/exp_id2_it2__opus")
EXP3_DIR = Path("/opt/ai-inventor/aii_pipeline/runs/run__20260222_005317/3_invention_loop/iter_2/gen_art/exp_id3_it2__opus")
FIGURES_DIR = WORKSPACE / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ---- Constants ----
BOOTSTRAP_N = int(os.environ.get("BOOTSTRAP_N", 1000))
BOOTSTRAP_AUC_N = int(os.environ.get("BOOTSTRAP_AUC_N", 500))
RNG = np.random.RandomState(42)
MAX_EXAMPLES = int(os.environ.get("MAX_EXAMPLES", 0)) or None


# ============================================================
# DATA LOADING
# ============================================================

def load_experiment(exp_dir: Path, label: str) -> dict:
    """Load a full experiment output JSON."""
    full_path = exp_dir / "full_method_out.json"
    logger.info(f"Loading {label} from {full_path}")
    data = json.loads(full_path.read_text())
    logger.info(f"  Loaded {label}: {len(data.get('datasets', []))} datasets")
    return data


def parse_json_field(s: str) -> dict:
    """Parse a JSON string field, returning empty dict on failure."""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return {}


def build_merged_df(exp1: dict, exp2: dict, exp3: dict) -> pd.DataFrame:
    """Build a merged per-graph DataFrame from all 3 experiments.

    All experiments process graphs in the same order within each dataset.
    We use sequential position (seq_idx) within each dataset for matching,
    since exp1 uses metadata_row_index (original dataset indices) while
    exp2/exp3 use sequential indices or graph_idx.
    """
    logger.info("Building merged per-graph DataFrame...")

    rows = []

    # Build per-dataset lists from each experiment for sequential matching
    # Exp1: primary data source
    exp1_by_ds = {}
    for ds in exp1["datasets"]:
        ds_name = ds["dataset"]
        exp1_by_ds[ds_name] = ds["examples"]

    # Exp2: SRWE/RWSE encodings
    exp2_by_ds = {}
    for ds in exp2["datasets"]:
        ds_name = ds["dataset"]
        exp2_by_ds[ds_name] = ds["examples"]

    # Exp3: spectral diagnostics
    exp3_by_ds = {}
    for ds in exp3["datasets"]:
        ds_name = ds["dataset"]
        exp3_by_ds[ds_name] = ds["examples"]

    # Merge by sequential position within each dataset
    for ds_name, exp1_examples in exp1_by_ds.items():
        exp2_examples = exp2_by_ds.get(ds_name, [])
        exp3_examples = exp3_by_ds.get(ds_name, [])

        n = len(exp1_examples)
        for seq_idx in range(n):
            ex1 = exp1_examples[seq_idx]
            inp1 = parse_json_field(ex1["input"])
            pred_ours = parse_json_field(ex1.get("predict_our_method", "{}"))

            row = {
                "dataset": ds_name,
                "graph_idx": ex1.get("metadata_row_index", seq_idx),
                "seq_idx": seq_idx,
                "num_nodes": ex1.get("metadata_num_nodes", inp1.get("num_nodes")),
                "num_edges": inp1.get("edge_index_len"),
                "sri_k20": ex1.get("metadata_sri_k20", pred_ours.get("sri_k20")),
                "delta_min": ex1.get("metadata_delta_min", pred_ours.get("delta_min")),
                "vander_k20": pred_ours.get("vander_k20"),
                "fold": ex1.get("metadata_fold"),
            }

            # Exp2 data (match by sequential position)
            if seq_idx < len(exp2_examples):
                ex2 = exp2_examples[seq_idx]
                try:
                    srwe = json.loads(ex2.get("predict_srwe", "[]"))
                    rwse = json.loads(ex2.get("predict_rwse", "[]"))
                    srwe_arr = np.array(srwe, dtype=np.float64)
                    rwse_arr = np.array(rwse, dtype=np.float64)
                    if len(srwe_arr) > 0 and len(rwse_arr) > 0:
                        row["srwe_rwse_l1"] = float(np.sum(np.abs(srwe_arr - rwse_arr)))
                        row["srwe_rwse_l2"] = float(np.sqrt(np.sum((srwe_arr - rwse_arr)**2)))
                except Exception:
                    pass

            # Exp3 data (match by sequential position)
            if seq_idx < len(exp3_examples):
                ex3 = exp3_examples[seq_idx]
                pred_diag = parse_json_field(ex3.get("predict_spectral_diagnostics", "{}"))
                pred_base = parse_json_field(ex3.get("predict_eigenvalue_clustering_baseline", "{}"))

                row["participation_ratio"] = pred_diag.get("sparsity_participation_ratio_median")
                row["vander_cond_log10"] = pred_diag.get("vandermonde_cond_K20_log10")
                row["resolution_diagnosis"] = pred_diag.get("resolution_diagnosis")
                row["eff_rank_1pct"] = pred_diag.get("sparsity_eff_rank_1pct_median")
                row["n_eigenvalue_clusters"] = pred_base.get("n_eigenvalue_clusters")
                row["spectral_gap"] = pred_base.get("spectral_gap")
                row["max_eigenvalue_gap"] = pred_base.get("max_eigenvalue_gap")

            rows.append(row)

    df = pd.DataFrame(rows)

    if MAX_EXAMPLES is not None:
        # Apply per-dataset limit to preserve dataset diversity
        dfs = []
        for ds_name in df["dataset"].unique():
            ds_df = df[df["dataset"] == ds_name].head(MAX_EXAMPLES)
            dfs.append(ds_df)
        df = pd.concat(dfs, ignore_index=True)

    # Compute derived fields
    df["log_vander_k20"] = np.log10(df["vander_k20"].clip(lower=1).astype(float))

    nn = df["num_nodes"].astype(float)
    ne = df["num_edges"].astype(float)
    df["density"] = (2.0 * ne / (nn * (nn - 1))).clip(0, 1)

    df["normalized_sri"] = df["sri_k20"].astype(float) / nn.clip(lower=1)

    # Binary: is graph aliased?
    df["is_aliased"] = (df["resolution_diagnosis"] == "aliased").astype(int)

    # Per-graph spectral participation ratio normalized
    pr = df["participation_ratio"].astype(float)
    df["pr_normalized"] = pr / nn.clip(lower=1)
    df["is_spectrally_sparse"] = df["pr_normalized"] < 0.1

    logger.info(f"Merged DataFrame: {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"Datasets: {df['dataset'].value_counts().to_dict()}")
    logger.info(f"Aliased fraction: {df['is_aliased'].mean():.3f}")

    return df


# ============================================================
# ANALYSIS 1: Discrepancy Resolution
# ============================================================

def analysis_1_discrepancy_resolution(exp1_meta: dict) -> dict:
    """
    Resolve discrepancy between model-free and MLP proxy gap measures.
    Uses aggregate results from exp1 metadata.
    """
    logger.info("=== Analysis 1: Discrepancy Resolution ===")
    results = {}

    phase2 = exp1_meta.get("phases", {}).get("phase2_model_free_quality", {})
    phase3 = exp1_meta.get("phases", {}).get("phase3_mlp_proxy", {})
    phase4 = exp1_meta.get("phases", {}).get("phase4_correlation", {})

    mf_rhos = []
    mlp_rhos = []
    ds_names = ["ZINC-subset", "Peptides-func", "Peptides-struct", "Synthetic-aliased-pairs"]

    for ds_name in ds_names:
        p2 = phase2.get(ds_name, {})
        p3 = phase3.get(ds_name, {})
        p4 = phase4.get(ds_name, {})

        model_free_rho = p2.get("spearman_sri_vs_gap", {}).get("rho", np.nan)
        model_free_p = p2.get("spearman_sri_vs_gap", {}).get("p", np.nan)
        mean_model_free_gap = p2.get("mean_quality_gap", np.nan)

        mlp_rho = p4.get("primary", {}).get("sri_vs_gap", {}).get("rho", np.nan)
        mlp_p = p4.get("primary", {}).get("sri_vs_gap", {}).get("p", np.nan)
        mlp_ci = p4.get("primary", {}).get("sri_vs_gap", {}).get("bootstrap", {})
        mean_mlp_gap = p3.get("mean_gap", np.nan)

        # Size-controlled results
        size_ctrl = p4.get("size_controlled", [])
        size_ctrl_summary = []
        for sc in size_ctrl:
            size_ctrl_summary.append({
                "size_range": sc.get("size_range", ""),
                "n": sc.get("n", 0),
                "rho": sc.get("rho", np.nan),
                "p": sc.get("p", np.nan),
            })

        # SRI vs num_nodes
        sri_vs_nodes = p4.get("confounder", {}).get("sri_vs_num_nodes", {})

        discrepancy = abs(model_free_rho) - abs(mlp_rho) if not (np.isnan(model_free_rho) or np.isnan(mlp_rho)) else np.nan

        ds_result = {
            "model_free_rho": model_free_rho,
            "model_free_p": model_free_p,
            "mean_model_free_gap": mean_model_free_gap,
            "mlp_proxy_rho": mlp_rho,
            "mlp_proxy_p": mlp_p,
            "mlp_proxy_ci_low": mlp_ci.get("ci_low", np.nan),
            "mlp_proxy_ci_high": mlp_ci.get("ci_high", np.nan),
            "mean_mlp_gap": mean_mlp_gap,
            "discrepancy_abs_rho": discrepancy,
            "sri_vs_num_nodes_rho": sri_vs_nodes.get("rho", np.nan),
            "size_controlled": size_ctrl_summary,
        }
        results[ds_name] = ds_result

        if not np.isnan(model_free_rho):
            mf_rhos.append(model_free_rho)
        if not np.isnan(mlp_rho):
            mlp_rhos.append(mlp_rho)

        logger.info(f"  {ds_name}: model_free_rho={model_free_rho:.3f}, mlp_rho={mlp_rho if not np.isnan(mlp_rho) else 'N/A'}, "
                     f"discrepancy={discrepancy if not np.isnan(discrepancy) else 'N/A'}")

    # Correlation between model_free and mlp rho across datasets
    if len(mf_rhos) >= 3 and len(mlp_rhos) >= 3:
        # Only compare datasets where both are available
        common_ds = [ds for ds in ds_names if ds in results
                     and not np.isnan(results[ds].get("model_free_rho", np.nan))
                     and not np.isnan(results[ds].get("mlp_proxy_rho", np.nan))]
        if len(common_ds) >= 3:
            mf = [results[ds]["model_free_rho"] for ds in common_ds]
            ml = [results[ds]["mlp_proxy_rho"] for ds in common_ds]
            try:
                corr, p_val = stats.spearmanr(mf, ml)
            except Exception:
                corr, p_val = np.nan, np.nan
            results["model_free_vs_mlp_gap_correlation"] = corr
            results["model_free_vs_mlp_gap_p"] = p_val
        else:
            results["model_free_vs_mlp_gap_correlation"] = np.nan
            results["model_free_vs_mlp_gap_p"] = np.nan
    else:
        results["model_free_vs_mlp_gap_correlation"] = np.nan
        results["model_free_vs_mlp_gap_p"] = np.nan

    # MLP gap coefficient of variation
    mlp_gaps = [results[ds].get("mean_mlp_gap", np.nan) for ds in ds_names
                if ds in results and not np.isnan(results[ds].get("mean_mlp_gap", np.nan))]
    if len(mlp_gaps) > 1:
        cv = float(np.std(mlp_gaps) / (np.mean(np.abs(mlp_gaps)) + 1e-10))
    else:
        cv = np.nan
    results["mlp_gap_coefficient_of_variation"] = cv

    logger.info(f"  Cross-dataset model_free vs mlp correlation: {results.get('model_free_vs_mlp_gap_correlation', 'N/A')}")
    logger.info(f"  MLP gap CV: {cv}")
    return results


# ============================================================
# ANALYSIS 2: Partial Correlations
# ============================================================

def bootstrap_spearman(x: np.ndarray, y: np.ndarray, n_boot: int = BOOTSTRAP_N) -> dict:
    """Compute Spearman rho with bootstrap CI."""
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    n = len(x)
    if n < 10:
        return {"rho": np.nan, "p": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": n}

    rho, p = stats.spearmanr(x, y)
    boot_rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.randint(0, n, n)
        try:
            boot_rhos[i], _ = stats.spearmanr(x[idx], y[idx])
        except Exception:
            boot_rhos[i] = np.nan
    boot_rhos = boot_rhos[np.isfinite(boot_rhos)]
    ci_low = float(np.percentile(boot_rhos, 2.5)) if len(boot_rhos) > 0 else np.nan
    ci_high = float(np.percentile(boot_rhos, 97.5)) if len(boot_rhos) > 0 else np.nan
    return {"rho": float(rho), "p": float(p), "ci_low": ci_low, "ci_high": ci_high, "n": n}


def partial_spearman(df_sub: pd.DataFrame, x_col: str, y_col: str,
                     covariates: list, n_boot: int = BOOTSTRAP_N) -> dict:
    """Compute partial Spearman correlation with bootstrap CI."""
    needed = [x_col, y_col] + covariates
    df_clean = df_sub[needed].dropna()
    n = len(df_clean)
    if n < 10:
        return {"rho": np.nan, "p": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": n}

    try:
        result = pg.partial_corr(data=df_clean, x=x_col, y=y_col,
                                  covar=covariates, method="spearman")
        rho = float(result["r"].values[0])
        p_val = float(result["p-val"].values[0])
    except Exception as e:
        logger.debug(f"Partial corr failed: {e}")
        return {"rho": np.nan, "p": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": n}

    boot_rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.randint(0, n, n)
        try:
            df_boot = df_clean.iloc[idx].reset_index(drop=True)
            res = pg.partial_corr(data=df_boot, x=x_col, y=y_col,
                                   covar=covariates, method="spearman")
            boot_rhos[i] = float(res["r"].values[0])
        except Exception:
            boot_rhos[i] = np.nan
    boot_rhos = boot_rhos[np.isfinite(boot_rhos)]
    ci_low = float(np.percentile(boot_rhos, 2.5)) if len(boot_rhos) > 0 else np.nan
    ci_high = float(np.percentile(boot_rhos, 97.5)) if len(boot_rhos) > 0 else np.nan
    return {"rho": rho, "p": p_val, "ci_low": ci_low, "ci_high": ci_high, "n": n}


def analysis_2_partial_correlations(df: pd.DataFrame) -> dict:
    """
    Partial Spearman correlations between SRI and encoding quality proxies,
    controlling for confounders.

    Two proxy measures:
      (a) srwe_rwse_l1: direct SRWE-RWSE L1 distance (n~100/dataset, from exp2)
      (b) vander_cond_log10: Vandermonde condition number log10 (all graphs, from exp3)
          Higher condition number → harder spectral recovery → larger encoding gap.
    """
    logger.info("=== Analysis 2: Partial Correlations ===")
    results = {}

    proxy_cols = ["srwe_rwse_l1", "vander_cond_log10"]

    covariate_sets = {
        "raw": [],
        "ctrl_num_nodes": ["num_nodes"],
        "ctrl_num_nodes_density": ["num_nodes", "density"],
    }
    if df["participation_ratio"].notna().sum() > 100:
        covariate_sets["ctrl_num_nodes_density_pr"] = ["num_nodes", "density", "participation_ratio"]

    datasets = sorted(df["dataset"].unique())
    for ds_name in datasets:
        ds_df = df[df["dataset"] == ds_name].copy()
        # Ensure numeric types
        for col in ["sri_k20"] + proxy_cols + ["num_nodes", "density", "participation_ratio"]:
            if col in ds_df.columns:
                ds_df[col] = pd.to_numeric(ds_df[col], errors="coerce")

        ds_results = {}
        for gap_col in proxy_cols:
            gap_results = {}
            n_avail = ds_df[gap_col].notna().sum()
            logger.info(f"  {ds_name} proxy={gap_col}: {n_avail} non-null values")

            for cov_name, covariates in covariate_sets.items():
                if cov_name == "raw":
                    res = bootstrap_spearman(
                        ds_df["sri_k20"].values.astype(float),
                        ds_df[gap_col].values.astype(float)
                    )
                else:
                    avail = all(c in ds_df.columns and ds_df[c].notna().sum() > 10 for c in covariates)
                    if not avail:
                        res = {"rho": np.nan, "p": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}
                    else:
                        res = partial_spearman(ds_df, "sri_k20", gap_col, covariates, n_boot=BOOTSTRAP_N)

                gap_results[cov_name] = res
                rho_str = f"{res['rho']:.4f}" if not np.isnan(res['rho']) else "NaN"
                ci_str = f"[{res['ci_low']:.4f}, {res['ci_high']:.4f}]" if not np.isnan(res['ci_low']) else "[NaN, NaN]"
                logger.info(f"    {ds_name} [{gap_col}|{cov_name}]: rho={rho_str}, CI={ci_str}, n={res['n']}")

            ds_results[gap_col] = gap_results

        results[ds_name] = ds_results

    return results


# ============================================================
# ANALYSIS 3: Sparsity-Stratified Analysis
# ============================================================

def analysis_3_sparsity_stratified(df: pd.DataFrame) -> dict:
    """
    Test hypothesis within spectrally-sparse and dense subsets.
    Sparse: pr_normalized < 0.1 (participation ratio / num_nodes).

    Uses vander_cond_log10 as primary proxy (available for all graphs)
    and srwe_rwse_l1 as secondary (available for ~100/dataset).
    """
    logger.info("=== Analysis 3: Sparsity-Stratified Analysis ===")
    results = {}
    proxy_cols = ["vander_cond_log10", "srwe_rwse_l1"]

    datasets = sorted(df["dataset"].unique())
    for ds_name in datasets:
        ds_df = df[df["dataset"] == ds_name].copy()
        for col in ["sri_k20", "vander_cond_log10", "srwe_rwse_l1", "num_nodes", "is_spectrally_sparse"]:
            if col in ds_df.columns:
                ds_df[col] = pd.to_numeric(ds_df[col], errors="coerce")

        ds_results = {}
        for subset_name, is_sparse in [("sparse", True), ("dense", False)]:
            subset = ds_df[ds_df["is_spectrally_sparse"] == (1 if is_sparse else 0)]
            n_subset = len(subset)

            if n_subset < 10:
                ds_results[subset_name] = {
                    "n": n_subset,
                }
                for gap_col in proxy_cols:
                    ds_results[subset_name][f"rho_raw_{gap_col}"] = np.nan
                    ds_results[subset_name][f"rho_raw_ci_{gap_col}"] = [np.nan, np.nan]
                    ds_results[subset_name][f"rho_partial_{gap_col}"] = np.nan
                    ds_results[subset_name][f"rho_partial_ci_{gap_col}"] = [np.nan, np.nan]
                logger.info(f"  {ds_name} [{subset_name}]: n={n_subset} (too few)")
                continue

            subset_result = {"n": n_subset}
            for gap_col in proxy_cols:
                raw = bootstrap_spearman(subset["sri_k20"].values.astype(float),
                                          subset[gap_col].values.astype(float))
                partial = partial_spearman(subset, "sri_k20", gap_col, ["num_nodes"], n_boot=BOOTSTRAP_N)

                subset_result[f"rho_raw_{gap_col}"] = raw["rho"]
                subset_result[f"rho_raw_ci_{gap_col}"] = [raw["ci_low"], raw["ci_high"]]
                subset_result[f"rho_partial_{gap_col}"] = partial["rho"]
                subset_result[f"rho_partial_ci_{gap_col}"] = [partial["ci_low"], partial["ci_high"]]
                subset_result[f"n_valid_{gap_col}"] = raw["n"]
                logger.info(f"  {ds_name} [{subset_name}|{gap_col}]: n_valid={raw['n']}, "
                           f"rho_raw={raw['rho']:.4f}, rho_partial={partial['rho']:.4f}")

            ds_results[subset_name] = subset_result

        results[ds_name] = ds_results

    return results


# ============================================================
# ANALYSIS 4: Predictor Comparison (AUC)
# ============================================================

def compute_auc_with_bootstrap(y_true: np.ndarray, scores: np.ndarray,
                                n_boot: int = BOOTSTRAP_AUC_N) -> dict:
    """Compute AUC with bootstrap CI. Also tries negated scores and picks best."""
    valid = np.isfinite(y_true) & np.isfinite(scores)
    y_true = y_true[valid].astype(float)
    scores = scores[valid].astype(float)
    n = len(y_true)

    if n < 10 or len(np.unique(y_true)) < 2:
        return {"auc": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": n, "direction": "N/A"}

    try:
        auc_pos = roc_auc_score(y_true, scores)
        auc_neg = roc_auc_score(y_true, -scores)
    except ValueError:
        return {"auc": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": n, "direction": "N/A"}

    if auc_neg > auc_pos:
        auc = auc_neg
        direction = "negative"
        use_scores = -scores
    else:
        auc = auc_pos
        direction = "positive"
        use_scores = scores

    boot_aucs = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.randint(0, n, n)
        y_b, s_b = y_true[idx], use_scores[idx]
        if len(np.unique(y_b)) < 2:
            boot_aucs[i] = np.nan
            continue
        try:
            boot_aucs[i] = roc_auc_score(y_b, s_b)
        except ValueError:
            boot_aucs[i] = np.nan

    boot_aucs = boot_aucs[np.isfinite(boot_aucs)]
    ci_low = float(np.percentile(boot_aucs, 2.5)) if len(boot_aucs) > 0 else np.nan
    ci_high = float(np.percentile(boot_aucs, 97.5)) if len(boot_aucs) > 0 else np.nan

    return {"auc": float(auc), "ci_low": ci_low, "ci_high": ci_high, "n": n, "direction": direction}


def analysis_4_predictor_comparison(df: pd.DataFrame) -> dict:
    """
    Compare predictors using AUC for binary classification.
    Two binary targets:
      (a) "is_aliased" (resolution_diagnosis == "aliased") — from exp3
      (b) "high_srwe_rwse_gap" (SRWE-RWSE L1 > median) — from exp2
    """
    logger.info("=== Analysis 4: Predictor Comparison (AUC) ===")
    results = {}

    predictor_specs = {
        "SRI": "sri_k20",
        "log_Vandermonde_kappa": "log_vander_k20",
        "delta_min": "delta_min",
        "normalized_SRI": "normalized_sri",
        "num_nodes": "num_nodes",
    }

    if df["participation_ratio"].notna().sum() > 50:
        predictor_specs["eff_spectral_rank"] = "participation_ratio"
    if df["vander_cond_log10"].notna().sum() > 50:
        predictor_specs["vander_cond_log10"] = "vander_cond_log10"
    if df["spectral_gap"].notna().sum() > 50:
        predictor_specs["spectral_gap"] = "spectral_gap"

    # Target (a): is_aliased
    targets = {"is_aliased": "is_aliased"}

    # Target (b): high SRWE-RWSE gap (above median per dataset)
    df = df.copy()
    df["high_srwe_gap"] = np.nan
    for ds_name in df["dataset"].unique():
        mask = df["dataset"] == ds_name
        vals = df.loc[mask, "srwe_rwse_l1"].dropna()
        if len(vals) > 10:
            med = vals.median()
            non_null = df["dataset"] == ds_name
            non_null &= df["srwe_rwse_l1"].notna()
            df.loc[non_null, "high_srwe_gap"] = (df.loc[non_null, "srwe_rwse_l1"] > med).astype(int)
    targets["high_srwe_rwse_gap"] = "high_srwe_gap"

    # Target (c): high Vandermonde condition (above median per dataset, available for all)
    df["high_vander_cond"] = 0
    for ds_name in df["dataset"].unique():
        mask = df["dataset"] == ds_name
        vals = df.loc[mask, "vander_cond_log10"].dropna()
        if len(vals) > 10:
            med = vals.median()
            df.loc[mask, "high_vander_cond"] = (df.loc[mask, "vander_cond_log10"] > med).astype(int)
    targets["high_vander_cond"] = "high_vander_cond"

    datasets = sorted(df["dataset"].unique())
    for target_name, target_col in targets.items():
        target_results = {}
        for ds_name in datasets:
            ds_df = df[df["dataset"] == ds_name]
            y_true = ds_df[target_col].values.astype(float)

            if len(np.unique(y_true[np.isfinite(y_true)])) < 2:
                logger.warning(f"  {ds_name} [{target_name}]: Only one class, skipping")
                target_results[ds_name] = {pred: {"auc": np.nan} for pred in predictor_specs}
                continue

            ds_pred_results = {}
            for pred_name, pred_col in predictor_specs.items():
                if pred_col not in ds_df.columns or ds_df[pred_col].notna().sum() < 10:
                    ds_pred_results[pred_name] = {"auc": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}
                    continue
                scores = ds_df[pred_col].values.astype(float)
                auc_res = compute_auc_with_bootstrap(y_true, scores, n_boot=BOOTSTRAP_AUC_N)
                ds_pred_results[pred_name] = auc_res
                logger.info(f"  {ds_name} [{target_name}] | {pred_name}: AUC={auc_res['auc']:.4f} "
                           f"[{auc_res['ci_low']:.4f}, {auc_res['ci_high']:.4f}] dir={auc_res['direction']}")

            target_results[ds_name] = ds_pred_results
        results[target_name] = target_results

    return results


# ============================================================
# ANALYSIS 5: Formal Hypothesis Assessment
# ============================================================

def interpret_bf(bf: float) -> str:
    """Interpret Bayes Factor."""
    if np.isnan(bf) or np.isinf(bf):
        return "N/A"
    if bf > 100:
        return "extreme_for_H1"
    if bf > 30:
        return "very_strong_for_H1"
    if bf > 10:
        return "strong_for_H1"
    if bf > 3:
        return "moderate_for_H1"
    if bf > 1:
        return "anecdotal_for_H1"
    if bf > 1/3:
        return "anecdotal_for_H0"
    if bf > 1/10:
        return "moderate_for_H0"
    return "strong_for_H0"


def bayesfactor_correlation(r: float, n: int) -> float:
    """Compute BF10 for correlation using pingouin."""
    if np.isnan(r) or n < 5:
        return np.nan
    try:
        bf = pg.bayesfactor_pearson(abs(r), n, alternative="two-sided")
        return float(bf)
    except Exception:
        return np.nan


def analysis_5_formal_assessment(exp1_meta: dict, exp2_meta: dict,
                                  analysis2_results: dict, analysis4_results: dict) -> dict:
    """
    Formal hypothesis assessment with Bayes factors.
    Criterion 1: rho(SRI, gap) > 0.5 (using exp1 metadata model-free results)
    Criterion 2: AUC > 0.65 (using analysis 4)
    Criterion 3: SRWE gap reduction >= 50% (using exp2 GNN benchmark)
    """
    logger.info("=== Analysis 5: Formal Hypothesis Assessment ===")
    results = {"criteria": {}, "bayes_factors": {}}

    # ---- Criterion 1: Correlation (from exp1 metadata) ----
    phase2 = exp1_meta.get("phases", {}).get("phase2_model_free_quality", {})
    phase4 = exp1_meta.get("phases", {}).get("phase4_correlation", {})

    criterion1 = {}
    ds_names = ["ZINC-subset", "Peptides-func", "Peptides-struct", "Synthetic-aliased-pairs"]

    for ds_name in ds_names:
        p2 = phase2.get(ds_name, {})
        p4 = phase4.get(ds_name, {})

        # Model-free rho (the raw correlation from the experiment)
        raw_rho = p2.get("spearman_sri_vs_gap", {}).get("rho", np.nan)
        n_phase2 = p2.get("n_valid", 0)

        # MLP proxy rho (partial/size-controlled)
        mlp_rho = p4.get("primary", {}).get("sri_vs_gap", {}).get("rho", np.nan)
        n_phase4 = p4.get("n_test", 0)

        # Size-controlled rhos (act as partial correlations)
        size_ctrl = p4.get("size_controlled", [])
        size_ctrl_rhos = [sc.get("rho", np.nan) for sc in size_ctrl]
        mean_size_ctrl = float(np.nanmean(size_ctrl_rhos)) if size_ctrl_rhos else np.nan

        raw_pass = abs(raw_rho) > 0.5 if not np.isnan(raw_rho) else False
        partial_pass = abs(mean_size_ctrl) > 0.5 if not np.isnan(mean_size_ctrl) else False

        # Also use our analysis2 partial correlation as supplementary evidence
        # Use vander_cond_log10 proxy (available for all graphs) as primary
        a2_vander = analysis2_results.get(ds_name, {}).get("vander_cond_log10", {})
        a2_partial = a2_vander.get("ctrl_num_nodes", {}).get("rho", np.nan)
        # Also grab SRWE-RWSE proxy (limited samples)
        a2_srwe = analysis2_results.get(ds_name, {}).get("srwe_rwse_l1", {})
        a2_partial_srwe = a2_srwe.get("ctrl_num_nodes", {}).get("rho", np.nan)

        bf_raw = bayesfactor_correlation(raw_rho, n_phase2)
        bf_mlp = bayesfactor_correlation(mlp_rho, n_phase4) if not np.isnan(mlp_rho) else np.nan

        criterion1[ds_name] = {
            "model_free_raw_rho": raw_rho,
            "model_free_n": n_phase2,
            "model_free_pass": raw_pass,
            "mlp_proxy_rho": mlp_rho,
            "mlp_proxy_n": n_phase4,
            "mean_size_controlled_rho": mean_size_ctrl,
            "size_controlled_pass": partial_pass,
            "analysis2_partial_rho_vander": a2_partial,
            "analysis2_partial_rho_srwe": a2_partial_srwe,
            "bf10_model_free": bf_raw,
            "bf10_mlp_proxy": bf_mlp,
        }
        logger.info(f"  C1 {ds_name}: raw_rho={raw_rho:.4f} pass={raw_pass}, "
                     f"mlp_rho={mlp_rho if not np.isnan(mlp_rho) else 'N/A'}, "
                     f"size_ctrl_mean={mean_size_ctrl if not np.isnan(mean_size_ctrl) else 'N/A'}")

    results["criteria"]["criterion_1_correlation"] = criterion1

    # ---- Criterion 2: AUC ----
    criterion2 = {}
    # Use "is_aliased" target from analysis 4
    a4_aliased = analysis4_results.get("is_aliased", {})
    for ds_name in ds_names:
        sri_res = a4_aliased.get(ds_name, {}).get("SRI", {})
        sri_auc = sri_res.get("auc", np.nan)
        auc_pass = sri_auc > 0.65 if not np.isnan(sri_auc) else False
        criterion2[ds_name] = {
            "sri_auc": sri_auc,
            "sri_auc_ci": [sri_res.get("ci_low", np.nan), sri_res.get("ci_high", np.nan)],
            "pass": auc_pass,
        }
        logger.info(f"  C2 {ds_name}: SRI AUC={sri_auc:.4f}" if not np.isnan(sri_auc) else f"  C2 {ds_name}: SRI AUC=N/A")

    results["criteria"]["criterion_2_auc"] = criterion2

    # ---- Criterion 3: SRWE gap reduction ----
    gnn = exp2_meta.get("phase4_gnn_benchmark", {}).get("encoding_results", {})
    rwse_mae = gnn.get("rwse", {}).get("mean_test_mae", np.nan)
    lape_mae = gnn.get("lappe", {}).get("mean_test_mae", np.nan)
    srwe_mae = gnn.get("srwe", {}).get("mean_test_mae", np.nan)

    if not any(np.isnan(x) for x in [rwse_mae, lape_mae, srwe_mae]):
        # LapPE is worst, RWSE is best. SRWE should close the gap.
        # gap_reduction = how much of (LapPE - RWSE) gap does SRWE close?
        # = (LapPE - SRWE) / (LapPE - RWSE) * 100
        total_gap = lape_mae - rwse_mae
        srwe_improvement = lape_mae - srwe_mae
        gap_reduction_pct = (srwe_improvement / total_gap * 100) if abs(total_gap) > 1e-10 else np.nan
    else:
        gap_reduction_pct = np.nan

    c3_pass = gap_reduction_pct >= 50 if not np.isnan(gap_reduction_pct) else False
    criterion3 = {
        "rwse_mae": rwse_mae,
        "lape_mae": lape_mae,
        "srwe_mae": srwe_mae,
        "srwe_gap_reduction_pct": gap_reduction_pct,
        "pass": c3_pass,
    }
    results["criteria"]["criterion_3_srwe"] = criterion3
    logger.info(f"  C3: RWSE={rwse_mae:.4f}, LapPE={lape_mae:.4f}, SRWE={srwe_mae:.4f}, "
                f"gap_red={gap_reduction_pct:.1f}%, pass={c3_pass}")

    # ---- Overall Verdict ----
    c1_raw_passes = sum(1 for ds in criterion1 if criterion1[ds].get("model_free_pass", False))
    c1_partial_passes = sum(1 for ds in criterion1 if criterion1[ds].get("size_controlled_pass", False))
    c1_total = len(criterion1)
    c2_passes = sum(1 for ds in criterion2 if criterion2[ds].get("pass", False))
    c2_total = len(criterion2)

    # Decision tree:
    # - All three criteria must substantially pass for "confirmed"
    # - At least one criterion passing substantially for "partially_confirmed"
    # - No criteria passing for "disconfirmed"
    if c1_raw_passes >= 2 and c2_passes >= 2 and c3_pass:
        verdict = "confirmed"
    elif c1_raw_passes >= 2 or (c1_raw_passes >= 1 and c3_pass):
        verdict = "partially_confirmed"
    elif c1_raw_passes >= 1 or c2_passes >= 1:
        verdict = "inconclusive"
    else:
        if c3_pass:
            verdict = "inconclusive"
        else:
            verdict = "disconfirmed"

    results["overall_verdict"] = verdict
    results["verdict_details"] = {
        "c1_raw_passes": f"{c1_raw_passes}/{c1_total}",
        "c1_size_ctrl_passes": f"{c1_partial_passes}/{c1_total}",
        "c2_auc_passes": f"{c2_passes}/{c2_total}",
        "c3_srwe_pass": c3_pass,
    }
    logger.info(f"  VERDICT: {verdict}")

    # Bayes factors summary
    for ds_name in criterion1:
        results["bayes_factors"][ds_name] = {
            "bf10_model_free": criterion1[ds_name].get("bf10_model_free", np.nan),
            "bf10_mlp_proxy": criterion1[ds_name].get("bf10_mlp_proxy", np.nan),
            "interpretation_model_free": interpret_bf(criterion1[ds_name].get("bf10_model_free", np.nan)),
            "interpretation_mlp_proxy": interpret_bf(criterion1[ds_name].get("bf10_mlp_proxy", np.nan)),
        }

    # Summary table
    summary_rows = []
    for ds_name in ds_names:
        c1d = criterion1.get(ds_name, {})
        c2d = criterion2.get(ds_name, {})
        summary_rows.append({
            "dataset": ds_name,
            "model_free_rho": c1d.get("model_free_raw_rho", np.nan),
            "mlp_proxy_rho": c1d.get("mlp_proxy_rho", np.nan),
            "size_ctrl_mean_rho": c1d.get("mean_size_controlled_rho", np.nan),
            "sri_auc": c2d.get("sri_auc", np.nan),
            "bf10_model_free": c1d.get("bf10_model_free", np.nan),
            "n_phase2": c1d.get("model_free_n", 0),
            "n_phase4": c1d.get("mlp_proxy_n", 0),
        })
    results["summary_table"] = summary_rows

    return results


# ============================================================
# FIGURE GENERATION
# ============================================================

def make_figures(df: pd.DataFrame, analysis1: dict, analysis2: dict,
                 analysis3: dict, analysis4: dict, analysis5: dict):
    """Generate publication-quality figures."""
    logger.info("=== Generating Figures ===")
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.dpi": 150})

    datasets = sorted(df["dataset"].unique())

    # ---- Figure 1: SRI vs proxy scatter (dual: SRWE-RWSE L1 and Vandermonde cond) ----
    try:
        n_ds = min(len(datasets), 4)
        fig, axes = plt.subplots(2, n_ds, figsize=(4 * n_ds, 7), squeeze=False)
        for row_idx, (gap_col, gap_label) in enumerate([
            ("srwe_rwse_l1", "SRWE-RWSE L1"),
            ("vander_cond_log10", "log₁₀(Vandermonde κ)"),
        ]):
            for i, ds_name in enumerate(datasets[:4]):
                ax = axes[row_idx, i]
                ds_df = df[df["dataset"] == ds_name].dropna(subset=["sri_k20", gap_col])
                if len(ds_df) > 2000:
                    plot_df = ds_df.sample(2000, random_state=42)
                else:
                    plot_df = ds_df
                if len(plot_df) == 0:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(f"{ds_name}\n({gap_label})")
                    continue
                ax.scatter(plot_df["sri_k20"].astype(float), plot_df[gap_col].astype(float),
                          alpha=0.3, s=5, c="steelblue" if row_idx == 0 else "darkorange")
                ax.set_xlabel("SRI (K=20)")
                ax.set_ylabel(gap_label)
                raw_res = analysis2.get(ds_name, {}).get(gap_col, {}).get("raw", {})
                rho = raw_res.get("rho", np.nan)
                rho_str = f"ρ={rho:.3f}" if not np.isnan(rho) else ""
                ax.set_title(f"{ds_name}\n(n={len(ds_df)}) {rho_str}")
        plt.suptitle("SRI vs Encoding Quality Proxies", fontsize=12, y=1.01)
        plt.tight_layout()
        fig.savefig(str(FIGURES_DIR / "scatter_sri_vs_gap.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Saved scatter_sri_vs_gap.png")
    except Exception:
        logger.exception("Failed scatter plot")

    # ---- Figure 2: Forest plot of partial correlations (dual proxy) ----
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(datasets) * 2)), sharey=True)
        colors = {"raw": "#1f77b4", "ctrl_num_nodes": "#ff7f0e",
                  "ctrl_num_nodes_density": "#2ca02c", "ctrl_num_nodes_density_pr": "#d62728"}

        for ax_idx, (proxy_name, proxy_label) in enumerate([
            ("srwe_rwse_l1", "SRI vs SRWE-RWSE L1"),
            ("vander_cond_log10", "SRI vs log₁₀(Vandermonde κ)"),
        ]):
            ax = axes[ax_idx]
            y_labels = []
            y_positions = []
            pos = 0
            for ds_name in sorted(analysis2.keys()):
                proxy_data = analysis2[ds_name].get(proxy_name, {})
                for cov_name in ["raw", "ctrl_num_nodes", "ctrl_num_nodes_density", "ctrl_num_nodes_density_pr"]:
                    res = proxy_data.get(cov_name, {})
                    rho = res.get("rho", np.nan)
                    if np.isnan(rho):
                        continue
                    ci_low = res.get("ci_low", np.nan)
                    ci_high = res.get("ci_high", np.nan)
                    color = colors.get(cov_name, "gray")
                    ax.plot(rho, pos, "o", color=color, markersize=6)
                    if not np.isnan(ci_low) and not np.isnan(ci_high):
                        ax.hlines(pos, ci_low, ci_high, color=color, linewidth=2)
                    cov_label = cov_name.replace("ctrl_", "| ").replace("_", ", ")
                    n_val = res.get("n", 0)
                    y_labels.append(f"{ds_name} {cov_label} (n={n_val})")
                    y_positions.append(pos)
                    pos += 1
                pos += 0.5

            ax.set_yticks(y_positions)
            ax.set_yticklabels(y_labels, fontsize=7)
            ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
            ax.axvline(0.5, color="red", linestyle="--", alpha=0.3, label="ρ=0.5 threshold")
            ax.axvline(-0.5, color="red", linestyle="--", alpha=0.3)
            ax.set_xlabel(f"Spearman ρ ({proxy_label})")
            ax.set_title(f"Proxy: {proxy_label}")

        axes[0].legend(loc="lower right")
        plt.suptitle("Partial Correlations with Bootstrap 95% CIs", fontsize=12)
        plt.tight_layout()
        fig.savefig(str(FIGURES_DIR / "forest_partial_correlations.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Saved forest_partial_correlations.png")
    except Exception:
        logger.exception("Failed forest plot")

    # ---- Figure 3: Predictor comparison ----
    try:
        a4_aliased = analysis4.get("is_aliased", {})
        valid_ds = [ds for ds in datasets if ds in a4_aliased]
        n_ds = min(len(valid_ds), 4)
        if n_ds > 0:
            fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 5), squeeze=False)
            for i, ds_name in enumerate(valid_ds[:4]):
                ax = axes[0, i]
                preds = a4_aliased[ds_name]
                names = []
                aucs = []
                errs_lo = []
                errs_hi = []
                for pred_name, pred_res in sorted(preds.items()):
                    auc = pred_res.get("auc", np.nan)
                    if np.isnan(auc):
                        continue
                    names.append(pred_name)
                    aucs.append(auc)
                    errs_lo.append(auc - pred_res.get("ci_low", auc))
                    errs_hi.append(pred_res.get("ci_high", auc) - auc)

                if len(names) > 0:
                    c = ["#e74c3c" if n == "SRI" else "#3498db" if n == "num_nodes" else "#95a5a6" for n in names]
                    y_pos = np.arange(len(names))
                    ax.barh(y_pos, aucs, xerr=[errs_lo, errs_hi], color=c, capsize=3, alpha=0.8)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(names, fontsize=8)
                    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
                    ax.axvline(0.65, color="red", linestyle="--", alpha=0.3)
                    ax.set_xlabel("AUC")
                    ax.set_title(f"{ds_name}\n(target: is_aliased)")
                    ax.set_xlim(0.3, 1.0)

            plt.suptitle("Predictor Comparison: AUC for 'is_aliased'", fontsize=12)
            plt.tight_layout()
            fig.savefig(str(FIGURES_DIR / "predictor_comparison_auc.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("  Saved predictor_comparison_auc.png")
    except Exception:
        logger.exception("Failed predictor plot")

    # ---- Figure 4: Quintile bar chart (Vandermonde cond - available for all graphs) ----
    try:
        gap_col_q = "vander_cond_log10"
        n_ds = min(len(datasets), 4)
        fig, axes = plt.subplots(1, n_ds, figsize=(4 * n_ds, 4), squeeze=False)
        for i, ds_name in enumerate(datasets[:4]):
            ax = axes[0, i]
            ds_df = df[df["dataset"] == ds_name].dropna(subset=["sri_k20", gap_col_q])
            if len(ds_df) < 20:
                ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{ds_name}")
                continue
            ds_df = ds_df.copy()
            ds_df["quintile"] = pd.qcut(ds_df["sri_k20"].astype(float), 5, labels=False, duplicates="drop")
            q_means = ds_df.groupby("quintile")[gap_col_q].mean()
            q_sems = ds_df.groupby("quintile")[gap_col_q].sem()
            ax.bar(q_means.index, q_means.values, yerr=q_sems.values, color="darkorange", alpha=0.7, capsize=3)
            ax.set_xlabel("SRI Quintile")
            ax.set_ylabel("Mean log₁₀(Vandermonde κ)")
            ax.set_title(f"{ds_name} (n={len(ds_df)})")
        plt.suptitle("Vandermonde Condition by SRI Quintile", fontsize=12)
        plt.tight_layout()
        fig.savefig(str(FIGURES_DIR / "quintile_gap.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Saved quintile_gap.png")
    except Exception:
        logger.exception("Failed quintile plot")

    # ---- Figure 5: Verdict summary ----
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axis("off")
        verdict = analysis5.get("overall_verdict", "N/A")
        vd = analysis5.get("verdict_details", {})
        c3 = analysis5.get("criteria", {}).get("criterion_3_srwe", {})

        text = f"OVERALL VERDICT: {verdict.upper()}\n\n"
        text += f"Criterion 1 (|ρ| > 0.5, model-free): {vd.get('c1_raw_passes', 'N/A')}\n"
        text += f"Criterion 1 (|ρ| > 0.5, size-ctrl):  {vd.get('c1_size_ctrl_passes', 'N/A')}\n"
        text += f"Criterion 2 (AUC > 0.65):             {vd.get('c2_auc_passes', 'N/A')}\n"
        text += f"Criterion 3 (SRWE gap ≥ 50%):         {vd.get('c3_srwe_pass', 'N/A')}\n\n"
        text += f"SRWE gap reduction: {c3.get('srwe_gap_reduction_pct', 0):.1f}%\n"
        text += f"(RWSE={c3.get('rwse_mae', 0):.4f}, LapPE={c3.get('lape_mae', 0):.4f}, SRWE={c3.get('srwe_mae', 0):.4f})"

        color = {"confirmed": "#27ae60", "partially_confirmed": "#f39c12",
                 "inconclusive": "#7f8c8d", "disconfirmed": "#e74c3c"}.get(verdict, "#7f8c8d")
        ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=11, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.3),
                family="monospace")
        fig.savefig(str(FIGURES_DIR / "verdict_summary.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Saved verdict_summary.png")
    except Exception:
        logger.exception("Failed verdict figure")


# ============================================================
# OUTPUT FORMATTING
# ============================================================

def sanitize_value(v):
    """Convert NaN/Inf/numpy types to JSON-safe values."""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, 6)
    if isinstance(v, np.floating):
        val = float(v)
        if math.isnan(val) or math.isinf(val):
            return None
        return round(val, 6)
    if isinstance(v, (np.integer, np.int64, np.int32)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, dict):
        return {str(k): sanitize_value(vv) for k, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [sanitize_value(vv) for vv in v]
    if isinstance(v, np.ndarray):
        return [sanitize_value(vv) for vv in v.tolist()]
    if isinstance(v, pd.Timestamp):
        return str(v)
    return v


def build_output(df: pd.DataFrame, analysis1: dict, analysis2: dict,
                 analysis3: dict, analysis4: dict, analysis5: dict) -> dict:
    """Build output JSON conforming to exp_eval_sol_out schema."""

    # ---- Aggregate metrics (numbers only, no None) ----
    metrics_agg = {}

    # Analysis 1
    mf_vs_mlp = analysis1.get("model_free_vs_mlp_gap_correlation", 0.0)
    metrics_agg["model_free_vs_mlp_gap_correlation"] = mf_vs_mlp if mf_vs_mlp is not None and not np.isnan(mf_vs_mlp) else 0.0
    cv = analysis1.get("mlp_gap_coefficient_of_variation", 0.0)
    metrics_agg["mlp_gap_coefficient_of_variation"] = cv if cv is not None and not np.isnan(cv) else 0.0

    # Analysis 2 - mean partial rhos (dual proxy)
    for proxy_name in ["srwe_rwse_l1", "vander_cond_log10"]:
        raw_rhos = []
        partial_rhos = []
        for ds in analysis2:
            proxy_data = analysis2[ds].get(proxy_name, {})
            r = proxy_data.get("raw", {}).get("rho", np.nan)
            if not np.isnan(r):
                raw_rhos.append(r)
            pr = proxy_data.get("ctrl_num_nodes", {}).get("rho", np.nan)
            if not np.isnan(pr):
                partial_rhos.append(pr)
        suffix = "_srwe" if proxy_name == "srwe_rwse_l1" else "_vander"
        metrics_agg[f"mean_raw_spearman_rho{suffix}"] = float(np.mean(raw_rhos)) if raw_rhos else 0.0
        metrics_agg[f"mean_partial_spearman_rho{suffix}"] = float(np.mean(partial_rhos)) if partial_rhos else 0.0

    # Analysis 2 - using exp1 metadata model-free rhos
    mf_rhos = []
    for ds_name in ["ZINC-subset", "Peptides-func", "Peptides-struct", "Synthetic-aliased-pairs"]:
        r = analysis1.get(ds_name, {}).get("model_free_rho", np.nan)
        if not np.isnan(r):
            mf_rhos.append(r)
    metrics_agg["mean_model_free_spearman_rho"] = float(np.mean(mf_rhos)) if mf_rhos else 0.0

    # Analysis 4 - SRI AUC
    a4_aliased = analysis4.get("is_aliased", {})
    sri_aucs = []
    for ds in a4_aliased:
        auc = a4_aliased[ds].get("SRI", {}).get("auc", np.nan)
        if not np.isnan(auc):
            sri_aucs.append(auc)
    metrics_agg["mean_sri_auc_is_aliased"] = float(np.mean(sri_aucs)) if sri_aucs else 0.0

    # Analysis 5
    c3 = analysis5.get("criteria", {}).get("criterion_3_srwe", {})
    gap_red = c3.get("srwe_gap_reduction_pct", 0.0)
    metrics_agg["srwe_gap_reduction_pct"] = float(gap_red) if gap_red is not None and not np.isnan(gap_red) else 0.0

    verdict_map = {"confirmed": 4, "partially_confirmed": 3, "inconclusive": 2, "disconfirmed": 1}
    metrics_agg["verdict_score"] = verdict_map.get(analysis5.get("overall_verdict", "inconclusive"), 2)

    c1 = analysis5.get("criteria", {}).get("criterion_1_correlation", {})
    c2 = analysis5.get("criteria", {}).get("criterion_2_auc", {})
    metrics_agg["n_datasets_c1_model_free_pass"] = sum(1 for ds in c1 if c1[ds].get("model_free_pass", False))
    metrics_agg["n_datasets_c2_auc_pass"] = sum(1 for ds in c2 if c2[ds].get("pass", False))
    metrics_agg["c3_srwe_pass"] = 1 if c3.get("pass", False) else 0

    # Sanitize metrics_agg: all values must be numbers
    for k in list(metrics_agg.keys()):
        v = metrics_agg[k]
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            metrics_agg[k] = 0.0
        else:
            metrics_agg[k] = float(v)

    # ---- Per-example datasets ----
    output_datasets = []
    for ds_name in sorted(df["dataset"].unique()):
        ds_df = df[df["dataset"] == ds_name]
        examples = []
        for _, row in ds_df.iterrows():
            input_dict = {
                "dataset": ds_name,
                "graph_idx": int(row["graph_idx"]) if pd.notna(row.get("graph_idx")) else 0,
                "num_nodes": int(row["num_nodes"]) if pd.notna(row.get("num_nodes")) else 0,
                "sri_k20": float(row["sri_k20"]) if pd.notna(row.get("sri_k20")) else 0.0,
            }
            output_dict = {
                "sri_k20": float(row["sri_k20"]) if pd.notna(row.get("sri_k20")) else 0.0,
                "delta_min": float(row["delta_min"]) if pd.notna(row.get("delta_min")) else 0.0,
                "is_aliased": int(row["is_aliased"]) if pd.notna(row.get("is_aliased")) else 0,
            }
            ex = {
                "input": json.dumps(input_dict),
                "output": json.dumps(output_dict),
                "predict_resolution_diagnosis": str(row.get("resolution_diagnosis", "unknown")),
                "eval_sri_k20": float(row["sri_k20"]) if pd.notna(row.get("sri_k20")) else 0.0,
                "eval_is_aliased": float(row["is_aliased"]) if pd.notna(row.get("is_aliased")) else 0.0,
                "eval_srwe_rwse_l1": float(row["srwe_rwse_l1"]) if pd.notna(row.get("srwe_rwse_l1")) else 0.0,
                "eval_vander_cond_log10": float(row["vander_cond_log10"]) if pd.notna(row.get("vander_cond_log10")) else 0.0,
            }
            examples.append(ex)
        output_datasets.append({"dataset": ds_name, "examples": examples})

    # ---- Metadata ----
    metadata = {
        "evaluation_name": "Confound Decomposition & Formal Hypothesis Assessment of Walk Resolution Limit",
        "n_experiments": 3,
        "n_total_graphs": len(df),
        "overall_verdict": analysis5.get("overall_verdict", "inconclusive"),
        "analyses": {
            "analysis_1_discrepancy_resolution": sanitize_value(analysis1),
            "analysis_2_partial_correlations": sanitize_value(analysis2),
            "analysis_3_sparsity_stratified": sanitize_value(analysis3),
            "analysis_4_predictor_comparison": sanitize_value(analysis4),
            "analysis_5_formal_assessment": sanitize_value(analysis5),
        },
        "figures": [
            "figures/scatter_sri_vs_gap.png",
            "figures/forest_partial_correlations.png",
            "figures/predictor_comparison_auc.png",
            "figures/quintile_gap.png",
            "figures/verdict_summary.png",
        ],
    }

    return {
        "metadata": metadata,
        "metrics_agg": metrics_agg,
        "datasets": output_datasets,
    }


# ============================================================
# MAIN
# ============================================================

@logger.catch
def main():
    t0 = time.time()

    logger.info("=" * 60)
    logger.info("Walk Resolution Limit: Confound Decomposition & Formal Assessment")
    logger.info(f"MAX_EXAMPLES: {MAX_EXAMPLES}")
    logger.info("=" * 60)

    # Load all experiments
    exp1 = load_experiment(EXP1_DIR, "Exp1: SRI-Performance Gap")
    exp2 = load_experiment(EXP2_DIR, "Exp2: SRWE via MPM")
    exp3 = load_experiment(EXP3_DIR, "Exp3: Spectral Diagnostics")

    exp1_meta = exp1.get("metadata", {})
    exp2_meta = exp2.get("metadata", {})

    # Build merged per-graph DataFrame
    df = build_merged_df(exp1, exp2, exp3)
    t_load = time.time() - t0
    logger.info(f"Data loading and merge: {t_load:.1f}s")

    # Analysis 1: Discrepancy Resolution (uses exp1 metadata)
    analysis1 = analysis_1_discrepancy_resolution(exp1_meta)
    t1 = time.time() - t0
    logger.info(f"Analysis 1: {t1 - t_load:.1f}s")

    # Analysis 2: Partial Correlations (uses per-graph data + SRWE-RWSE proxy)
    analysis2 = analysis_2_partial_correlations(df)
    t2 = time.time() - t0
    logger.info(f"Analysis 2: {t2 - t1:.1f}s")

    # Analysis 3: Sparsity-Stratified (uses per-graph data)
    analysis3 = analysis_3_sparsity_stratified(df)
    t3 = time.time() - t0
    logger.info(f"Analysis 3: {t3 - t2:.1f}s")

    # Analysis 4: Predictor Comparison AUC (uses per-graph data)
    analysis4 = analysis_4_predictor_comparison(df)
    t4 = time.time() - t0
    logger.info(f"Analysis 4: {t4 - t3:.1f}s")

    # Analysis 5: Formal Assessment (synthesizes all)
    analysis5 = analysis_5_formal_assessment(exp1_meta, exp2_meta, analysis2, analysis4)
    t5 = time.time() - t0
    logger.info(f"Analysis 5: {t5 - t4:.1f}s")

    # Generate figures
    make_figures(df, analysis1, analysis2, analysis3, analysis4, analysis5)
    t_fig = time.time() - t0
    logger.info(f"Figures: {t_fig - t5:.1f}s")

    # Build and save output
    logger.info("Building output JSON...")
    output = build_output(df, analysis1, analysis2, analysis3, analysis4, analysis5)
    output_path = WORKSPACE / "eval_out.json"
    output_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Saved output to {output_path}")

    # Log key results
    total = time.time() - t0
    logger.info("=" * 60)
    logger.info("KEY RESULTS:")
    logger.info(f"  Overall Verdict: {analysis5.get('overall_verdict', 'N/A')}")
    ma = output["metrics_agg"]
    logger.info(f"  Mean Model-Free Spearman ρ: {ma.get('mean_model_free_spearman_rho', 'N/A')}")
    logger.info(f"  Mean Proxy ρ (SRI vs SRWE-RWSE): {ma.get('mean_raw_spearman_rho_srwe', 'N/A')}")
    logger.info(f"  Mean Proxy ρ (SRI vs Vandermonde): {ma.get('mean_raw_spearman_rho_vander', 'N/A')}")
    logger.info(f"  Mean SRI AUC (is_aliased): {ma.get('mean_sri_auc_is_aliased', 'N/A')}")
    logger.info(f"  SRWE Gap Reduction: {ma.get('srwe_gap_reduction_pct', 'N/A')}%")
    logger.info(f"  Total time: {total:.1f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
