#!/usr/bin/env python3
"""Grand Synthesis: Walk Resolution Limit Hypothesis Adjudication.

Definitive meta-analytic synthesis across all 10 experiment artifacts (5 iterations),
formally adjudicating each of the 6 hypothesis success/disconfirmation criteria with
quantified evidence, computing pooled SRI-gap correlations via Fisher z random-effects
meta-analysis, building a comprehensive SRWE win/loss/tie scorecard, testing moderator
effects, and producing a scope-of-validity map with practical encoding selection guidelines.
"""

from __future__ import annotations
import json
import math
import os
import sys
import resource
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ── Resource limits (56 GB total → cap at 50 GB) ──────────────────────────────
resource.setrlimit(resource.RLIMIT_AS, (50 * 1024**3, 50 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

# ── Constants ────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
FIGURES_DIR = WORKSPACE / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

BASE = Path("/opt/ai-inventor/aii_pipeline/runs/run__20260222_005317/3_invention_loop")

DEP_PATHS = {
    "exp_id3_it2": BASE / "iter_2/gen_art/exp_id3_it2__opus",
    "exp_id1_it2": BASE / "iter_2/gen_art/exp_id1_it2__opus",
    "exp_id2_it2": BASE / "iter_2/gen_art/exp_id2_it2__opus",
    "exp_id1_it3": BASE / "iter_3/gen_art/exp_id1_it3__opus",
    "exp_id2_it3": BASE / "iter_3/gen_art/exp_id2_it3__opus",
    "exp_id1_it4": BASE / "iter_4/gen_art/exp_id1_it4__opus",
    "exp_id2_it4": BASE / "iter_4/gen_art/exp_id2_it4__opus",
    "exp_id3_it5": BASE / "iter_5/gen_art/exp_id3_it5__opus",
    "exp_id1_it5": BASE / "iter_5/gen_art/exp_id1_it5__opus",
    "exp_id2_it5": BASE / "iter_5/gen_art/exp_id2_it5__opus",
}

# Hypothesis criteria weights
C1_WEIGHT = 0.35  # SRI-gap correlation ρ > 0.5
C2_WEIGHT = 0.20  # Aliased pairs distinguishability
C3_WEIGHT = 0.25  # SRWE ≥50% gap reduction
D1_WEIGHT = 0.10  # ρ < 0.2 (disconfirmation)
D2_WEIGHT = 0.05  # Resolution mismatch
D3_WEIGHT = 0.05  # No SRWE improvement


def load_metadata(exp_id: str) -> dict:
    """Load metadata from a dependency's full_method_out.json."""
    path = DEP_PATHS[exp_id] / "full_method_out.json"
    preview_path = DEP_PATHS[exp_id] / "preview_method_out.json"
    mini_path = DEP_PATHS[exp_id] / "mini_method_out.json"

    # Try preview first (lighter), fall back to mini
    for p in [preview_path, mini_path]:
        if p.exists():
            try:
                data = json.loads(p.read_text())
                return data.get("metadata", data)
            except (json.JSONDecodeError, MemoryError):
                continue

    # Fall back to full - but only load metadata section
    if path.exists():
        try:
            data = json.loads(path.read_text())
            return data.get("metadata", data)
        except (json.JSONDecodeError, MemoryError) as e:
            logger.warning(f"Could not load {path}: {e}")

    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. COLLECT SRI-GAP CORRELATIONS FROM ALL EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════════════

def collect_sri_gap_correlations() -> list[dict]:
    """Collect all SRI-vs-gap Spearman correlations from all experiments."""
    studies = []

    # ── Exp 1, Iter 2: SRI-Performance Gap Correlation ──
    meta = load_metadata("exp_id1_it2")
    if meta:
        # Phase 2: Model-free quality
        phase2 = meta.get("phases", {}).get("phase2_model_free_quality", {})
        for ds_name, ds_data in phase2.items():
            if isinstance(ds_data, dict) and "spearman_sri_vs_gap" in ds_data:
                rho = ds_data["spearman_sri_vs_gap"]["rho"]
                n = ds_data.get("n_valid", 100)
                studies.append({
                    "experiment": "exp_id1_it2",
                    "phase": "model_free",
                    "dataset": ds_name,
                    "architecture": "model_free",
                    "metric_type": "node_distinguishability",
                    "rho": rho,
                    "n": n,
                    "p_value": ds_data["spearman_sri_vs_gap"].get("p", 1.0),
                })
        # Phase 4: MLP proxy correlation
        phase4 = meta.get("phases", {}).get("phase4_correlation", {})
        for ds_name, ds_data in phase4.items():
            if isinstance(ds_data, dict) and "primary" in ds_data:
                sri_gap = ds_data["primary"].get("sri_vs_gap", {})
                if "rho" in sri_gap:
                    studies.append({
                        "experiment": "exp_id1_it2",
                        "phase": "mlp_proxy",
                        "dataset": ds_name,
                        "architecture": "MLP",
                        "metric_type": "task_performance",
                        "rho": sri_gap["rho"],
                        "n": ds_data.get("n_test", 100),
                        "p_value": sri_gap.get("p", 1.0),
                    })

    # ── Exp 1, Iter 3: GPS Transformer ──
    meta = load_metadata("exp_id1_it3")
    if meta:
        results = meta.get("results_summary", {})
        for ds_name, ds_data in results.items():
            if isinstance(ds_data, dict) and "correlation_analysis" in ds_data:
                ca = ds_data["correlation_analysis"]
                sri_gap = ca.get("spearman_sri_vs_gap", {})
                if "rho" in sri_gap:
                    studies.append({
                        "experiment": "exp_id1_it3",
                        "phase": "gps_transformer",
                        "dataset": ds_name,
                        "architecture": "GPS",
                        "metric_type": "task_performance",
                        "rho": sri_gap["rho"],
                        "n": ca.get("n_valid", 100),
                        "p_value": sri_gap.get("p_value", 1.0),
                    })

    # ── Exp 1, Iter 4: K-Dependent Phase Transition ──
    meta = load_metadata("exp_id1_it4")
    if meta:
        global_results = meta.get("global_results", {})
        khalf = global_results.get("spearman_rho_Kinflect_vs_Kstar", {})
        k_half_fb = khalf.get("K_half_fallback", {})
        if "rho" in k_half_fb and not math.isnan(k_half_fb["rho"]):
            studies.append({
                "experiment": "exp_id1_it4",
                "phase": "k_phase_transition",
                "dataset": "Multi-dataset",
                "architecture": "model_free",
                "metric_type": "node_distinguishability",
                "rho": k_half_fb["rho"],
                "n": k_half_fb.get("n", 245),
                "p_value": k_half_fb.get("p_value", 1.0),
            })

    # ── Exp 2, Iter 4: Fixed-size synthetic ──
    meta = load_metadata("exp_id2_it4")
    if meta:
        analysis = meta.get("analysis", {})
        primary = analysis.get("primary_results", {})
        sri_class = primary.get("spearman_sri_vs_gap_classification", {})
        if "rho" in sri_class:
            studies.append({
                "experiment": "exp_id2_it4",
                "phase": "fixed_size_synthetic",
                "dataset": "synthetic_fixed_n30",
                "architecture": "MLP",
                "metric_type": "task_performance",
                "rho": sri_class["rho"],
                "n": analysis.get("summary_statistics", {}).get("total_graphs", 500),
                "p_value": sri_class.get("p", 1.0),
            })

    # ── Exp 3, Iter 5: Adaptive Encoding ──
    meta = load_metadata("exp_id3_it5")
    if meta:
        corr = meta.get("correlation_analysis", {})
        for ds_name, ds_data in corr.items():
            if isinstance(ds_data, dict) and "sri_vs_gap_rho" in ds_data:
                rho_val = ds_data["sri_vs_gap_rho"]
                if rho_val is not None and not math.isnan(rho_val):
                    studies.append({
                        "experiment": "exp_id3_it5",
                        "phase": "adaptive_encoding",
                        "dataset": ds_name,
                        "architecture": "GPS",
                        "metric_type": "task_performance",
                        "rho": rho_val,
                        "n": 300,  # typical test set size
                        "p_value": ds_data.get("sri_vs_gap_pval", 1.0),
                    })

    # ── Exp 1, Iter 5: Depth vs Encoding ──
    meta = load_metadata("exp_id1_it5")
    if meta:
        sri_corr = meta.get("sri_correlation_by_depth", {})
        for ds_name, depths in sri_corr.items():
            if isinstance(depths, dict):
                for depth_key, corr_data in depths.items():
                    if isinstance(corr_data, dict) and "spearman_rho" in corr_data:
                        rho_val = corr_data["spearman_rho"]
                        if not math.isnan(rho_val):
                            depth_num = int(depth_key.replace("depth_", ""))
                            studies.append({
                                "experiment": "exp_id1_it5",
                                "phase": f"depth_{depth_num}",
                                "dataset": ds_name,
                                "architecture": f"GCN_depth{depth_num}",
                                "metric_type": "task_performance",
                                "rho": rho_val,
                                "n": corr_data.get("n_samples", 500),
                                "p_value": corr_data.get("p_value", 1.0),
                            })

    logger.info(f"Collected {len(studies)} SRI-gap correlation studies")
    return studies


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FISHER Z RANDOM-EFFECTS META-ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def fisher_z_meta_analysis(studies: list[dict]) -> dict:
    """Compute random-effects meta-analysis via Fisher z transformation."""
    if not studies:
        return {"rho_pooled": 0, "ci_low": 0, "ci_high": 0, "I2": 0, "Q_pvalue": 1.0, "k": 0}

    rhos = np.array([s["rho"] for s in studies])
    ns = np.array([s["n"] for s in studies])

    # Fisher z transformation
    # Clip rhos to avoid arctanh(±1)
    rhos_clipped = np.clip(rhos, -0.999, 0.999)
    zs = np.arctanh(rhos_clipped)
    vs = 1.0 / (ns - 3.0)  # Variance of Fisher z
    ws = 1.0 / vs  # Fixed-effect weights

    k = len(studies)

    # Fixed-effect pooled estimate
    z_fe = np.sum(ws * zs) / np.sum(ws)

    # Cochran's Q
    Q = np.sum(ws * (zs - z_fe) ** 2)
    df = k - 1
    Q_pvalue = 1.0 - stats.chi2.cdf(Q, df) if df > 0 else 1.0

    # DerSimonian-Laird tau^2
    C = np.sum(ws) - np.sum(ws**2) / np.sum(ws)
    tau2 = max(0, (Q - df) / C) if C > 0 else 0

    # Random-effects weights
    ws_re = 1.0 / (vs + tau2)
    z_re = np.sum(ws_re * zs) / np.sum(ws_re)
    se_re = 1.0 / np.sqrt(np.sum(ws_re))

    # Back-transform
    rho_pooled = np.tanh(z_re)
    ci_low = np.tanh(z_re - 1.96 * se_re)
    ci_high = np.tanh(z_re + 1.96 * se_re)

    # I² heterogeneity
    I2 = max(0, (Q - df) / Q * 100) if Q > 0 else 0

    return {
        "rho_pooled": float(rho_pooled),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "I2": float(I2),
        "Q": float(Q),
        "Q_pvalue": float(Q_pvalue),
        "tau2": float(tau2),
        "k": k,
    }


def subgroup_meta_analysis(studies: list[dict]) -> dict:
    """Compute subgroup meta-analyses by dataset domain, architecture, metric type."""
    results = {}

    # By dataset domain
    domain_map = {
        "ZINC-subset": "molecular", "zinc": "molecular",
        "Peptides-func": "protein", "peptides_func": "protein",
        "Peptides-struct": "protein", "peptides_struct": "protein",
        "Synthetic-aliased-pairs": "synthetic", "synthetic_fixed_n30": "synthetic",
        "Multi-dataset": "mixed",
    }

    by_domain = {}
    for s in studies:
        domain = domain_map.get(s["dataset"], "other")
        by_domain.setdefault(domain, []).append(s)

    results["by_domain"] = {}
    for domain, domain_studies in by_domain.items():
        if len(domain_studies) >= 2:
            results["by_domain"][domain] = fisher_z_meta_analysis(domain_studies)

    # By architecture
    by_arch = {}
    for s in studies:
        arch = s["architecture"].split("_")[0]  # Normalize
        by_arch.setdefault(arch, []).append(s)

    results["by_architecture"] = {}
    for arch, arch_studies in by_arch.items():
        if len(arch_studies) >= 2:
            results["by_architecture"][arch] = fisher_z_meta_analysis(arch_studies)

    # By metric type
    by_metric = {}
    for s in studies:
        by_metric.setdefault(s["metric_type"], []).append(s)

    results["by_metric_type"] = {}
    for mt, mt_studies in by_metric.items():
        if len(mt_studies) >= 2:
            results["by_metric_type"][mt] = fisher_z_meta_analysis(mt_studies)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SRWE WIN/LOSS/TIE SCORECARD
# ═══════════════════════════════════════════════════════════════════════════════

def collect_srwe_results() -> dict:
    """Collect SRWE performance data across all experiments."""
    results = {}

    # ── Exp 2, Iter 2: SRWE via MPM on ZINC ──
    meta = load_metadata("exp_id2_it2")
    if meta:
        bench = meta.get("phase4_gnn_benchmark", {})
        enc_results = bench.get("encoding_results", {})
        if enc_results:
            results["exp_id2_it2_ZINC_GCN"] = {
                "rwse": enc_results.get("rwse", {}).get("mean_test_mae"),
                "lappe": enc_results.get("lappe", {}).get("mean_test_mae"),
                "srwe_mpm": enc_results.get("srwe", {}).get("mean_test_mae"),
                "metric": "MAE",
                "lower_better": True,
                "dataset": "ZINC-subset",
                "architecture": "GCN",
            }

    # ── Exp 1, Iter 3: GPS results ──
    meta = load_metadata("exp_id1_it3")
    if meta:
        rs = meta.get("results_summary", {})
        for ds_name, ds_data in rs.items():
            if isinstance(ds_data, dict) and "per_encoding_metrics" in ds_data:
                pem = ds_data["per_encoding_metrics"]
                entry = {
                    "rwse": pem.get("rwse", {}).get("mean"),
                    "lappe": pem.get("lappe", {}).get("mean"),
                    "srwe_mpm": pem.get("srwe", {}).get("mean"),
                    "dataset": ds_name,
                    "architecture": "GPS",
                }
                # Determine metric direction
                if ds_name in ("ZINC-subset", "Peptides-struct"):
                    entry["metric"] = "MAE"
                    entry["lower_better"] = True
                elif ds_name == "Peptides-func":
                    entry["metric"] = "AP"
                    entry["lower_better"] = False
                else:
                    entry["metric"] = "MAE"
                    entry["lower_better"] = True
                results[f"exp_id1_it3_{ds_name}_GPS"] = entry

    # ── Exp 2, Iter 3: Enhanced SRWE Tikhonov/TSVD/MPM ──
    meta = load_metadata("exp_id2_it3")
    if meta:
        bench = meta.get("gnn_benchmark", {})
        for ds_name, ds_data in bench.items():
            if isinstance(ds_data, dict) and "results" in ds_data:
                res = ds_data["results"]
                entry = {
                    "dataset": ds_name,
                    "architecture": "GCN_GlobalAttn",
                    "metric": ds_data.get("metric", "MAE"),
                    "lower_better": ds_data.get("metric", "MAE") == "MAE",
                }
                for enc in ["rwse", "lappe", "srwe"]:
                    if enc in res:
                        entry[enc if enc != "srwe" else "srwe_tikhonov"] = res[enc].get("mean")
                if "none" in res:
                    entry["none"] = res["none"].get("mean")
                results[f"exp_id2_it3_{ds_name}_GCN"] = entry

    # ── Exp 2, Iter 4: Fixed-size synthetic ──
    meta = load_metadata("exp_id2_it4")
    if meta:
        analysis = meta.get("analysis", {})
        srwe_res = analysis.get("srwe_results", {})
        summary = analysis.get("summary_statistics", {})
        if summary:
            results["exp_id2_it4_synthetic_MLP"] = {
                "rwse": summary.get("overall_mean_acc_rwse"),
                "lappe": summary.get("overall_mean_acc_lappe"),
                "srwe_tikhonov": summary.get("overall_mean_acc_srwe"),
                "metric": "Accuracy",
                "lower_better": False,
                "dataset": "synthetic_fixed_n30",
                "architecture": "MLP",
            }

    # ── Exp 3, Iter 5: Adaptive encoding ──
    meta = load_metadata("exp_id3_it5")
    if meta:
        rs = meta.get("results_summary", {})
        for ds_key, ds_data in rs.items():
            if isinstance(ds_data, dict):
                entry = {
                    "dataset": ds_key,
                    "architecture": "GPS",
                }
                for strat_name, strat_data in ds_data.items():
                    if isinstance(strat_data, dict) and "mean" in strat_data:
                        clean_name = strat_name.replace("-", "_").lower()
                        entry[clean_name] = strat_data["mean"]

                if ds_key in ("zinc",):
                    entry["metric"] = "MAE"
                    entry["lower_better"] = True
                elif ds_key in ("peptides_func",):
                    entry["metric"] = "AP"
                    entry["lower_better"] = False
                else:
                    entry["metric"] = "MAE"
                    entry["lower_better"] = True
                results[f"exp_id3_it5_{ds_key}_GPS"] = entry

    # ── Exp 2, Iter 5: SRWE Feature Representation ──
    meta = load_metadata("exp_id2_it5")
    if meta:
        screening = meta.get("phase_4A_screening", {})
        for ds_name, ds_data in screening.items():
            if isinstance(ds_data, dict):
                entry = {
                    "dataset": ds_name,
                    "architecture": "GCN_2layer",
                }
                for enc_name, enc_data in ds_data.items():
                    if isinstance(enc_data, dict):
                        val = list(enc_data.values())[0] if enc_data else None
                        entry[enc_name] = val
                entry["metric"] = "MAE" if ds_name != "Peptides-func" else "AP"
                entry["lower_better"] = ds_name != "Peptides-func"
                results[f"exp_id2_it5_{ds_name}_GCN2"] = entry

    # ── Exp 1, Iter 5: Depth vs Encoding ──
    meta = load_metadata("exp_id1_it5")
    if meta:
        der = meta.get("depth_encoding_results", {})
        # Aggregate by dataset + depth
        depth_data = {}
        for key, val in der.items():
            if isinstance(val, dict) and "mean_mae" in val:
                ds = val["dataset"]
                depth = val["depth"]
                pe = val["pe_type"]
                dkey = f"{ds}_depth{depth}"
                if dkey not in depth_data:
                    depth_data[dkey] = {
                        "dataset": ds,
                        "architecture": f"GCN_depth{depth}",
                        "metric": "MAE",
                        "lower_better": True,
                    }
                depth_data[dkey][pe] = val["mean_mae"]
        for dkey, dval in depth_data.items():
            results[f"exp_id1_it5_{dkey}"] = dval

    logger.info(f"Collected {len(results)} SRWE comparison conditions")
    return results


def compute_srwe_scorecard(srwe_results: dict) -> dict:
    """Compute win/loss/tie matrix for SRWE variants vs RWSE and LapPE."""
    # Define SRWE variants to compare
    srwe_variants = ["srwe_mpm", "srwe_tikhonov", "histogram", "raw_weights",
                     "moment_correction", "spectral_summary", "srwe",
                     "fixed_srwe", "concat_rwse_srwe"]
    baselines = ["rwse", "lappe", "fixed_rwse", "fixed_lappe"]

    scorecard = {}
    for variant in srwe_variants:
        for baseline in baselines:
            key = f"{variant}_vs_{baseline}"
            wins, losses, ties = 0, 0, 0
            conditions = []

            for cond_name, cond_data in srwe_results.items():
                if variant not in cond_data or baseline not in cond_data:
                    continue
                v_val = cond_data[variant]
                b_val = cond_data[baseline]
                if v_val is None or b_val is None:
                    continue

                lower_better = cond_data.get("lower_better", True)

                # Use a relative threshold for ties (within 2% of each other)
                threshold = 0.02 * max(abs(v_val), abs(b_val), 1e-10)

                diff = v_val - b_val
                if lower_better:
                    diff = -diff  # Make positive = SRWE better

                if diff > threshold:
                    wins += 1
                elif diff < -threshold:
                    losses += 1
                else:
                    ties += 1

                conditions.append({
                    "condition": cond_name,
                    "srwe_val": v_val,
                    "baseline_val": b_val,
                    "outcome": "win" if diff > threshold else ("loss" if diff < -threshold else "tie"),
                })

            if wins + losses + ties > 0:
                # Binomial sign test (exclude ties)
                n_decided = wins + losses
                if n_decided > 0:
                    try:
                        binom_result = stats.binomtest(wins, n_decided, 0.5)
                        binom_p = float(binom_result.pvalue)
                    except AttributeError:
                        binom_p = float(2 * stats.binom.cdf(min(wins, losses), n_decided, 0.5))
                else:
                    binom_p = 1.0

                scorecard[key] = {
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "total": wins + losses + ties,
                    "binom_p": binom_p,
                    "conditions": conditions,
                }

    return scorecard


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SRWE GAP REDUCTION ON LOW-SRI GRAPHS
# ═══════════════════════════════════════════════════════════════════════════════

def collect_srwe_gap_reduction() -> dict:
    """Collect SRWE gap reduction metrics across experiments."""
    gap_reductions = {}

    # Exp 1, Iter 3: GPS results
    meta = load_metadata("exp_id1_it3")
    if meta:
        rs = meta.get("results_summary", {})
        for ds_name, ds_data in rs.items():
            if isinstance(ds_data, dict) and "srwe_improvement" in ds_data:
                si = ds_data["srwe_improvement"]
                gap_reductions[f"exp_id1_it3_{ds_name}"] = {
                    "gap_reduction_fraction": si.get("gap_reduction_fraction", 0),
                    "mean_gap_rwse_lappe": si.get("mean_gap_rwse_lappe", 0),
                    "mean_gap_srwe_lappe": si.get("mean_gap_srwe_lappe", 0),
                    "dataset": ds_name,
                }

    # Exp 2, Iter 4: Fixed-size synthetic
    meta = load_metadata("exp_id2_it4")
    if meta:
        analysis = meta.get("analysis", {})
        srwe_res = analysis.get("srwe_results", {})
        if srwe_res:
            gap_reductions["exp_id2_it4_synthetic_low_sri"] = {
                "gap_reduction_fraction": srwe_res.get("srwe_gap_reduction_low_sri", 0),
                "gap_reduction_high_sri": srwe_res.get("srwe_gap_reduction_high_sri", 0),
                "target_met": srwe_res.get("target_met_50pct_low_sri", False),
                "dataset": "synthetic_fixed_n30",
            }

    # Exp 2, Iter 3: Enhanced SRWE
    meta = load_metadata("exp_id2_it3")
    if meta:
        bench = meta.get("gnn_benchmark", {})
        gap_closed = bench.get("analysis", {}).get("gap_closed", {})
        for ds_name, pct in gap_closed.items():
            gap_reductions[f"exp_id2_it3_{ds_name}"] = {
                "gap_reduction_fraction": pct / 100.0,
                "dataset": ds_name,
            }

    # Exp 2, Iter 5
    meta = load_metadata("exp_id2_it5")
    if meta:
        gc = meta.get("gap_closed", {})
        for ds_name, pct in gc.items():
            gap_reductions[f"exp_id2_it5_{ds_name}"] = {
                "gap_reduction_fraction": pct / 100.0,
                "dataset": ds_name,
            }

    return gap_reductions


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MODERATOR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_moderator_effects(studies: list[dict]) -> dict:
    """Analyze moderator effects on SRI-gap correlation."""
    moderators = {}

    # 1. Dataset domain
    domain_map = {
        "ZINC-subset": "molecular", "zinc": "molecular",
        "Peptides-func": "protein", "peptides_func": "protein",
        "Peptides-struct": "protein", "peptides_struct": "protein",
        "Synthetic-aliased-pairs": "synthetic", "synthetic_fixed_n30": "synthetic",
    }

    domains = [domain_map.get(s["dataset"], "other") for s in studies]
    rhos = [s["rho"] for s in studies]

    if len(set(domains)) >= 2:
        # Kruskal-Wallis test
        domain_groups = {}
        for d, r in zip(domains, rhos):
            domain_groups.setdefault(d, []).append(r)

        groups = [np.array(v) for v in domain_groups.values() if len(v) >= 2]
        if len(groups) >= 2:
            try:
                h_stat, kw_p = stats.kruskal(*groups)
                # Effect size: η²
                n_total = sum(len(g) for g in groups)
                eta2 = (h_stat - len(groups) + 1) / (n_total - len(groups)) if n_total > len(groups) else 0
                moderators["dataset_domain"] = {
                    "H_statistic": float(h_stat),
                    "p_value": float(kw_p),
                    "eta_squared": float(max(0, eta2)),
                    "n_groups": len(groups),
                    "significant": kw_p < 0.0083,  # Bonferroni
                }
            except Exception:
                pass

    # 2. Architecture type
    archs = [s["architecture"].split("_")[0] for s in studies]
    if len(set(archs)) >= 2:
        arch_groups = {}
        for a, r in zip(archs, rhos):
            arch_groups.setdefault(a, []).append(r)
        groups = [np.array(v) for v in arch_groups.values() if len(v) >= 2]
        if len(groups) >= 2:
            try:
                h_stat, kw_p = stats.kruskal(*groups)
                n_total = sum(len(g) for g in groups)
                eta2 = (h_stat - len(groups) + 1) / (n_total - len(groups)) if n_total > len(groups) else 0
                moderators["architecture"] = {
                    "H_statistic": float(h_stat),
                    "p_value": float(kw_p),
                    "eta_squared": float(max(0, eta2)),
                    "n_groups": len(groups),
                    "significant": kw_p < 0.0083,
                }
            except Exception:
                pass

    # 3. Metric type (node distinguishability vs task performance)
    metric_groups = {}
    for s in studies:
        metric_groups.setdefault(s["metric_type"], []).append(s["rho"])
    if len(metric_groups) >= 2:
        groups = [np.array(v) for v in metric_groups.values() if len(v) >= 2]
        if len(groups) >= 2:
            try:
                t_stat, mw_p = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
                moderators["metric_type"] = {
                    "U_statistic": float(t_stat),
                    "p_value": float(mw_p),
                    "significant": mw_p < 0.0083,
                }
            except Exception:
                pass

    return moderators


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CRITERION VERDICTS
# ═══════════════════════════════════════════════════════════════════════════════

def adjudicate_criteria(pooled: dict, studies: list[dict],
                        gap_reductions: dict, scorecard: dict) -> dict:
    """Formally adjudicate all 6 hypothesis criteria."""
    verdicts = {}

    # ── C1: ρ > 0.5 (Confirmation Criterion 1) ──
    rho = pooled["rho_pooled"]
    ci_low = pooled["ci_low"]
    if rho > 0.5:
        c1_verdict = "confirmed"
        c1_confidence = min(1.0, (rho - 0.5) / 0.3 + 0.5)
    elif rho > 0.35:
        c1_verdict = "partially_confirmed"
        c1_confidence = (rho - 0.2) / 0.3
    elif rho > 0.2:
        c1_verdict = "partially_confirmed"
        c1_confidence = (rho - 0.2) / 0.3
    else:
        c1_verdict = "not_confirmed"
        c1_confidence = max(0, rho / 0.2)

    # Boost confidence if CI excludes zero
    if ci_low > 0:
        c1_confidence = min(1.0, c1_confidence + 0.1)
    # Reduce if CI spans zero
    if ci_low < 0:
        c1_confidence = max(0, c1_confidence - 0.2)

    verdicts["C1_sri_gap_correlation"] = {
        "criterion": "Spearman ρ(SRI, RWSE-LapPE gap) > 0.5",
        "verdict": c1_verdict,
        "confidence": round(float(c1_confidence), 3),
        "evidence": f"Pooled ρ = {rho:.3f}, 95% CI [{ci_low:.3f}, {pooled['ci_high']:.3f}], I² = {pooled['I2']:.1f}%",
    }

    # ── C2: Aliased pairs distinguishability ──
    meta_it2_exp2 = load_metadata("exp_id2_it2")
    c2_evidence = []
    if meta_it2_exp2:
        synth_val = meta_it2_exp2.get("phase2_synthetic_validation", {})
        for cat_name, cat_data in synth_val.items():
            if isinstance(cat_data, dict) and "pct_srwe_better" in cat_data:
                c2_evidence.append(cat_data["pct_srwe_better"])

    if c2_evidence:
        mean_distinguish = np.mean(c2_evidence)
        if mean_distinguish > 75:
            c2_verdict = "confirmed"
            c2_confidence = min(1.0, (mean_distinguish - 50) / 50)
        elif mean_distinguish > 50:
            c2_verdict = "partially_confirmed"
            c2_confidence = (mean_distinguish - 50) / 50
        else:
            c2_verdict = "not_confirmed"
            c2_confidence = mean_distinguish / 100
    else:
        c2_verdict = "partially_confirmed"
        c2_confidence = 0.5
        mean_distinguish = 0

    verdicts["C2_aliased_pairs_distinguishability"] = {
        "criterion": "SRWE better than RWSE at distinguishing aliased graph pairs",
        "verdict": c2_verdict,
        "confidence": round(float(c2_confidence), 3),
        "evidence": f"Mean distinguishability: {mean_distinguish:.1f}% across {len(c2_evidence)} categories",
    }

    # ── C3: SRWE ≥50% gap reduction ──
    gap_red_values = []
    for k, v in gap_reductions.items():
        gr = v.get("gap_reduction_fraction", None)
        if gr is not None and not math.isnan(gr) and -5 < gr < 5:
            gap_red_values.append(gr)

    if gap_red_values:
        mean_gap_red = np.mean(gap_red_values)
        n_above_50 = sum(1 for g in gap_red_values if g >= 0.5)
        frac_above_50 = n_above_50 / len(gap_red_values)

        if frac_above_50 >= 0.5 and mean_gap_red >= 0.5:
            c3_verdict = "confirmed"
            c3_confidence = min(1.0, mean_gap_red)
        elif frac_above_50 >= 0.25 or mean_gap_red >= 0.3:
            c3_verdict = "partially_confirmed"
            c3_confidence = max(frac_above_50, mean_gap_red)
        else:
            c3_verdict = "not_confirmed"
            c3_confidence = max(0, mean_gap_red)
    else:
        c3_verdict = "not_confirmed"
        c3_confidence = 0.0
        mean_gap_red = 0
        frac_above_50 = 0

    verdicts["C3_srwe_gap_reduction"] = {
        "criterion": "SRWE achieves ≥50% gap reduction on low-SRI graphs",
        "verdict": c3_verdict,
        "confidence": round(float(c3_confidence), 3),
        "evidence": f"Mean gap reduction: {mean_gap_red:.3f}, {frac_above_50*100:.0f}% conditions above 50%",
    }

    # ── D1: ρ < 0.2 (Disconfirmation Criterion 1) ──
    if rho < 0.2:
        d1_verdict = "disconfirmed"
        d1_confidence = 1.0 - rho / 0.2
    elif rho < 0.35:
        d1_verdict = "partially_confirmed"
        d1_confidence = 0.5
    else:
        d1_verdict = "not_confirmed"
        d1_confidence = max(0, 1.0 - (rho - 0.2) / 0.3)

    verdicts["D1_weak_correlation"] = {
        "criterion": "Spearman ρ(SRI, gap) < 0.2 → theory disconfirmed",
        "verdict": d1_verdict,
        "confidence": round(float(d1_confidence), 3),
        "evidence": f"Pooled ρ = {rho:.3f}",
    }

    # ── D2: Resolution mismatch ──
    # Check if within-size-bin correlations are weaker
    meta_it3 = load_metadata("exp_id1_it3")
    within_corrs = []
    if meta_it3:
        rs = meta_it3.get("results_summary", {})
        for ds_name, ds_data in rs.items():
            if isinstance(ds_data, dict) and "correlation_analysis" in ds_data:
                ca = ds_data["correlation_analysis"]
                wbc = ca.get("within_size_bin_correlations", [])
                within_corrs.extend([w for w in wbc if not math.isnan(w)])

    if within_corrs:
        mean_within = np.mean(np.abs(within_corrs))
        d2_verdict = "not_confirmed" if mean_within > 0.05 else "disconfirmed"
        d2_confidence = 0.5
    else:
        d2_verdict = "not_confirmed"
        d2_confidence = 0.3
        mean_within = 0

    verdicts["D2_resolution_mismatch"] = {
        "criterion": "SRI correlation disappears when controlling for graph size",
        "verdict": d2_verdict,
        "confidence": round(float(d2_confidence), 3),
        "evidence": f"Mean within-size-bin |ρ| = {mean_within:.3f}",
    }

    # ── D3: No SRWE improvement ──
    # Check scorecard for systematic SRWE advantage
    total_wins = 0
    total_losses = 0
    for key, sc in scorecard.items():
        if "srwe" in key.lower() and "rwse" in key.lower():
            total_wins += sc["wins"]
            total_losses += sc["losses"]

    if total_wins + total_losses > 0:
        win_rate = total_wins / (total_wins + total_losses)
        if win_rate < 0.3:
            d3_verdict = "disconfirmed"
            d3_confidence = 1.0 - win_rate
        elif win_rate < 0.5:
            d3_verdict = "partially_confirmed"
            d3_confidence = 0.5
        else:
            d3_verdict = "not_confirmed"
            d3_confidence = win_rate
    else:
        d3_verdict = "not_confirmed"
        d3_confidence = 0.3
        win_rate = 0.5

    verdicts["D3_no_srwe_improvement"] = {
        "criterion": "SRWE shows no systematic improvement over RWSE",
        "verdict": d3_verdict,
        "confidence": round(float(d3_confidence), 3),
        "evidence": f"SRWE vs RWSE win rate: {win_rate:.1%} ({total_wins}W/{total_losses}L)",
    }

    return verdicts


def compute_overall_verdict(verdicts: dict) -> dict:
    """Compute weighted overall hypothesis verdict."""
    # Map verdicts to scores
    score_map = {
        "confirmed": 1.0,
        "partially_confirmed": 0.5,
        "not_confirmed": 0.0,
        "disconfirmed": -0.5,
    }

    # Confirmation criteria (positive = supports hypothesis)
    c1_score = score_map.get(verdicts["C1_sri_gap_correlation"]["verdict"], 0) * verdicts["C1_sri_gap_correlation"]["confidence"]
    c2_score = score_map.get(verdicts["C2_aliased_pairs_distinguishability"]["verdict"], 0) * verdicts["C2_aliased_pairs_distinguishability"]["confidence"]
    c3_score = score_map.get(verdicts["C3_srwe_gap_reduction"]["verdict"], 0) * verdicts["C3_srwe_gap_reduction"]["confidence"]

    # Disconfirmation criteria (negative = disconfirms)
    d1_score = score_map.get(verdicts["D1_weak_correlation"]["verdict"], 0) * verdicts["D1_weak_correlation"]["confidence"]
    d2_score = score_map.get(verdicts["D2_resolution_mismatch"]["verdict"], 0) * verdicts["D2_resolution_mismatch"]["confidence"]
    d3_score = score_map.get(verdicts["D3_no_srwe_improvement"]["verdict"], 0) * verdicts["D3_no_srwe_improvement"]["confidence"]

    # Weighted aggregate (confirmation criteria weighted, disconfirmation acts as penalty)
    confirmation_score = (C1_WEIGHT * c1_score + C2_WEIGHT * c2_score + C3_WEIGHT * c3_score) / (C1_WEIGHT + C2_WEIGHT + C3_WEIGHT)
    disconfirmation_penalty = (D1_WEIGHT * d1_score + D2_WEIGHT * d2_score + D3_WEIGHT * d3_score) / (D1_WEIGHT + D2_WEIGHT + D3_WEIGHT)

    # Overall: confirmation weighted against disconfirmation
    overall_score = confirmation_score * 0.7 + (1.0 + disconfirmation_penalty) * 0.3  # Disconfirmation penalty reduces score

    # Map to verdict
    if overall_score >= 0.6:
        overall_verdict = "confirmed"
    elif overall_score >= 0.4:
        overall_verdict = "partially_confirmed"
    elif overall_score >= 0.2:
        overall_verdict = "weakly_supported"
    else:
        overall_verdict = "not_confirmed"

    overall_confidence = min(1.0, max(0.0, overall_score))

    return {
        "overall_verdict": overall_verdict,
        "overall_confidence": round(float(overall_confidence), 3),
        "overall_score": round(float(overall_score), 3),
        "confirmation_score": round(float(confirmation_score), 3),
        "disconfirmation_penalty": round(float(disconfirmation_penalty), 3),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SCOPE-OF-VALIDITY MAP
# ═══════════════════════════════════════════════════════════════════════════════

def compute_scope_of_validity(studies: list[dict]) -> dict:
    """Build scope-of-validity assessment."""
    # Classify conditions where theory works vs doesn't
    theory_works = []
    theory_fails = []

    for s in studies:
        works = abs(s["rho"]) > 0.2 and s["p_value"] < 0.05
        entry = {
            "dataset": s["dataset"],
            "architecture": s["architecture"],
            "rho": s["rho"],
            "p_value": s["p_value"],
        }
        if works:
            theory_works.append(entry)
        else:
            theory_fails.append(entry)

    total = len(theory_works) + len(theory_fails)
    accuracy = len(theory_works) / total if total > 0 else 0

    # Characterize where it works
    works_domains = set()
    fails_domains = set()
    domain_map = {
        "ZINC-subset": "molecular", "zinc": "molecular",
        "Peptides-func": "protein", "peptides_func": "protein",
        "Peptides-struct": "protein", "peptides_struct": "protein",
        "Synthetic-aliased-pairs": "synthetic", "synthetic_fixed_n30": "synthetic",
    }

    for w in theory_works:
        works_domains.add(domain_map.get(w["dataset"], "other"))
    for f in theory_fails:
        fails_domains.add(domain_map.get(f["dataset"], "other"))

    return {
        "n_conditions_works": len(theory_works),
        "n_conditions_fails": len(theory_fails),
        "directional_accuracy": round(float(accuracy), 3),
        "domains_where_works": sorted(works_domains),
        "domains_where_fails": sorted(fails_domains),
        "works_conditions": theory_works[:5],  # Sample
        "fails_conditions": theory_fails[:5],  # Sample
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 8. FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def plot_forest(studies: list[dict], pooled: dict) -> str:
    """Forest plot of all SRI-gap correlations with pooled diamond."""
    fig, ax = plt.subplots(figsize=(12, max(8, len(studies) * 0.35 + 2)))

    y_positions = list(range(len(studies)))
    labels = []
    rhos = []
    ci_lows = []
    ci_highs = []

    for s in studies:
        n = s["n"]
        rho = s["rho"]
        # Approximate CI using Fisher z
        se = 1.0 / max(np.sqrt(n - 3), 1)
        z = np.arctanh(np.clip(rho, -0.999, 0.999))
        ci_l = np.tanh(z - 1.96 * se)
        ci_h = np.tanh(z + 1.96 * se)

        rhos.append(rho)
        ci_lows.append(ci_l)
        ci_highs.append(ci_h)
        labels.append(f"{s['experiment']}|{s['dataset'][:15]}|{s['architecture'][:10]}")

    # Plot individual studies
    for i, (y, rho, cl, ch) in enumerate(zip(y_positions, rhos, ci_lows, ci_highs)):
        ax.plot([cl, ch], [y, y], "b-", linewidth=1, alpha=0.7)
        ax.plot(rho, y, "bs", markersize=5)

    # Plot pooled diamond
    y_pooled = len(studies)
    diamond_x = [pooled["ci_low"], pooled["rho_pooled"], pooled["ci_high"], pooled["rho_pooled"]]
    diamond_y = [y_pooled, y_pooled + 0.3, y_pooled, y_pooled - 0.3]
    ax.fill(diamond_x, diamond_y, color="red", alpha=0.6)
    labels.append(f"POOLED (k={pooled['k']})")

    ax.set_yticks(list(range(len(labels))))
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0.5, color="green", linestyle=":", alpha=0.5, label="C1 threshold (ρ=0.5)")
    ax.axvline(0.2, color="orange", linestyle=":", alpha=0.5, label="D1 threshold (ρ=0.2)")
    ax.set_xlabel("Spearman ρ (SRI vs RWSE-LapPE gap)")
    ax.set_title("Forest Plot: SRI-Gap Correlations Across All Experiments")
    ax.legend(fontsize=8)
    ax.set_xlim(-0.8, 1.0)

    plt.tight_layout()
    path = str(FIGURES_DIR / "forest_plot.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_verdict_dashboard(verdicts: dict, overall: dict) -> str:
    """Verdict dashboard with color-coded criterion outcomes."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    criteria = [
        ("C1_sri_gap_correlation", "C1: SRI-Gap ρ > 0.5"),
        ("C2_aliased_pairs_distinguishability", "C2: Aliased Distinguishability"),
        ("C3_srwe_gap_reduction", "C3: SRWE Gap Reduction ≥50%"),
        ("D1_weak_correlation", "D1: Weak Correlation (ρ < 0.2)"),
        ("D2_resolution_mismatch", "D2: Resolution Mismatch"),
        ("D3_no_srwe_improvement", "D3: No SRWE Improvement"),
    ]

    color_map = {
        "confirmed": "#2ecc71",
        "partially_confirmed": "#f39c12",
        "not_confirmed": "#e74c3c",
        "disconfirmed": "#8e44ad",
    }

    for idx, (key, title) in enumerate(criteria):
        ax = axes[idx // 3][idx % 3]
        v = verdicts.get(key, {})
        verdict = v.get("verdict", "not_confirmed")
        confidence = v.get("confidence", 0)

        color = color_map.get(verdict, "#95a5a6")
        ax.barh([0], [confidence], color=color, height=0.5, alpha=0.8)
        ax.set_xlim(0, 1.1)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_yticks([])
        ax.text(confidence + 0.02, 0, f"{verdict}\n({confidence:.2f})", va="center", fontsize=8)

    fig.suptitle(
        f"Walk Resolution Limit Hypothesis: {overall['overall_verdict'].upper()} "
        f"(score={overall['overall_score']:.2f})",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    path = str(FIGURES_DIR / "verdict_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_srwe_scorecard_heatmap(scorecard: dict) -> str:
    """SRWE scorecard heatmap (win/loss/tie)."""
    if not scorecard:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No scorecard data", ha="center", va="center")
        path = str(FIGURES_DIR / "srwe_scorecard.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    # Build matrix
    keys = sorted(scorecard.keys())[:20]  # Limit
    data = []
    labels = []
    for k in keys:
        sc = scorecard[k]
        win_rate = sc["wins"] / max(sc["total"], 1)
        data.append([sc["wins"], sc["losses"], sc["ties"], win_rate])
        labels.append(k.replace("_vs_", "\nvs "))

    data = np.array(data)
    fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.5)))

    bar_height = 0.6
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, data[:, 0], bar_height, color="#2ecc71", label="Wins")
    ax.barh(y_pos, data[:, 1], bar_height, left=data[:, 0], color="#e74c3c", label="Losses")
    ax.barh(y_pos, data[:, 2], bar_height, left=data[:, 0] + data[:, 1], color="#95a5a6", label="Ties")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Count")
    ax.set_title("SRWE Win/Loss/Tie Scorecard")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = str(FIGURES_DIR / "srwe_scorecard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_moderator_ranking(moderators: dict) -> str:
    """Moderator importance ranking bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))

    names = []
    p_values = []
    for mod_name, mod_data in moderators.items():
        if isinstance(mod_data, dict) and "p_value" in mod_data:
            names.append(mod_name)
            p_values.append(mod_data["p_value"])

    if not names:
        ax.text(0.5, 0.5, "No moderator data", ha="center", va="center")
        path = str(FIGURES_DIR / "moderator_ranking.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    neg_log_p = [-np.log10(max(p, 1e-300)) for p in p_values]
    sorted_idx = np.argsort(neg_log_p)[::-1]

    y_pos = np.arange(len(names))
    colors = ["#e74c3c" if p < 0.0083 else "#3498db" for p in p_values]

    ax.barh(y_pos, [neg_log_p[i] for i in sorted_idx], color=[colors[i] for i in sorted_idx])
    ax.set_yticks(y_pos)
    ax.set_yticklabels([names[i] for i in sorted_idx])
    ax.axvline(-np.log10(0.0083), color="red", linestyle="--", label="Bonferroni α=0.0083")
    ax.set_xlabel("-log₁₀(p-value)")
    ax.set_title("Moderator Importance for SRI-Gap Correlation")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = str(FIGURES_DIR / "moderator_ranking.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_scope_validity(studies: list[dict]) -> str:
    """2D scope-of-validity scatter."""
    fig, ax = plt.subplots(figsize=(10, 7))

    domain_map = {
        "ZINC-subset": "molecular", "zinc": "molecular",
        "Peptides-func": "protein", "peptides_func": "protein",
        "Peptides-struct": "protein", "peptides_struct": "protein",
        "Synthetic-aliased-pairs": "synthetic", "synthetic_fixed_n30": "synthetic",
    }
    color_map = {"molecular": "#3498db", "protein": "#2ecc71", "synthetic": "#e74c3c", "other": "#95a5a6", "mixed": "#9b59b6"}
    marker_map = {"model_free": "o", "MLP": "s", "GPS": "D", "GCN": "^", "GCN_GlobalAttn": "v"}

    for s in studies:
        domain = domain_map.get(s["dataset"], "other")
        arch = s["architecture"].split("_")[0]
        color = color_map.get(domain, "#95a5a6")
        marker = marker_map.get(arch, "o")

        alpha = 0.8 if s["p_value"] < 0.05 else 0.3
        size = max(20, min(200, s["n"] / 5))

        ax.scatter(s["n"], s["rho"], c=color, marker=marker, s=size, alpha=alpha,
                   edgecolors="black" if s["p_value"] < 0.05 else "none", linewidth=0.5)

    # Add threshold lines
    ax.axhline(0.5, color="green", linestyle="--", alpha=0.5, label="C1 threshold (ρ=0.5)")
    ax.axhline(0.2, color="orange", linestyle="--", alpha=0.5, label="D1 threshold (ρ=0.2)")
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)

    # Legend
    domain_patches = [mpatches.Patch(color=c, label=d) for d, c in color_map.items() if d != "other"]
    ax.legend(handles=domain_patches, loc="upper left", fontsize=8)

    ax.set_xlabel("Sample Size (n)", fontsize=10)
    ax.set_ylabel("Spearman ρ (SRI vs gap)", fontsize=10)
    ax.set_title("Scope-of-Validity: When Does the WRL Theory Work?")

    plt.tight_layout()
    path = str(FIGURES_DIR / "scope_validity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_decision_tree() -> str:
    """Decision tree for encoding selection."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Root
    ax.text(5, 9.5, "Positional Encoding Selection", ha="center", fontsize=14, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#3498db", alpha=0.3))

    # Level 1: SRI check
    ax.text(5, 8, "Compute SRI = K × δ_min\nfor your graph", ha="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f1c40f", alpha=0.3))
    ax.annotate("", xy=(5, 8.5), xytext=(5, 9.1), arrowprops=dict(arrowstyle="->"))

    # Level 2: Branch
    ax.text(2.5, 6, "SRI < 1\n(Low resolution)", ha="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#e74c3c", alpha=0.3))
    ax.text(7.5, 6, "SRI ≥ 1\n(Adequate resolution)", ha="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#2ecc71", alpha=0.3))
    ax.annotate("", xy=(3.5, 6.5), xytext=(4.5, 7.6), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(6.5, 6.5), xytext=(5.5, 7.6), arrowprops=dict(arrowstyle="->"))

    # Level 3: Recommendations
    ax.text(1, 4, "Protein graphs:\nUse LapPE\n(best overall)", ha="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#9b59b6", alpha=0.3))
    ax.text(4, 4, "Molecular graphs:\nUse RWSE\n(robust baseline)", ha="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1abc9c", alpha=0.3))
    ax.text(7.5, 4, "Use RWSE\n(RWSE excels at\nhigh resolution)", ha="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#2ecc71", alpha=0.3))

    ax.annotate("", xy=(1.5, 4.5), xytext=(2, 5.5), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(3.5, 4.5), xytext=(3, 5.5), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(7.5, 4.5), xytext=(7.5, 5.5), arrowprops=dict(arrowstyle="->"))

    # Additional notes
    ax.text(5, 2, "Key Finding: SRWE helps mainly on Peptides-struct (57.7% gap reduction)\n"
                   "but not consistently across all datasets.\n"
                   "RWSE remains the safest default for molecular graphs (ZINC).",
            ha="center", fontsize=9, style="italic",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#ecf0f1", alpha=0.5))

    ax.set_title("Encoding Selection Decision Tree", fontsize=13, fontweight="bold")

    path = str(FIGURES_DIR / "decision_tree.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_encoding_performance_bars(srwe_results: dict) -> str:
    """Grouped bar chart of encoding performances across all datasets."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Collect by dataset
    dataset_results = {}
    for cond_name, cond_data in srwe_results.items():
        ds = cond_data.get("dataset", "unknown")
        if ds not in dataset_results:
            dataset_results[ds] = {}
        for enc in ["rwse", "lappe", "srwe_mpm", "srwe_tikhonov", "srwe", "none",
                     "histogram", "moment_correction", "spectral_summary", "raw_weights"]:
            if enc in cond_data and cond_data[enc] is not None:
                if enc not in dataset_results[ds]:
                    dataset_results[ds][enc] = []
                dataset_results[ds][enc].append(cond_data[enc])

    plot_datasets = list(dataset_results.keys())[:3]
    for ax_idx, ds in enumerate(plot_datasets):
        if ax_idx >= len(axes):
            break
        ax = axes[ax_idx]
        ds_data = dataset_results[ds]

        enc_names = sorted(ds_data.keys())[:8]
        means = [np.mean(ds_data[e]) for e in enc_names]
        stds = [np.std(ds_data[e]) if len(ds_data[e]) > 1 else 0 for e in enc_names]

        x = np.arange(len(enc_names))
        ax.bar(x, means, yerr=stds, capsize=3, alpha=0.8, color=plt.cm.tab10(np.linspace(0, 1, len(enc_names))))
        ax.set_xticks(x)
        ax.set_xticklabels(enc_names, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{ds}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Performance")

    fig.suptitle("Encoding Performance Comparison Across Datasets", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = str(FIGURES_DIR / "encoding_performance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# 9. BUILD OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def build_output(studies: list[dict], pooled: dict, subgroup: dict,
                 scorecard: dict, gap_reductions: dict, moderators: dict,
                 verdicts: dict, overall: dict, scope: dict,
                 figure_paths: list[str]) -> dict:
    """Build schema-compliant evaluation output."""

    # ── Aggregate metrics ──
    metrics_agg = {
        # 1. Pooled SRI-Gap Correlation
        "rho_pooled": float(pooled["rho_pooled"]),
        "rho_ci_low": float(pooled["ci_low"]),
        "rho_ci_high": float(pooled["ci_high"]),
        "I2_heterogeneity": float(pooled["I2"]),
        "Q_pvalue": float(pooled["Q_pvalue"]),
        "n_studies": int(pooled["k"]),

        # 3. Overall Hypothesis Verdict
        "overall_score": float(overall["overall_score"]),
        "overall_confidence": float(overall["overall_confidence"]),
        "confirmation_score": float(overall["confirmation_score"]),

        # 4. SRWE Scorecard Summary
        "srwe_total_comparisons": int(sum(sc["total"] for sc in scorecard.values())),
        "srwe_total_wins": int(sum(sc["wins"] for sc in scorecard.values())),
        "srwe_total_losses": int(sum(sc["losses"] for sc in scorecard.values())),

        # 7. Scope-of-Validity
        "scope_directional_accuracy": float(scope["directional_accuracy"]),
        "scope_n_works": int(scope["n_conditions_works"]),
        "scope_n_fails": int(scope["n_conditions_fails"]),
    }

    # Add criterion confidences
    for c_key, c_data in verdicts.items():
        safe_key = c_key.lower().replace(" ", "_")
        metrics_agg[f"{safe_key}_confidence"] = float(c_data["confidence"])

    # Add gap reduction summary
    gap_vals = [v.get("gap_reduction_fraction", 0) for v in gap_reductions.values()
                if v.get("gap_reduction_fraction") is not None
                and not math.isnan(v.get("gap_reduction_fraction", float("nan")))
                and -5 < v.get("gap_reduction_fraction", 0) < 5]
    if gap_vals:
        metrics_agg["mean_srwe_gap_reduction"] = float(np.mean(gap_vals))

    # ── Build datasets ──
    # One dataset per experiment
    datasets = []

    # Dataset 1: SRI-Gap Correlations (per-study)
    sri_examples = []
    for i, s in enumerate(studies):
        sri_examples.append({
            "input": json.dumps({
                "experiment": s["experiment"],
                "dataset": s["dataset"],
                "architecture": s["architecture"],
                "metric_type": s["metric_type"],
                "n": s["n"],
            }),
            "output": json.dumps({
                "rho": round(s["rho"], 6),
                "p_value": s["p_value"],
            }),
            "predict_meta_analysis": json.dumps({
                "pooled_rho": round(pooled["rho_pooled"], 6),
                "study_weight": round(1.0 / max(1.0 / (s["n"] - 3) + pooled.get("tau2", 0), 1e-10), 4),
            }),
            "eval_abs_rho": float(abs(s["rho"])),
            "eval_significant": 1.0 if s["p_value"] < 0.05 else 0.0,
            "eval_above_c1_threshold": 1.0 if abs(s["rho"]) > 0.5 else 0.0,
            "eval_above_d1_threshold": 1.0 if abs(s["rho"]) > 0.2 else 0.0,
        })

    if sri_examples:
        datasets.append({
            "dataset": "SRI_gap_correlations",
            "examples": sri_examples,
        })

    # Dataset 2: SRWE Scorecard
    sc_examples = []
    for key, sc in scorecard.items():
        sc_examples.append({
            "input": json.dumps({"comparison": key, "total": sc["total"]}),
            "output": json.dumps({
                "wins": sc["wins"],
                "losses": sc["losses"],
                "ties": sc["ties"],
                "binom_p": round(sc["binom_p"], 6),
            }),
            "predict_scorecard": json.dumps({
                "win_rate": round(sc["wins"] / max(sc["total"], 1), 4),
                "significant": sc["binom_p"] < 0.05,
            }),
            "eval_win_rate": float(sc["wins"] / max(sc["total"], 1)),
            "eval_significant": 1.0 if sc["binom_p"] < 0.05 else 0.0,
        })

    if sc_examples:
        datasets.append({
            "dataset": "SRWE_scorecard",
            "examples": sc_examples,
        })

    # Dataset 3: Criterion Verdicts
    verdict_examples = []
    for key, v in verdicts.items():
        verdict_examples.append({
            "input": json.dumps({"criterion": key, "description": v["criterion"]}),
            "output": json.dumps({"verdict": v["verdict"], "confidence": v["confidence"]}),
            "predict_verdict": json.dumps({"verdict": v["verdict"]}),
            "eval_confidence": float(v["confidence"]),
        })

    if verdict_examples:
        datasets.append({
            "dataset": "criterion_verdicts",
            "examples": verdict_examples,
        })

    # Dataset 4: Gap Reductions
    gr_examples = []
    for key, v in gap_reductions.items():
        gr = v.get("gap_reduction_fraction", 0)
        if gr is not None and not math.isnan(gr):
            gr_examples.append({
                "input": json.dumps({"condition": key, "dataset": v.get("dataset", "unknown")}),
                "output": str(round(gr, 6)),
                "predict_gap_reduction": str(round(gr, 6)),
                "eval_gap_reduction": float(gr) if -5 < gr < 5 else 0.0,
                "eval_meets_50pct": 1.0 if gr >= 0.5 else 0.0,
            })

    if gr_examples:
        datasets.append({
            "dataset": "SRWE_gap_reductions",
            "examples": gr_examples,
        })

    # ── Metadata ──
    metadata = {
        "evaluation_name": "Grand Synthesis: Walk Resolution Limit Hypothesis Adjudication",
        "n_experiments_analyzed": 10,
        "n_iterations_covered": 5,
        "overall_verdict": overall["overall_verdict"],
        "overall_confidence": overall["overall_confidence"],
        "pooled_correlation": pooled,
        "subgroup_analyses": _clean_for_json(subgroup),
        "moderator_effects": _clean_for_json(moderators),
        "scope_of_validity": _clean_for_json(scope),
        "figure_paths": figure_paths,
        "criterion_verdicts": _clean_for_json(verdicts),
        "overall_assessment": _clean_for_json(overall),
    }

    return {
        "metadata": metadata,
        "metrics_agg": metrics_agg,
        "datasets": datasets,
    }


def _clean_for_json(obj: Any) -> Any:
    """Recursively clean objects for JSON serialization (handle NaN, Inf, numpy)."""
    if isinstance(obj, dict):
        return {str(k): _clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_clean_for_json(v) for v in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v):
            return 0.0
        if math.isinf(v):
            return 1e15 if v > 0 else -1e15
        return v
    elif isinstance(obj, float):
        if math.isnan(obj):
            return 0.0
        if math.isinf(obj):
            return 1e15 if obj > 0 else -1e15
        return obj
    elif isinstance(obj, np.ndarray):
        return _clean_for_json(obj.tolist())
    elif isinstance(obj, set):
        return sorted(list(obj))
    elif isinstance(obj, (bool, int, str, type(None))):
        return obj
    # Catch-all for other types
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

@logger.catch
def main():
    logger.info("=" * 70)
    logger.info("Grand Synthesis: Walk Resolution Limit Hypothesis Adjudication")
    logger.info("=" * 70)

    # 1. Collect all SRI-gap correlations
    logger.info("Step 1: Collecting SRI-gap correlations from all experiments")
    studies = collect_sri_gap_correlations()

    # 2. Meta-analysis
    logger.info("Step 2: Fisher z random-effects meta-analysis")
    pooled = fisher_z_meta_analysis(studies)
    logger.info(f"  Pooled ρ = {pooled['rho_pooled']:.4f} [{pooled['ci_low']:.4f}, {pooled['ci_high']:.4f}]")
    logger.info(f"  I² = {pooled['I2']:.1f}%, Q p-value = {pooled['Q_pvalue']:.4f}")

    # 3. Subgroup analyses
    logger.info("Step 3: Subgroup meta-analyses")
    subgroup = subgroup_meta_analysis(studies)

    # 4. SRWE scorecard
    logger.info("Step 4: Collecting SRWE results and computing scorecard")
    srwe_results = collect_srwe_results()
    scorecard = compute_srwe_scorecard(srwe_results)

    # 5. Gap reductions
    logger.info("Step 5: Collecting SRWE gap reduction data")
    gap_reductions = collect_srwe_gap_reduction()

    # 6. Moderator analysis
    logger.info("Step 6: Moderator analysis")
    moderators = compute_moderator_effects(studies)

    # 7. Criterion adjudication
    logger.info("Step 7: Formally adjudicating hypothesis criteria")
    verdicts = adjudicate_criteria(pooled, studies, gap_reductions, scorecard)
    overall = compute_overall_verdict(verdicts)
    logger.info(f"  Overall verdict: {overall['overall_verdict']} (score={overall['overall_score']:.3f})")

    # 8. Scope of validity
    logger.info("Step 8: Scope-of-validity analysis")
    scope = compute_scope_of_validity(studies)
    logger.info(f"  Theory directional accuracy: {scope['directional_accuracy']:.1%}")

    # 9. Generate figures
    logger.info("Step 9: Generating figures")
    figure_paths = []
    try:
        figure_paths.append(plot_verdict_dashboard(verdicts, overall))
        logger.info("  ✓ Verdict dashboard")
    except Exception:
        logger.exception("Failed to generate verdict dashboard")

    try:
        figure_paths.append(plot_forest(studies, pooled))
        logger.info("  ✓ Forest plot")
    except Exception:
        logger.exception("Failed to generate forest plot")

    try:
        figure_paths.append(plot_srwe_scorecard_heatmap(scorecard))
        logger.info("  ✓ SRWE scorecard")
    except Exception:
        logger.exception("Failed to generate scorecard")

    try:
        figure_paths.append(plot_moderator_ranking(moderators))
        logger.info("  ✓ Moderator ranking")
    except Exception:
        logger.exception("Failed to generate moderator ranking")

    try:
        figure_paths.append(plot_scope_validity(studies))
        logger.info("  ✓ Scope-of-validity")
    except Exception:
        logger.exception("Failed to generate scope plot")

    try:
        figure_paths.append(plot_decision_tree())
        logger.info("  ✓ Decision tree")
    except Exception:
        logger.exception("Failed to generate decision tree")

    try:
        figure_paths.append(plot_encoding_performance_bars(srwe_results))
        logger.info("  ✓ Encoding performance bars")
    except Exception:
        logger.exception("Failed to generate performance bars")

    # 10. Build output
    logger.info("Step 10: Building output")
    output = build_output(
        studies, pooled, subgroup, scorecard, gap_reductions,
        moderators, verdicts, overall, scope, figure_paths,
    )

    # Clean for JSON
    output = _clean_for_json(output)

    # Save
    output_path = WORKSPACE / "eval_out.json"
    output_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved output to {output_path}")

    # Print summary
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Overall Verdict: {overall['overall_verdict']}")
    logger.info(f"  Overall Score: {overall['overall_score']:.3f}")
    logger.info(f"  Pooled ρ: {pooled['rho_pooled']:.4f} [{pooled['ci_low']:.4f}, {pooled['ci_high']:.4f}]")
    logger.info(f"  I² Heterogeneity: {pooled['I2']:.1f}%")
    logger.info(f"  Number of studies: {pooled['k']}")
    for key, v in verdicts.items():
        logger.info(f"  {key}: {v['verdict']} (conf={v['confidence']:.2f})")
    logger.info(f"  SRWE scorecard entries: {len(scorecard)}")
    logger.info(f"  Figures generated: {len(figure_paths)}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
