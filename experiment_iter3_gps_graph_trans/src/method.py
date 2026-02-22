#!/usr/bin/env python3
"""GPS Graph Transformer: RWSE vs LapPE vs SRWE with SRI-Performance Correlation Analysis.

Trains GPS Graph Transformers with three positional encodings (RWSE, LapPE, SRWE)
on ZINC-subset, Peptides-func, Peptides-struct, and Synthetic-aliased-pairs benchmarks.
Computes per-graph test losses to measure Spearman correlation between Spectral
Resolution Index (SRI) and the RWSE-vs-LapPE performance gap.

Output schema: exp_gen_sol_out.json (datasets array with per-graph predictions).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import sys
import os
import copy
import resource
import math
import warnings
from pathlib import Path
from collections import defaultdict

from loguru import logger
from scipy import sparse, stats
from scipy.linalg import eig
from scipy.sparse.linalg import eigsh

# ─── Logging ───────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parent
os.makedirs(WORKSPACE / "logs", exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(WORKSPACE / "logs" / "run.log"), rotation="30 MB", level="DEBUG")

warnings.filterwarnings("ignore")

# ─── Resource limits ───────────────────────────────────────────────────────
try:
    resource.setrlimit(resource.RLIMIT_AS, (50 * 1024**3, 50 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (3500, 3500))
except Exception:
    pass

# ─── Hardware detection ────────────────────────────────────────────────────
def _cgroup_cpus():
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
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

import psutil
NUM_CPUS = _cgroup_cpus() or os.cpu_count() or 1
HAS_GPU = torch.cuda.is_available()
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if HAS_GPU else 0
DEVICE = torch.device("cuda" if HAS_GPU else "cpu")
TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9
AVAILABLE_RAM_GB = min(psutil.virtual_memory().available / 1e9, TOTAL_RAM_GB)

# ─── Dependency paths ──────────────────────────────────────────────────────
DEP_DIR = Path("/opt/ai-inventor/aii_pipeline/runs/run__20260222_005317/"
               "3_invention_loop/iter_1/gen_art/data_id2_it1__opus")
DATA_DIR = DEP_DIR / "data_out"

# ─── Global timer ──────────────────────────────────────────────────────────
GLOBAL_START = time.time()
TIME_BUDGET_S = 3300  # 55 min hard limit to leave time for output

def time_remaining() -> float:
    return TIME_BUDGET_S - (time.time() - GLOBAL_START)

# ─── PyG imports ───────────────────────────────────────────────────────────
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GPSConv, GINEConv, global_add_pool
from torch_geometric.utils import to_scipy_sparse_matrix

# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_all_dependency_data() -> dict:
    """Load all examples from dependency JSON files grouped by dataset."""
    datasets = defaultdict(list)
    for i in range(1, 6):
        fpath = DATA_DIR / f"full_data_out_{i}.json"
        logger.info(f"Loading {fpath.name} ...")
        raw = json.loads(fpath.read_text())
        for ds in raw["datasets"]:
            ds_name = ds["dataset"]
            for ex in ds["examples"]:
                datasets[ds_name].append(ex)
    for ds_name, exs in datasets.items():
        logger.info(f"  {ds_name}: {len(exs)} examples")
    return dict(datasets)


def split_examples(examples: list, dataset_name: str) -> tuple:
    """Split examples into train/val/test based on metadata_fold or deterministic."""
    fold_map = defaultdict(list)
    for ex in examples:
        fold_map[ex["metadata_fold"]].append(ex)

    folds = sorted(fold_map.keys())
    if set(folds) >= {0, 1, 2}:
        # ZINC-style: fold 0=train, 1=val, 2=test
        return fold_map[0], fold_map[1], fold_map[2]
    elif dataset_name == "Synthetic-aliased-pairs":
        n = len(examples)
        n_train = int(0.6 * n)
        n_val = int(0.2 * n)
        return examples[:n_train], examples[n_train:n_train + n_val], examples[n_train + n_val:]
    else:
        # Peptides: all fold 0, use plan's 1400/300/300 subsample
        n = len(examples)
        if n >= 2000:
            return examples[:1400], examples[1400:1700], examples[1700:2000]
        else:
            n_train = int(0.7 * n)
            n_val = int(0.15 * n)
            return examples[:n_train], examples[n_train:n_train + n_val], examples[n_train + n_val:]


# ═══════════════════════════════════════════════════════════════════════════
# POSITIONAL ENCODING COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_lappe(edge_index_list: list, num_nodes: int, k: int = 8) -> np.ndarray:
    """Compute Laplacian Positional Encoding (top-k non-trivial eigenvectors)."""
    if num_nodes <= 1:
        return np.zeros((num_nodes, k), dtype=np.float32)

    row, col = edge_index_list
    n = num_nodes
    A = sparse.coo_matrix(
        (np.ones(len(row)), (np.array(row), np.array(col))), shape=(n, n)
    ).tocsr()
    A = (A + A.T) / 2
    deg = np.array(A.sum(axis=1)).flatten()
    D = sparse.diags(deg)
    L = D - A

    num_eigs = min(k + 1, n - 1)
    if num_eigs < 2:
        return np.zeros((n, k), dtype=np.float32)

    try:
        eigenvalues, eigenvectors = eigsh(L.tocsc(), k=num_eigs, which='SM', tol=1e-5)
        idx = np.argsort(eigenvalues)
        eigenvectors = eigenvectors[:, idx]
        pe = eigenvectors[:, 1:k + 1]
        if pe.shape[1] < k:
            pe = np.pad(pe, ((0, 0), (0, k - pe.shape[1])))
    except Exception:
        pe = np.zeros((n, k))
    return pe.astype(np.float32)


def compute_srwe_for_graph(rwse_moments: np.ndarray, num_bins: int = 20,
                            pencil_rank: int = 10) -> np.ndarray:
    """Matrix Pencil Method to get Super-Resolved Walk Encoding for one graph."""
    n_nodes, n_moments = rwse_moments.shape
    K = min(pencil_rank, n_moments // 2)
    srwe = np.zeros((n_nodes, num_bins))

    for v in range(n_nodes):
        m = rwse_moments[v]
        moments = np.concatenate([[1.0], m])

        H0 = np.zeros((K, K))
        H1 = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                idx0 = i + j
                idx1 = i + j + 1
                if idx0 < len(moments):
                    H0[i, j] = moments[idx0]
                if idx1 < len(moments):
                    H1[i, j] = moments[idx1]

        H0 += 1e-8 * np.eye(K)

        try:
            U, S, Vt = np.linalg.svd(H0)
        except np.linalg.LinAlgError:
            continue

        threshold = S[0] * 1e-3 if S[0] > 0 else 1e-12
        r = max(1, int(np.sum(S > threshold)))
        r = min(r, K)

        U_r = U[:, :r]
        S_r = np.diag(S[:r])
        Vt_r = Vt[:r, :]

        try:
            A_pencil = U_r.T @ H1 @ Vt_r.T
            eigenvalues, _ = eig(A_pencil, S_r)
            eigenvalues = np.real(eigenvalues)
        except Exception:
            continue

        valid = np.isfinite(eigenvalues)
        eigenvalues = eigenvalues[valid]
        if len(eigenvalues) == 0:
            continue

        try:
            V_mat = np.vander(eigenvalues, N=n_moments + 1, increasing=True)[:, 1:]
            V_mat = V_mat.T
            weights, _, _, _ = np.linalg.lstsq(V_mat, m, rcond=None)
            weights = np.maximum(weights, 0)
        except Exception:
            weights = np.ones(len(eigenvalues)) / max(len(eigenvalues), 1)

        bin_edges = np.linspace(-1.0, 2.0, num_bins + 1)
        for i_eig in range(len(eigenvalues)):
            lam = eigenvalues[i_eig]
            w = weights[i_eig] if i_eig < len(weights) else 0
            if not np.isfinite(lam) or not np.isfinite(w):
                continue
            bin_idx = int(np.clip(np.digitize(lam, bin_edges) - 1, 0, num_bins - 1))
            srwe[v, bin_idx] += w

    row_sums = srwe.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    srwe = srwe / row_sums
    return srwe.astype(np.float32)


def build_pyg_data(ex: dict, pe_type: str,
                   lappe_arr: np.ndarray = None,
                   srwe_arr: np.ndarray = None) -> Data:
    """Convert a dependency JSON example to a PyG Data object with PE."""
    inp = json.loads(ex["input"])
    edge_index = torch.tensor(inp["edge_index"], dtype=torch.long)
    num_nodes = inp["num_nodes"]

    node_feat_raw = inp["node_feat"]
    x = torch.tensor([nf[0] for nf in node_feat_raw], dtype=torch.long)

    if "edge_attr" in inp and inp["edge_attr"]:
        edge_attr = torch.tensor([ea[0] for ea in inp["edge_attr"]], dtype=torch.long)
    else:
        edge_attr = torch.zeros(edge_index.size(1), dtype=torch.long)

    # Target
    output_raw = ex["output"]
    if isinstance(output_raw, str):
        try:
            parsed = json.loads(output_raw)
            if isinstance(parsed, list):
                y = torch.tensor(parsed, dtype=torch.float32)
            else:
                y = torch.tensor([float(parsed)], dtype=torch.float32)
        except (json.JSONDecodeError, ValueError):
            y = torch.tensor([float(output_raw)], dtype=torch.float32)
    elif isinstance(output_raw, (int, float)):
        y = torch.tensor([float(output_raw)], dtype=torch.float32)
    elif isinstance(output_raw, list):
        y = torch.tensor(output_raw, dtype=torch.float32)
    else:
        y = torch.tensor([0.0], dtype=torch.float32)

    # PE
    spectral = inp.get("spectral", {})
    if pe_type == "rwse":
        rwse_data = spectral.get("rwse", None)
        pe = torch.tensor(rwse_data, dtype=torch.float32) if rwse_data else torch.zeros(num_nodes, 20)
    elif pe_type == "lappe":
        pe = torch.tensor(lappe_arr, dtype=torch.float32) if lappe_arr is not None else torch.zeros(num_nodes, 8)
    elif pe_type == "srwe":
        pe = torch.tensor(srwe_arr, dtype=torch.float32) if srwe_arr is not None else torch.zeros(num_nodes, 20)
    else:
        pe = torch.zeros(num_nodes, 20)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pe=pe,
                num_nodes=num_nodes)


# ═══════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════

class GPSModel(nn.Module):
    """GPS Graph Transformer with configurable PE type."""

    def __init__(self, pe_type: str, pe_dim: int, num_node_types: int,
                 num_edge_types: int, hidden_dim: int = 64,
                 num_layers: int = 5, num_heads: int = 4,
                 output_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        self.pe_type = pe_type
        pe_input_dim = 20 if pe_type in ("rwse", "srwe") else 8

        self.pe_norm = nn.BatchNorm1d(pe_input_dim)
        self.pe_lin = nn.Linear(pe_input_dim, pe_dim)

        node_emb_dim = hidden_dim - pe_dim
        self.node_emb = nn.Embedding(num_node_types, node_emb_dim)
        self.edge_emb = nn.Embedding(num_edge_types, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            nn_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            conv = GPSConv(
                channels=hidden_dim,
                conv=GINEConv(nn_layer),
                heads=num_heads,
                dropout=dropout,
                attn_type='multihead',
                attn_kwargs={'dropout': dropout},
            )
            self.convs.append(conv)
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, output_dim),
        )

    def forward(self, x, pe, edge_index, edge_attr, batch):
        if self.pe_type == "lappe" and self.training:
            signs = torch.randint(0, 2, (1, pe.size(1)), device=pe.device) * 2 - 1
            pe = pe * signs.float()

        pe_out = self.pe_lin(self.pe_norm(pe))
        h = torch.cat([self.node_emb(x), pe_out], dim=-1)
        ea = self.edge_emb(edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index, batch, edge_attr=ea)
            h = norm(h)

        h = global_add_pool(h, batch)
        return self.output_mlp(h)


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════

DATASET_CONFIGS = {
    "ZINC-subset": {
        "task": "regression", "output_dim": 1, "per_graph_loss_fn": "l1",
        "metric_name": "MAE", "metric_better": "lower",
        "num_node_types": 64, "num_edge_types": 16,
    },
    "Peptides-func": {
        "task": "classification", "output_dim": 10, "per_graph_loss_fn": "bce",
        "metric_name": "BCE", "metric_better": "lower",
        "num_node_types": 64, "num_edge_types": 16,
    },
    "Peptides-struct": {
        "task": "regression", "output_dim": 11, "per_graph_loss_fn": "l1",
        "metric_name": "MAE", "metric_better": "lower",
        "num_node_types": 64, "num_edge_types": 16,
    },
    "Synthetic-aliased-pairs": {
        "task": "regression", "output_dim": 1, "per_graph_loss_fn": "l1",
        "metric_name": "MAE", "metric_better": "lower",
        "num_node_types": 64, "num_edge_types": 16,
    },
}


def make_loss_fn(config: dict) -> nn.Module:
    if config["task"] == "classification":
        return nn.BCEWithLogitsLoss()
    return nn.L1Loss()


def per_graph_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> float:
    if loss_type == "l1":
        return (pred - target).abs().mean().item()
    elif loss_type == "bce":
        return F.binary_cross_entropy_with_logits(pred, target).item()
    return (pred - target).abs().mean().item()


def reshape_target(out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Ensure target shape matches output shape."""
    if out.shape == target.shape:
        return target
    if target.dim() == 1 and out.dim() == 2 and out.size(1) == 1:
        return target.unsqueeze(1)
    if target.dim() == 1 and out.dim() == 2:
        return target.view(out.shape)
    return target


def train_one_config(pe_type: str, seed: int,
                     train_data: list, val_data: list, test_data: list,
                     config: dict, device: torch.device,
                     max_epochs: int = 80, patience: int = 15,
                     lr: float = 1e-3, batch_size: int = 32,
                     hidden_dim: int = 64, num_layers: int = 5,
                     num_heads: int = 4, pe_dim: int = 16) -> dict:
    """Train one (pe_type, seed) configuration. Returns per-graph test losses."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    model = GPSModel(
        pe_type=pe_type, pe_dim=pe_dim,
        num_node_types=config["num_node_types"],
        num_edge_types=config["num_edge_types"],
        hidden_dim=hidden_dim, num_layers=num_layers,
        num_heads=num_heads, output_dim=config["output_dim"],
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size * 2)
    test_loader = DataLoader(test_data, batch_size=batch_size * 2)

    loss_fn = make_loss_fn(config).to(device)

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    best_epoch = 0
    start_time = time.time()

    for epoch in range(max_epochs):
        # Train
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch)
            target = reshape_target(out, batch.y)
            loss = loss_fn(out, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item() * batch.num_graphs
            train_count += batch.num_graphs

        scheduler.step()

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch)
                target = reshape_target(out, batch.y)
                val_loss_sum += loss_fn(out, target).item() * batch.num_graphs
                val_count += batch.num_graphs

        avg_val = val_loss_sum / max(val_count, 1)

        if epoch % 20 == 0:
            logger.debug(f"    Ep {epoch}: train={train_loss_sum / max(train_count,1):.4f} val={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        # Time check
        if time.time() - start_time > 600:  # 10 min per seed max
            logger.warning(f"    Seed time limit reached at epoch {epoch}")
            break

    wall_time = time.time() - start_time

    # Evaluate per-graph on test set
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()

    per_graph_losses = []
    per_graph_preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch)
            target = reshape_target(out, batch.y)
            for i in range(batch.num_graphs):
                pred_i = out[i]
                target_i = target[i]
                loss_i = per_graph_loss(pred_i, target_i, config["per_graph_loss_fn"])
                per_graph_losses.append(loss_i)
                per_graph_preds.append(pred_i.cpu().tolist() if pred_i.dim() > 0 else [pred_i.item()])

    return {
        "per_graph_losses": per_graph_losses,
        "per_graph_preds": per_graph_preds,
        "overall_metric": float(np.mean(per_graph_losses)) if per_graph_losses else 0.0,
        "best_epoch": best_epoch,
        "wall_time_seconds": wall_time,
        "final_epoch": epoch if 'epoch' in dir() else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def correlation_analysis(gaps: np.ndarray, sri_values: np.ndarray,
                         cond_values: np.ndarray,
                         num_nodes_values: np.ndarray) -> dict:
    """Compute SRI-gap correlations and quintile analysis."""
    results = {}
    valid = np.isfinite(gaps) & np.isfinite(sri_values) & np.isfinite(num_nodes_values)
    gaps = gaps[valid]
    sri_values = sri_values[valid]
    cond_values = cond_values[valid]
    num_nodes_values = num_nodes_values[valid]

    if len(gaps) < 10:
        return {"error": "Too few valid data points", "n_valid": int(valid.sum())}

    # Spearman: SRI vs gap
    rho_sri, p_sri = stats.spearmanr(sri_values, gaps)
    results["spearman_sri_vs_gap"] = {
        "rho": float(rho_sri) if np.isfinite(rho_sri) else 0.0,
        "p_value": float(p_sri) if np.isfinite(p_sri) else 1.0,
    }

    # Spearman: log(cond) vs gap
    log_cond = np.log10(np.clip(cond_values, 1, 1e15))
    vc = np.isfinite(log_cond)
    if vc.sum() > 10:
        rho_c, p_c = stats.spearmanr(log_cond[vc], gaps[vc])
        results["spearman_logcond_vs_gap"] = {
            "rho": float(rho_c) if np.isfinite(rho_c) else 0.0,
            "p_value": float(p_c) if np.isfinite(p_c) else 1.0,
        }

    # Partial correlation (size-controlled)
    try:
        import pingouin as pg
        import pandas as pd
        df = pd.DataFrame({"sri": sri_values, "gap": gaps, "num_nodes": num_nodes_values})
        partial = pg.partial_corr(data=df, x="sri", y="gap",
                                   covar="num_nodes", method="spearman")
        results["partial_spearman_sri_gap_ctrl_size"] = {
            "rho": float(partial["r"].values[0]),
            "p_value": float(partial["p-val"].values[0]),
        }
    except (ImportError, Exception):
        # Fallback: within-size-bin correlations
        try:
            pcts = np.percentile(num_nodes_values, [0, 25, 50, 75, 100])
            within = []
            for lo, hi in zip(pcts[:-1], pcts[1:]):
                mask = (num_nodes_values >= lo) & (num_nodes_values <= hi + 0.01)
                if mask.sum() > 10:
                    r, _ = stats.spearmanr(sri_values[mask], gaps[mask])
                    if np.isfinite(r):
                        within.append(float(r))
            results["within_size_bin_correlations"] = within
            results["mean_within_bin_corr"] = float(np.mean(within)) if within else 0.0
        except Exception:
            pass

    # Quintile analysis
    try:
        q_edges = np.percentile(sri_values, [0, 20, 40, 60, 80, 100])
    except Exception:
        q_edges = np.linspace(sri_values.min(), sri_values.max(), 6)

    quintiles = []
    for q in range(5):
        lo, hi = q_edges[q], q_edges[q + 1]
        mask = (sri_values >= lo) & (sri_values <= hi + 1e-10)
        n = int(mask.sum())
        quintiles.append({
            "quintile": q + 1,
            "sri_range": [float(lo), float(hi)],
            "mean_gap": float(np.mean(gaps[mask])) if n > 0 else 0.0,
            "std_gap": float(np.std(gaps[mask])) if n > 0 else 0.0,
            "count": n,
        })
    results["quintile_analysis"] = quintiles

    # Monotonicity check
    qmeans = [q["mean_gap"] for q in quintiles if q["count"] > 0]
    if len(qmeans) >= 3:
        diffs = [qmeans[i + 1] - qmeans[i] for i in range(len(qmeans) - 1)]
        results["quintile_monotonic_decreasing"] = all(d <= 0 for d in diffs)
        results["quintile_trend_direction"] = "decreasing" if np.mean(diffs) < 0 else "increasing"

    results["n_valid"] = int(valid.sum())
    return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

@logger.catch
def main():
    logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, "
                f"GPU={'Yes (' + f'{VRAM_GB:.0f}GB)' if HAS_GPU else 'No'}")
    logger.info(f"Device: {DEVICE}")

    # ─── Config ─────────────────────────────────────────────────────────
    SEEDS = [42, 123, 456]
    PE_TYPES = ["rwse", "lappe", "srwe"]
    HIDDEN_DIM = 64
    NUM_LAYERS = 5
    NUM_HEADS = 4
    PE_DIM = 16
    MAX_EPOCHS = 80
    PATIENCE = 15
    LR = 1e-3

    # ─── Load data ──────────────────────────────────────────────────────
    all_data = load_all_dependency_data()
    dataset_names = list(all_data.keys())
    logger.info(f"Datasets: {dataset_names}")

    all_results = {}
    all_output_datasets = []

    for ds_name in dataset_names:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Dataset: {ds_name}")
        logger.info(f"{'=' * 60}")

        examples = all_data[ds_name]
        config = DATASET_CONFIGS[ds_name]
        train_exs, val_exs, test_exs = split_examples(examples, ds_name)
        logger.info(f"Split: train={len(train_exs)} val={len(val_exs)} test={len(test_exs)}")

        # Batch size: larger for bigger datasets
        batch_size = 64 if len(train_exs) > 5000 else 32

        # Spectral metrics for test set
        test_sri, test_cond, test_nnodes = [], [], []
        for ex in test_exs:
            inp = json.loads(ex["input"])
            sp = inp.get("spectral", {})
            sri_d = sp.get("sri", {})
            cond_d = sp.get("vandermonde_cond", {})
            test_sri.append(float(sri_d.get("K=20", 0.0) or 0.0))
            test_cond.append(float(cond_d.get("K=20", 1.0) or 1.0))
            test_nnodes.append(float(ex.get("metadata_num_nodes", inp.get("num_nodes", 1))))
        test_sri = np.array(test_sri)
        test_cond = np.array(test_cond)
        test_nnodes = np.array(test_nnodes)

        # ─── Precompute PEs ─────────────────────────────────────────────
        logger.info("Precomputing PEs ...")
        t_pe = time.time()
        all_exs = train_exs + val_exs + test_exs
        lappe_cache = {}
        srwe_cache = {}

        for idx, ex in enumerate(all_exs):
            inp = json.loads(ex["input"])
            ei = inp["edge_index"]
            nn_ = inp["num_nodes"]
            sp = inp.get("spectral", {})

            lappe_cache[id(ex)] = compute_lappe(ei, nn_, k=8)

            rwse_data = sp.get("rwse", None)
            if rwse_data is not None:
                srwe_cache[id(ex)] = compute_srwe_for_graph(
                    np.array(rwse_data, dtype=np.float32), num_bins=20, pencil_rank=10
                )
            else:
                srwe_cache[id(ex)] = np.zeros((nn_, 20), dtype=np.float32)

            if (idx + 1) % 1000 == 0:
                logger.info(f"  PE {idx + 1}/{len(all_exs)} graphs ({time.time() - t_pe:.1f}s)")

        logger.info(f"PE done: {time.time() - t_pe:.1f}s for {len(all_exs)} graphs")

        # ─── Build PyG lists & train ─────────────────────────────────────
        def build_pyg_list(exs, pe_type):
            return [build_pyg_data(ex, pe_type,
                                   lappe_arr=lappe_cache.get(id(ex)),
                                   srwe_arr=srwe_cache.get(id(ex)))
                    for ex in exs]

        pe_results = {}
        for pe_type in PE_TYPES:
            if time_remaining() < 120:
                logger.warning(f"Time budget low ({time_remaining():.0f}s). Skipping {pe_type}.")
                pe_results[pe_type] = []
                continue

            logger.info(f"\n  PE: {pe_type}")
            pe_results[pe_type] = []

            train_pyg = build_pyg_list(train_exs, pe_type)
            val_pyg = build_pyg_list(val_exs, pe_type)
            test_pyg = build_pyg_list(test_exs, pe_type)

            for seed in SEEDS:
                if time_remaining() < 60:
                    logger.warning(f"Time budget low. Skipping seed {seed}.")
                    break

                logger.info(f"    Seed {seed} ...")
                result = train_one_config(
                    pe_type=pe_type, seed=seed,
                    train_data=train_pyg, val_data=val_pyg, test_data=test_pyg,
                    config=config, device=DEVICE,
                    max_epochs=MAX_EPOCHS, patience=PATIENCE, lr=LR,
                    batch_size=batch_size, hidden_dim=HIDDEN_DIM,
                    num_layers=NUM_LAYERS, num_heads=NUM_HEADS, pe_dim=PE_DIM,
                )
                logger.info(f"    Seed {seed}: metric={result['overall_metric']:.4f}, "
                            f"ep={result['best_epoch']}, wall={result['wall_time_seconds']:.1f}s")
                pe_results[pe_type].append(result)

        # ─── Aggregate ───────────────────────────────────────────────────
        n_test = len(test_exs)
        mean_losses = {}
        mean_preds = {}
        for pe_type in PE_TYPES:
            seeds = pe_results.get(pe_type, [])
            if not seeds:
                mean_losses[pe_type] = np.full(n_test, np.nan)
                mean_preds[pe_type] = [[0.0]] * n_test
                continue
            arrs = []
            for sd in seeds:
                a = sd["per_graph_losses"][:n_test]
                while len(a) < n_test:
                    a.append(float("nan"))
                arrs.append(a)
            mean_losses[pe_type] = np.nanmean(arrs, axis=0)
            mean_preds[pe_type] = seeds[-1]["per_graph_preds"][:n_test]
            while len(mean_preds[pe_type]) < n_test:
                mean_preds[pe_type].append([0.0])

        gap_rwse_lappe = mean_losses.get("rwse", np.zeros(n_test)) - mean_losses.get("lappe", np.zeros(n_test))
        gap_srwe_lappe = mean_losses.get("srwe", np.zeros(n_test)) - mean_losses.get("lappe", np.zeros(n_test))

        corr = correlation_analysis(gap_rwse_lappe, test_sri, test_cond, test_nnodes)

        mg_rwse = float(np.nanmean(gap_rwse_lappe))
        mg_srwe = float(np.nanmean(gap_srwe_lappe))
        gap_red = 1.0 - (mg_srwe / mg_rwse) if abs(mg_rwse) > 1e-8 else 0.0

        per_enc = {}
        per_seed = {}
        for pe_type in PE_TYPES:
            seeds = pe_results.get(pe_type, [])
            vals = [s["overall_metric"] for s in seeds]
            per_enc[pe_type] = {
                "mean": float(np.mean(vals)) if vals else 0.0,
                "std_across_seeds": float(np.std(vals)) if len(vals) > 1 else 0.0,
            }
            per_seed[pe_type] = {str(SEEDS[i]): v for i, v in enumerate(vals)}

        ds_result = {
            "per_encoding_metrics": per_enc,
            "per_seed_metrics": per_seed,
            "correlation_analysis": corr,
            "srwe_improvement": {
                "mean_gap_rwse_lappe": mg_rwse,
                "mean_gap_srwe_lappe": mg_srwe,
                "gap_reduction_fraction": float(gap_red),
            },
            "wall_times": {pe: [s["wall_time_seconds"] for s in pe_results.get(pe, [])]
                          for pe in PE_TYPES},
        }
        all_results[ds_name] = ds_result

        # Build output examples (schema-compliant)
        output_examples = []
        for idx, ex in enumerate(test_exs):
            out_ex = {
                "input": ex["input"],
                "output": str(ex["output"]),
            }
            for pe_type in PE_TYPES:
                p = mean_preds.get(pe_type, [[0.0]] * n_test)
                pv = p[idx] if idx < len(p) else [0.0]
                out_ex[f"predict_{pe_type}"] = json.dumps(pv)

            out_ex["metadata_sri_k20"] = float(test_sri[idx])
            out_ex["metadata_vandermonde_cond_k20"] = float(test_cond[idx])
            out_ex["metadata_num_nodes"] = int(test_nnodes[idx])
            out_ex["metadata_gap_rwse_lappe"] = float(gap_rwse_lappe[idx]) if np.isfinite(gap_rwse_lappe[idx]) else 0.0
            out_ex["metadata_gap_srwe_lappe"] = float(gap_srwe_lappe[idx]) if np.isfinite(gap_srwe_lappe[idx]) else 0.0
            for pe_type in PE_TYPES:
                lv = mean_losses.get(pe_type, np.zeros(n_test))
                out_ex[f"metadata_loss_{pe_type}"] = float(lv[idx]) if idx < len(lv) and np.isfinite(lv[idx]) else 0.0

            for k, v in ex.items():
                if k.startswith("metadata_") and k not in out_ex:
                    out_ex[k] = v

            output_examples.append(out_ex)

        all_output_datasets.append({"dataset": ds_name, "examples": output_examples})

        logger.info(f"\nResults for {ds_name}:")
        for pe_type in PE_TYPES:
            m = per_enc.get(pe_type, {})
            logger.info(f"  {pe_type}: {config['metric_name']} = {m.get('mean', 0):.4f} ± {m.get('std_across_seeds', 0):.4f}")
        if "spearman_sri_vs_gap" in corr:
            sr = corr["spearman_sri_vs_gap"]
            logger.info(f"  Spearman(SRI, gap): rho={sr['rho']:.4f}, p={sr['p_value']:.4e}")
        logger.info(f"  SRWE gap reduction: {gap_red:.2%}")

    # ─── Save output ─────────────────────────────────────────────────────
    total_time = time.time() - GLOBAL_START
    logger.info(f"\nTotal time: {total_time:.1f}s ({total_time / 60:.1f}min)")

    method_output = {
        "metadata": {
            "experiment": "GPS_RWSE_LapPE_SRWE_SRI_correlation",
            "architecture": f"GPS ({NUM_LAYERS} layers, {HIDDEN_DIM} dim, {NUM_HEADS} heads, GINEConv)",
            "hyperparameters": {
                "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS,
                "num_heads": NUM_HEADS, "pe_dim": PE_DIM,
                "lr": LR, "weight_decay": 1e-5,
                "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                "seeds": SEEDS, "pe_types": PE_TYPES,
            },
            "results_summary": all_results,
            "total_wall_time_seconds": total_time,
        },
        "datasets": all_output_datasets,
    }

    out_path = WORKSPACE / "method_out.json"
    out_path.write_text(json.dumps(method_output, indent=2, default=str))
    logger.info(f"Saved to {out_path} ({out_path.stat().st_size / 1e6:.1f}MB)")

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("EXPERIMENT SUMMARY")
    logger.info(f"{'=' * 60}")
    for ds_name, dr in all_results.items():
        logger.info(f"\n{ds_name}:")
        for pe, m in dr["per_encoding_metrics"].items():
            logger.info(f"  {pe}: {m['mean']:.4f} ± {m['std_across_seeds']:.4f}")
        c = dr.get("correlation_analysis", {})
        if "spearman_sri_vs_gap" in c:
            logger.info(f"  Spearman(SRI,gap): rho={c['spearman_sri_vs_gap']['rho']:.4f}")
        si = dr.get("srwe_improvement", {})
        logger.info(f"  SRWE gap reduction: {si.get('gap_reduction_fraction', 0):.2%}")


if __name__ == "__main__":
    main()
