#!/usr/bin/env python3
"""Depth vs Encoding Aliasing: Does GNN Message Passing Compensate for Walk Resolution Limits?

Tests whether increasing GNN depth (2->8 layers) compensates for RWSE spectral aliasing
by narrowing the RWSE-vs-LapPE performance gap. Trains GCN+GlobalAttention models at
5 depths x 3 encodings (RWSE, LapPE, SRWE) x 2 datasets (ZINC, Peptides-struct) x 3 seeds.
"""

import json
import glob
import math
import os
import sys
import time
import resource
from pathlib import Path

import numpy as np
import psutil
import scipy
import scipy.linalg
from scipy.stats import spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.data import Data, Batch

from loguru import logger

# ── LOGGING ──
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ── RESOURCE LIMITS ──
try:
    resource.setrlimit(resource.RLIMIT_AS, (48 * 1024**3, 48 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (3500, 3500))
except (ValueError, resource.error) as e:
    logger.warning(f"Could not set resource limits: {e}")

# ── HARDWARE DETECTION ──
def _cgroup_cpus():
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
TOTAL_RAM_GB = psutil.virtual_memory().total / 1e9

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, GPU={HAS_GPU} ({VRAM_GB:.1f}GB VRAM)")
logger.info(f"Device: {DEVICE}")

# ── CONSTANTS ──
WORKSPACE = Path(__file__).parent
DATA_DIR = Path("/opt/ai-inventor/aii_pipeline/runs/run__20260222_005317/3_invention_loop/iter_1/gen_art/data_id2_it1__opus")
MINI_DATA = DATA_DIR / "mini_data_out.json"
FULL_DATA_DIR = DATA_DIR / "data_out"

PE_DIM = 20
HIDDEN_DIM = 64
DEPTHS = [2, 3, 4, 6, 8]
PE_TYPES = ["rwse", "lappe", "srwe"]
SEEDS = [0, 1, 2]
MAX_EPOCHS = 100
PATIENCE = 20
LR = 1e-3
DROPOUT = 0.1
BATCH_SIZE = 64
ZINC_SUBSAMPLE = 2000
PEPTIDES_SUBSAMPLE = 1500
MAX_NODES_SPECTRAL = 200

TIME_BUDGET_SEC = 55 * 60
START_TIME = time.time()


def time_left():
    return TIME_BUDGET_SEC - (time.time() - START_TIME)


def time_check(label=""):
    remaining = time_left()
    if remaining < 60:
        logger.warning(f"Time budget nearly exhausted ({remaining:.0f}s left) at {label}")
        return True
    return False


# ── DATA LOADING ──
def load_all_data(max_examples_per_dataset=None):
    """Load all data from the split JSON files."""
    logger.info("Loading data from split files...")
    datasets = {}
    full_files = sorted(glob.glob(str(FULL_DATA_DIR / "full_data_out_*.json")))
    if not full_files:
        logger.warning("No full data files found, trying mini data")
        return load_mini_data()

    for fpath in full_files:
        logger.info(f"  Loading {Path(fpath).name}")
        with open(fpath) as f:
            raw = json.load(f)
        for ds_block in raw["datasets"]:
            name = ds_block["dataset"]
            if name not in datasets:
                datasets[name] = []
            datasets[name].extend(ds_block["examples"])

    for name, examples in datasets.items():
        logger.info(f"  {name}: {len(examples)} examples")
        if max_examples_per_dataset is not None and len(examples) > max_examples_per_dataset:
            datasets[name] = examples[:max_examples_per_dataset]
            logger.info(f"    Truncated to {max_examples_per_dataset}")

    return datasets


def load_mini_data():
    """Load mini data for testing."""
    with open(MINI_DATA) as f:
        raw = json.load(f)
    datasets = {}
    for ds_block in raw["datasets"]:
        datasets[ds_block["dataset"]] = ds_block["examples"]
    return datasets


# ── GRAPH PARSING ──
def parse_graph(example: dict) -> dict:
    """Parse a single example into graph components."""
    inp = json.loads(example["input"])
    edge_index = np.array(inp["edge_index"], dtype=np.int64)
    num_nodes = inp["num_nodes"]
    node_feat = np.array(inp["node_feat"], dtype=np.float32)
    spectral = inp["spectral"]
    eigenvalues = np.array(spectral["eigenvalues"], dtype=np.float64)
    rwse = np.array(spectral["rwse"], dtype=np.float32)  # [N, 20]
    sri = spectral.get("sri", {})

    output_str = example["output"]
    try:
        output = json.loads(output_str)
        if isinstance(output, list):
            target = np.array(output, dtype=np.float32)
        else:
            target = np.array([float(output)], dtype=np.float32)
    except (json.JSONDecodeError, ValueError):
        target = np.array([float(output_str)], dtype=np.float32)

    return {
        "edge_index": edge_index,
        "num_nodes": num_nodes,
        "node_feat": node_feat,
        "eigenvalues": eigenvalues,
        "rwse": rwse,
        "target": target,
        "sri": sri,
        "metadata_fold": example.get("metadata_fold", 0),
        "metadata_row_index": example.get("metadata_row_index", 0),
        "delta_min": spectral.get("delta_min", 0.0),
    }


# ── SPECTRAL PREPROCESSING ──
def compute_normalized_adjacency_eigen(edge_index: np.ndarray, num_nodes: int):
    """Compute eigenvalues/vectors of normalized adjacency D^{-1/2} A D^{-1/2}."""
    if num_nodes > MAX_NODES_SPECTRAL:
        return None, None

    # Build adjacency with vectorized indexing
    A = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    src = edge_index[0]
    dst = edge_index[1]
    A[src, dst] = 1.0

    deg = A.sum(axis=1)
    deg_inv_sqrt = np.zeros(num_nodes, dtype=np.float64)
    mask = deg > 0
    deg_inv_sqrt[mask] = 1.0 / np.sqrt(deg[mask])

    # Normalized adjacency: D^{-1/2} A D^{-1/2} using broadcasting
    P = A * deg_inv_sqrt[:, None] * deg_inv_sqrt[None, :]

    try:
        eigenvalues, eigenvectors = scipy.linalg.eigh(P)
    except Exception as e:
        logger.warning(f"Eigendecomposition failed: {e}")
        return None, None

    return eigenvalues, eigenvectors


def compute_srwe_batch(rwse_all: np.ndarray, eigenvalues_P: np.ndarray, alpha: float = 1e-4) -> np.ndarray:
    """Vectorized SRWE computation for ALL nodes in a graph at once.

    Args:
        rwse_all: [N_nodes, K] RWSE matrix for all nodes
        eigenvalues_P: [N_eig] eigenvalues of normalized adjacency
        alpha: Tikhonov regularization parameter

    Returns:
        weights: [N_nodes, N_eig] spectral weight matrix
    """
    K = rwse_all.shape[1]  # 20
    N_eig = eigenvalues_P.shape[0]
    N_nodes = rwse_all.shape[0]

    # Build Vandermonde-like matrix V[k,i] = eigenvalues_P[i]^(k+1)
    # Using broadcasting: shape (K, N_eig)
    powers = np.arange(1, K + 1, dtype=np.float64)[:, None]  # (K, 1)
    eigs = eigenvalues_P[None, :]  # (1, N_eig)
    V = np.power(eigs, powers)  # (K, N_eig)

    # Tikhonov: (V^T V + alpha I) W = V^T @ moments^T
    # VtV is (N_eig, N_eig), same for all nodes -> factorize once
    VtV = V.T @ V + alpha * np.eye(N_eig, dtype=np.float64)  # (N_eig, N_eig)
    VtM = V.T @ rwse_all.astype(np.float64).T  # (N_eig, N_nodes)

    try:
        # Solve for all nodes simultaneously
        W = scipy.linalg.solve(VtV, VtM, assume_a='pos')  # (N_eig, N_nodes)
    except np.linalg.LinAlgError:
        W = np.linalg.lstsq(VtV, VtM, rcond=None)[0]

    W = W.T  # (N_nodes, N_eig)

    # Project to non-negative, normalize per node
    W = np.maximum(W, 0.0)
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 1e-12, row_sums, 1.0)
    W = W / row_sums

    return W


def compute_pe(graph_data: dict, pe_type: str, eigenvalues_P=None, eigenvectors_P=None) -> np.ndarray:
    """Compute positional encoding for a graph. Returns [num_nodes, PE_DIM] array."""
    num_nodes = graph_data["num_nodes"]
    rwse = graph_data["rwse"]  # [N, 20]

    if pe_type == "rwse":
        pe = rwse[:, :PE_DIM]
        if pe.shape[1] < PE_DIM:
            pe = np.pad(pe, ((0, 0), (0, PE_DIM - pe.shape[1])))
        return pe.astype(np.float32)

    elif pe_type == "lappe":
        if eigenvectors_P is not None and eigenvectors_P.shape[0] == num_nodes:
            evecs_sq = eigenvectors_P ** 2  # [N, N]
            if evecs_sq.shape[1] >= PE_DIM:
                pe = evecs_sq[:, :PE_DIM]
            else:
                pe = np.pad(evecs_sq, ((0, 0), (0, PE_DIM - evecs_sq.shape[1])))
            return pe.astype(np.float32)
        else:
            pe = rwse[:, :PE_DIM]
            if pe.shape[1] < PE_DIM:
                pe = np.pad(pe, ((0, 0), (0, PE_DIM - pe.shape[1])))
            return pe.astype(np.float32)

    elif pe_type == "srwe":
        if eigenvalues_P is not None and eigenvalues_P.shape[0] == num_nodes:
            # Vectorized SRWE
            W = compute_srwe_batch(rwse, eigenvalues_P)  # [N_nodes, N_eig]

            # Histogram of each node's weights over eigenvalue bins in [-1, 1]
            bin_edges = np.linspace(-1.0, 1.0, PE_DIM + 1)
            bin_indices = np.digitize(eigenvalues_P, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, PE_DIM - 1)

            # Vectorized histogram: scatter-add weights into bins
            pe = np.zeros((num_nodes, PE_DIM), dtype=np.float64)
            for b in range(PE_DIM):
                mask = bin_indices == b
                if mask.any():
                    pe[:, b] = W[:, mask].sum(axis=1)
            return pe.astype(np.float32)
        else:
            pe = rwse[:, :PE_DIM]
            if pe.shape[1] < PE_DIM:
                pe = np.pad(pe, ((0, 0), (0, PE_DIM - pe.shape[1])))
            return pe.astype(np.float32)

    else:
        raise ValueError(f"Unknown pe_type: {pe_type}")


# ── MODEL: GCN + GlobalAttention ──
class GCN_GlobalAttention(nn.Module):
    """GCN with configurable depth and global attention pooling."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout

        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, edge_index, batch):
        h = F.relu(self.input_proj(x))
        for conv, bn in zip(self.convs, self.bns):
            h_new = conv(h, edge_index)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new  # residual

        gate = torch.sigmoid(self.gate_nn(h))
        h_weighted = h * gate
        h_graph = global_add_pool(h_weighted, batch)
        return self.head(h_graph)


# ── DATA PREPARATION WITH PE CACHING ──
def precompute_all_pe(graphs: list, eigen_cache: dict) -> dict:
    """Precompute PE arrays for all 3 PE types for all graphs. Returns pe_cache."""
    pe_cache = {}  # {(idx, pe_type): np.ndarray [N, PE_DIM]}
    for g in graphs:
        idx = g["_idx"]
        ev_P, evec_P = eigen_cache.get(idx, (None, None))
        for pe_type in PE_TYPES:
            pe = compute_pe(g, pe_type, eigenvalues_P=ev_P, eigenvectors_P=evec_P)
            pe_cache[(idx, pe_type)] = pe
    return pe_cache


def prepare_pyg_data_cached(graphs: list, pe_type: str, pe_cache: dict) -> list:
    """Convert parsed graph dicts to PyG Data using precomputed PE."""
    data_list = []
    for g in graphs:
        idx = g["_idx"]
        pe = pe_cache[(idx, pe_type)]
        x = np.concatenate([g["node_feat"], pe], axis=1)
        x_tensor = torch.tensor(x, dtype=torch.float32)
        edge_index_tensor = torch.tensor(g["edge_index"], dtype=torch.long)
        y_tensor = torch.tensor(g["target"], dtype=torch.float32)
        data = Data(x=x_tensor, edge_index=edge_index_tensor, y=y_tensor, num_nodes=g["num_nodes"])
        data_list.append(data)
    return data_list


def collate_batch(data_list):
    return Batch.from_data_list(data_list)


# ── TRAINING ──
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_samples = 0
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.batch)
        target = batch.y.view(pred.shape)
        loss = F.l1_loss(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * pred.size(0)
        n_samples += pred.size(0)
    return total_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_preds = []
    all_targets = []
    for batch in dataloader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.batch)
        target = batch.y.view(pred.shape)
        loss = F.l1_loss(pred, target)
        total_loss += loss.item() * pred.size(0)
        n_samples += pred.size(0)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(target.cpu().numpy())

    mae = total_loss / max(n_samples, 1)
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return mae, preds, targets


def make_dataloader(data_list, batch_size, shuffle=False):
    if shuffle:
        indices = np.random.permutation(len(data_list))
        data_list = [data_list[i] for i in indices]
    batches = []
    for i in range(0, len(data_list), batch_size):
        chunk = data_list[i:i + batch_size]
        batches.append(collate_batch(chunk))
    return batches


def train_model(train_data, val_data, input_dim, output_dim, depth, seed,
                max_epochs=MAX_EPOCHS, patience=PATIENCE, lr=LR,
                dropout=DROPOUT, device=DEVICE):
    """Train a GCN model with early stopping."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if HAS_GPU:
        torch.cuda.manual_seed(seed)

    actual_dropout = dropout
    actual_lr = lr
    if depth >= 6:
        actual_lr = 5e-4
        actual_dropout = 0.2

    model = GCN_GlobalAttention(
        input_dim=input_dim, hidden_dim=HIDDEN_DIM,
        output_dim=output_dim, num_layers=depth, dropout=actual_dropout,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=actual_lr)
    best_val_mae = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        train_loader = make_dataloader(train_data, BATCH_SIZE, shuffle=True)
        train_one_epoch(model, train_loader, optimizer, device)

        val_loader = make_dataloader(val_data, BATCH_SIZE, shuffle=False)
        val_mae, _, _ = evaluate(model, val_loader, device)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.debug(f"  Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 20 == 0 and time_check(f"epoch {epoch+1}"):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_mae


# ── SPLIT DATA ──
def split_zinc(graphs):
    train = [g for g in graphs if g["metadata_fold"] == 0]
    val = [g for g in graphs if g["metadata_fold"] == 1]
    test = [g for g in graphs if g["metadata_fold"] == 2]
    logger.info(f"ZINC split: train={len(train)}, val={len(val)}, test={len(test)}")
    if len(val) == 0 or len(test) == 0:
        logger.warning("ZINC folds not found properly, using random split")
        return random_split(graphs)
    return train, val, test


def random_split(graphs, train_frac=0.7, val_frac=0.15):
    np.random.seed(42)
    indices = np.random.permutation(len(graphs))
    n = len(graphs)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = [graphs[i] for i in indices[:n_train]]
    val = [graphs[i] for i in indices[n_train:n_train + n_val]]
    test = [graphs[i] for i in indices[n_train + n_val:]]
    return train, val, test


# ── SPECTRAL CACHE ──
def build_eigen_cache(graphs):
    """Precompute eigenvalues/vectors for all graphs."""
    logger.info(f"Building eigen cache for {len(graphs)} graphs...")
    cache = {}
    t0 = time.time()
    skipped = 0
    for i, g in enumerate(graphs):
        idx = g["_idx"]
        num_nodes = g["num_nodes"]
        if num_nodes > MAX_NODES_SPECTRAL:
            skipped += 1
            cache[idx] = (None, None)
            continue
        ev, evec = compute_normalized_adjacency_eigen(g["edge_index"], num_nodes)
        cache[idx] = (ev, evec)
        if i % 200 == 0 and i > 0:
            logger.debug(f"  Eigen cache: {i}/{len(graphs)} ({time.time()-t0:.1f}s)")
        if time_check("eigen_cache"):
            logger.warning("Eigen cache computation cut short by time budget")
            # Fill remaining with None
            for j in range(i + 1, len(graphs)):
                cache[graphs[j]["_idx"]] = (None, None)
            break

    elapsed = time.time() - t0
    logger.info(f"Eigen cache: {elapsed:.1f}s ({skipped} skipped for size, {len(cache)} total)")
    return cache


# ── MAIN EXPERIMENT ──
@logger.catch
def main(max_examples=None):
    """Run the full experiment."""
    global START_TIME
    START_TIME = time.time()

    logger.info("=" * 60)
    logger.info("Depth vs Encoding Aliasing Experiment")
    logger.info("=" * 60)

    # Load data
    if max_examples is not None and max_examples <= 3:
        raw_datasets = load_mini_data()
    else:
        raw_datasets = load_all_data()

    target_datasets = {
        "ZINC-subset": {
            "subsample": ZINC_SUBSAMPLE,
            "split_fn": split_zinc,
            "output_dim": 1,
        },
        "Peptides-struct": {
            "subsample": PEPTIDES_SUBSAMPLE,
            "split_fn": lambda g: random_split(g, 0.7, 0.15),
            "output_dim": 11,
        },
    }

    # Parse all graphs
    all_parsed = {}
    for ds_name, config in target_datasets.items():
        if ds_name not in raw_datasets:
            logger.warning(f"Dataset {ds_name} not found, skipping")
            continue

        examples = raw_datasets[ds_name]
        subsample = config["subsample"]
        if max_examples is not None:
            subsample = min(subsample, max_examples)

        np.random.seed(42)
        if len(examples) > subsample:
            indices = np.random.choice(len(examples), subsample, replace=False)
            examples = [examples[i] for i in sorted(indices)]
        else:
            examples = list(examples)

        logger.info(f"Parsing {ds_name}: {len(examples)} examples")
        parsed = []
        for i, ex in enumerate(examples):
            try:
                g = parse_graph(ex)
                g["_idx"] = i
                g["_dataset"] = ds_name
                parsed.append(g)
            except Exception:
                logger.exception(f"Failed to parse example {i} from {ds_name}")
                continue

        all_parsed[ds_name] = parsed
        logger.info(f"  Parsed {len(parsed)} graphs from {ds_name}")

    # Build eigen caches and precompute PE
    eigen_caches = {}
    pe_caches = {}
    for ds_name, graphs in all_parsed.items():
        logger.info(f"Building eigen cache for {ds_name}...")
        t0 = time.time()
        eigen_caches[ds_name] = build_eigen_cache(graphs)
        logger.info(f"  Eigen cache done in {time.time()-t0:.1f}s")

        logger.info(f"Precomputing PE for {ds_name}...")
        t1 = time.time()
        pe_caches[ds_name] = precompute_all_pe(graphs, eigen_caches[ds_name])
        logger.info(f"  PE precomputed in {time.time()-t1:.1f}s")

    # ── EXPERIMENT LOOP ──
    results = {}  # {(dataset, depth, pe_type): [mae per seed]}
    per_graph_preds = {}  # {(dataset, depth, pe_type, seed): {graph_idx: (pred, target)}}

    run_times = []
    total_runs_done = 0
    skipped_runs = 0
    active_seeds = list(SEEDS)
    active_depths = list(DEPTHS)
    active_pe_types = list(PE_TYPES)

    for ds_name in target_datasets:
        if ds_name not in all_parsed:
            continue

        graphs = all_parsed[ds_name]
        config = target_datasets[ds_name]
        output_dim = config["output_dim"]

        train_graphs, val_graphs, test_graphs = config["split_fn"](graphs)
        logger.info(f"\n{'='*40}")
        logger.info(f"Dataset: {ds_name}")
        logger.info(f"  Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

        pe_cache = pe_caches[ds_name]

        for depth in active_depths:
            for pe_type in active_pe_types:
                for seed in active_seeds:
                    if time_check(f"{ds_name}/d={depth}/pe={pe_type}/s={seed}"):
                        logger.warning("Skipping remaining runs due to time budget")
                        skipped_runs += 1
                        continue

                    run_key = (ds_name, depth, pe_type)
                    if run_key not in results:
                        results[run_key] = []

                    logger.info(f"  Run: {ds_name}, depth={depth}, pe={pe_type}, seed={seed}")
                    t_run = time.time()

                    try:
                        train_pyg = prepare_pyg_data_cached(train_graphs, pe_type, pe_cache)
                        val_pyg = prepare_pyg_data_cached(val_graphs, pe_type, pe_cache)
                        test_pyg = prepare_pyg_data_cached(test_graphs, pe_type, pe_cache)

                        input_dim = train_pyg[0].x.size(1) if train_pyg else PE_DIM + 1

                        model, best_val = train_model(
                            train_pyg, val_pyg, input_dim, output_dim, depth, seed
                        )

                        test_loader = make_dataloader(test_pyg, BATCH_SIZE, shuffle=False)
                        test_mae, test_preds, test_targets = evaluate(model, test_loader, DEVICE)

                        results[run_key].append(test_mae)
                        logger.info(f"    MAE: {test_mae:.6f} (val: {best_val:.6f})")

                        # Store per-graph predictions
                        pred_key = (ds_name, depth, pe_type, seed)
                        per_graph_preds[pred_key] = {}
                        for i, g in enumerate(test_graphs):
                            if i < len(test_preds):
                                p = test_preds[i].tolist() if test_preds[i].ndim > 0 else [float(test_preds[i])]
                                t = test_targets[i].tolist() if test_targets[i].ndim > 0 else [float(test_targets[i])]
                                per_graph_preds[pred_key][g["_idx"]] = (p, t)

                        del model
                        if HAS_GPU:
                            torch.cuda.empty_cache()

                    except torch.cuda.OutOfMemoryError:
                        logger.warning(f"  OOM at depth={depth}")
                        torch.cuda.empty_cache()
                        results[run_key].append(float("nan"))
                    except Exception:
                        logger.exception(f"  Failed: depth={depth}, pe={pe_type}, seed={seed}")
                        results[run_key].append(float("nan"))

                    elapsed_run = time.time() - t_run
                    run_times.append(elapsed_run)
                    total_runs_done += 1

                    # Adaptive fallback after first few runs
                    if total_runs_done == 6 and len(run_times) >= 6:
                        avg_time = np.mean(run_times)
                        total_planned = len(active_depths) * len(active_pe_types) * len(active_seeds) * 2
                        remaining = total_planned - total_runs_done
                        est_remaining = remaining * avg_time
                        logger.info(f"  Budget check: ~{est_remaining/60:.1f}min needed, {time_left()/60:.1f}min left")
                        if est_remaining > time_left() * 0.85:
                            if len(active_seeds) > 2:
                                active_seeds = [0, 1]
                                logger.warning("Reducing to 2 seeds for budget")
                            remaining = len(active_depths) * len(active_pe_types) * len(active_seeds) * 2 - total_runs_done
                            if remaining * avg_time > time_left() * 0.85:
                                active_depths = [2, 3, 4, 6]
                                logger.warning("Dropping depth=8 for budget")

    logger.info(f"\nCompleted {total_runs_done} runs, skipped {skipped_runs}")

    # ── ANALYSIS ──
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS")
    logger.info("=" * 60)

    # 1. Aggregate results
    depth_encoding_results = {}
    for (ds, d, pe), maes in results.items():
        valid = [m for m in maes if not np.isnan(m)]
        mean_mae = float(np.mean(valid)) if valid else float("nan")
        std_mae = float(np.std(valid)) if valid else float("nan")
        key_str = f"{ds}_depth{d}_{pe}"
        depth_encoding_results[key_str] = {
            "dataset": ds, "depth": d, "pe_type": pe,
            "mean_mae": round(mean_mae, 6), "std_mae": round(std_mae, 6),
            "n_seeds": len(valid), "raw_maes": [round(m, 6) for m in maes],
        }
        logger.info(f"  {key_str}: {mean_mae:.4f} +/- {std_mae:.4f}")

    # 2. Gap analysis
    gap_analysis = {}
    for ds_name in target_datasets:
        ds_gaps = {}
        for d in active_depths:
            rk = (ds_name, d, "rwse")
            lk = (ds_name, d, "lappe")
            if rk in results and lk in results:
                rv = [m for m in results[rk] if not np.isnan(m)]
                lv = [m for m in results[lk] if not np.isnan(m)]
                if rv and lv:
                    gap = float(np.mean(rv) - np.mean(lv))
                    ds_gaps[f"depth_{d}"] = round(gap, 6)
                    logger.info(f"  Gap {ds_name} depth={d}: {gap:.6f}")
        gap_analysis[ds_name] = ds_gaps

        if ds_gaps:
            sorted_depths = sorted(ds_gaps.keys())
            if len(sorted_depths) >= 2:
                first = ds_gaps[sorted_depths[0]]
                last = ds_gaps[sorted_depths[-1]]
                if abs(first) > 1e-8:
                    reduction = (first - last) / abs(first) * 100
                    gap_analysis[f"{ds_name}_gap_reduction_pct"] = round(reduction, 2)
                    gap_analysis[f"{ds_name}_gap_narrows"] = bool(reduction > 0)
                    logger.info(f"  Gap reduction {ds_name}: {reduction:.1f}%")
                else:
                    gap_analysis[f"{ds_name}_gap_reduction_pct"] = 0.0
                    gap_analysis[f"{ds_name}_gap_narrows"] = False

    # 3. SRI-gap correlation by depth
    sri_correlation_by_depth = {}
    for ds_name in target_datasets:
        if ds_name not in all_parsed:
            continue
        ds_corr = {}
        graphs = all_parsed[ds_name]
        _, _, test_graphs_ds = target_datasets[ds_name]["split_fn"](graphs)

        for d in active_depths:
            diff_errors = []
            sris = []
            for seed in active_seeds:
                rk = (ds_name, d, "rwse", seed)
                lk = (ds_name, d, "lappe", seed)
                if rk not in per_graph_preds or lk not in per_graph_preds:
                    continue
                rp = per_graph_preds[rk]
                lp = per_graph_preds[lk]
                for g in test_graphs_ds:
                    idx = g["_idx"]
                    if idx in rp and idx in lp:
                        re = float(np.mean(np.abs(np.array(rp[idx][0]) - np.array(rp[idx][1]))))
                        le = float(np.mean(np.abs(np.array(lp[idx][0]) - np.array(lp[idx][1]))))
                        diff_errors.append(re - le)
                        sris.append(g["sri"].get("K=20", g["sri"].get("K=16", 0.0)))

            if len(diff_errors) >= 5:
                try:
                    rho, pval = spearmanr(sris, diff_errors)
                    ds_corr[f"depth_{d}"] = {
                        "spearman_rho": round(float(rho), 6),
                        "p_value": round(float(pval), 6),
                        "n_samples": len(diff_errors),
                    }
                    logger.info(f"  SRI corr {ds_name} d={d}: rho={rho:.4f}, p={pval:.4f}")
                except Exception as e:
                    logger.warning(f"Spearman failed at depth={d}: {e}")
        sri_correlation_by_depth[ds_name] = ds_corr

    # 4. SRWE advantage by depth
    srwe_advantage_by_depth = {}
    for ds_name in target_datasets:
        ds_adv = {}
        for d in active_depths:
            sk = (ds_name, d, "srwe")
            rk = (ds_name, d, "rwse")
            if sk in results and rk in results:
                sv = [m for m in results[sk] if not np.isnan(m)]
                rv = [m for m in results[rk] if not np.isnan(m)]
                if sv and rv:
                    adv = float(np.mean(sv) - np.mean(rv))
                    ds_adv[f"depth_{d}"] = round(adv, 6)
                    logger.info(f"  SRWE adv {ds_name} d={d}: {adv:.6f}")
        srwe_advantage_by_depth[ds_name] = ds_adv

    # ── BUILD OUTPUT ──
    logger.info("Building output JSON...")
    output_datasets = []

    for ds_name in target_datasets:
        if ds_name not in all_parsed:
            continue
        graphs = all_parsed[ds_name]
        _, _, test_graphs_ds = target_datasets[ds_name]["split_fn"](graphs)

        examples = []
        for g in test_graphs_ds:
            idx = g["_idx"]
            target_val = g["target"].tolist()
            target_str = json.dumps(target_val) if len(target_val) > 1 else str(target_val[0])

            input_str = json.dumps({
                "graph_idx": idx,
                "num_nodes": g["num_nodes"],
                "delta_min": g["delta_min"],
                "sri_K20": g["sri"].get("K=20", 0.0),
                "dataset": ds_name,
            })

            entry = {
                "input": input_str,
                "output": target_str,
                "metadata_graph_idx": idx,
                "metadata_num_nodes": g["num_nodes"],
                "metadata_delta_min": g["delta_min"],
                "metadata_sri_K20": g["sri"].get("K=20", 0.0),
            }

            # Add predictions averaged over seeds
            for d in active_depths:
                for pe in active_pe_types:
                    preds = []
                    for seed in active_seeds:
                        pk = (ds_name, d, pe, seed)
                        if pk in per_graph_preds and idx in per_graph_preds[pk]:
                            preds.append(per_graph_preds[pk][idx][0])
                    if preds:
                        avg = np.mean(preds, axis=0).tolist()
                        ps = json.dumps([round(v, 6) for v in avg]) if len(avg) > 1 else str(round(avg[0], 6))
                        entry[f"predict_depth{d}_{pe}"] = ps

            examples.append(entry)

        output_datasets.append({"dataset": ds_name, "examples": examples})

    # Include other datasets for completeness
    for extra_name in ["Peptides-func", "Synthetic-aliased-pairs"]:
        if extra_name in raw_datasets:
            exs = raw_datasets[extra_name][:3]
            examples = []
            for ex in exs:
                examples.append({
                    "input": ex["input"][:500],
                    "output": ex["output"],
                    "predict_not_evaluated": "Not part of the depth-vs-encoding experiment; included for completeness.",
                })
            output_datasets.append({"dataset": extra_name, "examples": examples})

    output = {
        "metadata": {
            "experiment": "Depth vs Encoding Aliasing",
            "description": "Tests whether increasing GNN depth compensates for RWSE spectral aliasing by narrowing RWSE-vs-LapPE gap",
            "depths": active_depths,
            "pe_types": active_pe_types,
            "seeds": active_seeds,
            "hidden_dim": HIDDEN_DIM,
            "pe_dim": PE_DIM,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "zinc_subsample": ZINC_SUBSAMPLE,
            "peptides_subsample": PEPTIDES_SUBSAMPLE,
            "total_runs": total_runs_done,
            "skipped_runs": skipped_runs,
            "total_time_sec": round(time.time() - START_TIME, 1),
            "depth_encoding_results": depth_encoding_results,
            "gap_analysis": gap_analysis,
            "sri_correlation_by_depth": sri_correlation_by_depth,
            "srwe_advantage_by_depth": srwe_advantage_by_depth,
        },
        "datasets": output_datasets,
    }

    output_path = WORKSPACE / "method_out.json"
    output_path.write_text(json.dumps(output, indent=2))
    fsize = output_path.stat().st_size
    logger.info(f"Output saved: {output_path} ({fsize/1e6:.2f} MB)")
    logger.info(f"Total time: {time.time() - START_TIME:.1f}s")

    if fsize > 100 * 1e6:
        logger.warning("Output exceeds 100MB — splitting needed")

    return output


if __name__ == "__main__":
    main()
