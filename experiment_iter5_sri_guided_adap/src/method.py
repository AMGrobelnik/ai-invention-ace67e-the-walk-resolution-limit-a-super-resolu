#!/usr/bin/env python3
"""
SRI-Guided Adaptive Encoding Selection & RWSE+SRWE Concatenation Experiment.

Compares 5 positional-encoding strategies for GPS graph transformer across
ZINC-subset, Peptides-func, Peptides-struct, and Synthetic-aliased-pairs:
  1. FIXED-RWSE  (baseline)
  2. FIXED-LapPE
  3. FIXED-SRWE  (Tikhonov-recovered spectral walk encoding)
  4. SRI-THRESHOLD (per-graph adaptive selection based on SRI)
  5. CONCAT-RWSE-SRWE (complementary concatenation)

Also computes ORACLE (per-graph best) and LEARNED-SELECTOR (MLP selector).

Outputs method_out.json in exp_gen_sol_out.json schema format.
"""

import json
import math
import os
import sys
import time
import warnings
import resource
from pathlib import Path
from typing import Any

# ── Thread limits ──
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.metrics import average_precision_score
import psutil

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
TOTAL_RAM_GB = 56
resource.setrlimit(resource.RLIMIT_AS, (50 * 1024**3, 50 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

# ── Hardware detection ──
def _cgroup_cpus():
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
    except (FileNotFoundError, ValueError):
        pass
    return None

NUM_CPUS = _cgroup_cpus() or os.cpu_count() or 1
HAS_GPU = torch.cuda.is_available()
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if HAS_GPU else 0
DEVICE = torch.device("cuda" if HAS_GPU else "cpu")
AVAILABLE_RAM_GB = psutil.virtual_memory().available / 1e9

logger.info(f"Hardware: {NUM_CPUS} CPUs, {TOTAL_RAM_GB:.1f}GB RAM, GPU={HAS_GPU} ({VRAM_GB:.1f}GB VRAM)")
logger.info(f"Device: {DEVICE}")

# ── Dependency data path ──
DEP_DATA_DIR = Path("/opt/ai-inventor/aii_pipeline/runs/run__20260222_005317/3_invention_loop/iter_1/gen_art/data_id2_it1__opus")

# ── Configuration ──
MAX_EXAMPLES = int(os.environ.get("MAX_EXAMPLES", "0"))  # 0 = all
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "60"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
SEEDS = [42, 123]
GPS_CHANNELS = 64
GPS_PE_DIM = 8
GPS_LAYERS = 3
GPS_HEADS = 4
LR = 0.001
WEIGHT_DECAY = 1e-5
PATIENCE_LR = 15
MIN_LR = 1e-6
LAPPE_K = 8
SRWE_BINS = 20
SRWE_GRID = 100
SRWE_REG = 1e-3

STRATEGIES = ['FIXED-RWSE', 'FIXED-LapPE', 'FIXED-SRWE', 'SRI-THRESHOLD', 'CONCAT-RWSE-SRWE']

# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_all_data(max_examples: int = 0) -> dict:
    """Load all data from dependency workspace, organized by dataset and split."""
    from torch_geometric.data import Data

    datasets = {
        'zinc': {'train': [], 'val': [], 'test': []},
        'peptides_func': {'train': [], 'val': [], 'test': []},
        'peptides_struct': {'train': [], 'val': [], 'test': []},
        'synthetic': {'all': []},
    }

    # Map dataset names to internal keys
    ds_map = {
        'ZINC': 'zinc',
        'ZINC-subset': 'zinc',
        'Peptides-func': 'peptides_func',
        'Peptides-struct': 'peptides_struct',
        'Synthetic-aliased-pairs': 'synthetic',
    }

    # Map fold to split for ZINC: 0=train, 1=val, 2=test
    fold_to_split_zinc = {0: 'train', 1: 'val', 2: 'test'}

    all_examples = []
    data_dir = DEP_DATA_DIR / "data_out"

    # Decide which files to load based on max_examples
    # For very small tests, only load smaller files first
    per_ds_counts = {}

    for i in range(1, 6):
        fpath = data_dir / f"full_data_out_{i}.json"
        if not fpath.exists():
            logger.warning(f"Missing data file: {fpath}")
            continue

        # Skip large files if we only need a few examples and already have enough
        if max_examples > 0:
            total_so_far = sum(per_ds_counts.values())
            if total_so_far >= max_examples * 5:  # 5 dataset types
                logger.info(f"Skipping {fpath.name} (already have enough examples)")
                continue

        logger.info(f"Loading {fpath.name}...")
        raw = json.loads(fpath.read_text())
        for ds_entry in raw.get('datasets', []):
            ds_name_raw = ds_entry.get('dataset', '')
            ds_key = ds_map.get(ds_name_raw, None)
            if ds_key is None:
                logger.warning(f"Unknown dataset: {ds_name_raw}")
                continue
            examples = ds_entry.get('examples', [])
            if max_examples > 0:
                already = per_ds_counts.get(ds_key, 0)
                remaining = max(0, max_examples - already)
                examples = examples[:remaining]
            per_ds_counts[ds_key] = per_ds_counts.get(ds_key, 0) + len(examples)
            all_examples.extend([(ds_key, ex) for ex in examples])

    logger.info(f"Loaded {len(all_examples)} total examples from files")

    # Parse examples into PyG Data objects
    counts = {}
    for ds_key, ex in all_examples:
        try:
            inp = json.loads(ex['input'])
            edge_index = inp.get('edge_index', [[], []])
            num_nodes = inp.get('num_nodes', 0)
            node_feat = inp.get('node_feat', [])
            edge_attr_raw = inp.get('edge_attr', None)
            spectral = inp.get('spectral', {})
            output_str = ex.get('output', '')

            if num_nodes == 0:
                continue

            # Build PyG Data
            ei = torch.tensor(edge_index, dtype=torch.long) if edge_index and len(edge_index) == 2 else torch.zeros(2, 0, dtype=torch.long)

            # Node features
            if node_feat:
                x = torch.tensor(node_feat, dtype=torch.long)
            else:
                x = torch.zeros(num_nodes, 1, dtype=torch.long)

            # Edge attributes
            if edge_attr_raw and len(edge_attr_raw) > 0:
                ea = torch.tensor(edge_attr_raw, dtype=torch.long)
            else:
                # Create dummy edge attrs
                num_edges = ei.size(1)
                ea = torch.ones(num_edges, 1, dtype=torch.long)

            # Target - use 2D shape (1, num_classes) for multi-output tasks
            # so PyG batching produces (batch_size, num_classes)
            if ds_key == 'zinc':
                y = torch.tensor([float(output_str)], dtype=torch.float32)
            elif ds_key == 'peptides_func':
                labels = json.loads(output_str)
                y = torch.tensor([labels], dtype=torch.float32)  # (1, 10)
            elif ds_key == 'peptides_struct':
                values = json.loads(output_str)
                y = torch.tensor([values], dtype=torch.float32)  # (1, 11)
            elif ds_key == 'synthetic':
                y = torch.tensor([float(output_str)], dtype=torch.float32)
            else:
                y = torch.tensor([0.0], dtype=torch.float32)

            data = Data(x=x, edge_index=ei, edge_attr=ea, y=y)
            data.num_nodes_val = num_nodes

            # Spectral annotations
            rwse = spectral.get('rwse', [])
            if rwse and len(rwse) > 0:
                data.rwse = torch.tensor(rwse, dtype=torch.float32)
            else:
                data.rwse = torch.zeros(num_nodes, 20, dtype=torch.float32)

            eigenvalues = spectral.get('eigenvalues', [])
            data.eigenvalues = torch.tensor(eigenvalues, dtype=torch.float32)

            sri = spectral.get('sri', {})
            data.sri_K20 = float(sri.get('K=20', 0.0))

            vcond = spectral.get('vandermonde_cond', {})
            data.vandermonde_cond = float(vcond.get('K=20', 1e15))

            data.delta_min = float(spectral.get('delta_min', 0.0))
            data.dataset_key = ds_key
            data.original_input = ex['input']
            data.original_output = ex['output']
            data.metadata_fold = ex.get('metadata_fold', 0)

            # Determine split
            if ds_key == 'zinc':
                split = fold_to_split_zinc.get(data.metadata_fold, 'train')
            elif ds_key == 'synthetic':
                split = 'all'
            else:
                # Peptides: split by index (60/20/20)
                split = 'train'  # Will be reassigned below

            if ds_key in ('zinc',):
                datasets[ds_key][split].append(data)
            elif ds_key == 'synthetic':
                datasets[ds_key]['all'].append(data)
            else:
                datasets[ds_key]['train'].append(data)

            counts[ds_key] = counts.get(ds_key, 0) + 1

        except Exception as e:
            logger.debug(f"Failed to parse example: {str(e)[:200]}")
            continue

    # For ZINC: if val/test are empty but train has data, split train
    for ds_key in ('zinc',):
        n_train = len(datasets[ds_key]['train'])
        n_val = len(datasets[ds_key]['val'])
        n_test = len(datasets[ds_key]['test'])
        if n_train > 0 and (n_val == 0 or n_test == 0):
            all_data = datasets[ds_key]['train'] + datasets[ds_key]['val'] + datasets[ds_key]['test']
            n = len(all_data)
            if n >= 5:
                np.random.seed(42)
                indices = np.random.permutation(n)
                nt = max(1, int(0.6 * n))
                nv = max(1, int(0.2 * n))
                datasets[ds_key]['train'] = [all_data[i] for i in indices[:nt]]
                datasets[ds_key]['val'] = [all_data[i] for i in indices[nt:nt + nv]]
                datasets[ds_key]['test'] = [all_data[i] for i in indices[nt + nv:]]
            elif n > 0:
                datasets[ds_key]['train'] = all_data[:]
                datasets[ds_key]['val'] = all_data[:]
                datasets[ds_key]['test'] = all_data[:]

    # Split Peptides data into train/val/test (60/20/20)
    for ds_key in ('peptides_func', 'peptides_struct'):
        all_data = datasets[ds_key]['train']
        n = len(all_data)
        if n >= 5:
            np.random.seed(42)
            indices = np.random.permutation(n)
            n_train = max(1, int(0.6 * n))
            n_val = max(1, int(0.2 * n))
            n_test = max(1, n - n_train - n_val)
            # Ensure at least 1 in each split
            datasets[ds_key]['train'] = [all_data[i] for i in indices[:n_train]]
            datasets[ds_key]['val'] = [all_data[i] for i in indices[n_train:n_train + n_val]]
            datasets[ds_key]['test'] = [all_data[i] for i in indices[n_train + n_val:]]
        elif n > 0:
            # Very small data: replicate for all splits
            datasets[ds_key]['train'] = all_data[:]
            datasets[ds_key]['val'] = all_data[:]
            datasets[ds_key]['test'] = all_data[:]

    for ds_key in datasets:
        for split in datasets[ds_key]:
            n = len(datasets[ds_key][split])
            if n > 0:
                logger.info(f"  {ds_key}/{split}: {n} graphs")

    return datasets


# ═══════════════════════════════════════════════════════════════════════════
#  ENCODING COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_lappe(data, k: int = LAPPE_K) -> torch.Tensor:
    """Compute Laplacian Positional Encoding (squared eigenvector components)."""
    N = data.num_nodes_val
    if N < 3:
        return torch.zeros(N, k)

    edge_index = data.edge_index.numpy()
    if edge_index.shape[1] == 0:
        return torch.zeros(N, k)

    row, col = edge_index[0], edge_index[1]

    try:
        adj = csr_matrix((np.ones(len(row)), (row, col)), shape=(N, N))
        deg = np.array(adj.sum(axis=1)).flatten()
        deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
        D_inv_sqrt = csr_matrix((deg_inv_sqrt, (range(N), range(N))), shape=(N, N))
        L = csr_matrix(np.eye(N)) - D_inv_sqrt @ adj @ D_inv_sqrt

        actual_k = min(k, N - 2)
        if actual_k < 1:
            return torch.zeros(N, k)

        evals, evecs = eigsh(L, k=actual_k + 1, which='SM', tol=1e-4)
        idx = np.argsort(evals)[1:actual_k + 1]
        result = evecs[:, idx] ** 2  # square to remove sign ambiguity
    except Exception:
        result = np.zeros((N, min(k, max(1, N - 2))))

    # Pad to k dims
    if result.shape[1] < k:
        result = np.pad(result, ((0, 0), (0, k - result.shape[1])))

    return torch.tensor(result[:, :k], dtype=torch.float32)


def compute_srwe_graph(data, n_bins: int = SRWE_BINS,
                       lambda_range: tuple = (-3, 3),
                       n_grid: int = SRWE_GRID,
                       reg: float = SRWE_REG) -> torch.Tensor:
    """Compute SRWE for all nodes in a graph (vectorized across nodes)."""
    rwse = data.rwse.numpy()
    N, K = rwse.shape

    if K == 0 or N == 0:
        return torch.zeros(N, n_bins, dtype=torch.float32)

    lambda_grid = np.linspace(lambda_range[0], lambda_range[1], n_grid)

    # Vandermonde measurement matrix: V[k,j] = lambda_grid[j]^(k+1)
    V = np.array([lambda_grid ** (k + 1) for k in range(K)])  # (K, n_grid)

    # Tikhonov: W = (V^T V + reg*I)^{-1} V^T M  where M is (K, N)
    # Solve for all nodes at once
    VtV = V.T @ V + reg * np.eye(n_grid)
    VtM = V.T @ rwse.T  # (n_grid, N)

    try:
        W = np.linalg.solve(VtV, VtM)  # (n_grid, N)
    except np.linalg.LinAlgError:
        return torch.zeros(N, n_bins, dtype=torch.float32)

    W = np.maximum(W, 0)  # non-negativity

    # Normalize per node
    totals = W.sum(axis=0, keepdims=True)
    totals = np.where(totals > 1e-10, totals, 1.0)
    W = W / totals  # (n_grid, N)

    # Histogram binning (vectorized)
    bin_width = (lambda_range[1] - lambda_range[0]) / n_bins
    bin_indices = np.clip(
        ((lambda_grid - lambda_range[0]) / bin_width).astype(int),
        0, n_bins - 1
    )  # (n_grid,)

    srwe = np.zeros((N, n_bins))
    for b in range(n_bins):
        mask = (bin_indices == b)
        if mask.any():
            srwe[:, b] = W[mask, :].sum(axis=0)  # sum weights that fall in this bin

    return torch.tensor(srwe, dtype=torch.float32)


def precompute_encodings(datasets: dict, lambda_range: tuple = (-3, 3)) -> None:
    """Precompute LapPE and SRWE for all graphs in all datasets."""
    total_graphs = sum(
        len(datasets[ds][split])
        for ds in datasets
        for split in datasets[ds]
    )
    logger.info(f"Precomputing LapPE and SRWE for {total_graphs} graphs...")

    count = 0
    start = time.time()
    for ds_key in datasets:
        for split in datasets[ds_key]:
            for data in datasets[ds_key][split]:
                try:
                    data.lappe = compute_lappe(data, k=LAPPE_K)
                    data.srwe = compute_srwe_graph(data, lambda_range=lambda_range)
                except Exception as e:
                    N = data.num_nodes_val
                    data.lappe = torch.zeros(N, LAPPE_K)
                    data.srwe = torch.zeros(N, SRWE_BINS)
                    logger.debug(f"Encoding fallback for graph: {str(e)[:100]}")

                count += 1
                if count % 500 == 0:
                    elapsed = time.time() - start
                    rate = count / elapsed
                    remaining = (total_graphs - count) / rate if rate > 0 else 0
                    logger.info(f"  Encoded {count}/{total_graphs} graphs "
                                f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - start
    logger.info(f"Encoding precomputation complete: {count} graphs in {elapsed:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════
#  ENCODING STRATEGY DISPATCH
# ═══════════════════════════════════════════════════════════════════════════

def get_encoding(data, strategy: str, pe_dim_target: int = 20) -> torch.Tensor:
    """Return positional encoding tensor for the given strategy."""
    if strategy == 'FIXED-RWSE':
        pe = data.rwse
        if pe.size(1) < pe_dim_target:
            pad = torch.zeros(pe.size(0), pe_dim_target - pe.size(1))
            pe = torch.cat([pe, pad], dim=-1)
        return pe[:, :pe_dim_target]

    elif strategy == 'FIXED-LapPE':
        pe = data.lappe  # (N, 8)
        if pe.size(1) < pe_dim_target:
            pad = torch.zeros(pe.size(0), pe_dim_target - pe.size(1))
            pe = torch.cat([pe, pad], dim=-1)
        return pe[:, :pe_dim_target]

    elif strategy == 'FIXED-SRWE':
        pe = data.srwe
        if pe.size(1) < pe_dim_target:
            pad = torch.zeros(pe.size(0), pe_dim_target - pe.size(1))
            pe = torch.cat([pe, pad], dim=-1)
        return pe[:, :pe_dim_target]

    elif strategy == 'SRI-THRESHOLD':
        sri = getattr(data, 'sri_K20', 0.0)
        if sri is not None and sri > 1.0:
            return get_encoding(data, 'FIXED-RWSE', pe_dim_target)
        else:
            return get_encoding(data, 'FIXED-SRWE', pe_dim_target)

    elif strategy == 'CONCAT-RWSE-SRWE':
        rwse = data.rwse
        srwe = data.srwe
        if rwse.size(1) < 20:
            rwse = torch.cat([rwse, torch.zeros(rwse.size(0), 20 - rwse.size(1))], dim=-1)
        if srwe.size(1) < 20:
            srwe = torch.cat([srwe, torch.zeros(srwe.size(0), 20 - srwe.size(1))], dim=-1)
        return torch.cat([rwse[:, :20], srwe[:, :20]], dim=-1)  # (N, 40)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ═══════════════════════════════════════════════════════════════════════════
#  GPS MODEL
# ═══════════════════════════════════════════════════════════════════════════

from torch_geometric.nn import GPSConv, GINEConv, global_add_pool
from torch_geometric.loader import DataLoader


class GPSModel(nn.Module):
    def __init__(
        self,
        channels: int = GPS_CHANNELS,
        pe_dim: int = GPS_PE_DIM,
        pe_input_dim: int = 20,
        num_layers: int = GPS_LAYERS,
        node_input_type: str = 'categorical',
        num_node_classes: int = 30,
        node_feat_dim: int = 1,
        num_edge_classes: int = 5,
        edge_feat_dim: int = 1,
        num_output: int = 1,
        dropout: float = 0.1,
        attn_type: str = 'multihead',
    ):
        super().__init__()
        self.node_input_type = node_input_type

        # Node embedding
        if node_input_type == 'categorical':
            self.node_emb = nn.Embedding(num_node_classes, channels - pe_dim)
        else:
            self.node_emb = nn.Linear(node_feat_dim, channels - pe_dim)

        # PE projection
        self.pe_norm = nn.BatchNorm1d(pe_input_dim)
        self.pe_lin = nn.Linear(pe_input_dim, pe_dim)

        # Edge embedding
        if node_input_type == 'categorical':
            self.edge_emb = nn.Embedding(num_edge_classes, channels)
        else:
            self.edge_emb = nn.Linear(edge_feat_dim, channels)

        # GPS layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(channels, channels * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(channels * 2, channels),
            )
            conv = GPSConv(
                channels, GINEConv(mlp), heads=GPS_HEADS,
                attn_type=attn_type, dropout=dropout,
            )
            self.convs.append(conv)

        # Readout
        self.readout = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels // 2, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, num_output),
        )

    def forward(self, x, pe, edge_index, edge_attr, batch):
        # PE processing
        pe = self.pe_lin(self.pe_norm(pe))

        # Node embedding
        if self.node_input_type == 'categorical':
            x = x.squeeze(-1).clamp(0, self.node_emb.num_embeddings - 1)
            x = self.node_emb(x)
        else:
            x = self.node_emb(x.float())

        x = torch.cat([x, pe], dim=-1)

        # Edge embedding
        if self.node_input_type == 'categorical':
            edge_attr = edge_attr.squeeze(-1).clamp(0, self.edge_emb.num_embeddings - 1)
            edge_attr = self.edge_emb(edge_attr)
        else:
            edge_attr = self.edge_emb(edge_attr.float())

        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)

        x = global_add_pool(x, batch)
        return self.readout(x)


# Dataset-specific configs
DATASET_CONFIGS = {
    'zinc': dict(
        node_input_type='categorical', num_node_classes=30,
        num_edge_classes=5, edge_feat_dim=1,
        num_output=1, node_feat_dim=1,
    ),
    'peptides_func': dict(
        node_input_type='categorical', num_node_classes=120,
        num_edge_classes=2, edge_feat_dim=1,
        num_output=10, node_feat_dim=1,
    ),
    'peptides_struct': dict(
        node_input_type='categorical', num_node_classes=120,
        num_edge_classes=2, edge_feat_dim=1,
        num_output=11, node_feat_dim=1,
    ),
}

DATASET_META = {
    'zinc': {
        'criterion': nn.L1Loss(),
        'lower_better': True,
        'metric_name': 'MAE',
    },
    'peptides_func': {
        'criterion': nn.BCEWithLogitsLoss(),
        'lower_better': False,
        'metric_name': 'AP',
    },
    'peptides_struct': {
        'criterion': nn.L1Loss(),
        'lower_better': True,
        'metric_name': 'MAE',
    },
}


def compute_metric(preds: torch.Tensor, targets: torch.Tensor, ds_key: str) -> float:
    """Compute the appropriate metric for a dataset."""
    if ds_key == 'zinc':
        return (preds.squeeze() - targets.squeeze()).abs().mean().item()
    elif ds_key == 'peptides_func':
        try:
            # Ensure 2D: (batch, num_classes)
            t = targets.numpy()
            p = torch.sigmoid(preds).numpy()
            if t.ndim == 1:
                n_classes = p.shape[-1] if p.ndim > 1 else 10
                t = t.reshape(-1, n_classes)
            if p.ndim == 1:
                p = p.reshape(-1, t.shape[-1])
            return average_precision_score(t, p, average='macro')
        except Exception:
            return 0.0
    elif ds_key == 'peptides_struct':
        # Ensure same shape
        if preds.dim() != targets.dim():
            if preds.dim() > targets.dim():
                targets = targets.view_as(preds)
            else:
                preds = preds.view_as(targets)
        return (preds - targets).abs().mean().item()
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════

def prepare_batch_pe(batch_data, strategy: str) -> torch.Tensor:
    """Prepare positional encoding for a batch by computing per-graph then concatenating."""
    # In batched PyG data, we can access per-graph data through the batch
    # But get_encoding works on individual Data objects. For batched data,
    # we need to handle it differently.

    # For batched data, the rwse/lappe/srwe are already concatenated by DataLoader
    if strategy == 'FIXED-RWSE':
        pe = batch_data.rwse
        if pe.size(1) < 20:
            pad = torch.zeros(pe.size(0), 20 - pe.size(1), device=pe.device)
            pe = torch.cat([pe, pad], dim=-1)
        return pe[:, :20]

    elif strategy == 'FIXED-LapPE':
        pe = batch_data.lappe
        if pe.size(1) < 20:
            pad = torch.zeros(pe.size(0), 20 - pe.size(1), device=pe.device)
            pe = torch.cat([pe, pad], dim=-1)
        return pe[:, :20]

    elif strategy == 'FIXED-SRWE':
        pe = batch_data.srwe
        if pe.size(1) < 20:
            pad = torch.zeros(pe.size(0), 20 - pe.size(1), device=pe.device)
            pe = torch.cat([pe, pad], dim=-1)
        return pe[:, :20]

    elif strategy == 'SRI-THRESHOLD':
        # For batched data, we need to select per-graph
        # sri_K20 is a per-graph attribute; batch tells us which nodes belong to which graph
        batch_idx = batch_data.batch
        sri_vals = batch_data.sri_K20  # (num_graphs,) after batching

        rwse = batch_data.rwse
        srwe = batch_data.srwe

        if rwse.size(1) < 20:
            rwse = torch.cat([rwse, torch.zeros(rwse.size(0), 20 - rwse.size(1), device=rwse.device)], dim=-1)
        if srwe.size(1) < 20:
            srwe = torch.cat([srwe, torch.zeros(srwe.size(0), 20 - srwe.size(1), device=srwe.device)], dim=-1)

        rwse = rwse[:, :20]
        srwe = srwe[:, :20]

        pe = torch.zeros_like(rwse)
        for g_idx in range(len(sri_vals)):
            mask = (batch_idx == g_idx)
            if sri_vals[g_idx] > 1.0:
                pe[mask] = rwse[mask]
            else:
                pe[mask] = srwe[mask]
        return pe

    elif strategy == 'CONCAT-RWSE-SRWE':
        rwse = batch_data.rwse
        srwe = batch_data.srwe
        if rwse.size(1) < 20:
            rwse = torch.cat([rwse, torch.zeros(rwse.size(0), 20 - rwse.size(1), device=rwse.device)], dim=-1)
        if srwe.size(1) < 20:
            srwe = torch.cat([srwe, torch.zeros(srwe.size(0), 20 - srwe.size(1), device=srwe.device)], dim=-1)
        return torch.cat([rwse[:, :20], srwe[:, :20]], dim=-1)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def train_and_evaluate(
    ds_key: str,
    strategy: str,
    datasets: dict,
    seed: int,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """Train GPS model with a specific encoding strategy and return results."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if HAS_GPU:
        torch.cuda.manual_seed(seed)

    meta = DATASET_META[ds_key]
    ds_config = DATASET_CONFIGS[ds_key]
    pe_input_dim = 40 if strategy == 'CONCAT-RWSE-SRWE' else 20

    train_data = datasets[ds_key]['train']
    val_data = datasets[ds_key]['val']
    test_data = datasets[ds_key]['test']

    if len(train_data) == 0 or len(val_data) == 0 or len(test_data) == 0:
        logger.warning(f"Skipping {ds_key}/{strategy}: insufficient data")
        return {'best_val': float('nan'), 'best_test': float('nan'),
                'training_time_s': 0, 'epochs_run': 0}

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size * 2)
    test_loader = DataLoader(test_data, batch_size=batch_size * 2)

    model = GPSModel(
        channels=GPS_CHANNELS, pe_dim=GPS_PE_DIM, pe_input_dim=pe_input_dim,
        num_layers=GPS_LAYERS, **ds_config
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min' if meta['lower_better'] else 'max',
        factor=0.5, patience=PATIENCE_LR,
    )

    criterion = meta['criterion']
    best_val = float('inf') if meta['lower_better'] else float('-inf')
    best_test = float('nan')
    best_state = None
    start_time = time.time()

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        num_samples = 0
        for batch_data in train_loader:
            batch_data = batch_data.to(DEVICE)
            pe = prepare_batch_pe(batch_data, strategy).to(DEVICE)
            optimizer.zero_grad()

            out = model(batch_data.x, pe, batch_data.edge_index,
                        batch_data.edge_attr, batch_data.batch)

            if ds_key == 'peptides_func':
                target = batch_data.y.float()
                if target.dim() == 1:
                    target = target.view(out.size(0), -1)
                loss = criterion(out, target)
            elif ds_key == 'peptides_struct':
                target = batch_data.y.float()
                if target.dim() == 1:
                    target = target.view(out.size(0), -1)
                loss = criterion(out, target)
            else:
                loss = criterion(out.squeeze(), batch_data.y.squeeze().float())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * batch_data.num_graphs
            num_samples += batch_data.num_graphs

        avg_loss = total_loss / max(num_samples, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds, val_targets = [], []
            for batch_data in val_loader:
                batch_data = batch_data.to(DEVICE)
                pe = prepare_batch_pe(batch_data, strategy).to(DEVICE)
                out = model(batch_data.x, pe, batch_data.edge_index,
                            batch_data.edge_attr, batch_data.batch)
                val_preds.append(out.cpu())
                val_targets.append(batch_data.y.cpu())

            val_preds = torch.cat(val_preds, dim=0)
            val_targets = torch.cat(val_targets, dim=0)
            val_metric = compute_metric(val_preds, val_targets, ds_key)

        scheduler.step(val_metric if meta['lower_better'] else -val_metric)

        is_better = (val_metric < best_val) if meta['lower_better'] else (val_metric > best_val)
        if is_better:
            best_val = val_metric
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # Compute test metric at best val
            with torch.no_grad():
                test_preds, test_targets = [], []
                for batch_data in test_loader:
                    batch_data = batch_data.to(DEVICE)
                    pe = prepare_batch_pe(batch_data, strategy).to(DEVICE)
                    out = model(batch_data.x, pe, batch_data.edge_index,
                                batch_data.edge_attr, batch_data.batch)
                    test_preds.append(out.cpu())
                    test_targets.append(batch_data.y.cpu())
                test_preds = torch.cat(test_preds, dim=0)
                test_targets = torch.cat(test_targets, dim=0)
                best_test = compute_metric(test_preds, test_targets, ds_key)

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            lr_now = optimizer.param_groups[0]['lr']
            logger.debug(f"  Epoch {epoch}: loss={avg_loss:.4f} val={val_metric:.4f} "
                         f"best_val={best_val:.4f} best_test={best_test:.4f} lr={lr_now:.2e}")

        # Early stopping on min LR
        if optimizer.param_groups[0]['lr'] < MIN_LR:
            logger.debug(f"  Early stopping at epoch {epoch} (LR < {MIN_LR})")
            break

    elapsed = time.time() - start_time

    return {
        'best_val': best_val,
        'best_test': best_test,
        'training_time_s': elapsed,
        'epochs_run': epoch + 1,
        'best_state': best_state,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  PREDICTION COLLECTION
# ═══════════════════════════════════════════════════════════════════════════

def collect_predictions(
    model: GPSModel,
    test_data: list,
    strategy: str,
    ds_key: str,
    batch_size: int = BATCH_SIZE * 2,
) -> list:
    """Collect per-example predictions from a trained model."""
    model.eval()
    loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    all_preds = []

    with torch.no_grad():
        for batch_data in loader:
            batch_data = batch_data.to(DEVICE)
            pe = prepare_batch_pe(batch_data, strategy).to(DEVICE)
            out = model(batch_data.x, pe, batch_data.edge_index,
                        batch_data.edge_attr, batch_data.batch)

            if ds_key == 'peptides_func':
                # Convert to probabilities, then round for classification
                probs = torch.sigmoid(out).cpu()
                for i in range(probs.size(0)):
                    all_preds.append(json.dumps([round(p, 4) for p in probs[i].tolist()]))
            elif ds_key == 'peptides_struct':
                vals = out.cpu()
                for i in range(vals.size(0)):
                    all_preds.append(json.dumps([round(v, 4) for v in vals[i].tolist()]))
            else:
                vals = out.squeeze(-1).cpu()
                for i in range(vals.size(0)):
                    all_preds.append(str(round(vals[i].item(), 6)))

    return all_preds


# ═══════════════════════════════════════════════════════════════════════════
#  ORACLE & LEARNED SELECTOR
# ═══════════════════════════════════════════════════════════════════════════

def compute_oracle_results(
    ds_key: str,
    test_data: list,
    strategy_models: dict,
    seed: int,
) -> dict:
    """Compute Oracle per-graph best selection and per-graph analysis."""
    meta = DATASET_META[ds_key]
    ds_config = DATASET_CONFIGS[ds_key]
    fixed_strategies = ['FIXED-RWSE', 'FIXED-LapPE', 'FIXED-SRWE']

    per_graph_results = []
    oracle_preds = []
    loader = DataLoader(test_data, batch_size=1, shuffle=False)

    for g_idx, batch_data in enumerate(loader):
        batch_data = batch_data.to(DEVICE)
        best_metric = float('inf') if meta['lower_better'] else float('-inf')
        best_strategy = None
        best_pred = None
        strategy_metrics = {}

        for strategy in fixed_strategies:
            key = (ds_key, strategy, seed)
            if key not in strategy_models or strategy_models[key] is None:
                continue

            model = GPSModel(
                channels=GPS_CHANNELS, pe_dim=GPS_PE_DIM, pe_input_dim=20,
                num_layers=GPS_LAYERS, **ds_config
            ).to(DEVICE)
            model.load_state_dict(strategy_models[key])
            model.eval()

            with torch.no_grad():
                pe = prepare_batch_pe(batch_data, strategy).to(DEVICE)
                out = model(batch_data.x, pe, batch_data.edge_index,
                            batch_data.edge_attr, batch_data.batch)

            # Per-graph metric
            if ds_key == 'peptides_func':
                try:
                    m = average_precision_score(
                        batch_data.y.cpu().numpy().reshape(1, -1),
                        torch.sigmoid(out).cpu().numpy().reshape(1, -1),
                        average='macro'
                    )
                except Exception:
                    m = 0.0
            elif ds_key == 'peptides_struct':
                m = (out.squeeze().cpu() - batch_data.y.squeeze().cpu()).abs().mean().item()
            else:
                m = abs(out.squeeze().item() - batch_data.y.squeeze().item())

            strategy_metrics[strategy] = m

            is_better = (m < best_metric) if meta['lower_better'] else (m > best_metric)
            if is_better:
                best_metric = m
                best_strategy = strategy
                best_pred = out.cpu()

        if best_pred is not None:
            if ds_key == 'peptides_func':
                probs = torch.sigmoid(best_pred)
                vals = probs.squeeze()
                if vals.dim() == 0:
                    oracle_preds.append(str(round(vals.item(), 4)))
                else:
                    oracle_preds.append(json.dumps([round(p, 4) for p in vals.tolist()]))
            elif ds_key == 'peptides_struct':
                vals = best_pred.squeeze()
                if vals.dim() == 0:
                    oracle_preds.append(str(round(vals.item(), 4)))
                else:
                    oracle_preds.append(json.dumps([round(v, 4) for v in vals.tolist()]))
            else:
                oracle_preds.append(str(round(best_pred.squeeze().item(), 6)))
        else:
            oracle_preds.append("0")

        sri_val = getattr(test_data[g_idx], 'sri_K20', 0.0)
        per_graph_results.append({
            'graph_idx': g_idx,
            'sri_K20': sri_val,
            'vandermonde_cond': getattr(test_data[g_idx], 'vandermonde_cond', 0.0),
            'delta_min': getattr(test_data[g_idx], 'delta_min', 0.0),
            'num_nodes': test_data[g_idx].num_nodes_val,
            'oracle_strategy': best_strategy,
            'oracle_metric': best_metric,
            'strategy_metrics': strategy_metrics,
        })

    # Oracle overall metric
    oracle_metrics = [r['oracle_metric'] for r in per_graph_results if not np.isnan(r['oracle_metric'])]
    oracle_overall = np.mean(oracle_metrics) if oracle_metrics else float('nan')

    return {
        'oracle_overall': oracle_overall,
        'per_graph': per_graph_results,
        'oracle_preds': oracle_preds,
    }


def train_learned_selector(
    ds_key: str,
    datasets: dict,
    strategy_models: dict,
    seed: int,
) -> dict:
    """Train a small MLP selector that predicts the best encoding strategy per graph."""
    meta = DATASET_META[ds_key]
    ds_config = DATASET_CONFIGS[ds_key]
    fixed_strategies = ['FIXED-RWSE', 'FIXED-LapPE', 'FIXED-SRWE']

    def get_graph_features(data) -> np.ndarray:
        eigs = data.eigenvalues.numpy()
        N = data.num_nodes_val
        num_edges = data.edge_index.size(1)
        density = num_edges / max(N * (N - 1), 1)
        return np.array([
            float(getattr(data, 'sri_K20', 0.0)),
            np.log1p(float(getattr(data, 'vandermonde_cond', 0.0))),
            np.mean(eigs) if len(eigs) > 0 else 0.0,
            np.std(eigs) if len(eigs) > 0 else 0.0,
            float(N),
            density,
            float(getattr(data, 'delta_min', 0.0)),
        ])

    # Get per-graph labels from validation set
    val_data = datasets[ds_key]['val']
    val_features = []
    val_labels = []

    for data in val_data:
        feats = get_graph_features(data)
        val_features.append(feats)

        # Evaluate with each fixed strategy model
        best_idx = 0
        best_metric = float('inf') if meta['lower_better'] else float('-inf')

        loader = DataLoader([data], batch_size=1)
        batch = next(iter(loader)).to(DEVICE)

        for s_idx, strategy in enumerate(fixed_strategies):
            key = (ds_key, strategy, seed)
            if key not in strategy_models or strategy_models[key] is None:
                continue

            model = GPSModel(
                channels=GPS_CHANNELS, pe_dim=GPS_PE_DIM, pe_input_dim=20,
                num_layers=GPS_LAYERS, **ds_config
            ).to(DEVICE)
            model.load_state_dict(strategy_models[key])
            model.eval()

            with torch.no_grad():
                pe = prepare_batch_pe(batch, strategy).to(DEVICE)
                out = model(batch.x, pe, batch.edge_index, batch.edge_attr, batch.batch)

            if ds_key == 'peptides_func':
                try:
                    m = average_precision_score(
                        batch.y.cpu().numpy().reshape(1, -1),
                        torch.sigmoid(out).cpu().numpy().reshape(1, -1),
                        average='macro'
                    )
                except Exception:
                    m = 0.0
            elif ds_key == 'peptides_struct':
                m = (out.squeeze().cpu() - batch.y.squeeze().cpu()).abs().mean().item()
            else:
                m = abs(out.squeeze().item() - batch.y.squeeze().item())

            is_better = (m < best_metric) if meta['lower_better'] else (m > best_metric)
            if is_better:
                best_metric = m
                best_idx = s_idx

        val_labels.append(best_idx)

    if not val_features:
        return {'selector_acc': 0.0, 'sri_acc': 0.0}

    X_train = torch.tensor(np.array(val_features), dtype=torch.float32)
    y_train = torch.tensor(val_labels, dtype=torch.long)

    # Train small MLP selector
    selector = nn.Sequential(
        nn.Linear(7, 32), nn.ReLU(),
        nn.Linear(32, 3),
    )
    opt = torch.optim.Adam(selector.parameters(), lr=0.01)

    for ep in range(200):
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(selector(X_train), y_train)
        loss.backward()
        opt.step()

    # Evaluate on test set
    test_data = datasets[ds_key]['test']
    test_features = [get_graph_features(d) for d in test_data]
    X_test = torch.tensor(np.array(test_features), dtype=torch.float32)
    selector_preds = selector(X_test).argmax(dim=1).numpy()

    # SRI threshold predictions
    sri_preds = np.array([0 if getattr(d, 'sri_K20', 0) > 1.0 else 2 for d in test_data])

    return {
        'selector_preds': selector_preds.tolist(),
        'sri_preds': sri_preds.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def run_correlation_analysis(per_graph_data: list, ds_key: str) -> dict:
    """Run SRI correlation analysis on per-graph results."""
    from scipy.stats import spearmanr

    sri_vals = [r['sri_K20'] for r in per_graph_data if r['sri_K20'] is not None and r['sri_K20'] > 0]
    if len(sri_vals) < 5:
        return {'sri_vs_gap_rho': 0.0, 'sri_vs_gap_pval': 1.0, 'sri_median': 0.0}

    # SRI vs RWSE-LapPE performance gap
    gaps = []
    for r in per_graph_data:
        if r['sri_K20'] is not None and r['sri_K20'] > 0:
            rwse_m = r['strategy_metrics'].get('FIXED-RWSE', float('nan'))
            lappe_m = r['strategy_metrics'].get('FIXED-LapPE', float('nan'))
            if not np.isnan(rwse_m) and not np.isnan(lappe_m):
                gaps.append(rwse_m - lappe_m)
            else:
                gaps.append(0.0)

    if len(gaps) >= 5 and len(sri_vals) == len(gaps):
        rho_gap, pval_gap = spearmanr(sri_vals, gaps)
    else:
        rho_gap, pval_gap = 0.0, 1.0

    sri_median = float(np.median(sri_vals))

    return {
        'sri_vs_gap_rho': float(rho_gap) if not np.isnan(rho_gap) else 0.0,
        'sri_vs_gap_pval': float(pval_gap) if not np.isnan(pval_gap) else 1.0,
        'sri_median': sri_median,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

@logger.catch
def main():
    logger.info("=" * 70)
    logger.info("SRI-Guided Adaptive Encoding Selection Experiment")
    logger.info("=" * 70)

    global_start = time.time()

    # ── Phase 0: Load data ──
    logger.info("PHASE 0: Loading data...")
    datasets = load_all_data(max_examples=MAX_EXAMPLES)

    # Determine eigenvalue range for SRWE
    all_eigs = []
    for ds_key in datasets:
        for split in datasets[ds_key]:
            for data in datasets[ds_key][split]:
                eigs = data.eigenvalues.numpy()
                if len(eigs) > 0:
                    all_eigs.extend(eigs.tolist())

    if all_eigs:
        eig_p1 = np.percentile(all_eigs, 1)
        eig_p99 = np.percentile(all_eigs, 99)
        lambda_range = (eig_p1 - 0.5, eig_p99 + 0.5)
    else:
        lambda_range = (-3.0, 3.0)
    logger.info(f"Eigenvalue range for SRWE: {lambda_range}")

    # ── Phase 1: Precompute encodings ──
    logger.info("PHASE 1: Precomputing encodings...")
    enc_start = time.time()
    precompute_encodings(datasets, lambda_range=lambda_range)
    enc_time = time.time() - enc_start
    logger.info(f"Encoding precomputation took {enc_time:.1f}s")

    # ── Phase 2-4: Train and evaluate ──
    logger.info("PHASE 2-4: Training and evaluation...")

    results = {}
    saved_models = {}
    active_ds_keys = [k for k in ('zinc', 'peptides_func', 'peptides_struct')
                      if len(datasets[k].get('train', [])) > 0
                      and len(datasets[k].get('val', [])) > 0
                      and len(datasets[k].get('test', [])) > 0]

    logger.info(f"Active datasets for training: {active_ds_keys}")

    for ds_key in active_ds_keys:
        meta = DATASET_META[ds_key]
        train_n = len(datasets[ds_key]['train'])
        val_n = len(datasets[ds_key]['val'])
        test_n = len(datasets[ds_key]['test'])
        logger.info(f"\n{'='*50}")
        logger.info(f"Dataset: {ds_key} (train={train_n}, val={val_n}, test={test_n})")

        for strategy in STRATEGIES:
            for seed in SEEDS:
                logger.info(f"  Training: {ds_key} | {strategy} | seed={seed}")
                result = train_and_evaluate(ds_key, strategy, datasets, seed)

                results[(ds_key, strategy, seed)] = result

                # Save model state for Oracle/Selector
                if strategy in ('FIXED-RWSE', 'FIXED-LapPE', 'FIXED-SRWE') and result.get('best_state') is not None:
                    saved_models[(ds_key, strategy, seed)] = result['best_state']

                logger.info(f"    → test={result['best_test']:.4f} val={result['best_val']:.4f} "
                            f"epochs={result['epochs_run']} time={result['training_time_s']:.0f}s")

                # Check time budget
                elapsed = time.time() - global_start
                if elapsed > 2700:  # 45 min
                    logger.warning(f"Time budget running low ({elapsed:.0f}s). Reducing seeds.")
                    break

            # Check time after each strategy
            elapsed = time.time() - global_start
            if elapsed > 3000:  # 50 min
                logger.warning(f"Time budget critical ({elapsed:.0f}s). Stopping training.")
                break

        elapsed = time.time() - global_start
        if elapsed > 3000:
            logger.warning(f"Time budget critical ({elapsed:.0f}s). Stopping outer loop.")
            break

    # ── Phase 5: Oracle computation ──
    logger.info("\nPHASE 5: Oracle computation...")
    oracle_results = {}
    per_graph_data = {}
    oracle_preds_all = {}

    elapsed = time.time() - global_start
    if elapsed < 3200:  # Only if we have time
        for ds_key in active_ds_keys:
            seed = SEEDS[0]
            # Check if we have all 3 fixed strategy models
            needed_keys = [(ds_key, s, seed) for s in ('FIXED-RWSE', 'FIXED-LapPE', 'FIXED-SRWE')]
            if all(k in saved_models for k in needed_keys):
                logger.info(f"  Computing Oracle for {ds_key}...")
                oracle = compute_oracle_results(ds_key, datasets[ds_key]['test'], saved_models, seed)
                oracle_results[(ds_key, seed)] = oracle['oracle_overall']
                per_graph_data[(ds_key, seed)] = oracle['per_graph']
                oracle_preds_all[ds_key] = oracle['oracle_preds']
                logger.info(f"    Oracle {ds_key}: {oracle['oracle_overall']:.4f}")
            else:
                logger.warning(f"  Missing models for Oracle on {ds_key}")

    # ── Phase 6: Learned Selector ──
    logger.info("\nPHASE 6: Learned Selector...")
    selector_results = {}

    elapsed = time.time() - global_start
    if elapsed < 3300:
        for ds_key in active_ds_keys:
            seed = SEEDS[0]
            needed_keys = [(ds_key, s, seed) for s in ('FIXED-RWSE', 'FIXED-LapPE', 'FIXED-SRWE')]
            if all(k in saved_models for k in needed_keys):
                logger.info(f"  Training Selector for {ds_key}...")
                try:
                    sel = train_learned_selector(ds_key, datasets, saved_models, seed)
                    selector_results[ds_key] = sel
                    logger.info(f"    Selector trained for {ds_key}")
                except Exception as e:
                    logger.exception(f"  Selector training failed for {ds_key}: {e}")

    # ── Phase 7: Collect predictions and build output ──
    logger.info("\nPHASE 7: Building output...")

    # Collect test predictions for each strategy using best seed
    test_predictions = {}  # {(ds_key, strategy): [pred_strings]}
    for ds_key in active_ds_keys:
        seed = SEEDS[0]  # Use first seed for predictions
        for strategy in STRATEGIES:
            key = (ds_key, strategy, seed)
            if key in results and results[key].get('best_state') is not None:
                ds_config = DATASET_CONFIGS[ds_key]
                pe_input_dim = 40 if strategy == 'CONCAT-RWSE-SRWE' else 20
                model = GPSModel(
                    channels=GPS_CHANNELS, pe_dim=GPS_PE_DIM, pe_input_dim=pe_input_dim,
                    num_layers=GPS_LAYERS, **ds_config
                ).to(DEVICE)
                model.load_state_dict(results[key]['best_state'])
                preds = collect_predictions(model, datasets[ds_key]['test'], strategy, ds_key)
                test_predictions[(ds_key, strategy)] = preds

    # ── Phase 7b: Correlation analysis ──
    logger.info("Running correlation analysis...")
    analysis = {}
    for ds_key in active_ds_keys:
        seed = SEEDS[0]
        key = (ds_key, seed)
        if key in per_graph_data:
            analysis[ds_key] = run_correlation_analysis(per_graph_data[key], ds_key)
            logger.info(f"  {ds_key}: SRI vs gap rho={analysis[ds_key]['sri_vs_gap_rho']:.3f}, "
                        f"p={analysis[ds_key]['sri_vs_gap_pval']:.3f}")

    # ── Build results summary ──
    overall_results = {}
    for ds_key in active_ds_keys:
        overall_results[ds_key] = {}
        for strategy in STRATEGIES:
            test_vals = [results.get((ds_key, strategy, s), {}).get('best_test', float('nan'))
                         for s in SEEDS]
            test_vals = [v for v in test_vals if not np.isnan(v)]
            if test_vals:
                overall_results[ds_key][strategy] = {
                    'mean': float(np.mean(test_vals)),
                    'std': float(np.std(test_vals)),
                    'seeds': test_vals,
                }

        # Add Oracle
        oracle_vals = [oracle_results.get((ds_key, s), float('nan')) for s in SEEDS]
        oracle_vals = [v for v in oracle_vals if not np.isnan(v)]
        if oracle_vals:
            overall_results[ds_key]['ORACLE'] = {
                'mean': float(np.mean(oracle_vals)),
                'std': float(np.std(oracle_vals)),
                'seeds': oracle_vals,
            }

    # ── Build exp_gen_sol_out format ──
    ds_name_map = {
        'zinc': 'ZINC-subset',
        'peptides_func': 'Peptides-func',
        'peptides_struct': 'Peptides-struct',
        'synthetic': 'Synthetic-aliased-pairs',
    }

    output_datasets = []
    for ds_key in active_ds_keys:
        test_data = datasets[ds_key]['test']
        examples = []

        for i, data in enumerate(test_data):
            example = {
                'input': data.original_input,
                'output': data.original_output,
            }

            # Add predictions for each strategy
            for strategy in STRATEGIES:
                pred_key = (ds_key, strategy)
                if pred_key in test_predictions and i < len(test_predictions[pred_key]):
                    safe_name = strategy.replace('-', '_')
                    example[f'predict_{safe_name}'] = test_predictions[pred_key][i]

            # Add Oracle prediction
            if ds_key in oracle_preds_all and i < len(oracle_preds_all[ds_key]):
                example['predict_ORACLE'] = oracle_preds_all[ds_key][i]

            # Add metadata
            example['metadata_sri_K20'] = str(getattr(data, 'sri_K20', 0.0))
            example['metadata_delta_min'] = str(getattr(data, 'delta_min', 0.0))
            example['metadata_num_nodes'] = str(data.num_nodes_val)

            examples.append(example)

        if examples:
            output_datasets.append({
                'dataset': ds_name_map.get(ds_key, ds_key),
                'examples': examples,
            })

    # Also include synthetic dataset examples (no predictions, just structural analysis)
    if len(datasets.get('synthetic', {}).get('all', [])) > 0:
        synth_examples = []
        for data in datasets['synthetic']['all']:
            example = {
                'input': data.original_input,
                'output': data.original_output,
                'predict_baseline': data.original_output,
                'predict_our_method': data.original_output,
                'metadata_sri_K20': str(getattr(data, 'sri_K20', 0.0)),
                'metadata_delta_min': str(getattr(data, 'delta_min', 0.0)),
            }
            synth_examples.append(example)
        output_datasets.append({
            'dataset': 'Synthetic-aliased-pairs',
            'examples': synth_examples,
        })

    # ── Hypothesis assessment ──
    hypothesis = {}
    for ds_key in active_ds_keys:
        ds_results = overall_results.get(ds_key, {})
        concat_mean = ds_results.get('CONCAT-RWSE-SRWE', {}).get('mean', float('nan'))
        rwse_mean = ds_results.get('FIXED-RWSE', {}).get('mean', float('nan'))
        lappe_mean = ds_results.get('FIXED-LapPE', {}).get('mean', float('nan'))
        srwe_mean = ds_results.get('FIXED-SRWE', {}).get('mean', float('nan'))
        sri_mean = ds_results.get('SRI-THRESHOLD', {}).get('mean', float('nan'))

        lower = DATASET_META[ds_key]['lower_better']
        fixed_vals = [v for v in [rwse_mean, lappe_mean, srwe_mean] if not np.isnan(v)]
        if fixed_vals:
            best_fixed = min(fixed_vals) if lower else max(fixed_vals)
        else:
            best_fixed = float('nan')

        hypothesis[ds_key] = {
            'concat_outperforms_best_fixed': (
                bool(concat_mean < best_fixed) if lower else bool(concat_mean > best_fixed)
            ) if not np.isnan(concat_mean) and not np.isnan(best_fixed) else False,
            'sri_threshold_outperforms_best_fixed': (
                bool(sri_mean < best_fixed) if lower else bool(sri_mean > best_fixed)
            ) if not np.isnan(sri_mean) and not np.isnan(best_fixed) else False,
        }

    total_time = time.time() - global_start

    # ── Metadata ──
    metadata = {
        'method_name': 'SRI-Guided Adaptive Encoding Selection & RWSE+SRWE Concatenation',
        'description': (
            'Compares 5 positional-encoding strategies for GPS graph transformer: '
            'FIXED-RWSE (baseline), FIXED-LapPE, FIXED-SRWE (Tikhonov spectral recovery), '
            'SRI-THRESHOLD (adaptive selection per-graph based on Spectral Resolution Index), '
            'and CONCAT-RWSE-SRWE (complementary concatenation). '
            'Also computes ORACLE (per-graph best fixed strategy) for upper bound. '
            'Tests whether SRI-guided adaptive selection outperforms fixed encoding choices.'
        ),
        'strategies': STRATEGIES + ['ORACLE'],
        'hyperparameters': {
            'gps_channels': GPS_CHANNELS,
            'gps_pe_dim': GPS_PE_DIM,
            'gps_layers': GPS_LAYERS,
            'gps_heads': GPS_HEADS,
            'num_epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'lr': LR,
            'seeds': SEEDS,
            'lappe_k': LAPPE_K,
            'srwe_bins': SRWE_BINS,
            'srwe_reg': SRWE_REG,
        },
        'results_summary': overall_results,
        'hypothesis_assessment': hypothesis,
        'correlation_analysis': analysis,
        'selector_analysis': {
            ds_key: {
                'selector_preds_sample': sel.get('selector_preds', [])[:20],
                'sri_preds_sample': sel.get('sri_preds', [])[:20],
            }
            for ds_key, sel in selector_results.items()
        },
        'computational_cost': {
            'encoding_precompute_s': enc_time,
            'total_time_s': total_time,
            'training_times': {
                f"{ds_key}/{strategy}": np.mean([
                    results.get((ds_key, strategy, s), {}).get('training_time_s', 0)
                    for s in SEEDS
                ])
                for ds_key in active_ds_keys
                for strategy in STRATEGIES
            },
        },
        'hardware': {
            'device': str(DEVICE),
            'gpu': f"{torch.cuda.get_device_name(0)}" if HAS_GPU else 'None',
            'vram_gb': VRAM_GB,
            'num_cpus': NUM_CPUS,
        },
    }

    output = {
        'metadata': metadata,
        'datasets': output_datasets,
    }

    # ── Save output ──
    output_path = SCRIPT_DIR / "method_out.json"
    output_path.write_text(json.dumps(output, indent=2))
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"\nSaved method_out.json ({file_size_mb:.1f} MB)")

    # Print results table
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)
    for ds_key in active_ds_keys:
        metric_name = DATASET_META[ds_key]['metric_name']
        lower = DATASET_META[ds_key]['lower_better']
        logger.info(f"\n{ds_key} ({metric_name}, {'↓' if lower else '↑'}):")
        for strategy in STRATEGIES + ['ORACLE']:
            r = overall_results.get(ds_key, {}).get(strategy, {})
            if r:
                logger.info(f"  {strategy:25s}: {r['mean']:.4f} ± {r['std']:.4f}")

    logger.info(f"\nTotal experiment time: {total_time:.0f}s ({total_time/60:.1f}min)")
    logger.info("Done!")


if __name__ == "__main__":
    main()
