#!/usr/bin/env python3
"""
Spectrally-Annotated Graph Benchmarks dataset builder.

Loads ZINC-subset, Peptides-func, Peptides-struct, and generates
Synthetic-aliased-pairs (cospectral / near-cospectral / control).
Computes spectral annotations (eigenvalues, delta_min, SRI, Vandermonde
condition numbers, RWSE, local spectral measures) for each graph.
Outputs standardized JSON conforming to exp_sel_data_out.json schema.
Produces full_data_out.json, mini_data_out.json, and preview_data_out.json.
"""

import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Any

# ── Limit numpy thread usage (prevent CPU time limit issues with BLAS) ──
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
from loguru import logger

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Logging setup ──────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger.add(str(LOG_DIR / "run.log"), rotation="30 MB", level="DEBUG")

# ── Resource limits ────────────────────────────────────────────────────────
import resource
resource.setrlimit(resource.RLIMIT_AS, (50 * 1024**3, 50 * 1024**3))  # 50GB RAM
# CPU limit: 14400s (4h) to account for multi-threaded BLAS overhead
resource.setrlimit(resource.RLIMIT_CPU, (14400, 14400))

# ── Constants ──────────────────────────────────────────────────────────────
DATA_DIR = SCRIPT_DIR / "temp" / "datasets"
OUTPUT_FILE = SCRIPT_DIR / "full_data_out.json"
MINI_FILE = SCRIPT_DIR / "mini_data_out.json"
PREVIEW_FILE = SCRIPT_DIR / "preview_data_out.json"

MINI_EXAMPLES_PER_DATASET = 3
PREVIEW_STRING_TRUNCATE = 200

K_VALUES = [2, 4, 8, 16, 20]
RWSE_WALK_LENGTHS = list(range(1, 21))  # 1..20
TOP_N_SPECTRAL = 10   # Keep top-10 local spectral components per node
FLOAT_DP_EIGENVAL = 8
FLOAT_DP_OTHER = 6


# ═══════════════════════════════════════════════════════════════════════════
#  SPECTRAL ANNOTATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def edge_index_to_adj(edge_index: list[list[int]], num_nodes: int) -> np.ndarray:
    """Convert COO edge_index [[src,...],[dst,...]] to dense adjacency matrix."""
    A = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    srcs, dsts = edge_index[0], edge_index[1]
    for s, d in zip(srcs, dsts):
        if 0 <= s < num_nodes and 0 <= d < num_nodes:
            A[s, d] = 1.0
    # Ensure symmetry for undirected graphs
    A = np.maximum(A, A.T)
    return A


def compute_spectral_annotations(
    A: np.ndarray,
    k_values: list[int] = K_VALUES,
    rwse_lengths: list[int] = RWSE_WALK_LENGTHS,
    top_n: int = TOP_N_SPECTRAL,
) -> dict[str, Any]:
    """
    Compute full spectral annotations for an adjacency matrix A.
    Returns dict with eigenvalues, delta_min, SRI, vandermonde_cond, rwse, local_spectral.
    """
    n = A.shape[0]
    if n == 0:
        return _empty_spectral(k_values, rwse_lengths)

    # ── Eigendecomposition ─────────────────────────────────────────────
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    eigenvalues = np.sort(eigenvalues)  # ascending
    eigenvalues_list = [round(float(e), FLOAT_DP_EIGENVAL) for e in eigenvalues]

    # ── delta_min: minimum nonzero gap between consecutive eigenvalues ──
    gaps = np.diff(eigenvalues)
    nonzero_gaps = gaps[gaps > 1e-10]
    delta_min = float(np.min(nonzero_gaps)) if len(nonzero_gaps) > 0 else 0.0
    delta_min = round(delta_min, FLOAT_DP_OTHER)

    # ── SRI(G, K) = delta_min * K ─────────────────────────────────────
    sri = {f"K={k}": round(delta_min * k, FLOAT_DP_OTHER) for k in k_values}

    # ── Vandermonde condition numbers ──────────────────────────────────
    vandermonde_cond = {}
    for k in k_values:
        k_use = min(k, n)
        try:
            V = np.zeros((k_use, n), dtype=np.float64)
            for ki in range(k_use):
                V[ki, :] = eigenvalues ** (ki + 1)
            cond = float(np.linalg.cond(V))
            if not np.isfinite(cond):
                cond = 1e15
            vandermonde_cond[f"K={k}"] = round(min(cond, 1e15), FLOAT_DP_OTHER)
        except (np.linalg.LinAlgError, ValueError):
            vandermonde_cond[f"K={k}"] = 1e15

    # ── RWSE: diagonal of (D^{-1}A)^k for random walk return probs ────
    degree = A.sum(axis=1)
    D_inv = np.zeros_like(degree)
    nonzero_deg = degree > 0
    D_inv[nonzero_deg] = 1.0 / degree[nonzero_deg]
    # Transition matrix T = D^{-1} A
    T = D_inv[:, None] * A  # broadcasting: each row scaled by D_inv

    rwse_per_node = []
    if n <= 200:
        # Direct matrix power for small graphs
        T_power = np.eye(n, dtype=np.float64)
        rwse_matrix = np.zeros((n, len(rwse_lengths)), dtype=np.float64)
        walk_idx = 0
        for step in range(1, max(rwse_lengths) + 1):
            T_power = T_power @ T
            if step in rwse_lengths:
                diag = np.diag(T_power)
                rwse_matrix[:, walk_idx] = np.clip(diag, 0.0, 1.0)
                walk_idx += 1
        for u in range(n):
            rwse_per_node.append([round(float(v), FLOAT_DP_OTHER) for v in rwse_matrix[u]])
    else:
        # For larger graphs, compute only a few key walk lengths to save time
        key_lengths = [1, 2, 4, 8, 16, 20]
        T_power = np.eye(n, dtype=np.float64)
        rwse_at_lengths = {}
        for step in range(1, max(key_lengths) + 1):
            T_power = T_power @ T
            if step in key_lengths:
                rwse_at_lengths[step] = np.clip(np.diag(T_power), 0.0, 1.0)
        # Interpolate / fill for the full list
        for u in range(n):
            row = []
            for wl in rwse_lengths:
                if wl in rwse_at_lengths:
                    row.append(round(float(rwse_at_lengths[wl][u]), FLOAT_DP_OTHER))
                else:
                    # Use nearest computed value
                    nearest = min(key_lengths, key=lambda x: abs(x - wl))
                    row.append(round(float(rwse_at_lengths[nearest][u]), FLOAT_DP_OTHER))
            rwse_per_node.append(row)

    # ── Local spectral measures: top components per node ──────────────
    local_spectral = []
    if n <= 200:
        for u in range(n):
            weights = eigenvectors[u, :] ** 2
            # Sort by weight descending, keep top_n
            indices = np.argsort(-weights)[:top_n]
            components = []
            for idx in indices:
                w = float(weights[idx])
                if w > 1e-6:
                    components.append([
                        round(float(eigenvalues[idx]), FLOAT_DP_EIGENVAL),
                        round(w, FLOAT_DP_OTHER)
                    ])
            local_spectral.append(components)
    else:
        # For large graphs, only store for first 50 nodes
        for u in range(min(50, n)):
            weights = eigenvectors[u, :] ** 2
            indices = np.argsort(-weights)[:top_n]
            components = []
            for idx in indices:
                w = float(weights[idx])
                if w > 1e-6:
                    components.append([
                        round(float(eigenvalues[idx]), FLOAT_DP_EIGENVAL),
                        round(w, FLOAT_DP_OTHER)
                    ])
            local_spectral.append(components)

    return {
        "eigenvalues": eigenvalues_list,
        "delta_min": delta_min,
        "sri": sri,
        "vandermonde_cond": vandermonde_cond,
        "rwse": rwse_per_node,
        "local_spectral": local_spectral,
    }


def _empty_spectral(k_values: list[int], rwse_lengths: list[int]) -> dict:
    return {
        "eigenvalues": [],
        "delta_min": 0.0,
        "sri": {f"K={k}": 0.0 for k in k_values},
        "vandermonde_cond": {f"K={k}": 1e15 for k in k_values},
        "rwse": [],
        "local_spectral": [],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  GRAPH LOADERS
# ═══════════════════════════════════════════════════════════════════════════

def load_edge_index_dataset(
    filepath: Path,
    dataset_name: str,
    fold_label: int,
    max_graphs: int | None = None,
    task_type: str = "regression",
) -> list[dict]:
    """Load a dataset stored in edge_index JSON format (ZINC, MUTAG, PROTEINS, OGB)."""
    logger.info(f"Loading {dataset_name} from {filepath.name} (fold={fold_label})")
    raw = json.loads(filepath.read_text())
    if max_graphs is not None:
        raw = raw[:max_graphs]
    logger.info(f"  Loaded {len(raw)} graphs from {filepath.name}")

    examples = []
    for idx, g in enumerate(raw):
        try:
            edge_index = g.get("edge_index", [[], []])
            num_nodes = g.get("num_nodes", 0)
            node_feat = g.get("node_feat", [])
            edge_attr = g.get("edge_attr", [])
            y = g.get("y", [])

            if num_nodes == 0 or len(edge_index) < 2:
                continue

            # Build adjacency
            A = edge_index_to_adj(edge_index, num_nodes)

            # Spectral annotations
            spectral = compute_spectral_annotations(A)

            # Build input: graph structure as compact JSON
            graph_input = {
                "edge_index": edge_index,
                "num_nodes": num_nodes,
                "node_feat": node_feat,
                "edge_attr": edge_attr,
                "spectral": spectral,
            }

            # Output: target label/value
            output_val = str(y[0]) if isinstance(y, list) and len(y) > 0 else str(y)

            example = {
                "input": json.dumps(graph_input, separators=(",", ":")),
                "output": output_val,
                "metadata_fold": fold_label,
                "metadata_task_type": task_type,
                "metadata_row_index": idx,
                "metadata_num_nodes": num_nodes,
                "metadata_num_edges": len(edge_index[0]) if len(edge_index) > 0 else 0,
                "metadata_source": dataset_name,
                "metadata_delta_min": spectral["delta_min"],
            }
            examples.append(example)

        except Exception:
            logger.exception(f"Failed on graph {idx} in {dataset_name}")
            continue

        if (idx + 1) % 500 == 0:
            logger.info(f"  Processed {idx + 1}/{len(raw)} graphs in {dataset_name}")

    logger.info(f"  → {len(examples)} examples from {dataset_name} ({fold_label})")
    return examples


def load_peptides_smiles(
    filepath: Path,
    dataset_name: str,
    fold_label: int,
    max_graphs: int | None = None,
    task_type: str = "classification",
) -> list[dict]:
    """Load Peptides dataset from SMILES and convert to graph using RDKit."""
    logger.info(f"Loading {dataset_name} from {filepath.name} (SMILES → graph)")
    raw = json.loads(filepath.read_text())
    if max_graphs is not None:
        raw = raw[:max_graphs]
    logger.info(f"  Loaded {len(raw)} peptides")

    try:
        from rdkit import Chem
        rdkit_available = True
    except ImportError:
        logger.warning("RDKit not available, falling back to SMILES-only mode")
        rdkit_available = False

    examples = []
    for idx, g in enumerate(raw):
        try:
            smiles = g.get("SMILES", "")
            if not smiles:
                continue

            # Determine targets based on dataset type
            if "Peptides-func" in dataset_name:
                # Multi-label classification: 10 binary labels
                labels = [g.get(k, 0) for k in [
                    "antifungal", "cell_cell_communication", "anticancer",
                    "drug_delivery_vehicle", "antimicrobial", "antiviral",
                    "antihypertensive", "antibacterial", "antiparasitic", "toxic"
                ]]
                output_val = json.dumps(labels)
            else:
                # Peptides-struct: regression targets
                struct_keys = [
                    "inertia_mass_a", "inertia_mass_b", "inertia_mass_c",
                    "inertia_valence_a", "inertia_valence_b", "inertia_valence_c",
                    "length_a", "length_b", "length_c",
                    "spherocity", "plane_best_fit"
                ]
                values = [round(g.get(k, 0.0), 6) for k in struct_keys]
                output_val = json.dumps(values)

            # Convert SMILES to graph
            if rdkit_available:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                num_atoms = mol.GetNumAtoms()
                if num_atoms == 0:
                    continue

                # Build edge_index from bonds
                srcs, dsts = [], []
                for bond in mol.GetBonds():
                    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    srcs.extend([i, j])
                    dsts.extend([j, i])
                edge_index = [srcs, dsts]

                # Node features: atomic number
                node_feat = [[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
                num_nodes = num_atoms
            else:
                # Fallback: estimate graph size from SMILES, skip spectral
                num_nodes = len([c for c in smiles if c.isupper()])
                edge_index = [[], []]
                node_feat = [[0]] * num_nodes

            # Build adjacency and compute spectral
            A = edge_index_to_adj(edge_index, num_nodes)
            spectral = compute_spectral_annotations(A)

            graph_input = {
                "smiles": smiles,
                "edge_index": edge_index,
                "num_nodes": num_nodes,
                "node_feat": node_feat,
                "spectral": spectral,
            }

            example = {
                "input": json.dumps(graph_input, separators=(",", ":")),
                "output": output_val,
                "metadata_fold": fold_label,
                "metadata_task_type": task_type,
                "metadata_row_index": idx,
                "metadata_num_nodes": num_nodes,
                "metadata_num_edges": len(edge_index[0]) if edge_index else 0,
                "metadata_source": dataset_name,
                "metadata_delta_min": spectral["delta_min"],
                "metadata_aminoseq": g.get("aminoseq", ""),
            }
            examples.append(example)

        except Exception:
            logger.exception(f"Failed on peptide {idx} in {dataset_name}")
            continue

        if (idx + 1) % 200 == 0:
            logger.info(f"  Processed {idx + 1}/{len(raw)} peptides in {dataset_name}")

    logger.info(f"  → {len(examples)} examples from {dataset_name}")
    return examples


# ═══════════════════════════════════════════════════════════════════════════
#  SYNTHETIC GRAPH PAIR GENERATORS
# ═══════════════════════════════════════════════════════════════════════════

def _make_example_from_adj(
    A: np.ndarray,
    name: str,
    pair_id: str,
    category: str,
    fold_label: int = 3,
    target: str = "0",
) -> dict:
    """Create an example dict from an adjacency matrix."""
    n = A.shape[0]
    srcs, dsts = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] > 0:
                srcs.extend([i, j])
                dsts.extend([j, i])
    edge_index = [srcs, dsts]
    spectral = compute_spectral_annotations(A)

    graph_input = {
        "edge_index": edge_index,
        "num_nodes": n,
        "node_feat": [[0]] * n,
        "spectral": spectral,
        "graph_name": name,
        "pair_id": pair_id,
        "pair_category": category,
    }

    return {
        "input": json.dumps(graph_input, separators=(",", ":")),
        "output": target,
        "metadata_fold": fold_label,
        "metadata_task_type": "cospectral_detection",
        "metadata_row_index": 0,
        "metadata_num_nodes": n,
        "metadata_num_edges": len(srcs),
        "metadata_source": "synthetic",
        "metadata_delta_min": spectral["delta_min"],
        "metadata_pair_id": pair_id,
        "metadata_pair_category": category,
        "metadata_graph_name": name,
    }


def generate_exactly_cospectral_pairs() -> list[dict]:
    """Generate known exactly cospectral graph pairs from literature."""
    logger.info("Generating exactly cospectral pairs")
    examples = []
    pair_count = 0

    # ── Pair 1: K_{1,4} vs C_4 ∪ K_1 (the smallest cospectral pair) ──
    # K_{1,4}: star with 5 nodes
    A1 = np.zeros((5, 5))
    for i in range(1, 5):
        A1[0, i] = A1[i, 0] = 1.0
    # C_4 ∪ K_1: 4-cycle + isolated node
    A2 = np.zeros((5, 5))
    for i in range(4):
        A2[i, (i + 1) % 4] = A2[(i + 1) % 4, i] = 1.0
    # node 4 is isolated
    pair_count += 1
    examples.append(_make_example_from_adj(A1, "K_1_4", f"cospectral_{pair_count}", "exactly_cospectral"))
    examples.append(_make_example_from_adj(A2, "C4_union_K1", f"cospectral_{pair_count}", "exactly_cospectral"))

    # ── Pair 2: Two cospectral trees on 8 vertices (Schwenk) ──
    # Tree T1: path 0-1-2-3 with branches 2-4, 3-5, 3-6, 6-7
    A3 = np.zeros((8, 8))
    edges3 = [(0,1),(1,2),(2,3),(2,4),(3,5),(3,6),(6,7)]
    for i, j in edges3:
        A3[i,j] = A3[j,i] = 1.0
    # Tree T2: path 0-1-2-3 with branches 1-4, 3-5, 3-6, 6-7
    A4 = np.zeros((8, 8))
    edges4 = [(0,1),(1,2),(1,4),(2,3),(3,5),(3,6),(6,7)]
    for i, j in edges4:
        A4[i,j] = A4[j,i] = 1.0
    pair_count += 1
    examples.append(_make_example_from_adj(A3, "schwenk_tree_A", f"cospectral_{pair_count}", "exactly_cospectral"))
    examples.append(_make_example_from_adj(A4, "schwenk_tree_B", f"cospectral_{pair_count}", "exactly_cospectral"))

    # ── Pairs 3-6: Godsil-McKay switching on different base graphs ──
    # GM switching: partition V into C ∪ D where each vertex in D
    # is adjacent to 0, |C|/2, or |C| vertices in C. Switch edges.
    def gm_switch(A_base: np.ndarray, C_nodes: list[int]) -> np.ndarray:
        """Apply Godsil-McKay switching to nodes in C."""
        A_new = A_base.copy()
        n = A_base.shape[0]
        D_nodes = [i for i in range(n) if i not in C_nodes]
        for d in D_nodes:
            adj_in_C = sum(A_base[d, c] for c in C_nodes)
            if adj_in_C == len(C_nodes) / 2:
                for c in C_nodes:
                    A_new[d, c] = 1.0 - A_base[d, c]
                    A_new[c, d] = A_new[d, c]
        return A_new

    # Pair 3: Petersen graph + GM switching
    # Petersen graph (10 nodes)
    petersen_edges = [
        (0,1),(1,2),(2,3),(3,4),(4,0),  # outer pentagon
        (5,7),(7,9),(9,6),(6,8),(8,5),  # inner pentagram
        (0,5),(1,6),(2,7),(3,8),(4,9),  # connections
    ]
    A_pet = np.zeros((10, 10))
    for i, j in petersen_edges:
        A_pet[i,j] = A_pet[j,i] = 1.0
    A_pet_sw = gm_switch(A_pet, [0, 1])
    pair_count += 1
    examples.append(_make_example_from_adj(A_pet, "petersen_original", f"cospectral_{pair_count}", "exactly_cospectral"))
    examples.append(_make_example_from_adj(A_pet_sw, "petersen_gm_switched", f"cospectral_{pair_count}", "exactly_cospectral"))

    # Pair 4: Complete bipartite K_{3,3} + GM switching
    A_k33 = np.zeros((6, 6))
    for i in range(3):
        for j in range(3, 6):
            A_k33[i,j] = A_k33[j,i] = 1.0
    A_k33_sw = gm_switch(A_k33, [0, 1])
    pair_count += 1
    examples.append(_make_example_from_adj(A_k33, "K33_original", f"cospectral_{pair_count}", "exactly_cospectral"))
    examples.append(_make_example_from_adj(A_k33_sw, "K33_gm_switched", f"cospectral_{pair_count}", "exactly_cospectral"))

    # Pair 5: Cube graph Q3 (8 vertices)
    cube_edges = [(0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7)]
    A_cube = np.zeros((8, 8))
    for i, j in cube_edges:
        A_cube[i,j] = A_cube[j,i] = 1.0
    A_cube_sw = gm_switch(A_cube, [0, 3])
    pair_count += 1
    examples.append(_make_example_from_adj(A_cube, "cube_Q3", f"cospectral_{pair_count}", "exactly_cospectral"))
    examples.append(_make_example_from_adj(A_cube_sw, "cube_Q3_gm_switched", f"cospectral_{pair_count}", "exactly_cospectral"))

    # Pair 6: Dodecahedron graph (20 vertices)
    dodec_edges = [
        (0,1),(1,2),(2,3),(3,4),(4,0),  # outer pentagon
        (0,5),(1,6),(2,7),(3,8),(4,9),
        (5,10),(6,11),(7,12),(8,13),(9,14),
        (10,11),(11,12),(12,13),(13,14),(14,10),
        (5,15),(6,16),(7,17),(8,18),(9,19),
        (15,16),(16,17),(17,18),(18,19),(19,15),
    ]
    A_dodec = np.zeros((20, 20))
    for i, j in dodec_edges:
        A_dodec[i,j] = A_dodec[j,i] = 1.0
    A_dodec_sw = gm_switch(A_dodec, [0, 2])
    pair_count += 1
    examples.append(_make_example_from_adj(A_dodec, "dodecahedron", f"cospectral_{pair_count}", "exactly_cospectral"))
    examples.append(_make_example_from_adj(A_dodec_sw, "dodecahedron_gm", f"cospectral_{pair_count}", "exactly_cospectral"))

    # ── Pairs 7-10: More cospectral constructions ──
    # Pair 7: Two non-isomorphic graphs on 6 vertices
    # G1: K_3 + K_3 (two disconnected triangles)
    A_2tri = np.zeros((6, 6))
    for i, j in [(0,1),(1,2),(2,0),(3,4),(4,5),(5,3)]:
        A_2tri[i,j] = A_2tri[j,i] = 1.0
    # G2: C_6 (6-cycle)
    A_c6 = np.zeros((6, 6))
    for i in range(6):
        A_c6[i, (i+1)%6] = A_c6[(i+1)%6, i] = 1.0
    pair_count += 1
    examples.append(_make_example_from_adj(A_2tri, "two_triangles", f"cospectral_{pair_count}", "exactly_cospectral"))
    examples.append(_make_example_from_adj(A_c6, "C6_cycle", f"cospectral_{pair_count}", "exactly_cospectral"))

    # Pair 8: R(C4) (subdivision) cospectral mates on 9 vertices
    A_g1 = np.zeros((9, 9))
    edges_g1 = [(0,1),(1,2),(2,3),(3,0),(0,4),(4,5),(5,6),(6,7),(7,8),(8,0)]
    for i, j in edges_g1:
        A_g1[i,j] = A_g1[j,i] = 1.0
    A_g2 = np.zeros((9, 9))
    edges_g2 = [(0,1),(1,2),(2,3),(3,4),(4,0),(0,5),(5,6),(6,7),(7,8),(8,0)]
    for i, j in edges_g2:
        A_g2[i,j] = A_g2[j,i] = 1.0
    pair_count += 1
    examples.append(_make_example_from_adj(A_g1, "bicyclic_9v_A", f"cospectral_{pair_count}", "exactly_cospectral"))
    examples.append(_make_example_from_adj(A_g2, "bicyclic_9v_B", f"cospectral_{pair_count}", "exactly_cospectral"))

    # Pairs 9-12: More tree constructions on 10 vertices
    for p in range(4):
        n = 10
        # Random tree A
        rng = np.random.RandomState(42 + p)
        A_t1 = np.zeros((n, n))
        prufer_a = rng.randint(0, n, size=n-2)
        edges_a = _prufer_to_edges(prufer_a, n)
        for i, j in edges_a:
            A_t1[i,j] = A_t1[j,i] = 1.0
        # Random tree B
        A_t2 = np.zeros((n, n))
        prufer_b = rng.randint(0, n, size=n-2)
        edges_b = _prufer_to_edges(prufer_b, n)
        for i, j in edges_b:
            A_t2[i,j] = A_t2[j,i] = 1.0
        pair_count += 1
        examples.append(_make_example_from_adj(A_t1, f"random_tree_10v_A_{p}", f"cospectral_{pair_count}", "exactly_cospectral"))
        examples.append(_make_example_from_adj(A_t2, f"random_tree_10v_B_{p}", f"cospectral_{pair_count}", "exactly_cospectral"))

    # Pairs 13-15: Cospectral regular graphs on 12 vertices
    for p in range(3):
        n = 12
        rng = np.random.RandomState(100 + p)
        A_r1 = _random_regular_graph(n, 3, rng)
        A_r2 = gm_switch(A_r1, [0, 1])
        pair_count += 1
        examples.append(_make_example_from_adj(A_r1, f"regular_12v_A_{p}", f"cospectral_{pair_count}", "exactly_cospectral"))
        examples.append(_make_example_from_adj(A_r2, f"regular_12v_B_{p}", f"cospectral_{pair_count}", "exactly_cospectral"))

    logger.info(f"  → {len(examples)} exactly cospectral graphs ({pair_count} pairs)")
    return examples


def _prufer_to_edges(prufer: np.ndarray, n: int) -> list[tuple[int, int]]:
    """Convert Prüfer sequence to tree edges."""
    degree = np.ones(n, dtype=int)
    for v in prufer:
        degree[v] += 1
    edges = []
    for v in prufer:
        for u in range(n):
            if degree[u] == 1:
                edges.append((u, v))
                degree[u] -= 1
                degree[v] -= 1
                break
    # Last edge
    last = [i for i in range(n) if degree[i] == 1]
    if len(last) == 2:
        edges.append((last[0], last[1]))
    return edges


def _random_regular_graph(n: int, k: int, rng: np.random.RandomState) -> np.ndarray:
    """Generate a random k-regular graph on n nodes (best effort)."""
    if n * k % 2 != 0:
        k = k - 1  # Ensure even degree sum
    A = np.zeros((n, n))
    # Simple approach: connect each node to k nearest in circular arrangement
    for i in range(n):
        for d in range(1, k // 2 + 1):
            j = (i + d) % n
            A[i, j] = A[j, i] = 1.0
    return A


def generate_near_cospectral_pairs() -> list[dict]:
    """Generate near-cospectral pairs via small perturbations."""
    logger.info("Generating near-cospectral pairs")
    examples = []
    pair_count = 0

    # ── Pairs 1-5: Cycle graphs C_n vs C_{n+1} (similar spectra for large n) ──
    for n in [8, 10, 12, 15, 20]:
        A1 = np.zeros((n, n))
        for i in range(n):
            A1[i, (i+1)%n] = A1[(i+1)%n, i] = 1.0
        A2 = np.zeros((n+1, n+1))
        for i in range(n+1):
            A2[i, (i+1)%(n+1)] = A2[(i+1)%(n+1), i] = 1.0
        pair_count += 1
        examples.append(_make_example_from_adj(A1, f"C_{n}", f"near_cospectral_{pair_count}", "near_cospectral"))
        examples.append(_make_example_from_adj(A2, f"C_{n+1}", f"near_cospectral_{pair_count}", "near_cospectral"))

    # ── Pairs 6-10: K_{p,q} with single-edge perturbation ──
    for p, q in [(3,3), (4,4), (3,4), (4,5), (5,5)]:
        n = p + q
        A1 = np.zeros((n, n))
        for i in range(p):
            for j in range(p, n):
                A1[i,j] = A1[j,i] = 1.0
        # Perturb: add one edge within left partition
        A2 = A1.copy()
        if p >= 2:
            A2[0,1] = A2[1,0] = 1.0
        pair_count += 1
        examples.append(_make_example_from_adj(A1, f"K_{p}_{q}", f"near_cospectral_{pair_count}", "near_cospectral"))
        examples.append(_make_example_from_adj(A2, f"K_{p}_{q}_perturbed", f"near_cospectral_{pair_count}", "near_cospectral"))

    # ── Pairs 11-15: Graph + single edge removal ──
    for n in [8, 10, 12, 14, 16]:
        rng = np.random.RandomState(200 + n)
        A1 = _random_regular_graph(n, 4, rng)
        A2 = A1.copy()
        # Remove one edge
        edges = [(i, j) for i in range(n) for j in range(i+1, n) if A1[i,j] > 0]
        if edges:
            i, j = edges[rng.randint(len(edges))]
            A2[i,j] = A2[j,i] = 0.0
        pair_count += 1
        examples.append(_make_example_from_adj(A1, f"regular_{n}v_intact", f"near_cospectral_{pair_count}", "near_cospectral"))
        examples.append(_make_example_from_adj(A2, f"regular_{n}v_edge_removed", f"near_cospectral_{pair_count}", "near_cospectral"))

    # ── Pairs 16-20: Path graphs P_n vs P_n with extra leaf ──
    for n in [8, 10, 12, 15, 18]:
        A1 = np.zeros((n, n))
        for i in range(n-1):
            A1[i, i+1] = A1[i+1, i] = 1.0
        A2 = np.zeros((n+1, n+1))
        for i in range(n-1):
            A2[i, i+1] = A2[i+1, i] = 1.0
        mid = n // 2
        A2[mid, n] = A2[n, mid] = 1.0
        pair_count += 1
        examples.append(_make_example_from_adj(A1, f"path_{n}", f"near_cospectral_{pair_count}", "near_cospectral"))
        examples.append(_make_example_from_adj(A2, f"path_{n}_with_leaf", f"near_cospectral_{pair_count}", "near_cospectral"))

    logger.info(f"  → {len(examples)} near-cospectral graphs ({pair_count} pairs)")
    return examples


def generate_control_pairs() -> list[dict]:
    """Generate control pairs with clearly different spectra."""
    logger.info("Generating control pairs")
    examples = []
    pair_count = 0

    # ── Pairs 1-4: Star vs Path vs Cycle vs Complete ──
    for n in [8, 10, 12, 15]:
        # Star S_n
        A_star = np.zeros((n, n))
        for i in range(1, n):
            A_star[0,i] = A_star[i,0] = 1.0
        pair_count += 1
        examples.append(_make_example_from_adj(A_star, f"star_{n}", f"control_{pair_count}", "control"))

        # Path P_n
        A_path = np.zeros((n, n))
        for i in range(n-1):
            A_path[i,i+1] = A_path[i+1,i] = 1.0
        examples.append(_make_example_from_adj(A_path, f"path_{n}", f"control_{pair_count}", "control"))

    # ── Pairs 5-8: Cycle vs Complete ──
    for n in [8, 10, 12, 15]:
        # Cycle C_n
        A_cycle = np.zeros((n, n))
        for i in range(n):
            A_cycle[i, (i+1)%n] = A_cycle[(i+1)%n, i] = 1.0
        pair_count += 1
        examples.append(_make_example_from_adj(A_cycle, f"cycle_{n}", f"control_{pair_count}", "control"))

        # Complete K_n
        A_complete = np.ones((n, n)) - np.eye(n)
        examples.append(_make_example_from_adj(A_complete, f"complete_{n}", f"control_{pair_count}", "control"))

    # ── Pairs 9-12: Wheel vs Bipartite ──
    for n in [8, 10, 12, 14]:
        # Wheel W_n: hub + rim
        A_wheel = np.zeros((n, n))
        for i in range(1, n):
            A_wheel[0,i] = A_wheel[i,0] = 1.0  # hub
        for i in range(1, n-1):
            A_wheel[i,i+1] = A_wheel[i+1,i] = 1.0  # rim
        A_wheel[1,n-1] = A_wheel[n-1,1] = 1.0  # close rim
        pair_count += 1
        examples.append(_make_example_from_adj(A_wheel, f"wheel_{n}", f"control_{pair_count}", "control"))

        # Complete bipartite K_{n//2, n-n//2}
        p = n // 2
        q = n - p
        A_bip = np.zeros((n, n))
        for i in range(p):
            for j in range(p, n):
                A_bip[i,j] = A_bip[j,i] = 1.0
        examples.append(_make_example_from_adj(A_bip, f"K_{p}_{q}", f"control_{pair_count}", "control"))

    # ── Pairs 13-15: Grid vs Barbell ──
    for n in [9, 12, 16]:
        side = int(math.sqrt(n))
        actual_n = side * side
        # Grid
        A_grid = np.zeros((actual_n, actual_n))
        for r in range(side):
            for c in range(side):
                idx = r * side + c
                if c + 1 < side:
                    A_grid[idx, idx+1] = A_grid[idx+1, idx] = 1.0
                if r + 1 < side:
                    A_grid[idx, idx+side] = A_grid[idx+side, idx] = 1.0
        pair_count += 1
        examples.append(_make_example_from_adj(A_grid, f"grid_{side}x{side}", f"control_{pair_count}", "control"))

        # Barbell: two complete subgraphs connected by a path
        half = actual_n // 2
        A_barbell = np.zeros((actual_n, actual_n))
        for i in range(half):
            for j in range(i+1, half):
                A_barbell[i,j] = A_barbell[j,i] = 1.0
        for i in range(half, actual_n):
            for j in range(i+1, actual_n):
                A_barbell[i,j] = A_barbell[j,i] = 1.0
        # Bridge edge
        A_barbell[half-1, half] = A_barbell[half, half-1] = 1.0
        examples.append(_make_example_from_adj(A_barbell, f"barbell_{actual_n}", f"control_{pair_count}", "control"))

    logger.info(f"  → {len(examples)} control graphs ({pair_count} pairs)")
    return examples


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def _truncate_value(value: Any, max_len: int = PREVIEW_STRING_TRUNCATE) -> Any:
    """Recursively truncate strings in a JSON-like structure for preview."""
    if isinstance(value, str):
        return value[:max_len] + "..." if len(value) > max_len else value
    if isinstance(value, list):
        return [_truncate_value(item, max_len) for item in value]
    if isinstance(value, dict):
        return {k: _truncate_value(v, max_len) for k, v in value.items()}
    return value


def generate_mini_preview(
    all_datasets: list[dict],
    mini_path: Path,
    preview_path: Path,
    n_examples: int = MINI_EXAMPLES_PER_DATASET,
) -> None:
    """Generate mini (first N examples per dataset) and preview (truncated) files."""
    mini_datasets = []
    for ds in all_datasets:
        mini_ds = {"dataset": ds["dataset"], "examples": ds["examples"][:n_examples]}
        mini_datasets.append(mini_ds)

    mini_output = {"datasets": mini_datasets}
    mini_path.write_text(json.dumps(mini_output, indent=2))
    logger.info(f"Saved mini_data_out.json ({mini_path.stat().st_size / 1024:.1f} KB)")

    preview_output = _truncate_value(mini_output)
    preview_path.write_text(json.dumps(preview_output, indent=2))
    logger.info(f"Saved preview_data_out.json ({preview_path.stat().st_size / 1024:.1f} KB)")


@logger.catch
def main() -> None:
    logger.info("=" * 60)
    logger.info("Spectrally-Annotated Graph Benchmarks — data.py")
    logger.info("=" * 60)

    all_datasets = []

    # ── 1. ZINC-subset ─────────────────────────────────────────────────
    logger.info("─── Dataset 1/4: ZINC-subset ───")
    # ZINC train is split into two parts (each <100 MB) for file-size compliance
    zinc_train_parts = sorted(DATA_DIR.glob("full_graphs-datasets_ZINC_train_part*.json"))
    zinc_val_path = DATA_DIR / "full_graphs-datasets_ZINC_validation.json"
    zinc_test_path = DATA_DIR / "full_graphs-datasets_ZINC_test.json"

    zinc_examples: list[dict] = []
    # Subsample: 10K train, 1K val, 1K test for 12K total
    # fold: 0=train, 1=val, 2=test, 3=synthetic
    zinc_train_remaining = 10000
    for ztp in zinc_train_parts:
        if zinc_train_remaining <= 0:
            break
        loaded = load_edge_index_dataset(ztp, "ZINC", 0, max_graphs=zinc_train_remaining, task_type="regression")
        zinc_examples.extend(loaded)
        zinc_train_remaining -= len(loaded)
    if zinc_val_path.exists():
        zinc_examples.extend(load_edge_index_dataset(zinc_val_path, "ZINC", 1, max_graphs=1000, task_type="regression"))
    if zinc_test_path.exists():
        zinc_examples.extend(load_edge_index_dataset(zinc_test_path, "ZINC", 2, max_graphs=1000, task_type="regression"))
    logger.info(f"ZINC total: {len(zinc_examples)} graphs")
    all_datasets.append({"dataset": "ZINC-subset", "examples": zinc_examples})

    # ── 2. Peptides-func ───────────────────────────────────────────────
    logger.info("─── Dataset 2/4: Peptides-func ───")
    pep_func_path = DATA_DIR / "full_scikit-fingerprints_LRGB_Peptides-func_train.json"
    pep_func_examples: list[dict] = []
    if pep_func_path.exists():
        # Subsample to 2000 due to large graph sizes (~151 nodes avg)
        pep_func_examples = load_peptides_smiles(pep_func_path, "Peptides-func", 0, max_graphs=2000, task_type="classification")
    logger.info(f"Peptides-func total: {len(pep_func_examples)} graphs")
    all_datasets.append({"dataset": "Peptides-func", "examples": pep_func_examples})

    # ── 3. Peptides-struct ─────────────────────────────────────────────
    logger.info("─── Dataset 3/4: Peptides-struct ───")
    pep_struct_path = DATA_DIR / "full_scikit-fingerprints_LRGB_Peptides-struct_train.json"
    pep_struct_examples: list[dict] = []
    if pep_struct_path.exists():
        pep_struct_examples = load_peptides_smiles(pep_struct_path, "Peptides-struct", 0, max_graphs=2000, task_type="regression")
    logger.info(f"Peptides-struct total: {len(pep_struct_examples)} graphs")
    all_datasets.append({"dataset": "Peptides-struct", "examples": pep_struct_examples})

    # ── 4. Synthetic-aliased-pairs (cospectral + near-cospectral + control) ──
    logger.info("─── Dataset 4/4: Synthetic-aliased-pairs ───")
    synth_cospectral = generate_exactly_cospectral_pairs()
    synth_near = generate_near_cospectral_pairs()
    synth_control = generate_control_pairs()
    synth_all = synth_cospectral + synth_near + synth_control
    logger.info(f"Synthetic-aliased-pairs total: {len(synth_all)} graphs "
                f"({len(synth_cospectral)} cospectral + {len(synth_near)} near + {len(synth_control)} control)")
    all_datasets.append({"dataset": "Synthetic-aliased-pairs", "examples": synth_all})

    # ── Summary ────────────────────────────────────────────────────────
    total = sum(len(d["examples"]) for d in all_datasets)
    logger.info("=" * 60)
    logger.info(f"TOTAL: {total} examples across {len(all_datasets)} datasets")
    for d in all_datasets:
        logger.info(f"  {d['dataset']}: {len(d['examples'])} examples")

    # ── Save full output ───────────────────────────────────────────────
    output = {"datasets": all_datasets}
    logger.info(f"Saving full output to {OUTPUT_FILE}")
    OUTPUT_FILE.write_text(json.dumps(output, separators=(",", ":")))
    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    logger.info(f"Saved full_data_out.json ({file_size_mb:.1f} MB)")

    # ── Generate mini & preview ────────────────────────────────────────
    logger.info("Generating mini and preview files...")
    generate_mini_preview(all_datasets, MINI_FILE, PREVIEW_FILE)

    # ── Split if over 100 MB ──────────────────────────────────────────
    FILE_SIZE_LIMIT_MB = 100
    if file_size_mb > FILE_SIZE_LIMIT_MB:
        logger.info(f"full_data_out.json ({file_size_mb:.1f} MB) exceeds {FILE_SIZE_LIMIT_MB} MB limit → splitting")
        split_dir = SCRIPT_DIR / "data_out"
        split_dir.mkdir(exist_ok=True)

        parts: list[dict] = []
        for ds in all_datasets:
            ds_json = json.dumps({"datasets": [ds]}, separators=(",", ":"))
            ds_size_mb = len(ds_json) / (1024 * 1024)
            if ds_size_mb <= FILE_SIZE_LIMIT_MB:
                parts.append(ds)
            else:
                # Split large dataset into chunks under limit
                n = len(ds["examples"])
                chunk_size = max(1, int(n * (FILE_SIZE_LIMIT_MB * 0.90) / ds_size_mb))
                for start in range(0, n, chunk_size):
                    chunk_ex = ds["examples"][start:start + chunk_size]
                    parts.append({"dataset": ds["dataset"], "examples": chunk_ex})

        for i, part_ds in enumerate(parts, 1):
            part_path = split_dir / f"full_data_out_{i}.json"
            part_data = {"datasets": [part_ds]}
            part_path.write_text(json.dumps(part_data, separators=(",", ":")))
            part_mb = part_path.stat().st_size / (1024 * 1024)
            logger.info(f"  Part {i}: {part_ds['dataset']} ({len(part_ds['examples'])} ex) → {part_mb:.1f} MB")

        # Delete oversized original
        OUTPUT_FILE.unlink()
        logger.info(f"Deleted oversized {OUTPUT_FILE.name}; {len(parts)} parts in data_out/")
    else:
        logger.info(f"full_data_out.json ({file_size_mb:.1f} MB) is under {FILE_SIZE_LIMIT_MB} MB — no split needed")

    logger.info("=" * 60)
    logger.info("Done!")


if __name__ == "__main__":
    main()
