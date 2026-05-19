"""Compute the three feature-feature relations: co-activation (PMI/Jaccard),
decoder cosine, and label-embedding cosine."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from neograph.config import COACTIVATION_NMIN, COACTIVATION_PMI_MIN, KNN_K
from neograph.util import get_logger

log = get_logger("neograph.relations")


@dataclass
class CoactEdge:
    a: int
    b: int
    pmi: float
    jaccard: float
    n_co: int
    n_a: int
    n_b: int


def _coactivation_counts(active_features: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray, int]:
    """Given a boolean sparse matrix (n_positions, n_features) where entry (p, f) = 1
    means feature f fired at position p, return (n_f, n_pairs_ff, n_positions).

    Co-activation count for pair (i, j) is `(active.T @ active)[i, j]`.
    """
    if active_features.dtype != np.float32:
        active_features = active_features.astype(np.float32)
    n_f = np.asarray(active_features.sum(axis=0)).ravel().astype(np.int64)
    pair_counts = (active_features.T @ active_features).astype(np.int64)
    n_pos = int(active_features.shape[0])
    return n_f, pair_counts, n_pos


def coactivation_edges(
    active_features: sp.csr_matrix,
    top_k: int = KNN_K,
    pmi_min: float = COACTIVATION_PMI_MIN,
    n_min: int = COACTIVATION_NMIN,
) -> list[CoactEdge]:
    """Compute top-k co-activation neighbours per feature by PMI."""
    n_f, pair_counts, n_pos = _coactivation_counts(active_features)
    # PMI(i,j) = log( P(i,j) / (P(i)P(j)) ) = log( n_co * n_pos / (n_i * n_j) )
    pair_counts = pair_counts.tocoo()
    edges: list[CoactEdge] = []
    for a, b, n_co in zip(pair_counts.row, pair_counts.col, pair_counts.data):
        if a >= b:
            continue  # keep one direction per unordered pair
        if n_co < n_min:
            continue
        n_a = int(n_f[a])
        n_b = int(n_f[b])
        if n_a == 0 or n_b == 0:
            continue
        pmi = math.log(n_co * n_pos / (n_a * n_b))
        if pmi < pmi_min:
            continue
        union = n_a + n_b - int(n_co)
        jaccard = (int(n_co) / union) if union > 0 else 0.0
        edges.append(
            CoactEdge(a=int(a), b=int(b), pmi=pmi, jaccard=jaccard, n_co=int(n_co), n_a=n_a, n_b=n_b)
        )

    # Top-k per feature by PMI (both directions)
    by_feat: dict[int, list[CoactEdge]] = {}
    for e in edges:
        by_feat.setdefault(e.a, []).append(e)
        by_feat.setdefault(e.b, []).append(e)
    keep: set[tuple[int, int]] = set()
    for f, fedges in by_feat.items():
        fedges.sort(key=lambda x: x.pmi, reverse=True)
        for e in fedges[:top_k]:
            keep.add((e.a, e.b))
    return [e for e in edges if (e.a, e.b) in keep]


def topk_cosine_knn(
    vectors: np.ndarray,
    top_k: int = KNN_K,
    chunk_size: int = 1024,
    self_filter: bool = True,
) -> list[tuple[int, int, float]]:
    """Compute top-K cosine similarity edges for `vectors` (n, d)."""
    n, _d = vectors.shape
    vecs = vectors.astype(np.float32, copy=False)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    vn = vecs / norms

    edges: list[tuple[int, int, float]] = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sims = vn[start:end] @ vn.T  # (chunk, n)
        if self_filter:
            for i_local, i_global in enumerate(range(start, end)):
                sims[i_local, i_global] = -1.0
        idx = np.argpartition(-sims, top_k, axis=1)[:, :top_k]
        for i_local, i_global in enumerate(range(start, end)):
            row = sims[i_local]
            top_idx = idx[i_local]
            top_vals = row[top_idx]
            order = np.argsort(-top_vals)
            for j, val in zip(top_idx[order], top_vals[order]):
                if val <= 0:
                    continue
                edges.append((int(i_global), int(j), float(val)))
    return edges


def edges_to_cypher_rows(
    edges: list[tuple[int, int, float]],
    feature_id_fn,
    *,
    cosine_field: str = "cosine",
) -> list[dict]:
    """Convert (i, j, sim) tuples to dicts ready for UNWIND ingestion."""
    return [
        {"a": feature_id_fn(i), "b": feature_id_fn(j), cosine_field: sim}
        for i, j, sim in edges
    ]
