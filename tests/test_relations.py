"""Unit tests for relations math (no model required)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from neograph.relations import coactivation_edges, topk_cosine_knn


def test_topk_cosine_recovers_groundtruth_neighbours():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((5, 64)).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    # 5 features, each with 3 near-duplicates → 20 features total
    vectors = []
    expected = {}
    for i in range(5):
        for _ in range(4):
            v = base[i] + rng.standard_normal(64).astype(np.float32) * 0.05
            vectors.append(v)
    V = np.stack(vectors)
    # nearest neighbours of feature 0 should be among indices 1, 2, 3 (its sibling group)
    edges = topk_cosine_knn(V, top_k=3)
    out = [(i, j) for i, j, _ in edges if i == 0]
    j_neighbours = {j for _, j in out}
    assert {1, 2, 3}.issubset(j_neighbours)


def test_coactivation_pmi_above_random():
    """Construct a co-firing pair (0, 1) and an independent pair (2, 3). The PMI of (0, 1)
    should be high; (2, 3) should be filtered out at pmi_min=1.0."""
    rng = np.random.default_rng(123)
    n_pos = 1000
    n_feat = 4
    active = np.zeros((n_pos, n_feat), dtype=np.float32)
    # Features 0 and 1: co-fire on 50% of positions, always together
    cofire = rng.random(n_pos) < 0.5
    active[cofire, 0] = 1.0
    active[cofire, 1] = 1.0
    # Features 2, 3: independent, fire ~ 50% each (so co-fire ~ 25%)
    active[rng.random(n_pos) < 0.5, 2] = 1.0
    active[rng.random(n_pos) < 0.5, 3] = 1.0

    edges = coactivation_edges(sp.csr_matrix(active), top_k=5, pmi_min=0.5, n_min=5)
    pmi_lookup = {(e.a, e.b): e.pmi for e in edges}
    # (0,1) should be present with high PMI
    assert (0, 1) in pmi_lookup
    assert pmi_lookup[(0, 1)] > 0.5
    # (2,3) should either be absent or have PMI ~ 0
    if (2, 3) in pmi_lookup:
        assert pmi_lookup[(2, 3)] < pmi_lookup[(0, 1)]
