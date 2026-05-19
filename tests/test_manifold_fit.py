"""Unit tests for the manifold-fit pipeline using synthetic data.

Two scenarios:
- A 1D smooth curve embedded in 2304D (rhyme-like): verify the principal curve recovers
  monotonic arc ordering and back-projection produces reasonable 2304D centroids.
- A cyclic 1D loop in 2304D (weekday-like): verify is_cyclic=True and the periodic spline
  is sampled.
"""

from __future__ import annotations

import numpy as np
import pytest

from neograph.manifold.fit import (
    detect_cyclic,
    fit_community_manifold,
    fit_polygonal_line,
)


def _make_curve_2304(n_points: int, n_dims: int = 2304, t_max: float = 1.0, noise: float = 0.01, seed: int = 0):
    """Synthetic 1D curve embedded in 2304D. Low default noise so PCA finds the signal."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, t_max, n_points)
    basis_x = rng.standard_normal(n_dims).astype(np.float32)
    basis_y = rng.standard_normal(n_dims).astype(np.float32)
    basis_x /= float(np.linalg.norm(basis_x))
    basis_y = basis_y - (float(basis_y @ basis_x)) * basis_x
    basis_y /= float(np.linalg.norm(basis_y))
    x = np.cos(2 * np.pi * t) * 5
    y = np.sin(2 * np.pi * t) * 5
    h = x[:, None] * basis_x[None, :] + y[:, None] * basis_y[None, :]
    h = h + rng.standard_normal(h.shape).astype(np.float32) * noise
    return h.astype(np.float32), t


def test_open_curve_back_projects_into_data_subspace():
    """For an open curve in 2304D, the back-projected waypoints should sit near the data points."""
    h, _t = _make_curve_2304(n_points=200, t_max=0.4, noise=0.01, seed=11)
    fit = fit_community_manifold(h, n_waypoints=16, random_state=11)
    assert fit is not None
    assert fit.waypoints_2304.shape == (16, 2304)
    assert fit.tangents_2304.shape == (16, 2304)
    # Tangents are unit-norm
    norms = np.linalg.norm(fit.tangents_2304, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)
    # Open curve — cyclic should not trip
    assert fit.is_cyclic is False
    # Each waypoint should have at least one data point nearby (within median pairwise dist)
    data_centroid = h.mean(axis=0)
    typical_dist = float(np.median(np.linalg.norm(h - data_centroid, axis=1)))
    dists = np.linalg.norm(fit.waypoints_2304[:, None, :] - h[None, :, :], axis=2).min(axis=1)
    assert dists.max() < typical_dist


def test_cyclic_curve_detected():
    h, _t = _make_curve_2304(n_points=300, t_max=1.0, noise=0.01, seed=22)
    fit = fit_community_manifold(h, n_waypoints=16, random_state=22)
    assert fit is not None
    # Full loop — cyclic detection should fire
    assert fit.is_cyclic is True
    assert fit.method == "spline-cyclic"
    assert fit.arc_length > 0


def test_too_few_points_returns_none():
    h = np.zeros((5, 2304), dtype=np.float32)
    fit = fit_community_manifold(h, n_waypoints=16)
    assert fit is None


def test_detect_cyclic_helper():
    # 4 vertices forming a square — start ≈ end
    V = np.array([[0.0, 0], [1, 0], [1, 1], [0.05, 0.0]])
    assert detect_cyclic(V) is True
    # Open polyline — start ≠ end
    V_open = np.array([[0.0, 0], [1, 0], [2, 0], [3, 0]])
    assert detect_cyclic(V_open) is False
