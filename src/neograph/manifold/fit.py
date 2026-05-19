"""Per-community manifold fitting: PCA → Kégl polygonal-line principal curve
→ 16 waypoints in 2304-D, with cyclic detection.

Design (ratified by the Plan-agent design review, see memory feedback_manifold_fit.md):
- Fit in PCA space, never UMAP. UMAP is for viz only (manifold/viz.py).
- PCA dim d = clip(min{k : cumvar(k) ≥ 0.95}, 8, 32).
- Kégl polygonal-line is the default principal-curve algorithm (robust to curvature).
- Cyclic detection: if endpoint-to-startpoint distance < median inter-waypoint distance, refit
  with scipy.interpolate.splprep(per=True) periodic cubic B-spline.
- Tangents come from spline derivative when available, else central differences in PCA space,
  then mapped to 2304D via pca.components_.T @ tangent_d, L2-normalized.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import interpolate
from sklearn.decomposition import PCA

from neograph.config import (
    N_WAYPOINTS,
    PCA_DIM_CEIL,
    PCA_DIM_FLOOR,
    PCA_VARIANCE_TARGET,
)
from neograph.util import get_logger

log = get_logger("neograph.manifold.fit")


@dataclass
class FittedManifold:
    n_points: int
    pca_dim: int
    waypoints_pca: np.ndarray  # (n_wp, d)
    waypoints_2304: np.ndarray  # (n_wp, d_model)
    tangents_2304: np.ndarray  # (n_wp, d_model) unit-norm
    arc_positions: np.ndarray  # (n_wp,), 0..1
    arc_length: float
    fit_residual: float  # mean L2 of holdout to nearest waypoint (PCA space) / within-scatter
    is_cyclic: bool
    method: str  # "polygonal+spline" | "polygonal" | "spline-cyclic"
    pca_mean: np.ndarray = field(repr=False)  # (d_model,)
    pca_components: np.ndarray = field(repr=False)  # (d, d_model)


# ============================================================================
# PCA
# ============================================================================


def fit_pca(
    h: np.ndarray,
    variance_target: float = PCA_VARIANCE_TARGET,
    d_floor: int = PCA_DIM_FLOOR,
    d_ceil: int = PCA_DIM_CEIL,
) -> tuple[PCA, int]:
    """PCA on activations `h` (n_points, d_model). Returns (fitted PCA, chosen d)."""
    max_d = min(d_ceil, h.shape[0] - 1, h.shape[1])
    pca = PCA(n_components=max_d, svd_solver="auto", random_state=42).fit(h)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, variance_target) + 1)
    d = int(np.clip(k, d_floor, d_ceil))
    if d > max_d:
        d = max_d
    pca_d = PCA(n_components=d, svd_solver="auto", random_state=42).fit(h)
    return pca_d, d


# ============================================================================
# Kégl polygonal-line principal curve (vendored)
# Based on Kégl, Krzyżak, Linder, Zeger (2000) — "Learning and Design of Principal Curves".
# Simplified: start with PCA-1 axis chord, iteratively add vertices at high-variance points,
# project data to the polyline, refit vertices as projections onto curve.
# ============================================================================


def _project_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Project `p` onto segment a→b. Returns (foot, t∈[0,1] of foot, distance)."""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return a, 0.0, float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    foot = a + t * ab
    return foot, t, float(np.linalg.norm(p - foot))


def _polyline_projections(X: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each point, find its nearest segment of the polyline V (n_v, d).
    Returns (assignments=segment_idx, arc_positions, residuals).
    """
    n = X.shape[0]
    n_seg = V.shape[0] - 1
    seg_assign = np.zeros(n, dtype=np.int64)
    arc_pos = np.zeros(n, dtype=np.float64)
    resid = np.full(n, np.inf, dtype=np.float64)

    seg_lens = np.linalg.norm(V[1:] - V[:-1], axis=1)
    seg_arc_start = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = float(seg_arc_start[-1]) if seg_arc_start[-1] > 0 else 1.0

    for i in range(n):
        for s in range(n_seg):
            _, t, dist = _project_to_segment(X[i], V[s], V[s + 1])
            if dist < resid[i]:
                resid[i] = dist
                seg_assign[i] = s
                arc_pos[i] = (seg_arc_start[s] + t * seg_lens[s]) / total_len
    return seg_assign, arc_pos, resid


def fit_polygonal_line(
    X: np.ndarray,
    n_segments_initial: int = 4,
    n_segments_max: int = 16,
    max_outer_iters: int = 20,
    max_inner_iters: int = 10,
    tol: float = 1e-4,
    random_state: int = 0,
) -> np.ndarray:
    """Kégl-style polygonal-line fit. Returns vertices (n_v, d), n_v ≤ n_segments_max + 1.

    Strategy:
    - Initialise with the longest principal axis of X.
    - Outer loop: project data to polyline, recompute vertex positions as mean of points
      assigned to incident segments.
    - Periodically add a new vertex at the segment with highest residual sum-of-squares
      until we reach n_segments_max.
    """
    rng = np.random.default_rng(random_state)
    Xc = X.astype(np.float64, copy=False)
    n, d = Xc.shape

    # Initial polyline: line along the first principal axis, between projection extremes.
    mean = Xc.mean(axis=0)
    centered = Xc - mean
    _U, _S, Vt = np.linalg.svd(centered, full_matrices=False)
    axis = Vt[0]
    proj = centered @ axis
    lo, hi = float(proj.min()), float(proj.max())
    V = np.array([mean + lo * axis, mean + hi * axis])
    # Distribute initial segments
    for _ in range(n_segments_initial - 1):
        V = _split_longest_segment(V)

    prev_loss = np.inf
    for outer in range(max_outer_iters):
        # Inner: alternating projection / vertex update
        for _inner in range(max_inner_iters):
            seg_assign, arc_pos, _ = _polyline_projections(Xc, V)
            new_V = V.copy()
            # Each vertex k is influenced by points assigned to segments k-1 (right end) and k (left end).
            for k in range(V.shape[0]):
                left_pts = Xc[(seg_assign == k - 1)] if k - 1 >= 0 else None
                right_pts = Xc[(seg_assign == k)] if k < V.shape[0] - 1 else None
                pts = []
                if left_pts is not None and left_pts.size:
                    pts.append(left_pts)
                if right_pts is not None and right_pts.size:
                    pts.append(right_pts)
                if pts:
                    new_V[k] = np.concatenate(pts, axis=0).mean(axis=0)
                else:
                    # No points — nudge slightly toward the centroid
                    new_V[k] = 0.5 * V[k] + 0.5 * mean + 1e-6 * rng.standard_normal(d)
            shift = np.linalg.norm(new_V - V)
            V = new_V
            if shift < tol:
                break

        # Compute current loss (sum of squared residuals)
        _, _, resid = _polyline_projections(Xc, V)
        loss = float((resid**2).sum())
        log.debug("outer=%d n_seg=%d loss=%.3f", outer, V.shape[0] - 1, loss)
        if abs(prev_loss - loss) < tol * max(1.0, prev_loss):
            # Try to grow if we haven't hit the cap
            if V.shape[0] - 1 < n_segments_max:
                V = _split_highest_residual_segment(V, Xc)
                prev_loss = np.inf
            else:
                break
        else:
            prev_loss = loss

        # Allow growing every couple of iterations
        if outer % 2 == 1 and V.shape[0] - 1 < n_segments_max:
            V = _split_highest_residual_segment(V, Xc)

    return V


def _split_longest_segment(V: np.ndarray) -> np.ndarray:
    seg_lens = np.linalg.norm(V[1:] - V[:-1], axis=1)
    k = int(np.argmax(seg_lens))
    midpoint = 0.5 * (V[k] + V[k + 1])
    return np.insert(V, k + 1, midpoint, axis=0)


def _split_highest_residual_segment(V: np.ndarray, X: np.ndarray) -> np.ndarray:
    seg_assign, _arc, resid = _polyline_projections(X, V)
    n_seg = V.shape[0] - 1
    seg_resid = np.zeros(n_seg)
    for s in range(n_seg):
        mask = seg_assign == s
        if mask.any():
            seg_resid[s] = float((resid[mask] ** 2).sum())
    k = int(np.argmax(seg_resid))
    midpoint = 0.5 * (V[k] + V[k + 1])
    return np.insert(V, k + 1, midpoint, axis=0)


# ============================================================================
# Cyclic detection + periodic B-spline fallback
# ============================================================================


def detect_cyclic(V: np.ndarray) -> bool:
    """V is (n_v, d) polyline vertices. True if endpoint-to-startpoint distance is below
    the median inter-vertex distance — suggesting a closed loop."""
    if V.shape[0] < 4:
        return False
    inter = np.linalg.norm(V[1:] - V[:-1], axis=1)
    close_gap = float(np.linalg.norm(V[0] - V[-1]))
    return close_gap < float(np.median(inter))


def detect_cyclic_from_data(X: np.ndarray, angular_coverage_min: float = 0.85) -> bool:
    """Robust cyclic detection from the raw points in PCA space.

    Project to the top-2 PCs, compute the angular distribution of points around their centroid.
    If the angular range covers ≥ `angular_coverage_min` of the full circle, treat as cyclic.
    """
    if X.shape[0] < 16 or X.shape[1] < 2:
        return False
    # Use first two PCs of the data (already in PCA space, so just take cols 0,1)
    pts = X[:, :2] - X[:, :2].mean(axis=0)
    radii = np.linalg.norm(pts, axis=1)
    if float(np.median(radii)) < 1e-6:
        return False
    thetas = np.arctan2(pts[:, 1], pts[:, 0])
    # Bin angles into 24 sectors and check fraction non-empty
    bins, _ = np.histogram(thetas, bins=24, range=(-np.pi, np.pi))
    covered = float((bins > 0).sum()) / 24.0
    if covered < angular_coverage_min:
        return False
    # Also check that the radial coordinate is concentrated (i.e., a ring, not a disk)
    radial_cv = float(np.std(radii) / (np.mean(radii) + 1e-9))
    return radial_cv < 0.5


def periodic_bspline_waypoints(V: np.ndarray, n_waypoints: int) -> np.ndarray:
    """Fit a periodic cubic B-spline through V (treating last point ≈ first) and sample n_waypoints
    uniformly along arc length."""
    pts = V.copy()
    # Close the loop if not already
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-6:
        pts = np.vstack([pts, pts[:1]])
    # splprep wants (d, n) shape
    tck, _u = interpolate.splprep(pts.T.tolist(), s=0, per=True, k=min(3, pts.shape[0] - 1))
    u_sample = np.linspace(0.0, 1.0, n_waypoints, endpoint=False)
    samples = np.array(interpolate.splev(u_sample, tck)).T  # (n_wp, d)
    return samples


# ============================================================================
# Sampling waypoints uniformly along arc length (polyline)
# ============================================================================


def sample_waypoints_polyline(V: np.ndarray, n_waypoints: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample `n_waypoints` waypoints uniformly along the arc length of polyline V.

    Returns (waypoints (n_wp, d), arc_positions (n_wp,) in [0, 1]).
    """
    seg_lens = np.linalg.norm(V[1:] - V[:-1], axis=1)
    total = float(seg_lens.sum())
    if total < 1e-9:
        return np.repeat(V[:1], n_waypoints, axis=0), np.linspace(0, 1, n_waypoints)
    seg_arc_start = np.concatenate([[0.0], np.cumsum(seg_lens)]) / total
    arc_targets = np.linspace(0.0, 1.0, n_waypoints)
    wp = np.zeros((n_waypoints, V.shape[1]))
    for i, a in enumerate(arc_targets):
        # Find segment containing arc-position a
        s = int(np.searchsorted(seg_arc_start, a, side="right") - 1)
        s = max(0, min(s, V.shape[0] - 2))
        seg_a, seg_b = seg_arc_start[s], seg_arc_start[s + 1]
        t = (a - seg_a) / max(seg_b - seg_a, 1e-12)
        wp[i] = V[s] + t * (V[s + 1] - V[s])
    return wp, arc_targets


# ============================================================================
# Tangent computation
# ============================================================================


def _tangents_central_diff(waypoints: np.ndarray, is_cyclic: bool) -> np.ndarray:
    """Central differences (cyclic wrap-around if applicable)."""
    n_wp = waypoints.shape[0]
    tangents = np.zeros_like(waypoints)
    for i in range(n_wp):
        if is_cyclic:
            prev = waypoints[(i - 1) % n_wp]
            nxt = waypoints[(i + 1) % n_wp]
        else:
            prev = waypoints[max(i - 1, 0)]
            nxt = waypoints[min(i + 1, n_wp - 1)]
        t = nxt - prev
        norm = float(np.linalg.norm(t))
        if norm > 1e-9:
            tangents[i] = t / norm
    return tangents


# ============================================================================
# Top-level: fit a community
# ============================================================================


def fit_community_manifold(
    activations_2304: np.ndarray,
    n_waypoints: int = N_WAYPOINTS,
    holdout_frac: float = 0.2,
    n_segments_initial: int = 4,
    n_segments_max: int = 16,
    method: Literal["auto", "polygonal", "spline"] = "auto",
    random_state: int = 42,
) -> FittedManifold | None:
    """Top-level: PCA → polygonal-line → cyclic-detect → 16 waypoints in PCA → back-project to 2304.

    Returns None if too few points to fit (< n_waypoints + 4).
    """
    h = np.asarray(activations_2304, dtype=np.float32)
    if h.shape[0] < n_waypoints + 4:
        log.warning("Too few points (%d) to fit manifold; skipping", h.shape[0])
        return None

    rng = np.random.default_rng(random_state)
    idx = rng.permutation(h.shape[0])
    n_train = int((1 - holdout_frac) * h.shape[0])
    train_idx, hold_idx = idx[:n_train], idx[n_train:]
    h_train = h[train_idx]
    h_hold = h[hold_idx]

    # 1. PCA
    pca, d = fit_pca(h_train)
    X_train = pca.transform(h_train)
    X_hold = pca.transform(h_hold)

    # 2. Polygonal-line fit in PCA space
    V = fit_polygonal_line(
        X_train,
        n_segments_initial=n_segments_initial,
        n_segments_max=n_segments_max,
        random_state=random_state,
    )

    # 3. Cyclic detection + waypoint sampling
    is_cyclic = detect_cyclic(V) or detect_cyclic_from_data(X_train)
    # If data is cyclic but the polyline isn't closed, manually close it before spline fitting
    if is_cyclic and not detect_cyclic(V):
        # Reorder vertices by angular position in PCA-2 plane around the centroid
        center = V[:, :2].mean(axis=0)
        thetas = np.arctan2(V[:, 1] - center[1], V[:, 0] - center[0])
        order = np.argsort(thetas)
        V = V[order]
    fit_method = "polygonal"
    if is_cyclic and method != "polygonal":
        try:
            wp_pca = periodic_bspline_waypoints(V, n_waypoints)
            fit_method = "spline-cyclic"
        except Exception as exc:  # noqa: BLE001
            log.warning("Periodic spline fit failed (%s) — falling back to polyline", exc)
            wp_pca, _arc = sample_waypoints_polyline(V, n_waypoints)
            fit_method = "polygonal"
    else:
        wp_pca, _arc = sample_waypoints_polyline(V, n_waypoints)
    arc_positions = (
        np.linspace(0.0, 1.0, n_waypoints, endpoint=not is_cyclic)
        if not is_cyclic
        else np.linspace(0.0, 1.0, n_waypoints, endpoint=False)
    )
    seg_lens = np.linalg.norm(wp_pca[1:] - wp_pca[:-1], axis=1)
    arc_length = float(seg_lens.sum())
    if is_cyclic:
        arc_length += float(np.linalg.norm(wp_pca[0] - wp_pca[-1]))

    # 4. Back-project to 2304 (PCA-inverse is exact)
    wp_2304 = pca.inverse_transform(wp_pca).astype(np.float32)

    # 5. Tangents (PCA-space central differences, then map to 2304)
    tangents_pca = _tangents_central_diff(wp_pca, is_cyclic)
    # Mapping: tangent in 2304D = pca.components_.T @ tangent_d (components_ is (d, d_model))
    tangents_2304 = (tangents_pca @ pca.components_).astype(np.float32)
    norms = np.linalg.norm(tangents_2304, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    tangents_2304 = tangents_2304 / norms

    # 6. Fit residual on holdout, normalised by within-scatter
    if X_hold.shape[0] > 0:
        # Distance to nearest waypoint in PCA space
        diffs = X_hold[:, None, :] - wp_pca[None, :, :]
        d2 = (diffs**2).sum(-1)
        nearest = np.sqrt(d2.min(axis=1))
        residual = float(np.median(nearest))
    else:
        residual = 0.0
    within_scatter = float(np.median(np.linalg.norm(X_train - X_train.mean(axis=0), axis=1)))
    fit_residual = residual / max(within_scatter, 1e-9)

    return FittedManifold(
        n_points=int(h.shape[0]),
        pca_dim=d,
        waypoints_pca=wp_pca.astype(np.float32),
        waypoints_2304=wp_2304,
        tangents_2304=tangents_2304,
        arc_positions=arc_positions.astype(np.float32),
        arc_length=float(arc_length),
        fit_residual=fit_residual,
        is_cyclic=bool(is_cyclic),
        method=fit_method,
        pca_mean=pca.mean_.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
    )
