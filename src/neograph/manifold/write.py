"""Persist fitted manifolds to Neo4j (Manifold, Waypoint, :HAS_WAYPOINT, :NEXT, :LIES_ON)."""

from __future__ import annotations

import numpy as np

from neograph.config import SAE
from neograph.cypher import NeographClient
from neograph.manifold.fit import FittedManifold
from neograph.util import get_logger

log = get_logger("neograph.manifold.write")


def write_manifold(
    client: NeographClient,
    *,
    manifold_id: str,
    fit: FittedManifold,
    feature_indices: list[int],
    feature_positions_pca: np.ndarray,
    layer_id: str | None = None,
    notes: str = "",
) -> None:
    """Write Manifold + Waypoint nodes and project features via :LIES_ON.

    feature_positions_pca: (n_features, d_pca) — each community feature's mean activation
        in PCA space, used to compute its closest waypoint and arc position.
    """
    layer_id = layer_id or f"{SAE.release}/L{SAE.layer}"

    # 1. Manifold + Waypoint nodes
    wp_rows = [
        {
            "index": int(i),
            "arc": float(fit.arc_positions[i]),
            "centroid": fit.waypoints_2304[i].tolist(),
            "tangent": fit.tangents_2304[i].tolist(),
        }
        for i in range(fit.waypoints_2304.shape[0])
    ]

    with client.session() as s:
        s.run(
            """
            MERGE (m:Manifold {id: $mid})
              SET m.layer_id = $layer_id,
                  m.intrinsic_dim = 1,
                  m.pca_dim = $pca_dim,
                  m.method = $method,
                  m.n_waypoints = $n_wp,
                  m.arc_length = $arc_len,
                  m.fit_residual = $resid,
                  m.is_cyclic = $cyclic,
                  m.n_points = $n_pts,
                  m.notes = $notes
            """,
            mid=manifold_id,
            layer_id=layer_id,
            pca_dim=int(fit.pca_dim),
            method=fit.method,
            n_wp=int(fit.waypoints_2304.shape[0]),
            arc_len=float(fit.arc_length),
            resid=float(fit.fit_residual),
            cyclic=bool(fit.is_cyclic),
            n_pts=int(fit.n_points),
            notes=notes,
        )
        # Waypoints (batch)
        s.run(
            """
            MATCH (m:Manifold {id: $mid})
            UNWIND $rows AS r
            MERGE (w:Waypoint {id: $mid + '/w' + toString(r.index)})
              SET w.index = r.index,
                  w.arc_position = r.arc
            WITH m, r, w
            CALL db.create.setNodeVectorProperty(w, 'centroid', r.centroid)
            CALL db.create.setNodeVectorProperty(w, 'tangent', r.tangent)
            MERGE (m)-[:HAS_WAYPOINT]->(w)
            """,
            mid=manifold_id,
            rows=wp_rows,
        )
        # Chain :NEXT waypoints in order
        s.run(
            """
            MATCH (m:Manifold {id: $mid})-[:HAS_WAYPOINT]->(w:Waypoint)
            WITH m, w ORDER BY w.index
            WITH m, collect(w) AS wps
            UNWIND range(0, size(wps)-2) AS i
            WITH wps[i] AS a, wps[i+1] AS b
            MERGE (a)-[r:NEXT]->(b)
              SET r.arc_delta = b.arc_position - a.arc_position
            """,
            mid=manifold_id,
        )
        if fit.is_cyclic:
            s.run(
                """
                MATCH (m:Manifold {id: $mid})-[:HAS_WAYPOINT]->(w:Waypoint)
                WITH m, w ORDER BY w.index
                WITH m, collect(w) AS wps
                WITH wps[size(wps)-1] AS a, wps[0] AS b
                MERGE (a)-[r:NEXT]->(b)
                  SET r.arc_delta = (1.0 - a.arc_position) + b.arc_position
                """,
                mid=manifold_id,
            )

    # 2. Project features onto waypoints
    if feature_positions_pca.shape[0] == 0:
        return
    # Closest waypoint per feature
    wp_pca = fit.waypoints_pca
    diffs = feature_positions_pca[:, None, :] - wp_pca[None, :, :]
    d2 = (diffs**2).sum(-1)
    nearest = np.argmin(d2, axis=1)
    perp = np.sqrt(d2.min(axis=1))
    arc = np.array([fit.arc_positions[int(j)] for j in nearest], dtype=np.float32)

    rows = [
        {
            "feat_id": _feature_id(int(feature_indices[i])),
            "wp": int(nearest[i]),
            "perp": float(perp[i]),
            "arc": float(arc[i]),
        }
        for i in range(len(feature_indices))
    ]
    with client.session() as s:
        s.run(
            """
            MATCH (m:Manifold {id: $mid})
            UNWIND $rows AS r
            MATCH (f:SAEFeature {id: r.feat_id})
            MERGE (f)-[lo:LIES_ON]->(m)
              SET lo.closest_waypoint = r.wp,
                  lo.perp_distance = r.perp,
                  lo.arc_position = r.arc
            """,
            mid=manifold_id,
            rows=rows,
        )


def _feature_id(idx: int) -> str:
    return f"{SAE.neograph_id}/F{idx:05d}"
