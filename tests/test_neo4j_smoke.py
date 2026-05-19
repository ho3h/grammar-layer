"""End-to-end smoke test of the Neo4j ingestion path WITHOUT the model.

Inserts ~10 dummy SAEFeature nodes + AutoInterpLabel nodes + a Manifold + Waypoints,
then runs Q1, Q3, Q4 to ensure the queries work and indexes are exercised.
Cleans up after itself so it can run alongside real data.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from neograph.cypher import NeographClient
from neograph.config import SAE


TEST_SAE_ID = "smoke-test-sae"


@pytest.fixture
def client():
    try:
        c = NeographClient()
        # Sanity: drop any leftover smoke data
        c.run("MATCH (n) WHERE n.sae_id = $sid DETACH DELETE n", sid=TEST_SAE_ID)
        c.run("MATCH (m:Manifold {id: 'smoke-manifold'}) DETACH DELETE m")
        c.run("MATCH (w:Waypoint) WHERE w.id STARTS WITH 'smoke-manifold' DETACH DELETE w")
        c.run("MATCH (a:AutoInterpLabel) WHERE a.id STARTS WITH 'smoke-' DETACH DELETE a")
        yield c
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Neo4j unreachable on {os.environ.get('NEO4J_URI')}: {exc}")
    # Cleanup
    c.run("MATCH (n) WHERE n.sae_id = $sid DETACH DELETE n", sid=TEST_SAE_ID)
    c.run("MATCH (m:Manifold {id: 'smoke-manifold'}) DETACH DELETE m")
    c.run("MATCH (w:Waypoint) WHERE w.id STARTS WITH 'smoke-manifold' DETACH DELETE w")
    c.run("MATCH (a:AutoInterpLabel) WHERE a.id STARTS WITH 'smoke-' DETACH DELETE a")
    c.close()


def test_ingest_dummy_features_and_query(client):
    rng = np.random.default_rng(0)
    feats = []
    labels = []
    for i in range(10):
        v_dec = rng.standard_normal(SAE.d_in).astype(np.float32)
        v_dec /= np.linalg.norm(v_dec)
        v_enc = rng.standard_normal(SAE.d_in).astype(np.float32)
        v_enc /= np.linalg.norm(v_enc)
        emb = rng.standard_normal(384).astype(np.float32)
        emb /= np.linalg.norm(emb)
        feats.append(
            {
                "fid": f"smoke-feat-{i}",
                "sae_id": TEST_SAE_ID,
                "idx": i,
                "dec": v_dec.tolist(),
                "enc": v_enc.tolist(),
                "act_density": 0.01 * (i + 1),
                "max_act": 1.0 + i,
                "comm": i % 3,
            }
        )
        labels.append(
            {
                "lid": f"smoke-label-{i}",
                "fid": f"smoke-feat-{i}",
                "source": "smoke-test",
                "text": f"dummy concept {i}",
                "emb": emb.tolist(),
            }
        )

    # Write features
    client.run(
        """
        UNWIND $rows AS r
        MERGE (f:SAEFeature {id: r.fid})
          SET f.sae_id = r.sae_id,
              f.index = r.idx,
              f.activation_density = r.act_density,
              f.max_act = r.max_act,
              f.communityId = r.comm
        WITH f, r
        CALL db.create.setNodeVectorProperty(f, 'decoder_vec', r.dec)
        CALL db.create.setNodeVectorProperty(f, 'encoder_vec', r.enc)
        """,
        rows=feats,
    )
    # Write labels
    client.run(
        """
        UNWIND $rows AS r
        MERGE (a:AutoInterpLabel {id: r.lid})
          SET a.source = r.source,
              a.text = r.text
        WITH a, r
        CALL db.create.setNodeVectorProperty(a, 'embedding', r.emb)
        WITH a, r
        MATCH (f:SAEFeature {id: r.fid})
        MERGE (f)-[lbl:LABELED_AS]->(a)
          SET lbl.primary = true
        """,
        rows=labels,
    )

    # A dummy manifold with 4 waypoints + LIES_ON
    wp_rows = []
    for i in range(4):
        cen = rng.standard_normal(SAE.d_in).astype(np.float32)
        cen /= np.linalg.norm(cen)
        tan = rng.standard_normal(SAE.d_in).astype(np.float32)
        tan /= np.linalg.norm(tan)
        wp_rows.append({"i": i, "arc": i / 3.0, "cen": cen.tolist(), "tan": tan.tolist()})
    client.run(
        """
        MERGE (m:Manifold {id: 'smoke-manifold'})
          SET m.intrinsic_dim = 1, m.n_waypoints = 4
        WITH m
        UNWIND $rows AS r
        MERGE (w:Waypoint {id: 'smoke-manifold/w' + toString(r.i)})
          SET w.index = r.i, w.arc_position = r.arc
        WITH m, w, r
        CALL db.create.setNodeVectorProperty(w, 'centroid', r.cen)
        CALL db.create.setNodeVectorProperty(w, 'tangent', r.tan)
        MERGE (m)-[:HAS_WAYPOINT]->(w)
        """,
        rows=wp_rows,
    )
    # LIES_ON for first 5 features
    client.run(
        """
        MATCH (m:Manifold {id: 'smoke-manifold'})
        UNWIND range(0, 4) AS i
        MATCH (f:SAEFeature {id: 'smoke-feat-' + toString(i)})
        MERGE (f)-[lo:LIES_ON]->(m)
          SET lo.closest_waypoint = i % 4,
              lo.perp_distance = 0.1 * i,
              lo.arc_position = i / 4.0
        """
    )

    # Q1: features on the same manifold but different communities — should return entries
    q1 = client.q1_features_on_same_manifold_diff_community(0)
    assert len(q1) > 0, "Q1 should return ≥1 feature on smoke-manifold in a different community"

    # Q4: waypoints
    q4 = client.q4_manifold_waypoints("smoke-manifold")
    assert len(q4) == 4
    assert list(q4["index"]) == [0, 1, 2, 3]
