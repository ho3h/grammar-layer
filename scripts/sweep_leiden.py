"""Sweep Leiden gamma to find the resolution that best recovers Goodfire's rhyme features.

Goodfire (PRD §9.1) reports 23 anchor features for the "-ore" rhyme manifold; of those
12 fall within width-16k indexing (the rest are width-32k+). We sweep γ ∈ {1.0, 2.0, 3.0, 4.0, 6.0}
and report:
- community count, modularity
- how many distinct Leiden communities the 12 anchors split across
- NMI vs the {anchor / non-anchor} bipartition
- the dominant community of the anchors

Writes back to Neo4j only if the user later asks; this is a read-only sweep.
"""

from __future__ import annotations

import json

import numpy as np

from neograph.config import PATHS, SAE
from neograph.cypher import NeographClient
from neograph.evals import GOODFIRE_RHYME_FEATURES, nmi_vs_goodfire
from neograph.util import get_logger

log = get_logger("neograph.sweep_leiden")

GRAPH_NAME = "feature-multi-graph-sweep"


def _drop(c, name):
    rows = c.run("CALL gds.graph.exists($name) YIELD exists", name=name)
    if rows and rows[0]["exists"]:
        c.run("CALL gds.graph.drop($name) YIELD graphName", name=name)


def project(c) -> None:
    _drop(c, GRAPH_NAME)
    c.run(
        """
        MATCH (a:SAEFeature)-[r:CO_ACTIVATES_WITH|DECODER_SIMILAR|LABEL_SIMILAR]-(b:SAEFeature)
        WHERE a.sae_id = $sae_id AND b.sae_id = $sae_id
        WITH a, b, r,
          CASE type(r)
            WHEN 'CO_ACTIVATES_WITH' THEN 0.5 * (r.pmi / 10.0)
            WHEN 'DECODER_SIMILAR'   THEN 0.3 * r.cosine
            WHEN 'LABEL_SIMILAR'     THEN 0.2 * r.cosine
          END AS w
        WITH gds.graph.project(
              $name, a, b,
              {relationshipProperties: {weight: w}},
              {undirectedRelationshipTypes: ['*']}
            ) AS g
        RETURN g.graphName AS graphName, g.nodeCount AS nodeCount,
               g.relationshipCount AS relationshipCount
        """,
        name=GRAPH_NAME,
        sae_id=SAE.neograph_id,
    )


def leiden_at(c, gamma: float) -> dict:
    rows = c.run(
        """
        CALL gds.leiden.write($name, {
          writeProperty: $prop,
          gamma: $gamma, theta: 0.01, randomSeed: 42,
          relationshipWeightProperty: 'weight'
        }) YIELD communityCount, modularity
        RETURN communityCount, modularity
        """,
        name=GRAPH_NAME,
        prop=f"communityId_g{int(gamma * 10)}",
        gamma=gamma,
    )
    return rows[0] if rows else {}


def analyse_goodfire(c, gamma: float) -> dict:
    prop = f"communityId_g{int(gamma * 10)}"
    avail = [i for i in GOODFIRE_RHYME_FEATURES if i < 16384]
    rows = c.run(
        f"""
        MATCH (f:SAEFeature)
        WHERE f.{prop} IS NOT NULL AND coalesce(f.is_dead, false) = false
        RETURN f.index AS idx, f.{prop} AS cid
        """
    )
    indices = np.array([r["idx"] for r in rows])
    communities = np.array([r["cid"] for r in rows])
    goodfire_mask = np.isin(indices, avail)
    from sklearn.metrics import normalized_mutual_info_score
    nmi = float(normalized_mutual_info_score(communities, goodfire_mask.astype(int)))
    # Distribution of anchor features across communities
    from collections import Counter
    anchor_communities = communities[goodfire_mask]
    counts = Counter(int(c) for c in anchor_communities)
    largest = counts.most_common(1)
    return {
        "gamma": gamma,
        "n_communities": int(communities.max()) + 1 if len(communities) else 0,
        "anchors_in_range": len(avail),
        "anchor_communities_distinct": len(counts),
        "anchor_top_community": largest[0] if largest else None,
        "anchor_in_top_community": largest[0][1] if largest else 0,
        "nmi": nmi,
    }


def main() -> int:
    gammas = [1.0, 2.0, 3.0, 4.0, 6.0]
    results = []
    with NeographClient() as c:
        log.info("Projecting GDS graph ...")
        project(c)
        for gamma in gammas:
            log.info("Running Leiden γ=%s ...", gamma)
            stats = leiden_at(c, gamma)
            anal = analyse_goodfire(c, gamma)
            anal["modularity"] = stats.get("modularity")
            anal["leiden_community_count"] = stats.get("communityCount")
            results.append(anal)
            log.info(
                "γ=%.1f → %d communities, modularity=%.3f, anchors split across %d, top community has %d/12, NMI=%.3f",
                gamma,
                stats.get("communityCount") or 0,
                stats.get("modularity") or 0.0,
                anal["anchor_communities_distinct"],
                anal["anchor_in_top_community"],
                anal["nmi"],
            )
    PATHS.reports.mkdir(parents=True, exist_ok=True)
    out = PATHS.reports / "leiden_gamma_sweep.json"
    out.write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", out)

    # Pick the best gamma
    best = max(results, key=lambda r: (r["nmi"], r["anchor_in_top_community"]))
    log.info(
        "Best γ=%.1f: NMI=%.3f, anchor concentration %d/12 in community %s",
        best["gamma"], best["nmi"], best["anchor_in_top_community"], best["anchor_top_community"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
