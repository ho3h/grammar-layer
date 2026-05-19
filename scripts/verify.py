"""Read every exit criterion (PRD §9) and print a summary."""

from __future__ import annotations

import json
from pathlib import Path

from neograph.config import PATHS, SAE
from neograph.cypher import NeographClient
from neograph.evals import nmi_vs_goodfire, rhyme_community_summary, steering_summary
from neograph.util import get_logger

log = get_logger("neograph.verify")


def main() -> int:
    out: dict[str, dict] = {}

    with NeographClient() as c:
        # P1
        out["p1_neo4j"] = c.run("RETURN gds.version() AS gds, apoc.version() AS apoc")[0]
        idx_rows = c.run("SHOW INDEXES YIELD name, type, state WHERE type='VECTOR' RETURN count(*) AS n")
        out["p1_vector_indexes"] = {"n": idx_rows[0]["n"]}

        # P2
        counts = c.run(
            """
            CALL () { MATCH (f:SAEFeature) RETURN count(f) AS nf }
            CALL () { MATCH (a:AutoInterpLabel) RETURN count(a) AS na }
            CALL () { MATCH (p:Prompt) RETURN count(p) AS np }
            CALL () { MATCH (a:Activation) RETURN count(a) AS nact }
            RETURN nf, na, np, nact
            """
        )[0]
        out["p2_features"] = counts

        # P3
        out["p3_relations"] = c.run(
            """
            CALL () { OPTIONAL MATCH ()-[r:CO_ACTIVATES_WITH]->() RETURN count(r) AS n_co }
            CALL () { OPTIONAL MATCH ()-[r:DECODER_SIMILAR]->()  RETURN count(r) AS n_dec }
            CALL () { OPTIONAL MATCH ()-[r:LABEL_SIMILAR]->()    RETURN count(r) AS n_lab }
            CALL () {
              MATCH (f:SAEFeature) WHERE f.communityId IS NOT NULL
              RETURN count(DISTINCT f.communityId) AS n_communities
            }
            RETURN n_co, n_dec, n_lab, n_communities
            """
        )[0]

        # P4
        out["p4_manifolds"] = c.run(
            """
            CALL () { MATCH (m:Manifold) RETURN count(m) AS n_mf }
            CALL () { MATCH (w:Waypoint) RETURN count(w) AS n_wp }
            CALL () { MATCH (m:Manifold) RETURN avg(m.fit_residual) AS avg_resid }
            CALL () { MATCH (c:Concept) RETURN count(c) AS n_concepts }
            RETURN n_mf, n_wp, avg_resid, n_concepts
            """
        )[0]
        out["p4_rhyme"] = rhyme_community_summary(c)

    # P6
    out["p6_steering"] = steering_summary(PATHS.reports / "p6_steering.json")

    # Print
    def line(msg: str) -> None:
        print(msg)

    line("=" * 70)
    line("Neograph end-to-end verification")
    line("=" * 70)
    p1 = out["p1_neo4j"]
    line(f"P1  Neo4j ✓  GDS={p1['gds']}  APOC={p1['apoc']}  vector_indexes={out['p1_vector_indexes']['n']}/4")
    p2 = out["p2_features"]
    line(f"P2  features={p2['nf']:>6}/{SAE.d_sae}  labels={p2['na']:>5}  prompts={p2['np']:>5}  activations={p2['nact']:>6}")
    p3 = out["p3_relations"]
    line(f"P3  CO={p3['n_co']:>6}  DEC={p3['n_dec']:>6}  LAB={p3['n_lab']:>6}  communities={p3['n_communities']}")
    p4 = out["p4_manifolds"]
    line(f"P4  manifolds={p4['n_mf']}  waypoints={p4['n_wp']}  concepts={p4['n_concepts']}  avg_resid={p4['avg_resid']}")
    rh = out["p4_rhyme"]
    line(f"    NMI vs. Goodfire (23 rhyme features): {rh.get('nmi', 0.0):.3f}  matched={rh.get('matched', 0)}/{rh.get('total_in_sae', 0)}")
    if "top_community_labels" in rh:
        labels = rh["top_community_labels"][:5]
        line(f"    Top rhyme community {rh.get('top_community_id')} labels: {labels}")
    p6 = out["p6_steering"]
    if p6 and "error" not in p6 and "linear" in p6 and "manifold" in p6:
        line(f"P6  steering: linear hit={p6['linear']['target_hit_rate']:.2f}  "
             f"manifold hit={p6['manifold']['target_hit_rate']:.2f}  "
             f"ratio={p6.get('manifold_vs_linear_hit_ratio', 0):.2f}x  "
             f"entropy Δ={p6.get('entropy_delta_nat', 0):.2f} nat")
    else:
        line("P6  (no steering report yet)")
    line("=" * 70)

    # Dump JSON
    out_path = PATHS.reports / "verify.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    line(f"Full report: {out_path}")

    return 0


if __name__ == "__main__":
    main()
