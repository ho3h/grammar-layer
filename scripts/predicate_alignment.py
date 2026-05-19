"""Cross-model predicate-feature alignment — the structural test.

The shallow Hungarian-on-label-centroids approach was too coarse (Theo's critique 2026-05-12).
The structural test is: for features that participate in MULTIPLE circuits in Gemma
(the predicate backbone), do they have label-similar counterparts in GPT-2 that ALSO
participate in multiple GPT-2 circuits?

If yes → predicate-level features are cross-model universal at the circuit level.
If no → the Gemma predicate backbone is Gemma-specific. Either is publishable.

Required input:
- N≥8 Circuits per model populated by scripts/causal_attribution_v2.py
- Both models' AutoInterpLabel embeddings already in the label_emb vector index

Output:
- reports/predicate_alignment.json — per-Gemma-feature: GPT-2 counterparts + their circuit
  participation
- reports/predicate_alignment_summary.txt — readable verdict
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from neograph.config import PATHS, SAE as GEMMA_SAE
from neograph.cypher import NeographClient
from neograph.util import get_logger

log = get_logger("neograph.predicate")

GPT2_SAE_ID = "gpt2-small-res-jb/L8"


def predicate_features(c: NeographClient, model_prefix: str, sae_id: str, min_circuits: int = 3,
                       top_k: int = 25) -> list[dict]:
    """Find features appearing in ≥min_circuits circuits for this model, ranked by total |attribution|."""
    rows = c.run(
        """
        MATCH (cir:Circuit)-[inc:INCLUDES]->(f:SAEFeature)
        WHERE cir.model = $model AND f.sae_id = $sae_id
        WITH f, count(DISTINCT cir) AS n_circuits, collect(DISTINCT cir.id) AS circuits,
             sum(inc.attribution) AS total_attr, avg(abs(inc.attribution)) AS mean_abs_attr
        WHERE n_circuits >= $min_circuits
        OPTIONAL MATCH (f)-[:LABELED_AS]->(a:AutoInterpLabel)
        RETURN f.index AS feature, f.id AS fid, n_circuits, circuits, total_attr, mean_abs_attr,
               a.text AS label, a.embedding AS embedding
        ORDER BY n_circuits DESC, mean_abs_attr DESC LIMIT $top_k
        """,
        model=model_prefix, sae_id=sae_id, min_circuits=min_circuits, top_k=top_k,
    )
    return rows


def find_cross_model_counterparts(c: NeographClient, gemma_pred: list[dict],
                                  k_neighbors: int = 10, cos_threshold: float = 0.7,
                                  min_gpt2_circuits: int = 2) -> list[dict]:
    """For each Gemma predicate feature, find GPT-2 features with similar autointerp label,
    and check whether THEY participate in multiple GPT-2 circuits."""
    results = []
    for gf in gemma_pred:
        if not gf.get("embedding"):
            continue
        # Vector-index query restricted to GPT-2 features
        nbrs = c.run(
            """
            CALL db.index.vector.queryNodes('label_emb', $k, $emb) YIELD node, score
            MATCH (g:SAEFeature)-[:LABELED_AS]->(node)
            WHERE g.sae_id = $sae_id AND score >= $threshold
            OPTIONAL MATCH (cir:Circuit)-[inc:INCLUDES]->(g)
            WHERE cir.model = 'gpt2'
            WITH g, node, score,
                 count(DISTINCT cir) AS n_circuits,
                 collect(DISTINCT cir.id) AS circuits,
                 sum(inc.attribution) AS total_attr
            RETURN g.index AS gpt2_feature, g.id AS gpt2_fid,
                   node.text AS gpt2_label, score AS label_cosine,
                   n_circuits AS gpt2_n_circuits, circuits AS gpt2_circuits, total_attr
            ORDER BY score DESC LIMIT $k
            """,
            k=k_neighbors, emb=gf["embedding"], sae_id=GPT2_SAE_ID, threshold=cos_threshold,
        )
        # Find the GPT-2 counterpart with the most circuit participation
        promoted = [n for n in nbrs if n.get("gpt2_n_circuits", 0) >= min_gpt2_circuits]
        promoted.sort(key=lambda n: (-int(n.get("gpt2_n_circuits", 0)), -float(n.get("label_cosine", 0))))
        results.append({
            "gemma_feature": gf["feature"],
            "gemma_label": gf.get("label"),
            "gemma_n_circuits": gf["n_circuits"],
            "gemma_circuits": gf["circuits"],
            "gemma_mean_abs_attr": gf["mean_abs_attr"],
            "n_gpt2_candidates": len(nbrs),
            "n_gpt2_predicate_candidates": len(promoted),
            "best_gpt2_counterpart": promoted[0] if promoted else None,
            "all_gpt2_candidates": nbrs[:5],
        })
    return results


def main() -> int:
    with NeographClient() as c:
        log.info("Finding Gemma predicate features (≥3 circuits) ...")
        gemma_pred = predicate_features(c, "gemma", GEMMA_SAE.neograph_id, min_circuits=3, top_k=30)
        log.info("Gemma predicate features: %d", len(gemma_pred))
        log.info("Finding GPT-2 predicate features (≥3 circuits) ...")
        gpt2_pred = predicate_features(c, "gpt2", GPT2_SAE_ID, min_circuits=3, top_k=30)
        log.info("GPT-2 predicate features: %d", len(gpt2_pred))

        if not gemma_pred:
            log.warning("No Gemma predicate features yet — has causal_attribution_v2 finished for Gemma?")
            return 1

        log.info("Looking for cross-model counterparts ...")
        alignment = find_cross_model_counterparts(c, gemma_pred, k_neighbors=8, cos_threshold=0.6,
                                                  min_gpt2_circuits=2)

    # Tally
    n_aligned = sum(1 for r in alignment if r["best_gpt2_counterpart"] is not None)
    log.info("Aligned (Gemma predicate has a multi-circuit GPT-2 counterpart): %d / %d",
             n_aligned, len(alignment))

    print("\n=== Gemma predicate features (≥3 circuits) and their GPT-2 counterparts ===")
    print(f"{'gFeat':>6} {'gCirc':>5}  Gemma label / best GPT-2 counterpart")
    for r in alignment:
        g_label = (r["gemma_label"] or "")[:55]
        best = r["best_gpt2_counterpart"]
        if best:
            p_label = (best.get("gpt2_label") or "")[:55]
            print(f"{r['gemma_feature']:>6} {r['gemma_n_circuits']:>5d}  "
                  f"{g_label}")
            print(f"       ↔ gpt2 feat {best['gpt2_feature']:>6} in {best['gpt2_n_circuits']} circuits, "
                  f"cos={best['label_cosine']:+.3f}  {p_label}")
        else:
            print(f"{r['gemma_feature']:>6} {r['gemma_n_circuits']:>5d}  "
                  f"{g_label}   [no multi-circuit GPT-2 counterpart at cos≥0.6]")

    print("\n=== GPT-2 predicate features (≥3 circuits), for comparison ===")
    for r in gpt2_pred[:15]:
        print(f"  feat {r['feature']:>5}  in_{r['n_circuits']} circuits  "
              f"Σattr={r['total_attr']:+.3f}  {r.get('label', '')[:70]}")

    PATHS.reports.mkdir(parents=True, exist_ok=True)
    out = PATHS.reports / "predicate_alignment.json"
    # Strip embedding from output (large + not human-readable)
    clean_gemma = [{k: v for k, v in r.items() if k != "embedding"} for r in gemma_pred]
    clean_gpt2 = [{k: v for k, v in r.items() if k != "embedding"} for r in gpt2_pred]
    out.write_text(json.dumps({
        "gemma_predicate_features": clean_gemma,
        "gpt2_predicate_features": clean_gpt2,
        "alignment": alignment,
        "n_gemma_predicates": len(gemma_pred),
        "n_gpt2_predicates": len(gpt2_pred),
        "n_aligned": n_aligned,
    }, indent=2, default=str))
    log.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
