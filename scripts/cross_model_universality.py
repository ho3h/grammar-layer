"""Proper cross-model universality test (the test of Theo's meta-question).

For every (gemma_community, gpt2_community) pair, compute:
- Label-embedding centroid cosine similarity (how similar are the auto-interp narratives?)
- Intra-community density ratio (do both communities have similar clustering structure?)
- A "concept match" score that combines them.

Output:
- For each concept pattern (weekday, money, code, prefix), show the best-matching
  gemma↔gpt2 community pair.
- Heatmap data of all pairwise community-cosine similarities (saved as .json
  for downstream rendering).
- Hungarian assignment between Gemma and GPT-2 communities for universality.

Run AFTER scripts/cross_model_gpt2.py completes Leiden.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from neograph.config import PATHS, SAE as GEMMA_SAE
from neograph.cypher import NeographClient
from neograph.util import get_logger

log = get_logger("neograph.universality")

GPT2_SAE_ID = "gpt2-small-res-jb/L8"


def _community_centroids(c: NeographClient, sae_id: str, community_prop: str) -> dict[int, np.ndarray]:
    """For each community, average the MiniLM label embeddings of its features."""
    rows = c.run(
        f"""
        MATCH (f:SAEFeature)-[:LABELED_AS]->(a:AutoInterpLabel)
        WHERE f.sae_id = $sae_id AND f.{community_prop} IS NOT NULL
        RETURN f.{community_prop} AS cid, a.embedding AS emb
        """,
        sae_id=sae_id,
    )
    by_cid: dict[int, list[np.ndarray]] = defaultdict(list)
    for r in rows:
        by_cid[int(r["cid"])].append(np.asarray(r["emb"], dtype=np.float32))
    return {cid: np.stack(embs).mean(axis=0) for cid, embs in by_cid.items() if embs}


def _community_sizes(c: NeographClient, sae_id: str, prop: str) -> dict[int, int]:
    rows = c.run(
        f"""
        MATCH (f:SAEFeature)
        WHERE f.sae_id = $sae_id AND f.{prop} IS NOT NULL
        RETURN f.{prop} AS cid, count(f) AS n
        """,
        sae_id=sae_id,
    )
    return {int(r["cid"]): int(r["n"]) for r in rows}


def _community_density(c: NeographClient, sae_id: str, prop: str) -> dict[int, float]:
    """Intra-community edge density across the three relation types."""
    rows = c.run(
        f"""
        MATCH (a:SAEFeature)-[r:CO_ACTIVATES_WITH|DECODER_SIMILAR|LABEL_SIMILAR]->(b:SAEFeature)
        WHERE a.sae_id = $sae_id AND b.sae_id = $sae_id
              AND a.{prop} = b.{prop} AND a.{prop} IS NOT NULL
        RETURN a.{prop} AS cid, count(r) AS n_intra
        """,
        sae_id=sae_id,
    )
    intra = {int(r["cid"]): int(r["n_intra"]) for r in rows}
    sizes = _community_sizes(c, sae_id, prop)
    density = {}
    for cid, n in sizes.items():
        max_possible = n * (n - 1)
        density[cid] = (intra.get(cid, 0) / max_possible) if max_possible > 0 else 0.0
    return density


def _representative_labels(c: NeographClient, sae_id: str, prop: str, cid: int, k: int = 5) -> list[str]:
    rows = c.run(
        f"""
        MATCH (f:SAEFeature)-[:LABELED_AS]->(a:AutoInterpLabel)
        WHERE f.sae_id = $sae_id AND f.{prop} = $cid
        RETURN a.text AS text, f.activation_density AS dens
        ORDER BY dens DESC LIMIT $k
        """,
        sae_id=sae_id, cid=cid, k=k,
    )
    return [r["text"] for r in rows if r.get("text")]


def main() -> int:
    with NeographClient() as c:
        log.info("Building Gemma community centroids ...")
        gemma_cents = _community_centroids(c, GEMMA_SAE.neograph_id, "communityId")
        gemma_sizes = _community_sizes(c, GEMMA_SAE.neograph_id, "communityId")
        gemma_density = _community_density(c, GEMMA_SAE.neograph_id, "communityId")
        log.info("Gemma: %d communities", len(gemma_cents))

        log.info("Building GPT-2 community centroids ...")
        gpt2_cents = _community_centroids(c, GPT2_SAE_ID, "communityId_gpt2")
        gpt2_sizes = _community_sizes(c, GPT2_SAE_ID, "communityId_gpt2")
        gpt2_density = _community_density(c, GPT2_SAE_ID, "communityId_gpt2")
        log.info("GPT-2: %d communities", len(gpt2_cents))

        if not gemma_cents or not gpt2_cents:
            log.error("missing communities in one model — was scripts/cross_model_gpt2.py run?")
            return 1

        # Normalize for cosine
        gemma_ids = sorted(gemma_cents)
        gpt2_ids = sorted(gpt2_cents)
        Cg = np.stack([gemma_cents[i] for i in gemma_ids])
        Cp = np.stack([gpt2_cents[i] for i in gpt2_ids])
        Cg = Cg / (np.linalg.norm(Cg, axis=1, keepdims=True) + 1e-9)
        Cp = Cp / (np.linalg.norm(Cp, axis=1, keepdims=True) + 1e-9)
        sim = Cg @ Cp.T  # (n_gemma, n_gpt2), cosine similarity in MiniLM-label space

        # Hungarian: best one-to-one assignment maximising similarity
        cost = -sim
        gi, pi = linear_sum_assignment(cost)
        assignment = []
        for g_idx, p_idx in zip(gi, pi):
            g_cid, p_cid = gemma_ids[g_idx], gpt2_ids[p_idx]
            g_labels = _representative_labels(c, GEMMA_SAE.neograph_id, "communityId", g_cid)
            p_labels = _representative_labels(c, GPT2_SAE_ID, "communityId_gpt2", p_cid)
            assignment.append({
                "gemma_cid": g_cid, "gpt2_cid": p_cid,
                "centroid_cosine": float(sim[g_idx, p_idx]),
                "gemma_size": gemma_sizes[g_cid], "gpt2_size": gpt2_sizes[p_cid],
                "gemma_density": gemma_density[g_cid], "gpt2_density": gpt2_density[p_cid],
                "gemma_top_labels": g_labels[:3],
                "gpt2_top_labels": p_labels[:3],
            })
        assignment.sort(key=lambda r: -r["centroid_cosine"])

        # Concept-level matching: for each labelled concept, where does it land in each model
        concept_match = []
        patterns = {
            "weekday": ["day of the week", "weekday", "weekend", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
            "money": ["monetary", "financial", "currency", "dollar", "salary"],
            "programming": ["programming", "code", "function", "variable", "syntax", "import"],
            "word_prefix": ["beginning with", "starting with", "starts with"],
        }
        for name, terms in patterns.items():
            cy = " OR ".join(f"toLower(a.text) CONTAINS '{t}'" for t in terms)
            g_rows = c.run(
                f"""
                MATCH (f:SAEFeature)-[:LABELED_AS]->(a:AutoInterpLabel)
                WHERE f.sae_id = $sid AND ({cy})
                WITH f.communityId AS cid, count(f) AS n
                WHERE cid IS NOT NULL
                RETURN cid, n ORDER BY n DESC LIMIT 3
                """,
                sid=GEMMA_SAE.neograph_id,
            )
            p_rows = c.run(
                f"""
                MATCH (f:SAEFeature)-[:LABELED_AS]->(a:AutoInterpLabel)
                WHERE f.sae_id = $sid AND ({cy})
                WITH f.communityId_gpt2 AS cid, count(f) AS n
                WHERE cid IS NOT NULL
                RETURN cid, n ORDER BY n DESC LIMIT 3
                """,
                sid=GPT2_SAE_ID,
            )
            concept_match.append({
                "concept": name,
                "gemma_top": [(int(r["cid"]), int(r["n"])) for r in g_rows],
                "gpt2_top": [(int(r["cid"]), int(r["n"])) for r in p_rows],
            })

        # Cross-reference: do the matched-by-Hungarian pairs hold the same concepts?
        concept_alignment = []
        for concept in concept_match:
            g_top_cid = concept["gemma_top"][0][0] if concept["gemma_top"] else None
            p_top_cid = concept["gpt2_top"][0][0] if concept["gpt2_top"] else None
            hungarian_pair = next((a for a in assignment if a["gemma_cid"] == g_top_cid), None)
            aligned = bool(hungarian_pair and hungarian_pair["gpt2_cid"] == p_top_cid)
            concept_alignment.append({
                "concept": concept["concept"],
                "gemma_concept_community": g_top_cid,
                "gpt2_concept_community": p_top_cid,
                "hungarian_says": hungarian_pair["gpt2_cid"] if hungarian_pair else None,
                "aligned": aligned,
                "hungarian_centroid_cos": hungarian_pair["centroid_cosine"] if hungarian_pair else None,
            })

    print("\n=== Hungarian community alignment (top 10 by label-centroid cosine) ===")
    print(f"{'gCID':>5} {'pCID':>5} {'cos':>6} {'gSize':>6} {'pSize':>6}  gemma_label / gpt2_label")
    for a in assignment[:10]:
        g_lab = (a['gemma_top_labels'][0] or '')[:38]
        p_lab = (a['gpt2_top_labels'][0] or '')[:38]
        print(f"{a['gemma_cid']:>5} {a['gpt2_cid']:>5} {a['centroid_cosine']:+.3f} {a['gemma_size']:>6} {a['gpt2_size']:>6}  {g_lab:<40}| {p_lab}")

    print("\n=== Concept-level alignment (does the same concept land in Hungarian-paired communities?) ===")
    for r in concept_alignment:
        verdict = "✅ aligned" if r["aligned"] else "❌ misaligned"
        print(f"  {r['concept']:14s}  gemma cid {r['gemma_concept_community']} → gpt2 cid {r['gpt2_concept_community']} "
              f"(Hungarian: {r['hungarian_says']})  {verdict}")

    out = PATHS.reports / "cross_model_universality.json"
    out.write_text(json.dumps({
        "hungarian_assignment": assignment,
        "concept_match": concept_match,
        "concept_alignment": concept_alignment,
        "n_gemma_communities": len(gemma_cents),
        "n_gpt2_communities": len(gpt2_cents),
    }, indent=2, default=str))
    log.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
