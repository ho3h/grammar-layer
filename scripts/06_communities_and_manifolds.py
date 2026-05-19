"""P3+P4: GDS Leiden community detection + per-community manifold fitting.

Steps:
1. Project multi-relation graph into GDS.
2. Run Leiden, write `communityId` to SAEFeature nodes.
3. For each community of size ≥ COMMUNITY_SIZE_MIN, pull residual-stream activations,
   fit principal curve in PCA space, write Manifold + Waypoint + LIES_ON.
4. Attach Concepts.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from neograph.config import COMMUNITY_SIZE_MIN, MODEL, PATHS, SAE
from neograph.cypher import NeographClient
from neograph.manifold.concept import attach_concept
from neograph.manifold.fit import fit_community_manifold
from neograph.manifold.write import write_manifold
from neograph.util import exit_marker, get_logger

log = get_logger("neograph.communities")

GRAPH_NAME = "feature-multi-graph"


def _drop_existing_projection(c: NeographClient) -> None:
    rows = c.run("CALL gds.graph.exists($name) YIELD exists", name=GRAPH_NAME)
    if rows and rows[0]["exists"]:
        c.run("CALL gds.graph.drop($name) YIELD graphName", name=GRAPH_NAME)


def project_graph(c: NeographClient) -> None:
    """Project a single UNDIRECTED multi-relation graph for Leiden via the
    aggregating ``gds.graph.project`` function (modern API)."""
    _drop_existing_projection(c)
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


def run_leiden(c: NeographClient) -> dict:
    rows = c.run(
        """
        CALL gds.leiden.write($name, {
          writeProperty: 'communityId',
          gamma: 1.0, theta: 0.01, randomSeed: 42,
          relationshipWeightProperty: 'weight'
        }) YIELD communityCount, modularity, modularities
        RETURN communityCount, modularity, modularities
        """,
        name=GRAPH_NAME,
    )
    return rows[0] if rows else {}


def list_communities(c: NeographClient) -> list[dict]:
    return c.run(
        """
        MATCH (f:SAEFeature)
        WHERE f.communityId IS NOT NULL
        RETURN f.communityId AS cid, count(f) AS n
        ORDER BY n DESC
        """
    )


def fit_one_community(
    c: NeographClient,
    model,
    sae,
    cid: int,
    feature_indices: list[int],
    synth_df: pd.DataFrame,
    *,
    prompts: pd.DataFrame,
    residual_cache: dict[tuple[str, int], np.ndarray],
) -> bool:
    # Collect (prompt_id, position) where any community feature fires
    sub = synth_df[synth_df["feature_index"].isin(feature_indices)]
    if sub.empty:
        log.info("Community %d: no synth activations — skipping", cid)
        return False
    positions = sub[["prompt_id", "position"]].drop_duplicates()
    log.info("Community %d: %d features, %d positions", cid, len(feature_indices), len(positions))

    h_rows: list[np.ndarray] = []
    needed_prompts = {pid for pid in positions["prompt_id"].unique() if pid in prompts.index}
    for pid in needed_prompts:
        text = prompts.at[pid, "text"]
        # Cheap test: if every needed (pid, pos) is already cached, skip the forward pass.
        wanted = positions[positions["prompt_id"] == pid]["position"].astype(int).tolist()
        if all((pid, p) in residual_cache for p in wanted):
            continue
        with torch.no_grad():
            tokens = model.to_tokens(text, prepend_bos=True)[:, :128]
            _logits, cache_act = model.run_with_cache(tokens, names_filter=[SAE.hook_name])
        h_all = cache_act[SAE.hook_name][0].float().cpu().numpy()  # (seq, d_model)
        for p in wanted:
            if p < h_all.shape[0]:
                residual_cache[(pid, p)] = h_all[p]

    # Build h_rows from cache
    for _, row in positions.iterrows():
        key = (row["prompt_id"], int(row["position"]))
        v = residual_cache.get(key)
        if v is not None:
            h_rows.append(v)

    if len(h_rows) < 16:
        log.info("Community %d: too few activations (%d) — skipping", cid, len(h_rows))
        return False

    H = np.stack(h_rows)
    fit = fit_community_manifold(H, n_waypoints=16, random_state=42)
    if fit is None:
        return False

    # Per-feature centroid in PCA space (for :LIES_ON)
    feat_centroids = []
    for fidx in feature_indices:
        fsub = sub[sub["feature_index"] == fidx][["prompt_id", "position"]].drop_duplicates()
        vecs = []
        for _, row in fsub.iterrows():
            v = residual_cache.get((row["prompt_id"], int(row["position"])))
            if v is not None:
                vecs.append(v)
        if not vecs:
            continue
        cen = np.mean(vecs, axis=0)
        cen_pca = (cen - fit.pca_mean) @ fit.pca_components.T  # (d,)
        feat_centroids.append((fidx, cen_pca))

    if not feat_centroids:
        return False
    idx_arr = [t[0] for t in feat_centroids]
    pca_arr = np.stack([t[1] for t in feat_centroids])

    manifold_id = f"community-{cid}/L{SAE.layer}"
    write_manifold(
        c,
        manifold_id=manifold_id,
        fit=fit,
        feature_indices=idx_arr,
        feature_positions_pca=pca_arr,
        notes=f"Leiden community {cid}, {len(feature_indices)} features",
    )

    # Concept assignment
    label_rows = c.run(
        """
        MATCH (f:SAEFeature)-[:LABELED_AS]->(a:AutoInterpLabel)
        WHERE f.communityId = $cid
        RETURN a.text AS text
        LIMIT 30
        """,
        cid=cid,
    )
    labels = [r["text"] for r in label_rows if r.get("text")]
    attach_concept(
        c,
        manifold_id=manifold_id,
        autointerp_labels=labels,
        representative_tokens=[],
        name_seed=f"Community {cid}",
    )
    return True


def main() -> int:
    synth_path = PATHS.staging / "activations_synth.parquet"
    synth_df = pd.read_parquet(synth_path) if synth_path.exists() else pd.DataFrame()
    if synth_df.empty:
        log.error("No synthetic activations — run 03 first")
        exit_marker("communities-and-manifolds", ok=False, stage="missing-synth")
        return 1

    from sae_lens import SAE as SaeLensSAE
    from sae_lens import HookedSAETransformer as HookedTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(MODEL.name, device=device)
    model.eval()
    sae = SaeLensSAE.from_pretrained(release=SAE.release, sae_id=SAE.sae_id, device=device)

    prompts = pd.read_parquet(PATHS.staging / "prompts.parquet").set_index("id")
    residual_cache: dict[tuple[str, int], np.ndarray] = {}

    with NeographClient() as c:
        log.info("Projecting GDS graph ...")
        project_graph(c)
        log.info("Running Leiden ...")
        stats = run_leiden(c)
        modularity = float(stats.get("modularity", 0.0))
        ncomm = int(stats.get("communityCount", 0))
        log.info("Leiden: %d communities, modularity=%.3f", ncomm, modularity)
        communities = list_communities(c)
        n_fit = 0
        for row in tqdm(communities, desc="manifold-fit"):
            if row["n"] < COMMUNITY_SIZE_MIN:
                continue
            cid = int(row["cid"])
            feature_rows = c.run(
                "MATCH (f:SAEFeature) WHERE f.communityId = $cid RETURN f.index AS idx",
                cid=cid,
            )
            indices = [int(r["idx"]) for r in feature_rows]
            if fit_one_community(
                c, model, sae, cid, indices, synth_df,
                prompts=prompts, residual_cache=residual_cache,
            ):
                n_fit += 1
            log.info("residual_cache size: %d", len(residual_cache))
    exit_marker(
        "communities-and-manifolds",
        ok=(modularity > 0.5 and n_fit > 0),
        communities=ncomm,
        modularity=modularity,
        manifolds_written=n_fit,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
