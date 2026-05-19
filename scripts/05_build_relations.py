"""P3: Build co-activation (PMI/Jaccard), decoder cosine, label cosine relations."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from tqdm import tqdm

from neograph.config import KNN_K, PATHS, SAE
from neograph.cypher import NeographClient
from neograph.relations import coactivation_edges, topk_cosine_knn
from neograph.util import chunk, exit_marker, get_logger

log = get_logger("neograph.relations.build")


def _feature_id(idx: int) -> str:
    return f"{SAE.neograph_id}/F{idx:05d}"


def build_coactivation(c: NeographClient, synth_df: pd.DataFrame) -> int:
    if synth_df.empty:
        log.warning("No synthetic activations on disk — skipping co-activation")
        return 0
    log.info("Computing co-activation PMI/Jaccard from %d synth rows ...", len(synth_df))
    # Build (prompt, position) → row id, feature → col id
    pos_keys = synth_df.assign(key=synth_df["prompt_id"] + ":" + synth_df["position"].astype(str))
    unique_pos = pos_keys["key"].unique()
    pos_idx = {k: i for i, k in enumerate(unique_pos)}
    rows = pos_keys["key"].map(pos_idx).to_numpy()
    cols = pos_keys["feature_index"].to_numpy()
    data = np.ones(len(rows), dtype=np.float32)
    M = sp.csr_matrix((data, (rows, cols)), shape=(len(unique_pos), SAE.d_sae))
    edges = coactivation_edges(M, top_k=KNN_K)
    log.info("Co-activation edges: %d", len(edges))

    write_rows = [
        {
            "a": _feature_id(e.a),
            "b": _feature_id(e.b),
            "pmi": e.pmi,
            "jaccard": e.jaccard,
            "n_co": e.n_co,
            "n_a": e.n_a,
            "n_b": e.n_b,
        }
        for e in edges
    ]
    for batch in chunk(write_rows, 5000):
        c.run(
            """
            UNWIND $rows AS r
            MATCH (a:SAEFeature {id: r.a}), (b:SAEFeature {id: r.b})
            MERGE (a)-[e:CO_ACTIVATES_WITH]->(b)
              SET e.pmi = r.pmi,
                  e.jaccard = r.jaccard,
                  e.n_co = r.n_co,
                  e.n_a = r.n_a,
                  e.n_b = r.n_b
            """,
            rows=batch,
        )
    return len(edges)


def build_decoder_cosine(c: NeographClient, sae) -> int:
    log.info("Computing decoder cosine top-%d ...", KNN_K)
    W_dec = sae.W_dec.detach().float().cpu().numpy()  # (d_sae, d_in)
    edges = topk_cosine_knn(W_dec, top_k=KNN_K)
    log.info("Decoder edges: %d", len(edges))
    write_rows = [
        {"a": _feature_id(i), "b": _feature_id(j), "cosine": float(sim)}
        for i, j, sim in edges
    ]
    for batch in chunk(write_rows, 5000):
        c.run(
            """
            UNWIND $rows AS r
            MATCH (a:SAEFeature {id: r.a}), (b:SAEFeature {id: r.b})
            MERGE (a)-[e:DECODER_SIMILAR]->(b)
              SET e.cosine = r.cosine
            """,
            rows=batch,
        )
    return len(edges)


def build_label_cosine(c: NeographClient) -> int:
    """Use the Neo4j vector index on AutoInterpLabel.embedding to query top-K for each feature."""
    log.info("Computing label cosine top-%d via vector index ...", KNN_K)
    n_written = 0
    # For each (SAEFeature)-[:LABELED_AS]->(AutoInterpLabel), find the top-K nearest labels and link
    # their owning features.
    res = c.run(
        """
        MATCH (f:SAEFeature)-[:LABELED_AS]->(a:AutoInterpLabel)
        RETURN f.id AS fid, a.embedding AS emb
        """
    )
    log.info("Indexing %d features with labels", len(res))
    for batch in tqdm(list(chunk(res, 128)), desc="label-cosine"):
        rows = []
        for r in batch:
            sim_rows = c.run(
                """
                CALL db.index.vector.queryNodes('label_emb', $k, $emb) YIELD node, score
                MATCH (g:SAEFeature)-[:LABELED_AS]->(node)
                WHERE g.id <> $fid
                RETURN g.id AS gid, score
                LIMIT $k
                """,
                emb=r["emb"],
                k=KNN_K + 1,
                fid=r["fid"],
            )
            for sr in sim_rows[:KNN_K]:
                rows.append({"a": r["fid"], "b": sr["gid"], "cosine": float(sr["score"])})
        if rows:
            c.run(
                """
                UNWIND $rows AS r
                MATCH (a:SAEFeature {id: r.a}), (b:SAEFeature {id: r.b})
                MERGE (a)-[e:LABEL_SIMILAR]->(b)
                  SET e.cosine = r.cosine
                """,
                rows=rows,
            )
            n_written += len(rows)
    return n_written


def main() -> int:
    from sae_lens import SAE as SaeLensSAE

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    sae = SaeLensSAE.from_pretrained(release=SAE.release, sae_id=SAE.sae_id, device=device)

    synth_path = PATHS.staging / "activations_synth.parquet"
    synth_df = pd.read_parquet(synth_path) if synth_path.exists() else pd.DataFrame()

    with NeographClient() as c:
        n_co = build_coactivation(c, synth_df)
        n_dec = build_decoder_cosine(c, sae)
        n_lab = build_label_cosine(c)

    exit_marker(
        "relations-built",
        ok=(n_dec > 100_000 and n_co > 0),
        coactivation=n_co,
        decoder=n_dec,
        label=n_lab,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
