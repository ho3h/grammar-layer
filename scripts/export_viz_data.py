"""Export the data the Three.js prototype needs as one JSON blob.

Output:
- reports/viz_data.json — per-model: 3D coords (UMAP), n_circuits_per_feature, label for top
  backbone features, per-prompt active feature indices, cross-model label-similarity pairs.

The Three.js front-end (apps/grammar_layer/index.html) loads this once and renders.
"""

from __future__ import annotations

import json

import numpy as np

from neograph.config import PATHS, SAE as GEMMA_SAE
from neograph.cypher import NeographClient
from neograph.util import get_logger

log = get_logger("neograph.viz.export")

GPT2_SAE_ID = "gpt2-small-res-jb/L8"


def pull_circuits(c: NeographClient, model: str, sae_id: str) -> dict:
    rows = c.run(
        """
        MATCH (cir:Circuit {model: $m})-[:INCLUDES]->(f:SAEFeature)
        WHERE f.sae_id = $sid
        RETURN cir.prompt_id AS prompt_id, cir.category AS category, cir.prompt AS prompt,
               cir.target_token AS target, collect(f.index) AS features
        """,
        m=model, sid=sae_id,
    )
    return [
        {
            "prompt_id": r["prompt_id"],
            "category": r.get("category", "unknown"),
            "prompt": r.get("prompt", ""),
            "target": r.get("target", ""),
            "features": sorted(set(int(x) for x in r["features"])),
        }
        for r in rows
    ]


def pull_top_labels(c: NeographClient, model: str, sae_id: str,
                    backbone_idx: np.ndarray, k: int = 20) -> dict[int, str]:
    rows = c.run(
        """
        MATCH (f:SAEFeature)-[:LABELED_AS]->(a:AutoInterpLabel)
        WHERE f.sae_id = $sid AND f.index IN $ids
        RETURN f.index AS idx, a.text AS label
        """,
        sid=sae_id, ids=[int(x) for x in backbone_idx[:k]],
    )
    return {int(r["idx"]): r["label"] for r in rows}


def cross_model_pairs(c: NeographClient, min_cos: float = 0.85, limit: int = 200) -> list[dict]:
    rows = c.run(
        """
        MATCH (g:SAEFeature)-[:LABELED_AS]->(a:AutoInterpLabel)
        WHERE g.sae_id CONTAINS 'gemma'
        CALL db.index.vector.queryNodes('label_emb', 3, a.embedding) YIELD node, score
        MATCH (p:SAEFeature)-[:LABELED_AS]->(node)
        WHERE p.sae_id = $sid AND score >= $cos
        RETURN g.index AS gemma_idx, p.index AS gpt2_idx, score AS cos
        ORDER BY score DESC LIMIT $limit
        """,
        sid=GPT2_SAE_ID, cos=min_cos, limit=limit,
    )
    return [{"g": int(r["gemma_idx"]), "p": int(r["gpt2_idx"]), "cos": float(r["cos"])} for r in rows]


def main() -> int:
    log.info("Loading saved UMAP coords + circuit counts ...")
    Z_g = np.load(PATHS.reports / "umap_gemma_3d.npy").astype(np.float32)
    Z_p = np.load(PATHS.reports / "umap_gpt2_3d.npy").astype(np.float32)
    counts_g = np.load(PATHS.reports / "circuits_per_feature_gemma.npy").astype(int)
    counts_p = np.load(PATHS.reports / "circuits_per_feature_gpt2.npy").astype(int)

    # Top-20 backbone per model
    bb_g = np.argsort(-counts_g)[:20]
    bb_p = np.argsort(-counts_p)[:20]

    with NeographClient() as c:
        circuits_g = pull_circuits(c, "gemma", GEMMA_SAE.neograph_id)
        circuits_p = pull_circuits(c, "gpt2", GPT2_SAE_ID)
        labels_g = pull_top_labels(c, "gemma", GEMMA_SAE.neograph_id, bb_g, k=20)
        labels_p = pull_top_labels(c, "gpt2", GPT2_SAE_ID, bb_p, k=20)
        pairs = cross_model_pairs(c, min_cos=0.85, limit=200)

    # Round coords to 3 decimals for smaller JSON
    def pack(Z, counts, labels, backbone_idx):
        return {
            "n_features": int(Z.shape[0]),
            "coords": np.round(Z, 3).tolist(),
            "n_circuits": counts.tolist(),
            "max_n_circuits": int(counts.max()),
            "backbone": [
                {"index": int(i), "n_circuits": int(counts[i]), "label": labels.get(int(i), "")}
                for i in backbone_idx if counts[i] > 0
            ],
        }

    blob = {
        "gemma": {
            **pack(Z_g, counts_g, labels_g, bb_g),
            "model_pretty": "Gemma 2 2B  ·  Gemma Scope L20  ·  width-16k",
            "circuits": sorted(circuits_g, key=lambda r: r["prompt_id"]),
        },
        "gpt2": {
            **pack(Z_p, counts_p, labels_p, bb_p),
            "model_pretty": "GPT-2 small  ·  RES-JB L8  ·  24,576 features",
            "circuits": sorted(circuits_p, key=lambda r: r["prompt_id"]),
        },
        "vocab_links": pairs,
        "schema_version": 1,
    }

    out = PATHS.reports / "viz_data.json"
    out.write_text(json.dumps(blob))
    size_mb = out.stat().st_size / 1024 / 1024
    log.info("Wrote %s (%.1f MB)", out, size_mb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
