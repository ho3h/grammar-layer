"""Cross-model ingest: GPT-2 small + RES-JB SAEs (layer 8 resid_pre) into the same Neograph DB,
under sae_id='gpt2-small-res-jb/L8'. After ingest, run Leiden separately for GPT-2 and produce
a community-by-concept comparison table against Gemma 2 2B.

This is the highest-leverage experiment Neograph hasn't tested yet (Theo's priority #1, 2026-05-12):
the question is whether the multi-relation graph schema is doing something pandas-equivalent
or something paradigm-different. Cross-model motif matching is the test.

Pipeline:
1. Load GPT-2 small + RES-JB L8 (24576 features, d_model=768, d_sae=24576)
2. Re-run the existing corpus (prompts.parquet) — capture activations, top-K, synth Activations
3. Ingest SAEFeature nodes with the new sae_id; Neuronpedia labels via gpt2-small/8-res-jb slug
4. Build relations under the new sae_id namespace
5. Project a separate GDS graph + Leiden → property `communityId_gpt2`
6. Cross-model matching: for each concept pattern, which community holds it in each model?
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import httpx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from tqdm import tqdm

from neograph.config import KNN_K, PATHS, COMMUNITY_SIZE_MIN, COACTIVATION_NMIN, COACTIVATION_PMI_MIN
from neograph.cypher import NeographClient
from neograph.labels import LabelEmbedder, LabelsCache, fetch_neuronpedia_label
from neograph.relations import coactivation_edges, topk_cosine_knn
from neograph.util import chunk, exit_marker, get_logger

log = get_logger("neograph.gpt2")


# ============================================================================
# Cross-model config (overrides neograph.config)
# ============================================================================


@dataclass(frozen=True)
class GPT2Config:
    hf_repo: str = "gpt2"
    name: str = "gpt2-small"
    family: str = "gpt2"
    d_model: int = 768
    n_layers: int = 12
    vocab_size: int = 50257
    layer: int = 8
    hook_name: str = "blocks.8.hook_resid_pre"
    site: str = "resid_pre"
    release: str = "gpt2-small-res-jb"
    sae_id_attr: str = "blocks.8.hook_resid_pre"  # SAELens attribute
    d_in: int = 768
    d_sae: int = 24576
    architecture: str = "standard"
    activation_threshold: float = 1e-3

    @property
    def neograph_id(self) -> str:
        return "gpt2-small-res-jb/L8"


GPT2 = GPT2Config()


def _feature_id(idx: int) -> str:
    return f"{GPT2.neograph_id}/F{idx:05d}"


def neuronpedia_url(idx: int) -> str:
    return f"https://www.neuronpedia.org/api/feature/gpt2-small/8-res-jb/{idx}"


# ============================================================================
# Schema additions: vector index for d_in=768
# ============================================================================


def add_schema(c: NeographClient) -> None:
    log.info("Adding GPT-2 vector indexes (768-dim) ...")
    c.run(
        """
        CREATE VECTOR INDEX feat_decoder_768 IF NOT EXISTS
          FOR (f:SAEFeature) ON (f.decoder_vec_768)
          OPTIONS {indexConfig: { `vector.dimensions`: 768, `vector.similarity_function`: 'cosine' }}
        """
    )
    c.run(
        """
        CREATE VECTOR INDEX feat_encoder_768 IF NOT EXISTS
          FOR (f:SAEFeature) ON (f.encoder_vec_768)
          OPTIONS {indexConfig: { `vector.dimensions`: 768, `vector.similarity_function`: 'cosine' }}
        """
    )


# ============================================================================
# Activation capture (compact version)
# ============================================================================


def capture_activations(model, sae, prompts_df, top_k=32, batch_size=16, max_seq=128):
    """Return (stats_df, topk_dict, synth_rows)."""
    d_sae = GPT2.d_sae
    max_act = np.zeros(d_sae, dtype=np.float32)
    n_active = np.zeros(d_sae, dtype=np.int64)
    n_positions = 0
    topk_per_feat: dict[int, list[tuple[float, str, int]]] = {}
    synth_rows: list[dict] = []
    synth_sources = {"rhyme-ore", "weekday"}
    thr = GPT2.activation_threshold

    for batch_start in tqdm(range(0, len(prompts_df), batch_size), desc="gpt2-capture"):
        batch = prompts_df.iloc[batch_start : batch_start + batch_size]
        texts = batch["text"].tolist()
        ids = batch["id"].tolist()
        sources = batch["source"].tolist()

        with torch.no_grad():
            tokens = model.to_tokens(texts, prepend_bos=True)[:, :max_seq]
            _logits, cache = model.run_with_cache_with_saes(tokens, saes=[sae])

        feat_key = next(k for k in cache.keys() if "sae" in k and "acts_post" in k)
        feat_acts = cache[feat_key].float().cpu().numpy()
        b, seq, _ = feat_acts.shape
        n_positions += b * seq

        active_mask = feat_acts > thr
        max_act = np.maximum(max_act, feat_acts.reshape(-1, d_sae).max(axis=0))
        n_active += active_mask.reshape(-1, d_sae).sum(axis=0).astype(np.int64)

        # Top-K per feature
        flat = feat_acts.reshape(-1, d_sae)
        for f_start in range(0, d_sae, 1024):
            f_end = min(f_start + 1024, d_sae)
            chunk_acts = flat[:, f_start:f_end]
            top_idx = np.argpartition(-chunk_acts, min(top_k, chunk_acts.shape[0] - 1), axis=0)[:top_k]
            for ci, fidx in enumerate(range(f_start, f_end)):
                vals = chunk_acts[top_idx[:, ci], ci]
                heap = topk_per_feat.setdefault(fidx, [])
                for mag, flat_pos in zip(vals.tolist(), top_idx[:, ci].tolist()):
                    if mag <= thr:
                        continue
                    b_idx = flat_pos // seq
                    p_idx = flat_pos % seq
                    if b_idx >= len(ids):
                        continue
                    heap.append((float(mag), ids[b_idx], int(p_idx)))
                heap.sort(reverse=True)
                del heap[top_k:]

        # Synthetic Activation rows
        for bi, (pid, src) in enumerate(zip(ids, sources)):
            if src not in synth_sources:
                continue
            pos_mask = active_mask[bi].any(axis=-1)
            for p in np.where(pos_mask)[0]:
                feat_indices = np.where(active_mask[bi, p])[0]
                for f in feat_indices:
                    synth_rows.append(
                        {
                            "prompt_id": pid,
                            "position": int(p),
                            "feature_index": int(f),
                            "magnitude": float(feat_acts[bi, p, f]),
                        }
                    )

    density = n_active / max(n_positions, 1)
    stats_df = pd.DataFrame(
        {
            "feature_index": np.arange(d_sae),
            "max_act": max_act,
            "activation_density": density.astype(np.float32),
            "is_dead": max_act < thr,
        }
    ).set_index("feature_index")
    return stats_df, topk_per_feat, synth_rows


# ============================================================================
# Ingest
# ============================================================================


def write_meta(c: NeographClient) -> None:
    c.run(
        """
        MERGE (m:Model {id: $mid})
          SET m.family = $fam, m.d_model = $dm, m.n_layers = $nl,
              m.vocab_size = $vs, m.source = $src
        """,
        mid=GPT2.name, fam=GPT2.family, dm=GPT2.d_model, nl=GPT2.n_layers,
        vs=GPT2.vocab_size, src=GPT2.hf_repo,
    )
    c.run(
        """
        MERGE (l:Layer {id: $lid})
          SET l.index = $idx, l.site = $site, l.hook_name = $hook
        """,
        lid=f"{GPT2.name}/L{GPT2.layer}", idx=GPT2.layer, site=GPT2.site, hook=GPT2.hook_name,
    )
    c.run(
        """
        MERGE (s:SAE {id: $sid})
          SET s.release = $rel, s.sae_id = $sa, s.d_in = $din, s.d_sae = $dsae, s.architecture = $arch
        """,
        sid=GPT2.neograph_id, rel=GPT2.release, sa=GPT2.sae_id_attr,
        din=GPT2.d_in, dsae=GPT2.d_sae, arch=GPT2.architecture,
    )


def ingest_features(c: NeographClient, sae, stats_df: pd.DataFrame) -> None:
    W_enc = sae.W_enc.detach().float().cpu().numpy()  # (d_in, d_sae)
    W_dec = sae.W_dec.detach().float().cpu().numpy()  # (d_sae, d_in)
    decoder_norms = np.linalg.norm(W_dec, axis=1)
    log.info("Writing %d GPT-2 SAEFeature nodes ...", GPT2.d_sae)
    rows_iter = (
        {
            "fid": _feature_id(i),
            "sae_id": GPT2.neograph_id,
            "idx": int(i),
            "dec": W_dec[i].tolist(),
            "enc": W_enc[:, i].tolist(),
            "dec_norm": float(decoder_norms[i]),
            "act_density": float(stats_df.at[i, "activation_density"]),
            "max_act": float(stats_df.at[i, "max_act"]),
            "is_dead": bool(stats_df.at[i, "is_dead"]),
        }
        for i in range(GPT2.d_sae)
    )
    pbar = tqdm(total=GPT2.d_sae, desc="gpt2-feat-write")
    for batch in chunk(rows_iter, 256):
        c.run(
            """
            UNWIND $rows AS r
            MERGE (f:SAEFeature {id: r.fid})
              SET f.sae_id = r.sae_id,
                  f.index = r.idx,
                  f.decoder_norm = r.dec_norm,
                  f.activation_density = r.act_density,
                  f.max_act = r.max_act,
                  f.is_dead = r.is_dead
            WITH f, r
            CALL db.create.setNodeVectorProperty(f, 'decoder_vec_768', r.dec)
            CALL db.create.setNodeVectorProperty(f, 'encoder_vec_768', r.enc)
            WITH f
            MATCH (sae:SAE {id: f.sae_id}), (layer:Layer {id: $lid})
            MERGE (f)-[:DEFINED_BY]->(sae)
            MERGE (f)-[:LIVES_IN]->(layer)
            """,
            rows=batch, lid=f"{GPT2.name}/L{GPT2.layer}",
        )
        pbar.update(len(batch))
    pbar.close()


def ingest_labels(c: NeographClient, embedder: LabelEmbedder) -> int:
    cache = LabelsCache(PATHS.data / "labels_cache_gpt2.json")
    written = 0
    with httpx.Client(timeout=20.0) as http:
        for batch_idx in tqdm(list(chunk(range(GPT2.d_sae), 256)), desc="gpt2-labels"):
            labels = []
            for idx in batch_idx:
                cached = cache.get(idx)
                if cached:
                    from neograph.labels import Label
                    lab = Label(feature_index=idx, text=cached["text"], source=cached["source"], score=cached.get("score"))
                else:
                    url = neuronpedia_url(idx)
                    try:
                        r = http.get(url)
                        if r.status_code != 200:
                            continue
                        data = r.json()
                        exps = data.get("explanations") or []
                        if not exps:
                            continue
                        exp = exps[0]
                        text = (exp.get("description") or "").strip()
                        if not text:
                            continue
                        from neograph.labels import Label
                        lab = Label(
                            feature_index=idx, text=text,
                            source=f"neuronpedia:{exp.get('typeName', 'unknown')}",
                            score=None,
                        )
                        cache.put(idx, {"text": lab.text, "source": lab.source, "score": None})
                    except Exception as exc:  # noqa: BLE001
                        log.debug("label fetch fail for %d: %s", idx, exc)
                        continue
                labels.append(lab)
            if not labels:
                continue
            texts = [lab.text for lab in labels]
            embs = embedder.embed(texts)
            cache.flush()
            rows = [
                {
                    "lid": f"{_feature_id(lab.feature_index)}#{lab.source}",
                    "fid": _feature_id(lab.feature_index),
                    "source": lab.source, "text": lab.text, "score": lab.score,
                    "emb": embs[i].tolist(),
                }
                for i, lab in enumerate(labels)
            ]
            c.run(
                """
                UNWIND $rows AS r
                MERGE (a:AutoInterpLabel {id: r.lid})
                  SET a.source = r.source, a.text = r.text, a.score = r.score
                WITH a, r
                CALL db.create.setNodeVectorProperty(a, 'embedding', r.emb)
                WITH a, r
                MATCH (f:SAEFeature {id: r.fid})
                MERGE (f)-[lbl:LABELED_AS]->(a) SET lbl.primary = true
                """,
                rows=rows,
            )
            written += len(rows)
    return written


# ============================================================================
# Relations
# ============================================================================


def build_coactivation(c: NeographClient, synth_rows: list[dict]) -> int:
    if not synth_rows:
        return 0
    df = pd.DataFrame(synth_rows)
    log.info("GPT-2 co-activation from %d synth rows ...", len(df))
    df["key"] = df["prompt_id"] + ":" + df["position"].astype(str)
    unique_pos = df["key"].unique()
    pos_idx = {k: i for i, k in enumerate(unique_pos)}
    rows = df["key"].map(pos_idx).to_numpy()
    cols = df["feature_index"].to_numpy()
    data = np.ones(len(rows), dtype=np.float32)
    M = sp.csr_matrix((data, (rows, cols)), shape=(len(unique_pos), GPT2.d_sae))
    edges = coactivation_edges(M, top_k=KNN_K, pmi_min=COACTIVATION_PMI_MIN, n_min=COACTIVATION_NMIN)
    log.info("GPT-2 co-act edges: %d", len(edges))
    for batch in chunk(edges, 5000):
        rows_batch = [
            {"a": _feature_id(e.a), "b": _feature_id(e.b), "pmi": e.pmi,
             "jaccard": e.jaccard, "n_co": e.n_co, "n_a": e.n_a, "n_b": e.n_b}
            for e in batch
        ]
        c.run(
            """
            UNWIND $rows AS r
            MATCH (a:SAEFeature {id: r.a}), (b:SAEFeature {id: r.b})
            MERGE (a)-[e:CO_ACTIVATES_WITH]->(b)
              SET e.pmi = r.pmi, e.jaccard = r.jaccard,
                  e.n_co = r.n_co, e.n_a = r.n_a, e.n_b = r.n_b
            """,
            rows=rows_batch,
        )
    return len(edges)


def build_decoder_cosine(c: NeographClient, sae) -> int:
    W_dec = sae.W_dec.detach().float().cpu().numpy()
    edges = topk_cosine_knn(W_dec, top_k=KNN_K)
    log.info("GPT-2 decoder edges: %d", len(edges))
    for batch in chunk(edges, 5000):
        rows = [{"a": _feature_id(i), "b": _feature_id(j), "cosine": float(s)} for i, j, s in batch]
        c.run(
            """
            UNWIND $rows AS r
            MATCH (a:SAEFeature {id: r.a}), (b:SAEFeature {id: r.b})
            MERGE (a)-[e:DECODER_SIMILAR]->(b) SET e.cosine = r.cosine
            """,
            rows=rows,
        )
    return len(edges)


def build_label_cosine(c: NeographClient) -> int:
    res = c.run(
        """
        MATCH (f:SAEFeature)-[:LABELED_AS]->(a:AutoInterpLabel)
        WHERE f.sae_id = $sae_id
        RETURN f.id AS fid, a.embedding AS emb
        """,
        sae_id=GPT2.neograph_id,
    )
    log.info("GPT-2 label cosine on %d labeled features ...", len(res))
    n_written = 0
    for batch in tqdm(list(chunk(res, 128)), desc="gpt2-label-cos"):
        rows = []
        for r in batch:
            sim_rows = c.run(
                """
                CALL db.index.vector.queryNodes('label_emb', $k, $emb) YIELD node, score
                MATCH (g:SAEFeature)-[:LABELED_AS]->(node)
                WHERE g.id <> $fid AND g.sae_id = $sae_id
                RETURN g.id AS gid, score LIMIT $k
                """,
                emb=r["emb"], k=KNN_K + 1, fid=r["fid"], sae_id=GPT2.neograph_id,
            )
            for sr in sim_rows[:KNN_K]:
                rows.append({"a": r["fid"], "b": sr["gid"], "cosine": float(sr["score"])})
        if rows:
            c.run(
                """
                UNWIND $rows AS r
                MATCH (a:SAEFeature {id: r.a}), (b:SAEFeature {id: r.b})
                MERGE (a)-[e:LABEL_SIMILAR]->(b) SET e.cosine = r.cosine
                """,
                rows=rows,
            )
            n_written += len(rows)
    return n_written


# ============================================================================
# Leiden under a separate writeProperty so it coexists with Gemma's
# ============================================================================


GRAPH_NAME = "gpt2-multi-graph"


def project_and_leiden(c: NeographClient) -> dict:
    exists = c.run("CALL gds.graph.exists($name) YIELD exists RETURN exists", name=GRAPH_NAME)
    if exists and exists[0].get("exists"):
        c.run("CALL gds.graph.drop($name) YIELD graphName RETURN graphName", name=GRAPH_NAME)

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
        RETURN g.graphName, g.nodeCount, g.relationshipCount
        """,
        name=GRAPH_NAME, sae_id=GPT2.neograph_id,
    )
    rows = c.run(
        """
        CALL gds.leiden.write($name, {
          writeProperty: 'communityId_gpt2',
          gamma: 1.0, theta: 0.01, randomSeed: 42,
          relationshipWeightProperty: 'weight'
        }) YIELD communityCount, modularity
        RETURN communityCount, modularity
        """,
        name=GRAPH_NAME,
    )
    return rows[0] if rows else {}


# ============================================================================
# Cross-model matching query
# ============================================================================


def cross_model_matching(c: NeographClient) -> dict:
    """For each concept pattern, report top community in each model."""
    patterns = {
        "weekday": ["day of the week", "weekday", "weekend", "monday", "tuesday", "wednesday",
                    "thursday", "friday", "saturday", "sunday"],
        "money": ["monetary", "financial", "currency", "dollar", "salary"],
        "programming": ["programming", "code", "function", "variable", "syntax", "import"],
        "word_prefix": ["beginning with", "starting with", "starts with"],
    }
    results = {}
    for name, terms in patterns.items():
        any_clauses = " OR ".join(f"toLower(a.text) CONTAINS '{t}'" for t in terms)
        # Gemma
        gemma_rows = c.run(
            f"""
            MATCH (f:SAEFeature)-[:LABELED_AS]->(a:AutoInterpLabel)
            WHERE f.sae_id STARTS WITH 'gemma' AND ({any_clauses})
            WITH f.communityId AS cid, count(f) AS n
            WHERE cid IS NOT NULL
            RETURN cid, n ORDER BY n DESC LIMIT 3
            """
        )
        gpt2_rows = c.run(
            f"""
            MATCH (f:SAEFeature)-[:LABELED_AS]->(a:AutoInterpLabel)
            WHERE f.sae_id = $sid AND ({any_clauses})
            WITH f.communityId_gpt2 AS cid, count(f) AS n
            WHERE cid IS NOT NULL
            RETURN cid, n ORDER BY n DESC LIMIT 3
            """,
            sid=GPT2.neograph_id,
        )
        results[name] = {"gemma": gemma_rows, "gpt2": gpt2_rows}
    return results


def main() -> int:
    log.info("=== Cross-model GPT-2 small + RES-JB L8 ===")
    prompts_path = PATHS.staging / "prompts.parquet"
    if not prompts_path.exists():
        log.error("prompts.parquet missing — run scripts/02_seed_corpus.py first")
        return 1
    prompts_df = pd.read_parquet(prompts_path)
    log.info("Corpus: %d prompts (reusing Gemma's)", len(prompts_df))

    from sae_lens import SAE as SaeLensSAE, HookedSAETransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("Loading GPT-2 small on %s ...", device)
    model = HookedSAETransformer.from_pretrained(GPT2.name, device=device)
    model.eval()
    log.info("Loading SAE %s / %s ...", GPT2.release, GPT2.sae_id_attr)
    sae = SaeLensSAE.from_pretrained(release=GPT2.release, sae_id=GPT2.sae_id_attr, device=device)
    log.info("SAE: d_in=%d d_sae=%d", sae.cfg.d_in, sae.cfg.d_sae)

    with NeographClient() as c:
        add_schema(c)
        write_meta(c)
        log.info("Capturing GPT-2 activations ...")
        stats_df, topk_per_feat, synth_rows = capture_activations(model, sae, prompts_df, batch_size=16)
        log.info("synth Activation rows: %d", len(synth_rows))
        ingest_features(c, sae, stats_df)
        embedder = LabelEmbedder()
        n_labels = ingest_labels(c, embedder)
        log.info("GPT-2 labels written: %d", n_labels)
        n_co = build_coactivation(c, synth_rows)
        n_dec = build_decoder_cosine(c, sae)
        n_lab = build_label_cosine(c)
        log.info("relations: co=%d dec=%d lab=%d", n_co, n_dec, n_lab)
        stats = project_and_leiden(c)
        log.info("GPT-2 Leiden: %d communities, modularity=%.3f",
                 int(stats.get("communityCount", 0)), float(stats.get("modularity", 0.0)))
        match = cross_model_matching(c)

    print("\n=== Cross-model community matching ===")
    for name, rec in match.items():
        gemma_top = rec["gemma"][0] if rec["gemma"] else None
        gpt2_top = rec["gpt2"][0] if rec["gpt2"] else None
        print(f"  {name:14s}  Gemma: {gemma_top}     GPT-2: {gpt2_top}")
    import json
    out = PATHS.reports / "cross_model_matching.json"
    out.write_text(json.dumps({"matching": match, "gpt2_leiden": stats}, indent=2, default=str))
    log.info("Wrote %s", out)
    exit_marker("cross-model-ingest", ok=(stats.get("modularity", 0) > 0.3 and n_dec > 100_000),
                gpt2_communities=stats.get("communityCount"),
                gpt2_modularity=stats.get("modularity"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
