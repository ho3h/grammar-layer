"""P2: Ingest Model + Layer + SAE + SAEFeature + AutoInterpLabel + Token + Prompt nodes.

- Writes 16,384 SAEFeature nodes with decoder/encoder vectors via setNodeVectorProperty.
- Pulls autointerp from Neuronpedia (with Anthropic fallback for empty ones).
- Embeds labels with MiniLM, populates the label_emb vector index.
- Writes Prompt nodes from data/staging/prompts.parquet.
- Writes Activation rows for synthetic prompts (apoc.periodic.iterate).
"""

from __future__ import annotations

import sys
from itertools import islice

import httpx
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from neograph.config import MODEL, PATHS, SAE
from neograph.cypher import NeographClient
from neograph.labels import LabelEmbedder, LabelsCache, fetch_claude_label, fetch_neuronpedia_label
from neograph.util import chunk, exit_marker, get_logger

log = get_logger("neograph.ingest")


def write_meta_nodes(c: NeographClient) -> None:
    c.run(
        """
        MERGE (m:Model {id: $mid})
          SET m.family = $fam, m.d_model = $dm, m.n_layers = $nl,
              m.vocab_size = $vs, m.source = $src
        """,
        mid=MODEL.name,
        fam=MODEL.family,
        dm=MODEL.d_model,
        nl=MODEL.n_layers,
        vs=MODEL.vocab_size,
        src=MODEL.hf_repo,
    )
    c.run(
        """
        MERGE (l:Layer {id: $lid})
          SET l.index = $idx, l.site = $site, l.hook_name = $hook
        """,
        lid=f"{MODEL.name}/L{SAE.layer}",
        idx=SAE.layer,
        site=SAE.site,
        hook=SAE.hook_name,
    )
    c.run(
        """
        MERGE (s:SAE {id: $sid})
          SET s.release = $rel, s.sae_id = $saeid,
              s.d_in = $din, s.d_sae = $dsae,
              s.architecture = $arch, s.l0_target = $l0
        """,
        sid=SAE.neograph_id,
        rel=SAE.release,
        saeid=SAE.sae_id,
        din=SAE.d_in,
        dsae=SAE.d_sae,
        arch=SAE.architecture,
        l0=71.0,
    )


def _feature_id(idx: int) -> str:
    return f"{SAE.neograph_id}/F{idx:05d}"


def ingest_features(c: NeographClient, sae, stats_df: pd.DataFrame) -> None:
    """Write SAEFeature nodes in batches with decoder/encoder vectors."""
    W_enc = sae.W_enc.detach().float().cpu().numpy()  # (d_in, d_sae)
    W_dec = sae.W_dec.detach().float().cpu().numpy()  # (d_sae, d_in)
    decoder_norms = np.linalg.norm(W_dec, axis=1)

    log.info("Writing %d SAEFeature nodes ...", SAE.d_sae)
    rows_iter = (
        {
            "fid": _feature_id(i),
            "sae_id": SAE.neograph_id,
            "idx": int(i),
            "dec": W_dec[i].tolist(),
            "enc": W_enc[:, i].tolist(),
            "dec_norm": float(decoder_norms[i]),
            "act_density": float(stats_df.at[i, "activation_density"]),
            "max_act": float(stats_df.at[i, "max_act"]),
            "is_dead": bool(stats_df.at[i, "is_dead"]),
        }
        for i in range(SAE.d_sae)
    )
    pbar = tqdm(total=SAE.d_sae, desc="SAEFeature")
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
            CALL db.create.setNodeVectorProperty(f, 'decoder_vec', r.dec)
            CALL db.create.setNodeVectorProperty(f, 'encoder_vec', r.enc)
            WITH f
            MATCH (sae:SAE {id: f.sae_id}), (layer:Layer {id: $lid})
            MERGE (f)-[:DEFINED_BY]->(sae)
            MERGE (f)-[:LIVES_IN]->(layer)
            """,
            rows=batch,
            lid=f"{MODEL.name}/L{SAE.layer}",
        )
        pbar.update(len(batch))
    pbar.close()


def ingest_labels(
    c: NeographClient,
    topk_df: pd.DataFrame,
    embedder: LabelEmbedder,
    n_features: int,
    max_features: int | None = None,
) -> int:
    """Pull autointerp from Neuronpedia (with fallback), embed, and write nodes.

    Returns the number of features labelled.
    """
    cache = LabelsCache()
    written = 0
    # Build top-K rows by feature for the Claude fallback
    topk_by_feat: dict[int, list[str]] = {}
    if topk_df is not None and len(topk_df):
        for fidx, sub in topk_df.groupby("feature_index"):
            topk_by_feat[int(fidx)] = [f"prompt={r['prompt_id']} pos={r['position']} mag={r['magnitude']:.2f}" for _, r in sub.iterrows()]

    indices = range(n_features) if max_features is None else range(min(n_features, max_features))

    with httpx.Client() as http:
        for batch in chunk(indices, 64):
            labels = []
            for idx in batch:
                lab = fetch_neuronpedia_label(idx, http, cache=cache, sleep_s=0.02)
                if lab is None:
                    lab = fetch_claude_label(idx, topk_by_feat.get(idx, []), cache=cache)
                if lab is None:
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
                    "source": lab.source,
                    "text": lab.text,
                    "score": lab.score,
                    "emb": embs[i].tolist(),
                }
                for i, lab in enumerate(labels)
            ]
            c.run(
                """
                UNWIND $rows AS r
                MERGE (a:AutoInterpLabel {id: r.lid})
                  SET a.source = r.source,
                      a.text = r.text,
                      a.score = r.score
                WITH a, r
                CALL db.create.setNodeVectorProperty(a, 'embedding', r.emb)
                WITH a, r
                MATCH (f:SAEFeature {id: r.fid})
                MERGE (f)-[lbl:LABELED_AS]->(a)
                  SET lbl.primary = true
                """,
                rows=rows,
            )
            written += len(rows)
    cache.flush()
    return written


def ingest_prompts(c: NeographClient, prompts_df: pd.DataFrame) -> None:
    log.info("Writing %d Prompt nodes ...", len(prompts_df))
    for batch in chunk(prompts_df.to_dict("records"), 512):
        c.run(
            """
            UNWIND $rows AS r
            MERGE (p:Prompt {id: r.id})
              SET p.text = r.text,
                  p.source = r.source
            """,
            rows=batch,
        )


def ingest_synth_activations(c: NeographClient, synth_df: pd.DataFrame) -> int:
    if synth_df is None or not len(synth_df):
        return 0
    log.info("Writing %d Activation rows via apoc.periodic.iterate ...", len(synth_df))
    # Stage to a temporary parquet read via apoc — but apoc can't read parquet directly.
    # Use a chunked python iterator instead.
    n = 0
    rows = synth_df.to_dict("records")
    for batch in chunk(rows, 5000):
        c.run(
            """
            UNWIND $rows AS r
            MATCH (f:SAEFeature {sae_id: $sae_id, index: r.feature_index})
            MERGE (a:Activation {id: r.prompt_id + ':' + toString(r.position) + ':' + f.id})
              SET a.position = r.position,
                  a.magnitude = r.magnitude
            MERGE (p:Prompt {id: r.prompt_id})
            MERGE (p)-[:HAS_ACTIVATION]->(a)
            MERGE (a)-[act:OF_FEATURE]->(f)
            """,
            rows=batch,
            sae_id=SAE.neograph_id,
        )
        n += len(batch)
    return n


def main() -> int:
    stats_path = PATHS.staging / "feature_stats.parquet"
    if not stats_path.exists():
        log.error("Run scripts/03_capture_activations.py first.")
        exit_marker("features-ingested", ok=False, stage="missing-stats")
        return 1
    stats_df = pd.read_parquet(stats_path).set_index("feature_index")
    topk_path = PATHS.staging / "feature_topk.parquet"
    topk_df = pd.read_parquet(topk_path) if topk_path.exists() else pd.DataFrame()
    synth_path = PATHS.staging / "activations_synth.parquet"
    synth_df = pd.read_parquet(synth_path) if synth_path.exists() else pd.DataFrame()
    prompts_df = pd.read_parquet(PATHS.staging / "prompts.parquet")

    from sae_lens import SAE as SaeLensSAE

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    sae = SaeLensSAE.from_pretrained(release=SAE.release, sae_id=SAE.sae_id, device=device)

    embedder = LabelEmbedder()

    with NeographClient() as c:
        write_meta_nodes(c)
        ingest_features(c, sae, stats_df)
        n_labels = ingest_labels(c, topk_df, embedder, SAE.d_sae)
        ingest_prompts(c, prompts_df)
        n_acts = ingest_synth_activations(c, synth_df)
        counts = c.run(
            """
            CALL { MATCH (f:SAEFeature) RETURN count(f) AS nf }
            CALL { MATCH (a:AutoInterpLabel) RETURN count(a) AS na }
            CALL { MATCH (p:Prompt) RETURN count(p) AS np }
            CALL { MATCH (a:Activation) RETURN count(a) AS nact }
            RETURN nf, na, np, nact
            """
        )[0]
    ok = counts["nf"] == SAE.d_sae and counts["na"] >= 8000
    exit_marker(
        "features-ingested",
        ok=ok,
        features=counts["nf"],
        labels=counts["na"],
        prompts=counts["np"],
        activations=counts["nact"],
    )
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
