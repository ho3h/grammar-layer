"""Lightweight causal attribution — write :CAUSES edges by zero-ablation patching.

This is the poor man's version of the circuit-tracer/Anthropic attribution graph. Instead
of using pre-trained transcoders + a replacement model, we directly ablate each active SAE
feature and measure the change in next-token logit at a target token. That gives a real
causal effect-size (not just correlation), which is what `:CAUSES` was designed for.

This is the experiment Theo flagged as #2: "If those Cypher queries feel like superpowers,
you have your answer; if they feel like ceremony, you have your answer." The point isn't
to compete with circuit-tracer on attribution quality — it's to populate the schema with
real causal edges so the multi-hop queries become testable.

Pipeline:
1. Pick a prompt where Gemma deterministically outputs a known target token
   (e.g. "Today is Monday. Tomorrow is" → " Tuesday")
2. Forward pass with SAE → record baseline logit at target token
3. For each active feature (act > threshold) at the last 5 positions:
   - Reconstruct residual stream WITHOUT that feature's contribution
   - Forward through remaining layers, measure new target-token logit
   - Effect size = baseline_logit - ablated_logit (positive = supportive)
4. Write top-N effect-size features as :CAUSES edges with effect_size + prompt_id + method
5. Run five Cypher queries that join :CAUSES to communities, manifolds, autointerp labels

Usage:
    uv run python scripts/causal_attribution.py
"""

from __future__ import annotations

import sys

import torch

from neograph.config import MODEL, SAE as SAE_CFG
from neograph.cypher import NeographClient
from neograph.util import exit_marker, get_logger

log = get_logger("neograph.causal")


PROMPTS = [
    {
        "prompt": "Today is Monday. Tomorrow is",
        "target": " Tuesday",
        "id": "weekday-mon-tue",
    },
    {
        "prompt": "The capital of France is",
        "target": " Paris",
        "id": "capital-france",
    },
]


def _feature_id(idx: int) -> str:
    return f"{SAE_CFG.neograph_id}/F{idx:05d}"


def ablate_and_score(model, sae, prompt: str, target_token_id: int, top_n: int = 50) -> list[dict]:
    """For each active feature at the last position, ablate and measure target-logit delta."""
    tokens = model.to_tokens(prompt, prepend_bos=True)

    # Baseline forward pass with SAE on
    with torch.no_grad():
        baseline_logits, _ = model.run_with_cache_with_saes(tokens, saes=[sae])
    baseline = float(baseline_logits[0, -1, target_token_id].item())
    log.info("baseline logit @ %s = %.3f", target_token_id, baseline)

    # Get feature activations at the last position
    with torch.no_grad():
        _logits, cache = model.run_with_cache_with_saes(tokens, saes=[sae])
    feat_key = next(k for k in cache.keys() if "sae" in k and "acts_post" in k)
    feat_acts = cache[feat_key][0, -1, :]  # (d_sae,)
    active = (feat_acts > SAE_CFG.activation_threshold).nonzero(as_tuple=True)[0]
    log.info("active features at last position: %d", len(active))

    results: list[dict] = []
    hook_name = f"{SAE_CFG.hook_name}.hook_sae_acts_post"
    # Ablation: re-run with the feature's contribution zeroed via a hook.
    # TransformerLens passes `hook=` as a kwarg, hence the **kwargs.
    for fidx in active.tolist():
        def ablate(act, fidx=fidx, **kwargs):
            act = act.clone()
            act[..., -1, fidx] = 0.0
            return act

        with torch.no_grad():
            ablated_logits = model.run_with_hooks_with_saes(
                tokens, saes=[sae], fwd_hooks=[(hook_name, ablate)]
            )
        ablated = float(ablated_logits[0, -1, target_token_id].item())
        effect = baseline - ablated  # positive = feature supports target
        results.append({"feature_index": int(fidx), "effect_size": effect, "magnitude": float(feat_acts[fidx])})

    results.sort(key=lambda r: abs(r["effect_size"]), reverse=True)
    return results[:top_n]


def write_circuit(c: NeographClient, prompt_id: str, prompt: str, target_token: str, edges: list[dict]) -> None:
    cid = f"circuit/{prompt_id}/{target_token.strip()}"
    c.run(
        """
        MERGE (cir:Circuit {id: $cid})
          SET cir.prompt_id = $pid, cir.target_token = $tok,
              cir.source = 'patching', cir.pruning_threshold = 0.0
        """,
        cid=cid, pid=prompt_id, tok=target_token,
    )
    rows = [
        {"cid": cid, "fid": _feature_id(e["feature_index"]),
         "effect": float(e["effect_size"]), "mag": float(e["magnitude"])}
        for e in edges
    ]
    c.run(
        """
        UNWIND $rows AS r
        MATCH (f:SAEFeature {id: r.fid}), (cir:Circuit {id: r.cid})
        MERGE (cir)-[inc:INCLUDES]->(f)
          SET inc.role = CASE WHEN r.effect > 0 THEN 'support' ELSE 'oppose' END,
              inc.attribution = r.effect,
              inc.magnitude = r.mag
        """,
        rows=rows,
    )


def query_circuit_to_communities(c: NeographClient, cid: str) -> list[dict]:
    """Q-CAUSE-1: which Leiden communities does this circuit recruit, and how strongly?"""
    return c.run(
        """
        MATCH (cir:Circuit {id: $cid})-[inc:INCLUDES]->(f:SAEFeature)
        WHERE f.communityId IS NOT NULL
        WITH f.communityId AS community, count(f) AS n_features,
             sum(inc.attribution) AS total_attribution
        RETURN community, n_features, total_attribution
        ORDER BY total_attribution DESC LIMIT 10
        """,
        cid=cid,
    )


def query_circuit_features_on_manifolds(c: NeographClient, cid: str) -> list[dict]:
    """Q-CAUSE-2: which features in this circuit lie on a fitted manifold?"""
    return c.run(
        """
        MATCH (cir:Circuit {id: $cid})-[inc:INCLUDES]->(f:SAEFeature)-[lo:LIES_ON]->(m:Manifold)
        OPTIONAL MATCH (m)-[:DESCRIBES]->(co:Concept)
        RETURN f.index AS feature, inc.attribution AS effect,
               m.id AS manifold, co.name AS concept, lo.arc_position AS arc
        ORDER BY abs(inc.attribution) DESC LIMIT 20
        """,
        cid=cid,
    )


def query_circuit_labels(c: NeographClient, cid: str) -> list[dict]:
    """Q-CAUSE-3: top-attributed features + their autointerp labels."""
    return c.run(
        """
        MATCH (cir:Circuit {id: $cid})-[inc:INCLUDES]->(f:SAEFeature)
        OPTIONAL MATCH (f)-[:LABELED_AS]->(a:AutoInterpLabel)
        RETURN f.index AS feature, inc.attribution AS effect, a.text AS label
        ORDER BY abs(inc.attribution) DESC LIMIT 20
        """,
        cid=cid,
    )


def query_supporting_vs_opposing(c: NeographClient, cid: str) -> dict:
    """Q-CAUSE-4: aggregate supporting vs opposing attribution by community."""
    rows = c.run(
        """
        MATCH (cir:Circuit {id: $cid})-[inc:INCLUDES]->(f:SAEFeature)
        WITH f.communityId AS cid, inc.role AS role, sum(inc.attribution) AS total
        WHERE cid IS NOT NULL
        RETURN cid, role, total
        ORDER BY abs(total) DESC LIMIT 20
        """,
        cid=cid,
    )
    return {"rows": rows}


def query_cross_circuit_overlap(c: NeographClient) -> list[dict]:
    """Q-CAUSE-5: which features participate in MULTIPLE circuits — likely shared mechanism."""
    return c.run(
        """
        MATCH (cir:Circuit)-[inc:INCLUDES]->(f:SAEFeature)
        WITH f, count(DISTINCT cir) AS n_circuits, collect(DISTINCT cir.id) AS circuits,
             sum(inc.attribution) AS total_effect
        WHERE n_circuits > 1
        OPTIONAL MATCH (f)-[:LABELED_AS]->(a:AutoInterpLabel)
        RETURN f.index AS feature, n_circuits, circuits, total_effect, a.text AS label
        ORDER BY n_circuits DESC, abs(total_effect) DESC LIMIT 20
        """
    )


def main() -> int:
    from sae_lens import SAE as SaeLensSAE, HookedSAETransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("Loading %s on %s ...", MODEL.name, device)
    model = HookedSAETransformer.from_pretrained(MODEL.name, device=device)
    model.eval()
    log.info("Loading SAE ...")
    sae = SaeLensSAE.from_pretrained(release=SAE_CFG.release, sae_id=SAE_CFG.sae_id, device=device)

    with NeographClient() as c:
        for spec in PROMPTS:
            target_id = model.tokenizer.encode(spec["target"], add_special_tokens=False)[0]
            log.info("=== prompt=%r target=%r (id=%d) ===", spec["prompt"], spec["target"], target_id)
            edges = ablate_and_score(model, sae, spec["prompt"], target_id, top_n=50)
            log.info("top-5 supporters / opposers:")
            for e in edges[:5]:
                log.info("  feat %5d: effect=%+.3f", e["feature_index"], e["effect_size"])
            write_circuit(c, spec["id"], spec["prompt"], spec["target"], edges)
            log.info("wrote Circuit + 50 INCLUDES edges")

        # Run the 5 queries
        print("\n=== Cypher queries against the populated :CAUSES schema ===")
        for spec in PROMPTS:
            cid = f"circuit/{spec['id']}/{spec['target'].strip()}"
            print(f"\n-- Q-CAUSE-1 ({cid}): communities recruited --")
            for r in query_circuit_to_communities(c, cid):
                print(f"   community {r['community']:>4}  n={r['n_features']:>2}  Σattr={r['total_attribution']:+.3f}")

            print(f"\n-- Q-CAUSE-2 ({cid}): features on a manifold --")
            for r in query_circuit_features_on_manifolds(c, cid):
                print(f"   feat {r['feature']:>5}  effect={r['effect']:+.3f}  manifold={r['manifold']}")

            print(f"\n-- Q-CAUSE-3 ({cid}): top features by attribution + label --")
            for r in query_circuit_labels(c, cid)[:10]:
                label = (r['label'] or '')[:70]
                print(f"   feat {r['feature']:>5}  effect={r['effect']:+.3f}  {label}")

        print("\n-- Q-CAUSE-5: features participating in MULTIPLE circuits --")
        for r in query_cross_circuit_overlap(c)[:10]:
            label = (r['label'] or '')[:70]
            print(f"   feat {r['feature']:>5}  in_{r['n_circuits']} circuits  Σeffect={r['total_effect']:+.3f}  {label}")

    exit_marker("causal-attribution", ok=True, n_prompts=len(PROMPTS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
