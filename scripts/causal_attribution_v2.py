"""Causal attribution v2 — zero-ablation patching across N diverse prompts on EITHER Gemma 2 2B
or GPT-2 small. Writes one Circuit per prompt + INCLUDES edges per (circuit, feature).

The point of the v2 expansion (Theo's note, 2026-05-12): N=2 circuits is anecdote. We need
8-12 to know whether the predicate-feature backbone (15596, 13414, 12927, 6631, 9768 in Gemma)
is real or coincidence.

Usage:
    uv run python scripts/causal_attribution_v2.py --model gemma
    uv run python scripts/causal_attribution_v2.py --model gpt2

Both models can coexist in the same Neo4j store under different sae_id namespaces. Circuit
ids are namespaced (e.g. `circuit/gemma/capital-fr/Paris`) so cross-model overlap queries
filter by model prefix.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

import torch

from neograph.config import PATHS, SAE as GEMMA_SAE
from neograph.cypher import NeographClient
from neograph.util import exit_marker, get_logger

log = get_logger("neograph.causal.v2")


@dataclass(frozen=True)
class ModelSpec:
    nickname: str  # "gemma" or "gpt2"
    hf_name: str
    sae_release: str
    sae_id_attr: str
    sae_neograph_id: str
    hook_name: str
    activation_threshold: float = 1e-3


GEMMA = ModelSpec(
    nickname="gemma",
    hf_name="gemma-2-2b",
    sae_release=GEMMA_SAE.release,
    sae_id_attr=GEMMA_SAE.sae_id,
    sae_neograph_id=GEMMA_SAE.neograph_id,
    hook_name=GEMMA_SAE.hook_name,
)

GPT2 = ModelSpec(
    nickname="gpt2",
    hf_name="gpt2",
    sae_release="gpt2-small-res-jb",
    sae_id_attr="blocks.8.hook_resid_pre",
    sae_neograph_id="gpt2-small-res-jb/L8",
    hook_name="blocks.8.hook_resid_pre",
)


def _feature_id(spec: ModelSpec, idx: int) -> str:
    return f"{spec.sae_neograph_id}/F{idx:05d}"


def _circuit_id(spec: ModelSpec, prompt_id: str, target_token: str) -> str:
    return f"circuit/{spec.nickname}/{prompt_id}/{target_token.strip()}"


def ablate_and_score(model, sae, spec: ModelSpec, prompt: str, target_token_id: int) -> list[dict]:
    """Zero-ablate each active feature at the last position; record next-token logit deltas.

    Returns list of {feature_index, effect_size, magnitude} for ALL active features (no top-N cap).
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    with torch.no_grad():
        baseline_logits = model.run_with_hooks_with_saes(tokens, saes=[sae], fwd_hooks=[])
    baseline = float(baseline_logits[0, -1, target_token_id].item())

    with torch.no_grad():
        _logits, cache = model.run_with_cache_with_saes(tokens, saes=[sae])
    feat_key = next(k for k in cache.keys() if "sae" in k and "acts_post" in k)
    feat_acts = cache[feat_key][0, -1, :].float().cpu()
    active = (feat_acts > spec.activation_threshold).nonzero(as_tuple=True)[0]

    hook_name = f"{spec.hook_name}.hook_sae_acts_post"
    results: list[dict] = []
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
        results.append({
            "feature_index": int(fidx),
            "effect_size": baseline - ablated,
            "magnitude": float(feat_acts[fidx].item()),
            "baseline_logit": baseline,
        })
    return results


def write_circuit(c: NeographClient, spec: ModelSpec, prompt_meta: dict, edges: list[dict],
                  baseline_logit: float, top_k: int = 50) -> str:
    cid = _circuit_id(spec, prompt_meta["id"], prompt_meta["target"])
    c.run(
        """
        MERGE (cir:Circuit {id: $cid})
          SET cir.prompt_id = $pid, cir.prompt = $prompt, cir.target_token = $tok,
              cir.category = $cat, cir.model = $model, cir.sae_id = $sae,
              cir.source = 'zero-ablation-patching', cir.baseline_logit = $bl
        """,
        cid=cid, pid=prompt_meta["id"], prompt=prompt_meta["prompt"],
        tok=prompt_meta["target"], cat=prompt_meta.get("category", "unknown"),
        model=spec.nickname, sae=spec.sae_neograph_id, bl=float(baseline_logit),
    )
    # Keep all features but truncate visible edges to top-k by |effect|
    sorted_edges = sorted(edges, key=lambda e: abs(e["effect_size"]), reverse=True)[:top_k]
    rows = [
        {"cid": cid, "fid": _feature_id(spec, e["feature_index"]),
         "effect": float(e["effect_size"]), "mag": float(e["magnitude"]),
         "rank": int(rank)}
        for rank, e in enumerate(sorted_edges)
    ]
    c.run(
        """
        UNWIND $rows AS r
        MATCH (f:SAEFeature {id: r.fid}), (cir:Circuit {id: r.cid})
        MERGE (cir)-[inc:INCLUDES]->(f)
          SET inc.role = CASE WHEN r.effect > 0 THEN 'support' ELSE 'oppose' END,
              inc.attribution = r.effect, inc.magnitude = r.mag, inc.rank = r.rank
        """,
        rows=rows,
    )
    return cid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gemma", "gpt2"], required=True)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()
    spec = GEMMA if args.model == "gemma" else GPT2

    prompts = json.loads((PATHS.data / "causal_prompts.json").read_text())
    log.info("Loaded %d prompts", len(prompts))

    from sae_lens import SAE as SaeLensSAE, HookedSAETransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("Loading %s on %s ...", spec.hf_name, device)
    model = HookedSAETransformer.from_pretrained(spec.hf_name, device=device)
    model.eval()
    log.info("Loading SAE %s / %s ...", spec.sae_release, spec.sae_id_attr)
    sae = SaeLensSAE.from_pretrained(release=spec.sae_release, sae_id=spec.sae_id_attr, device=device)

    written_circuits = []
    with NeographClient() as c:
        for prompt_meta in prompts:
            # Tokenise target; require single-token target for clean attribution
            target_ids = model.tokenizer.encode(prompt_meta["target"], add_special_tokens=False)
            if not target_ids:
                log.warning("skip %s: empty target", prompt_meta["id"])
                continue
            target_id = target_ids[0]
            target_str = model.tokenizer.decode([target_id])
            log.info("=== %s: %r → %r (id %d) ===", prompt_meta["id"], prompt_meta["prompt"],
                     target_str, target_id)
            edges = ablate_and_score(model, sae, spec, prompt_meta["prompt"], target_id)
            if not edges:
                log.warning("no active features at last position for %s — skipping", prompt_meta["id"])
                continue
            baseline = edges[0]["baseline_logit"] if edges else 0.0
            top5 = sorted(edges, key=lambda e: abs(e["effect_size"]), reverse=True)[:5]
            for e in top5:
                log.info("  feat %5d  effect=%+.3f  mag=%.2f", e["feature_index"], e["effect_size"], e["magnitude"])
            cid = write_circuit(c, spec, prompt_meta, edges, baseline, top_k=args.top_k)
            written_circuits.append(cid)
            log.info("  → wrote %s with %d INCLUDES edges", cid, min(len(edges), args.top_k))

    log.info("Done. Wrote %d circuits for %s.", len(written_circuits), spec.nickname)
    out = PATHS.reports / f"causal_circuits_{spec.nickname}.json"
    out.write_text(json.dumps(written_circuits, indent=2))
    exit_marker(f"causal-attribution-{spec.nickname}", ok=len(written_circuits) >= 8,
                model=spec.nickname, n_circuits=len(written_circuits))
    return 0


if __name__ == "__main__":
    sys.exit(main())
