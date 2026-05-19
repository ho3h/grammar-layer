"""Find the per-prompt top-K features by attribution magnitude, then jointly ablate them.

This is the diagnostic that distinguishes "the named backbone (6631, 9768, 13414…) isn't
load-bearing" (already shown by causal_ablation.py) from "nothing is load-bearing in joint
zero-ablation for this model+SAE setup".

Method per prompt:
1. Compute per-feature attribution to target log-P via zero-ablation at last position (one
   forward pass per active feature, same as causal_attribution_v2.py but on log-P).
2. Sort active features by |Δlog P(target)|, descending.
3. Take top-K (default K=10).
4. Joint zero-ablate those K features, record baseline/joint hit rate, target log-P,
   target rank, entropy, argmax shift.
5. Also record the K feature labels (if a label cache is provided).

Output: per-prompt {top_features:[{idx, single_effect}], baseline, joint, drop}.

Usage:
    uv run python scripts/load_bearing_topk.py \
        --model gemma \
        --prompts-file data/causal_prompts.json \
        --top-k 10 \
        --output reports/load_bearing_topk_gemma_12.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from neograph.config import PATHS, SAE as GEMMA_SAE
from neograph.util import get_logger

log = get_logger("neograph.load_bearing")


GEMMA = {
    "nickname": "gemma",
    "hf_name": "gemma-2-2b",
    "sae_release": GEMMA_SAE.release,
    "sae_id_attr": GEMMA_SAE.sae_id,
    "hook_name": GEMMA_SAE.hook_name,
    "activation_threshold": 1e-3,
}

# Gemma 2 9B — 42 layers, layer 20 is roughly mid-network (analogous depth to
# Gemma 2 2B layer 20 in a 26-layer model). The pt-res-canonical release has
# 16k-width SAEs at every layer. Picking layer 20 matches the 2B run for the
# clearest within-family scale comparison.
GEMMA_9B = {
    "nickname": "gemma_9b",
    "hf_name": "gemma-2-9b",
    "sae_release": "gemma-scope-9b-pt-res-canonical",
    "sae_id_attr": "layer_20/width_16k/canonical",
    "hook_name": "blocks.20.hook_resid_post",
    "activation_threshold": 1e-3,
}

GPT2 = {
    "nickname": "gpt2",
    "hf_name": "gpt2",
    "sae_release": "gpt2-small-res-jb",
    "sae_id_attr": "blocks.8.hook_resid_pre",
    "hook_name": "blocks.8.hook_resid_pre",
    "activation_threshold": 1e-3,
}

# Pythia 70M deduped — 6 layers. Layer 5 is the deepest residual SAE available.
PYTHIA_70M = {
    "nickname": "pythia_70m",
    "hf_name": "EleutherAI/pythia-70m-deduped",
    "sae_release": "pythia-70m-deduped-res-sm",
    "sae_id_attr": "blocks.5.hook_resid_post",
    "hook_name": "blocks.5.hook_resid_post",
    "activation_threshold": 1e-3,
}

# Gemma 1 2B (older generation, instruction-tuned base). Joseph Bloom's SAEs.
# 18 transformer layers. Pick block 12 (~67% depth) to roughly match GPT-2 L8.
GEMMA_1_2B = {
    "nickname": "gemma_1_2b",
    "hf_name": "gemma-2b",
    "sae_release": "gemma-2b-res-jb",
    "sae_id_attr": "blocks.12.hook_resid_post",
    "hook_name": "blocks.12.hook_resid_post",
    "activation_threshold": 1e-3,
}

# Qwen 3 1.7B base (Qwen Scope 3 release, w32k, l50 = layer 50% depth — wait,
# actually "l50" here means "loss=50" sparsity, not layer depth). The sae_id is
# `layer14`. 1.7B has 28 layers; layer 14 ≈ 50% depth. We'll try mid-late by
# picking a deeper sae_id at run time if available.
QWEN3_1_7B = {
    "nickname": "qwen3_1_7b",
    "hf_name": "Qwen/Qwen3-1.7B",
    "sae_release": "qwen-scope-3-1.7b-base-w32k-l50",
    "sae_id_attr": "layer20",
    "hook_name": "blocks.20.hook_resid_post",
    "activation_threshold": 1e-3,
}

# Mistral 7B base — SAE only at blocks 8/16/24. Block 24 is deepest (75% of 32).
MISTRAL_7B = {
    "nickname": "mistral_7b",
    "hf_name": "mistralai/Mistral-7B-v0.1",
    "sae_release": "mistral-7b-res-wg",
    "sae_id_attr": "blocks.24.hook_resid_pre",
    "hook_name": "blocks.24.hook_resid_pre",
    "activation_threshold": 1e-3,
}

# Gemma 2 27B — 46 layers, SAE at layers 10/22/34 only. Pick L22 (~48% depth)
# to match the relative depth of Gemma 9B L20 (48%). Width 131k.
# Memory-heavy: ~54GB at fp16; we may need bf16 or 4-bit on a 128GB box.
GEMMA_27B = {
    "nickname": "gemma_27b",
    "hf_name": "gemma-2-27b",
    "sae_release": "gemma-scope-27b-pt-res-canonical",
    "sae_id_attr": "layer_22/width_131k/canonical",
    "hook_name": "blocks.22.hook_resid_post",
    "activation_threshold": 1e-3,
}

# Gemma 2 2B at width 65k — the width-stability check for the v3 fingerprint.
# Same model, same layer, larger SAE — tests whether the f15596/f10142 fingerprint
# survives at higher SAE width or fragments into finer features.
GEMMA_W65K = {
    "nickname": "gemma_w65k",
    "hf_name": "gemma-2-2b",
    "sae_release": "gemma-scope-2b-pt-res-canonical",
    "sae_id_attr": "layer_20/width_65k/canonical",
    "hook_name": "blocks.20.hook_resid_post",
    "activation_threshold": 1e-3,
}

MODEL_SPECS = {
    "gemma":       GEMMA,
    "gemma_w65k":  GEMMA_W65K,
    "gemma_9b":    GEMMA_9B,
    "gemma_27b":   GEMMA_27B,
    "gemma_1_2b":  GEMMA_1_2B,
    "gpt2":        GPT2,
    "pythia_70m":  PYTHIA_70M,
    "qwen3_1_7b":  QWEN3_1_7B,
    "mistral_7b":  MISTRAL_7B,
}


def _score(logits_at_last: torch.Tensor, target_id: int) -> dict:
    log_probs = F.log_softmax(logits_at_last.float(), dim=-1)
    argmax_id = int(log_probs.argmax().item())
    sorted_ids = torch.argsort(log_probs, descending=True)
    target_rank = int((sorted_ids == target_id).nonzero(as_tuple=True)[0].item())
    probs = log_probs.exp()
    entropy = float(-(probs * log_probs).sum().item())
    return {
        "argmax_token_id": argmax_id,
        "log_p_target": float(log_probs[target_id].item()),
        "log_p_argmax": float(log_probs[argmax_id].item()),
        "target_rank": target_rank,
        "entropy": entropy,
        "hit": argmax_id == target_id,
    }


def per_feature_attribution(model, sae, spec, prompt: str, target_id: int) -> tuple[dict, list[dict]]:
    """One forward pass per active feature → per-feature Δlog P(target). Returns (baseline_score, [{idx, effect, mag}])."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    with torch.no_grad():
        baseline_logits = model.run_with_hooks_with_saes(tokens, saes=[sae], fwd_hooks=[])
    baseline_score = _score(baseline_logits[0, -1, :], target_id)
    baseline_lp = baseline_score["log_p_target"]

    with torch.no_grad():
        _logits, cache = model.run_with_cache_with_saes(tokens, saes=[sae])
    feat_key = next(k for k in cache.keys() if "sae" in k and "acts_post" in k)
    feat_acts = cache[feat_key][0, -1, :].float().cpu()
    active = (feat_acts > spec["activation_threshold"]).nonzero(as_tuple=True)[0].tolist()

    hook_name = f"{spec['hook_name']}.hook_sae_acts_post"
    effects: list[dict] = []
    for fidx in active:
        def ablate(act, fidx=fidx, **kwargs):
            act = act.clone()
            act[..., -1, fidx] = 0.0
            return act
        with torch.no_grad():
            abl_logits = model.run_with_hooks_with_saes(tokens, saes=[sae], fwd_hooks=[(hook_name, ablate)])
        abl_lp = float(F.log_softmax(abl_logits[0, -1, :].float(), dim=-1)[target_id].item())
        effects.append({
            "feature_index": int(fidx),
            "single_log_p_drop": baseline_lp - abl_lp,
            "magnitude": float(feat_acts[fidx].item()),
        })
    return baseline_score, effects


def joint_ablate(model, sae, spec, prompt: str, target_id: int, feature_indices: list[int]) -> dict:
    tokens = model.to_tokens(prompt, prepend_bos=True)
    hook_name = f"{spec['hook_name']}.hook_sae_acts_post"

    def ablate(act, _fidxs=tuple(feature_indices), **kwargs):
        act = act.clone()
        for fidx in _fidxs:
            act[..., -1, fidx] = 0.0
        return act

    with torch.no_grad():
        logits = model.run_with_hooks_with_saes(tokens, saes=[sae], fwd_hooks=[(hook_name, ablate)])
    return _score(logits[0, -1, :], target_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_SPECS.keys()), default="gemma")
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--sign", choices=["abs", "positive", "negative"], default="abs",
                        help="Rank features by signed attribution (positive = supporting target, "
                             "negative = opposing) instead of |attribution|.")
    parser.add_argument("--labels-cache", default=None,
                        help="Optional path to labels_cache.json for human-readable labels.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    spec = MODEL_SPECS[args.model]

    prompts_path = Path(args.prompts_file)
    if not prompts_path.is_absolute():
        prompts_path = PATHS.root / prompts_path
    prompts = json.loads(prompts_path.read_text())
    log.info("Loaded %d prompts from %s", len(prompts), prompts_path)

    labels: dict[int, str] = {}
    if args.labels_cache:
        lc = Path(args.labels_cache)
        if not lc.is_absolute():
            lc = PATHS.root / lc
        if lc.exists():
            raw = json.loads(lc.read_text())
            # The cache key format may be feature-index strings or full SAE-feature IDs.
            for k, v in raw.items():
                try:
                    if isinstance(v, dict) and "label" in v:
                        labels[int(k.split("/")[-1].replace("F", "").lstrip("0") or "0")] = v["label"]
                    elif isinstance(v, str):
                        labels[int(k.split("/")[-1].replace("F", "").lstrip("0") or "0")] = v
                except (ValueError, KeyError):
                    pass
            log.info("Loaded %d labels", len(labels))

    from sae_lens import SAE as SaeLensSAE, HookedSAETransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("Loading %s on %s ...", spec["hf_name"], device)
    model = HookedSAETransformer.from_pretrained(spec["hf_name"], device=device)
    model.eval()
    log.info("Loading SAE %s / %s ...", spec["sae_release"], spec["sae_id_attr"])
    sae = SaeLensSAE.from_pretrained(release=spec["sae_release"], sae_id=spec["sae_id_attr"], device=device)

    # Trust the SAE's own hook_name over the spec — auto-corrects spec typos
    # when adding new model families.
    sae_hook = getattr(sae.cfg, "hook_name", None) if hasattr(sae, "cfg") else None
    if sae_hook and sae_hook != spec["hook_name"]:
        log.warning("Overriding spec hook_name %r with sae.cfg.hook_name %r",
                    spec["hook_name"], sae_hook)
        spec = {**spec, "hook_name": sae_hook}

    results: list[dict] = []
    for i, p in enumerate(prompts):
        target_ids = model.tokenizer.encode(p["target"], add_special_tokens=False)
        if not target_ids:
            continue
        tid = target_ids[0]
        target_str = model.tokenizer.decode([tid])
        log.info("[%d/%d] %s  target=%r (id=%d)  active features...", i + 1, len(prompts), p["id"], target_str, tid)

        baseline, effects = per_feature_attribution(model, sae, spec, p["prompt"], tid)
        if args.sign == "positive":
            filtered = [e for e in effects if e["single_log_p_drop"] > 0]
            filtered.sort(key=lambda e: e["single_log_p_drop"], reverse=True)
        elif args.sign == "negative":
            filtered = [e for e in effects if e["single_log_p_drop"] < 0]
            filtered.sort(key=lambda e: e["single_log_p_drop"])
        else:
            filtered = list(effects)
            filtered.sort(key=lambda e: abs(e["single_log_p_drop"]), reverse=True)
        topk = filtered[: args.top_k]
        topk_ids = [e["feature_index"] for e in topk]

        joint = joint_ablate(model, sae, spec, p["prompt"], tid, topk_ids)

        # Attach labels if available
        for e in topk:
            e["label"] = labels.get(e["feature_index"], None)

        # Also record the top-K by the OTHER signs so a single run yields enough data
        # for downstream "supporting vs opposing" analysis without re-running attribution.
        topk_pos = sorted([e for e in effects if e["single_log_p_drop"] > 0],
                          key=lambda e: e["single_log_p_drop"], reverse=True)[: args.top_k]
        topk_neg = sorted([e for e in effects if e["single_log_p_drop"] < 0],
                          key=lambda e: e["single_log_p_drop"])[: args.top_k]
        topk_abs = sorted(effects, key=lambda e: abs(e["single_log_p_drop"]), reverse=True)[: args.top_k]

        results.append({
            "id": p["id"],
            "category": p["category"],
            "prompt": p["prompt"],
            "target": p["target"],
            "target_token_id": tid,
            "target_token_str": target_str,
            "n_active_features": len(effects),
            "topk_features": topk,            # primary set (per --sign flag) — also used for joint ablation
            "topk_supporting": topk_pos,      # always populated, regardless of --sign
            "topk_opposing": topk_neg,        # always populated, regardless of --sign
            "topk_by_abs": topk_abs,
            "baseline": {**baseline, "argmax_token_str": model.tokenizer.decode([baseline["argmax_token_id"]])},
            "joint_topk_ablated": {**joint, "argmax_token_str": model.tokenizer.decode([joint["argmax_token_id"]])},
            "log_p_drop_vs_baseline": baseline["log_p_target"] - joint["log_p_target"],
        })

        log.info(
            "    %d active feats, top-%d single drops sum=%+.2f  joint drop=%+.2f  hit %s→%s",
            len(effects), args.top_k,
            sum(e["single_log_p_drop"] for e in topk),
            results[-1]["log_p_drop_vs_baseline"],
            baseline["hit"], joint["hit"],
        )

    # Aggregate
    cats = sorted({r["category"] for r in results})
    summary: dict = {"per_category": {}, "overall": {}}
    for cat in cats + ["__all__"]:
        rows = results if cat == "__all__" else [r for r in results if r["category"] == cat]
        if not rows:
            continue
        baseline_hits = sum(r["baseline"]["hit"] for r in rows)
        joint_hits = sum(r["joint_topk_ablated"]["hit"] for r in rows)
        mean_drop = sum(r["log_p_drop_vs_baseline"] for r in rows) / len(rows)
        mean_n_active = sum(r["n_active_features"] for r in rows) / len(rows)
        bucket = summary["overall"] if cat == "__all__" else summary["per_category"].setdefault(cat, {})
        bucket.update({
            "n": len(rows),
            "baseline_hit_rate": baseline_hits / len(rows),
            "joint_hit_rate": joint_hits / len(rows),
            "mean_log_p_drop": mean_drop,
            "mean_n_active_features": mean_n_active,
        })

    out = {
        "model": spec["nickname"],
        "prompts_file": str(prompts_path.relative_to(PATHS.root)),
        "top_k": args.top_k,
        "sign": args.sign,
        "results": results,
        "summary": summary,
    }
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PATHS.root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", out_path)
    log.info("=== Summary ===")
    log.info("Overall:  baseline_hit=%.2f  joint_top%d_hit=%.2f  mean_logP_drop=%+0.2f  mean_n_active=%.1f",
             summary["overall"]["baseline_hit_rate"], args.top_k, summary["overall"]["joint_hit_rate"],
             summary["overall"]["mean_log_p_drop"], summary["overall"]["mean_n_active_features"])
    for cat, s in summary["per_category"].items():
        log.info("  %-24s n=%d  baseline=%.2f  joint=%.2f  Δ=%.2f  Δlog P=%+0.2f",
                 cat, s["n"], s["baseline_hit_rate"], s["joint_hit_rate"],
                 s["baseline_hit_rate"] - s["joint_hit_rate"], s["mean_log_p_drop"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
