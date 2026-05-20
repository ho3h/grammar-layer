"""Causal amplification: bidirectional steering of fingerprint features.

If a feature like Gemma f15596 ("forms of to-be") is a real suppressor of the specific
capital completion, then *amplifying* it should:
- Lower log P(specific target) further
- Raise log P(generic 'a' / 'the' / 'a country' / 'a city' completions)
- Change the argmax from the specific target to a generic noun phrase

If the feature is just incidentally co-active, amplifying should do nothing or push the
output in a random direction.

This closes the interpretability-illusion critique (Makelov et al.; Heimersheim 2024) more
cleanly than ablation alone: bidirectional control is a stronger causal claim than
unidirectional ablation. It also produces a steering primitive — an explicit recipe for
making the model output more generic / hedged completions.

Method per (model, feature, prompt):
1. Baseline forward pass: record log P(target), log P(generic alternatives), argmax.
2. For each scale ∈ {0.0 [=ablate], 1.0 [=baseline], 2.0, 5.0, 10.0}:
   - Hook the SAE acts_post at the last position, multiply that feature by the scale.
   - Record log P(target), log P(generics), argmax.
3. Report: monotonic relationship between scale and log P(target) (negative slope) and
   between scale and log P(generic) (positive slope).

Targets: 'generic' alternatives for capital prompts are tokens that complete "X is __" with
a generic noun phrase rather than a specific city. We use a small fixed list and report
each model's tokeniser-encoded versions.

Usage:
    uv run python scripts/causal_amplification.py --model gemma --feature 15596
    uv run python scripts/causal_amplification.py --model pythia_70m --feature 23527
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from neograph.config import PATHS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_bearing_topk import MODEL_SPECS  # noqa: E402

# Generic noun-phrase tokens to score against the specific capital target. The argmax-
# shift test is "did the top-1 token become a generic continuation when we amplified?"
GENERIC_CANDIDATES = [
    " a", " the", " an",
    " a city", " a country", " a place", " a small",
    " an island", " a kingdom", " a region",
    " one", " not", " also", " known",
]


def _encode_first_token(tokenizer, text: str) -> int | None:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if ids else None


def _score(logits_at_last: torch.Tensor, target_id: int, generic_ids: list[int]) -> dict:
    log_probs = F.log_softmax(logits_at_last.float(), dim=-1)
    argmax_id = int(log_probs.argmax().item())
    return {
        "argmax_token_id": argmax_id,
        "log_p_target": float(log_probs[target_id].item()),
        "log_p_generic_max": float(max(log_probs[g].item() for g in generic_ids)),
        "log_p_generic_sum": float(torch.logsumexp(log_probs[generic_ids], dim=0).item()),
        "log_p_argmax": float(log_probs[argmax_id].item()),
    }


def steer_and_score(model, sae, spec, prompt: str, feature_idx: int, scales: list[float],
                    target_id: int, generic_ids: list[int]) -> list[dict]:
    """For each scale, multiply the feature's last-position activation by scale and score."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    hook_name = f"{spec['hook_name']}.hook_sae_acts_post"

    results = []
    for scale in scales:
        def steer(act, _scale=scale, _fidx=feature_idx, **kwargs):
            act = act.clone()
            act[..., -1, _fidx] = act[..., -1, _fidx] * _scale
            return act

        with torch.no_grad():
            logits = model.run_with_hooks_with_saes(
                tokens, saes=[sae], fwd_hooks=[(hook_name, steer)]
            )
        s = _score(logits[0, -1, :], target_id, generic_ids)
        s["scale"] = scale
        results.append(s)
    return results


def baseline_active_mag(model, sae, spec, prompt: str, feature_idx: int) -> float:
    """Get baseline activation magnitude of the feature at the last position."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    with torch.no_grad():
        _logits, cache = model.run_with_cache_with_saes(tokens, saes=[sae])
    feat_key = next(k for k in cache.keys() if "sae" in k and "acts_post" in k)
    return float(cache[feat_key][0, -1, feature_idx].item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_SPECS.keys()), required=True)
    parser.add_argument("--feature", type=int, required=True,
                        help="SAE feature index to amplify.")
    parser.add_argument("--prompts-file", default="data/prompts_50.json")
    parser.add_argument("--categories", nargs="+", default=["capital"],
                        help="Prompt categories to evaluate (default: capital).")
    parser.add_argument("--scales", nargs="+", type=float,
                        default=[0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
    parser.add_argument("--output", default=None,
                        help="Where to write results (default reports/amp_<model>_f<feat>.json).")
    args = parser.parse_args()

    spec = MODEL_SPECS[args.model]
    if args.output is None:
        args.output = f"reports/amp_{args.model}_f{args.feature}.json"

    prompts_path = Path(args.prompts_file)
    if not prompts_path.is_absolute():
        prompts_path = PATHS.root / prompts_path
    all_prompts = json.loads(prompts_path.read_text())
    prompts = [p for p in all_prompts if p["category"] in args.categories]
    print(f"Loaded {len(prompts)} prompts in categories {args.categories}")

    from sae_lens import SAE as SaeLensSAE, HookedSAETransformer
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading {spec['hf_name']} on {device}...")
    model = HookedSAETransformer.from_pretrained(spec["hf_name"], device=device)
    model.eval()
    print(f"Loading SAE {spec['sae_release']} / {spec['sae_id_attr']}...")
    sae = SaeLensSAE.from_pretrained(
        release=spec["sae_release"], sae_id=spec["sae_id_attr"], device=device
    )
    sae_hook = getattr(sae.cfg, "hook_name", None) if hasattr(sae, "cfg") else None
    if sae_hook and sae_hook != spec["hook_name"]:
        spec = {**spec, "hook_name": sae_hook}

    # Encode the generic alternatives in this tokeniser
    generic_ids: list[int] = []
    generic_strs: list[str] = []
    for cand in GENERIC_CANDIDATES:
        tid = _encode_first_token(model.tokenizer, cand)
        if tid is not None and tid not in generic_ids:
            generic_ids.append(tid)
            generic_strs.append(model.tokenizer.decode([tid]))
    print(f"Generic candidates resolved to {len(generic_ids)} unique token ids:")
    for tid, s in zip(generic_ids, generic_strs):
        print(f"  id={tid:>6}  -> {s!r}")

    per_prompt: list[dict] = []
    for p in prompts:
        target_ids = model.tokenizer.encode(p["target"], add_special_tokens=False)
        if not target_ids:
            continue
        tid = target_ids[0]
        target_str = model.tokenizer.decode([tid])
        baseline_mag = baseline_active_mag(model, sae, spec, p["prompt"], args.feature)
        scored = steer_and_score(
            model, sae, spec, p["prompt"], args.feature, args.scales, tid, generic_ids
        )
        per_prompt.append({
            "id": p["id"],
            "prompt": p["prompt"],
            "target": p["target"],
            "target_token_id": tid,
            "target_token_str": target_str,
            "baseline_feature_mag": baseline_mag,
            "scales": scored,
        })
        # Compact print
        print(f"\n  {p['id']}: target={target_str!r}  baseline_feature_mag={baseline_mag:+.3f}")
        for s in scored:
            argmax_str = model.tokenizer.decode([s["argmax_token_id"]])
            print(
                f"    scale={s['scale']:>5.1f}  "
                f"log_p(target)={s['log_p_target']:+.3f}  "
                f"log_p(generic_max)={s['log_p_generic_max']:+.3f}  "
                f"argmax={argmax_str!r}"
            )

    # Aggregate: monotonicity per prompt + mean per scale
    scales_seen = args.scales
    agg_log_p_target = {s: [] for s in scales_seen}
    agg_log_p_generic_max = {s: [] for s in scales_seen}
    agg_argmax_is_target = {s: [] for s in scales_seen}
    agg_argmax_is_generic = {s: [] for s in scales_seen}
    for r in per_prompt:
        gid_set = set(generic_ids)
        for s in r["scales"]:
            agg_log_p_target[s["scale"]].append(s["log_p_target"])
            agg_log_p_generic_max[s["scale"]].append(s["log_p_generic_max"])
            agg_argmax_is_target[s["scale"]].append(s["argmax_token_id"] == r["target_token_id"])
            agg_argmax_is_generic[s["scale"]].append(s["argmax_token_id"] in gid_set)

    summary = {
        "model": args.model,
        "feature": args.feature,
        "n_prompts": len(per_prompt),
        "scales": scales_seen,
        "mean_log_p_target_by_scale": {
            str(s): sum(vals) / len(vals) if vals else None
            for s, vals in agg_log_p_target.items()
        },
        "mean_log_p_generic_max_by_scale": {
            str(s): sum(vals) / len(vals) if vals else None
            for s, vals in agg_log_p_generic_max.items()
        },
        "hit_rate_target_by_scale": {
            str(s): sum(vals) / len(vals) if vals else None
            for s, vals in agg_argmax_is_target.items()
        },
        "generic_rate_by_scale": {
            str(s): sum(vals) / len(vals) if vals else None
            for s, vals in agg_argmax_is_generic.items()
        },
    }
    print("\n=== Aggregate over %d prompts ===" % len(per_prompt))
    print(f"{'scale':>6}  {'logP(target)':>14}  {'logP(generic_max)':>18}  "
          f"{'hit_target':>11}  {'argmax_generic':>14}")
    for s in scales_seen:
        print(
            f"{s:>6.1f}  "
            f"{summary['mean_log_p_target_by_scale'][str(s)]:>14.3f}  "
            f"{summary['mean_log_p_generic_max_by_scale'][str(s)]:>18.3f}  "
            f"{summary['hit_rate_target_by_scale'][str(s)]:>11.2f}  "
            f"{summary['generic_rate_by_scale'][str(s)]:>14.2f}"
        )

    # Monotonicity check: does mean log P(target) decrease monotonically with scale?
    means_t = [summary["mean_log_p_target_by_scale"][str(s)] for s in scales_seen]
    monotone_decreasing_target = all(
        means_t[i] >= means_t[i + 1] for i in range(len(means_t) - 1)
    )
    means_g = [summary["mean_log_p_generic_max_by_scale"][str(s)] for s in scales_seen]
    monotone_increasing_generic = all(
        means_g[i] <= means_g[i + 1] for i in range(len(means_g) - 1)
    )
    summary["monotone_decreasing_target"] = monotone_decreasing_target
    summary["monotone_increasing_generic"] = monotone_increasing_generic

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PATHS.root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "summary": summary,
        "generic_candidates": list(zip(generic_ids, generic_strs)),
        "per_prompt": per_prompt,
    }, indent=2))
    print(f"\nWrote {out_path}")
    print(f"monotone_decreasing_target = {monotone_decreasing_target}")
    print(f"monotone_increasing_generic = {monotone_increasing_generic}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
