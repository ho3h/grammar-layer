"""Causal ablation (P1) — joint and single SAE-feature ablation on Gemma 2 2B or GPT-2 small.

Adapts the hook pattern from causal_attribution_v2.py but flips the question: instead of
ranking features by attribution score, this measures whether ablating the candidate backbone
features DESTROYS completion. If joint-ablation of {6631, 9768, 13414} drops target hit rate
uniformly across categories with ≥1 nat log-P drop on target, the backbone is *load-bearing*,
not just correlated.

Methods:
    --method zero  → set feature activation to 0 at the last position
    --method mean  → set feature activation to its mean across the rest of the prompt set at the
                     last position (leave-one-out style; the corpus-mean counterfactual)

Conditions per prompt:
    baseline                                      (no hook)
    ablate_<f> for each f in --single-ablate     (one feature at a time)
    joint_<f1>_<f2>_…  for the --features set    (joint)

Metrics per condition: argmax token, hit (argmax==target), log P(target), log P(argmax),
target rank in sorted logits, entropy over vocab.

Usage:
    uv run python scripts/causal_ablation.py \
        --model gemma \
        --prompts-file data/causal_prompts.json \
        --features 6631 9768 13414 \
        --single-ablate 6631 9768 15596 13414 12927 \
        --method zero \
        --output reports/ablation_zero_gemma_12.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from neograph.config import PATHS, SAE as GEMMA_SAE
from neograph.util import exit_marker, get_logger

log = get_logger("neograph.causal.ablation")


@dataclass(frozen=True)
class ModelSpec:
    nickname: str
    hf_name: str
    sae_release: str
    sae_id_attr: str
    sae_neograph_id: str
    hook_name: str


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


def _score(logits_at_last: torch.Tensor, target_id: int) -> dict:
    """Reduce per-position logits to a scalar metric bundle for the last position."""
    log_probs = F.log_softmax(logits_at_last.float(), dim=-1)
    log_p_target = float(log_probs[target_id].item())
    argmax_id = int(log_probs.argmax().item())
    log_p_argmax = float(log_probs[argmax_id].item())
    sorted_ids = torch.argsort(log_probs, descending=True)
    target_rank = int((sorted_ids == target_id).nonzero(as_tuple=True)[0].item())
    probs = log_probs.exp()
    entropy = float(-(probs * log_probs).sum().item())
    return {
        "argmax_token_id": argmax_id,
        "log_p_target": log_p_target,
        "log_p_argmax": log_p_argmax,
        "target_rank": target_rank,
        "entropy": entropy,
        "hit": argmax_id == target_id,
    }


def collect_baseline_acts(model, sae, spec: ModelSpec, prompts: list[dict],
                          features: list[int]) -> dict[int, list[float]]:
    """For each feature in `features`, record its last-position activation on every prompt.

    Returned mapping {fidx: [acts]} is used to build the leave-one-out mean for --method mean.
    """
    feat_key_cache: list[str] = []
    out: dict[int, list[float]] = {f: [] for f in features}
    for p in prompts:
        tokens = model.to_tokens(p["prompt"], prepend_bos=True)
        with torch.no_grad():
            _logits, cache = model.run_with_cache_with_saes(tokens, saes=[sae])
        if not feat_key_cache:
            feat_key_cache.append(next(k for k in cache.keys() if "sae" in k and "acts_post" in k))
        feat_key = feat_key_cache[0]
        last_acts = cache[feat_key][0, -1, :].float().cpu()
        for f in features:
            out[f].append(float(last_acts[f].item()))
    return out


def make_hook(feature_indices: list[int], replacement_values: list[float] | None):
    """Build a hook that replaces the named features' last-position activation.

    `replacement_values=None` means zero-ablate. Otherwise replacement_values[i] is the value to
    place at feature_indices[i].
    """
    if replacement_values is None:
        repl = [0.0] * len(feature_indices)
    else:
        assert len(replacement_values) == len(feature_indices)
        repl = replacement_values

    def hook(act, _fidxs=tuple(feature_indices), _repls=tuple(repl), **kwargs):
        act = act.clone()
        for fidx, val in zip(_fidxs, _repls):
            act[..., -1, fidx] = val
        return act

    return hook


def score_condition(model, sae, spec: ModelSpec, prompt: str, target_id: int,
                    hook_features: list[int], replacement_values: list[float] | None) -> dict:
    tokens = model.to_tokens(prompt, prepend_bos=True)
    hook_name = f"{spec.hook_name}.hook_sae_acts_post"
    fwd_hooks = []
    if hook_features:
        fwd_hooks = [(hook_name, make_hook(hook_features, replacement_values))]
    with torch.no_grad():
        logits = model.run_with_hooks_with_saes(tokens, saes=[sae], fwd_hooks=fwd_hooks)
    return _score(logits[0, -1, :], target_id)


def summarise(per_prompt: list[dict], condition_keys: list[str]) -> dict:
    """Compute per-condition aggregates: hit rate, mean log P drop, per-category drop."""
    categories = sorted({r["category"] for r in per_prompt})
    summary: dict = {"overall": {}, "per_category": {c: {} for c in categories}}

    baseline_hits = [r["conditions"]["baseline"]["hit"] for r in per_prompt]
    baseline_logps = [r["conditions"]["baseline"]["log_p_target"] for r in per_prompt]

    def _stats(hits: list[bool], logps: list[float], baseline_lp: list[float]) -> dict:
        hit_rate = sum(hits) / len(hits) if hits else 0.0
        mean_logp = sum(logps) / len(logps) if logps else 0.0
        mean_logp_drop = sum(bl - lp for bl, lp in zip(baseline_lp, logps)) / len(logps) if logps else 0.0
        return {
            "n": len(hits),
            "hit_rate": hit_rate,
            "mean_log_p_target": mean_logp,
            "mean_log_p_drop_vs_baseline": mean_logp_drop,
        }

    for cond in condition_keys:
        hits = [r["conditions"][cond]["hit"] for r in per_prompt]
        logps = [r["conditions"][cond]["log_p_target"] for r in per_prompt]
        summary["overall"][cond] = _stats(hits, logps, baseline_logps)
        for cat in categories:
            cat_rows = [r for r in per_prompt if r["category"] == cat]
            cat_hits = [r["conditions"][cond]["hit"] for r in cat_rows]
            cat_logps = [r["conditions"][cond]["log_p_target"] for r in cat_rows]
            cat_baseline = [r["conditions"]["baseline"]["log_p_target"] for r in cat_rows]
            summary["per_category"][cat][cond] = _stats(cat_hits, cat_logps, cat_baseline)

    # Load-bearing index = baseline hit_rate − joint hit_rate (largest such delta across conditions)
    joint_cond = next((c for c in condition_keys if c.startswith("joint_")), None)
    if joint_cond:
        summary["load_bearing_index"] = (
            summary["overall"]["baseline"]["hit_rate"]
            - summary["overall"][joint_cond]["hit_rate"]
        )
        summary["uniform_drop"] = all(
            summary["per_category"][cat]["baseline"]["hit_rate"]
            - summary["per_category"][cat][joint_cond]["hit_rate"]
            >= 0.4
            for cat in categories
            if summary["per_category"][cat]["baseline"]["hit_rate"] >= 0.5
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gemma", "gpt2"], default="gemma")
    parser.add_argument("--prompts-file", required=True,
                        help="Path to a JSON list of {id, prompt, target, category}.")
    parser.add_argument("--features", type=int, nargs="+", required=True,
                        help="Feature indices to ablate JOINTLY (the load-bearing test).")
    parser.add_argument("--single-ablate", type=int, nargs="*", default=[],
                        help="Features to also ablate individually for per-feature breakdown.")
    parser.add_argument("--method", choices=["zero", "mean"], default="zero")
    parser.add_argument("--output", required=True, help="Where to write the results JSON.")
    args = parser.parse_args()

    spec = GEMMA if args.model == "gemma" else GPT2
    prompts_path = Path(args.prompts_file)
    if not prompts_path.is_absolute():
        prompts_path = PATHS.root / prompts_path
    prompts = json.loads(prompts_path.read_text())
    log.info("Loaded %d prompts from %s", len(prompts), prompts_path)

    all_features = sorted({*args.features, *args.single_ablate})
    log.info("Conditions: baseline + %d single + joint(%s)",
             len(args.single_ablate), "+".join(str(f) for f in args.features))

    from sae_lens import SAE as SaeLensSAE, HookedSAETransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("Loading %s on %s ...", spec.hf_name, device)
    model = HookedSAETransformer.from_pretrained(spec.hf_name, device=device)
    model.eval()
    log.info("Loading SAE %s / %s ...", spec.sae_release, spec.sae_id_attr)
    sae = SaeLensSAE.from_pretrained(release=spec.sae_release, sae_id=spec.sae_id_attr, device=device)

    # For mean-ablation we need per-feature activation across prompts (leave-one-out mean).
    feat_acts_by_prompt: dict[int, list[float]] = {}
    if args.method == "mean":
        log.info("Collecting baseline activations on %d prompts for mean-ablation counterfactual",
                 len(prompts))
        feat_acts_by_prompt = collect_baseline_acts(model, sae, spec, prompts, all_features)

    # Resolve target token ids
    enriched: list[dict] = []
    for p in prompts:
        target_ids = model.tokenizer.encode(p["target"], add_special_tokens=False)
        if not target_ids:
            log.warning("skip %s: empty target", p["id"])
            continue
        tid = target_ids[0]
        tok_str = model.tokenizer.decode([tid])
        enriched.append({**p, "target_token_id": tid, "target_token_str": tok_str})

    per_prompt: list[dict] = []
    joint_key = f"joint_{'_'.join(str(f) for f in args.features)}"
    condition_keys = ["baseline"] + [f"ablate_{f}" for f in args.single_ablate] + [joint_key]

    for idx, p in enumerate(enriched):
        tid = p["target_token_id"]
        log.info("[%d/%d] %s  prompt=%r  target=%r (id=%d)",
                 idx + 1, len(enriched), p["id"], p["prompt"], p["target_token_str"], tid)

        conditions: dict[str, dict] = {}
        # Baseline
        conditions["baseline"] = score_condition(model, sae, spec, p["prompt"], tid, [], None)

        # Single ablations
        for f in args.single_ablate:
            if args.method == "zero":
                repl: list[float] | None = None
            else:
                # leave-one-out mean across the OTHER prompts
                acts = feat_acts_by_prompt[f]
                other = [a for j, a in enumerate(acts) if j != idx]
                repl = [sum(other) / len(other) if other else 0.0]
            conditions[f"ablate_{f}"] = score_condition(model, sae, spec, p["prompt"], tid, [f], repl)

        # Joint ablation
        if args.method == "zero":
            repl_joint: list[float] | None = None
        else:
            repl_joint = []
            for f in args.features:
                acts = feat_acts_by_prompt[f]
                other = [a for j, a in enumerate(acts) if j != idx]
                repl_joint.append(sum(other) / len(other) if other else 0.0)
        conditions[joint_key] = score_condition(
            model, sae, spec, p["prompt"], tid, list(args.features), repl_joint
        )

        # Attach decoded argmax for readability
        for cond in conditions.values():
            cond["argmax_token_str"] = model.tokenizer.decode([cond["argmax_token_id"]])

        per_prompt.append({**p, "conditions": conditions})

        baseline = conditions["baseline"]
        joint = conditions[joint_key]
        log.info(
            "    baseline: argmax=%r logP=%+0.2f hit=%s  →  joint: argmax=%r logP=%+0.2f hit=%s  (drop logP=%+0.2f)",
            baseline["argmax_token_str"], baseline["log_p_target"], baseline["hit"],
            joint["argmax_token_str"], joint["log_p_target"], joint["hit"],
            baseline["log_p_target"] - joint["log_p_target"],
        )

    summary = summarise(per_prompt, condition_keys)

    out = {
        "model": spec.nickname,
        "sae": spec.sae_neograph_id,
        "prompts_file": str(prompts_path.relative_to(PATHS.root)),
        "method": args.method,
        "joint_features": list(args.features),
        "single_ablate_features": list(args.single_ablate),
        "conditions": condition_keys,
        "results": per_prompt,
        "summary": summary,
    }
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PATHS.root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", out_path)
    log.info("=== Summary ===")
    log.info("Baseline hit rate: %.2f  |  Joint hit rate: %.2f  |  Load-bearing index: %.2f",
             summary["overall"]["baseline"]["hit_rate"],
             summary["overall"][joint_key]["hit_rate"],
             summary.get("load_bearing_index", 0.0))
    for cat, conds in summary["per_category"].items():
        b = conds["baseline"]["hit_rate"]
        j = conds[joint_key]["hit_rate"]
        dlp = conds[joint_key]["mean_log_p_drop_vs_baseline"]
        log.info("  %-24s  baseline=%.2f  joint=%.2f  Δ=%.2f  Δlog P(target)=%+0.2f",
                 cat, b, j, b - j, dlp)

    exit_marker(
        f"causal-ablation-{spec.nickname}-{args.method}",
        ok=summary.get("load_bearing_index", 0.0) >= 0.4 and summary.get("uniform_drop", False),
        model=spec.nickname,
        method=args.method,
        load_bearing_index=summary.get("load_bearing_index", 0.0),
        baseline_hit_rate=summary["overall"]["baseline"]["hit_rate"],
        joint_hit_rate=summary["overall"][joint_key]["hit_rate"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
