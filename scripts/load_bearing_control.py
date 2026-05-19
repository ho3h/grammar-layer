"""Control experiment for load_bearing_topk.

For each prompt, compute per-feature attribution (same as load_bearing_topk.py), then
jointly ablate FOUR feature sets and compare:

1. supporting_top10  — top-10 by positive single_log_p_drop   (the headline condition)
2. bottom10_by_attr  — 10 *active* features with the smallest |single_log_p_drop|
3. random10_active   — 10 randomly sampled active features, averaged over N seeds
4. all_supporting    — every feature with positive single_log_p_drop (upper bound)

The relevant comparison is (1) vs (2)/(3). If (1) collapses the target but (2)/(3) don't,
the attribution-based selection is doing real work. If they all collapse similarly, the
"load-bearing" claim is an artifact of ablating ~10 features regardless of which 10.

Usage:
    uv run python scripts/load_bearing_control.py \
        --model gemma --prompts-file data/prompts_50.json \
        --top-k 10 --n-random-seeds 5 \
        --output reports/load_bearing_control_gemma_50.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

from neograph.config import PATHS
from neograph.util import get_logger

# Reuse model specs + helpers from the headline script — keeps the controls
# locked to the exact same model+SAE+layer setup as the original results.
sys.path.insert(0, str(Path(__file__).parent))
from load_bearing_topk import MODEL_SPECS, _score, joint_ablate, per_feature_attribution  # noqa: E402

log = get_logger("neograph.load_bearing_control")


def pick_random_active(effects: list[dict], k: int, rng: random.Random) -> list[int]:
    pool = [e["feature_index"] for e in effects]
    if len(pool) <= k:
        return pool
    return rng.sample(pool, k)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_SPECS.keys()), default="gemma")
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--n-random-seeds", type=int, default=5,
                        help="Number of random feature sets to average over per prompt.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    spec = MODEL_SPECS[args.model]

    prompts_path = Path(args.prompts_file)
    if not prompts_path.is_absolute():
        prompts_path = PATHS.root / prompts_path
    prompts = json.loads(prompts_path.read_text())
    log.info("Loaded %d prompts from %s", len(prompts), prompts_path)

    from sae_lens import SAE as SaeLensSAE, HookedSAETransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("Loading %s on %s ...", spec["hf_name"], device)
    model = HookedSAETransformer.from_pretrained(spec["hf_name"], device=device)
    model.eval()
    log.info("Loading SAE %s / %s ...", spec["sae_release"], spec["sae_id_attr"])
    sae = SaeLensSAE.from_pretrained(release=spec["sae_release"], sae_id=spec["sae_id_attr"], device=device)

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
        log.info("[%d/%d] %s  target=%r", i + 1, len(prompts), p["id"], p["target"])

        baseline, effects = per_feature_attribution(model, sae, spec, p["prompt"], tid)
        baseline_lp = baseline["log_p_target"]

        # Condition 1: supporting top-K (the headline)
        supporting = sorted([e for e in effects if e["single_log_p_drop"] > 0],
                            key=lambda e: e["single_log_p_drop"], reverse=True)[: args.top_k]
        sup_ids = [e["feature_index"] for e in supporting]
        sup_score = joint_ablate(model, sae, spec, p["prompt"], tid, sup_ids) if sup_ids else None

        # Condition 2: bottom-K by |single_log_p_drop| among active features
        bottom = sorted(effects, key=lambda e: abs(e["single_log_p_drop"]))[: args.top_k]
        bottom_ids = [e["feature_index"] for e in bottom]
        bottom_score = joint_ablate(model, sae, spec, p["prompt"], tid, bottom_ids)

        # Condition 3: random K from active, averaged over N seeds
        random_runs = []
        for s in range(args.n_random_seeds):
            rng = random.Random(1000 * i + s)
            rand_ids = pick_random_active(effects, args.top_k, rng)
            r_score = joint_ablate(model, sae, spec, p["prompt"], tid, rand_ids)
            random_runs.append({
                "seed": s,
                "feature_indices": rand_ids,
                "log_p_target": r_score["log_p_target"],
                "log_p_drop": baseline_lp - r_score["log_p_target"],
                "hit": r_score["hit"],
            })
        random_drops = [r["log_p_drop"] for r in random_runs]
        random_hits = [r["hit"] for r in random_runs]

        # Condition 4: all supporting features (upper bound on how much you can ablate)
        all_sup_ids = [e["feature_index"] for e in effects if e["single_log_p_drop"] > 0]
        all_sup_score = (joint_ablate(model, sae, spec, p["prompt"], tid, all_sup_ids)
                         if all_sup_ids else None)

        row = {
            "id": p["id"],
            "category": p["category"],
            "prompt": p["prompt"],
            "target": p["target"],
            "target_token_id": tid,
            "n_active_features": len(effects),
            "baseline": baseline,
            "supporting_top10": {
                "feature_indices": sup_ids,
                "score": sup_score,
                "log_p_drop": (baseline_lp - sup_score["log_p_target"]) if sup_score else None,
            },
            "bottom10_by_attr": {
                "feature_indices": bottom_ids,
                "score": bottom_score,
                "log_p_drop": baseline_lp - bottom_score["log_p_target"],
            },
            "random10_active": {
                "n_seeds": args.n_random_seeds,
                "runs": random_runs,
                "mean_log_p_drop": sum(random_drops) / len(random_drops),
                "std_log_p_drop": (sum((d - sum(random_drops) / len(random_drops)) ** 2 for d in random_drops) / len(random_drops)) ** 0.5,
                "hit_rate": sum(random_hits) / len(random_hits),
            },
            "all_supporting": {
                "n_features": len(all_sup_ids),
                "score": all_sup_score,
                "log_p_drop": (baseline_lp - all_sup_score["log_p_target"]) if all_sup_score else None,
            },
        }
        results.append(row)

        log.info(
            "    n_active=%d  baseline_lp=%+.2f  sup_drop=%+.2f  bottom_drop=%+.2f  rand_drop=%+.2f±%.2f",
            len(effects), baseline_lp,
            row["supporting_top10"]["log_p_drop"] or 0.0,
            row["bottom10_by_attr"]["log_p_drop"],
            row["random10_active"]["mean_log_p_drop"],
            row["random10_active"]["std_log_p_drop"],
        )

    # Aggregate per condition
    def _agg(rows, key, score_path, drop_key=None):
        n = len(rows)
        if n == 0:
            return {}
        hits = 0
        drops = []
        for r in rows:
            cond = r[key]
            if score_path == "score" and cond.get("score"):
                hits += int(cond["score"]["hit"])
            elif score_path == "hit_rate":
                hits += cond.get("hit_rate", 0.0)
            dk = drop_key or "log_p_drop"
            if dk in cond and cond[dk] is not None:
                drops.append(cond[dk])
            elif "mean_log_p_drop" in cond:
                drops.append(cond["mean_log_p_drop"])
        return {
            "n": n,
            "hit_rate": hits / n if score_path == "score" else hits / n,
            "mean_log_p_drop": sum(drops) / len(drops) if drops else 0.0,
        }

    baseline_hits = sum(r["baseline"]["hit"] for r in results)
    overall = {
        "baseline": {
            "n": len(results),
            "hit_rate": baseline_hits / len(results),
            "mean_log_p_target": sum(r["baseline"]["log_p_target"] for r in results) / len(results),
        },
        "supporting_top10": _agg(results, "supporting_top10", "score"),
        "bottom10_by_attr": _agg(results, "bottom10_by_attr", "score"),
        "random10_active": _agg(results, "random10_active", "hit_rate"),
        "all_supporting": _agg(results, "all_supporting", "score"),
    }

    per_category = {}
    cats = sorted({r["category"] for r in results})
    for cat in cats:
        rows = [r for r in results if r["category"] == cat]
        per_category[cat] = {
            "n": len(rows),
            "baseline": {
                "hit_rate": sum(r["baseline"]["hit"] for r in rows) / len(rows),
            },
            "supporting_top10": _agg(rows, "supporting_top10", "score"),
            "bottom10_by_attr": _agg(rows, "bottom10_by_attr", "score"),
            "random10_active": _agg(rows, "random10_active", "hit_rate"),
        }

    out = {
        "model": spec["nickname"],
        "prompts_file": str(prompts_path.relative_to(PATHS.root)),
        "top_k": args.top_k,
        "n_random_seeds": args.n_random_seeds,
        "results": results,
        "summary": {"overall": overall, "per_category": per_category},
    }
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PATHS.root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", out_path)
    log.info("=== Overall ===")
    log.info("  baseline           hit=%.2f  mean_lp_target=%+.2f",
             overall["baseline"]["hit_rate"], overall["baseline"]["mean_log_p_target"])
    log.info("  supporting_top10   hit=%.2f  Δlog P=%+.2f",
             overall["supporting_top10"]["hit_rate"], overall["supporting_top10"]["mean_log_p_drop"])
    log.info("  bottom10_by_attr   hit=%.2f  Δlog P=%+.2f",
             overall["bottom10_by_attr"]["hit_rate"], overall["bottom10_by_attr"]["mean_log_p_drop"])
    log.info("  random10_active    hit=%.2f  Δlog P=%+.2f",
             overall["random10_active"]["hit_rate"], overall["random10_active"]["mean_log_p_drop"])
    log.info("  all_supporting     hit=%.2f  Δlog P=%+.2f",
             overall["all_supporting"]["hit_rate"], overall["all_supporting"]["mean_log_p_drop"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
