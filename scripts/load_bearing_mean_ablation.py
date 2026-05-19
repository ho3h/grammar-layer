"""Mean-ablation replication of the supporting-top10 collapse.

Tests the "zero-ablation is OOD" methodological objection. For each prompt, identify the
same top-10 supporting features as the zero-ablation analysis, but replace their activation
at the last position with the feature's **mean** activation (computed over a held-out
corpus) instead of with zero. If the model still collapses, the load-bearing effect is
method-robust; if it doesn't, the original effect was partly an artifact of zero being
out-of-distribution.

Corpus for mean estimation: by default, the same prompts file, using activations at all
non-last positions across all prompts. This is a slight contamination (same prompts being
used both for stats and for the test) but the position split eliminates direct leakage:
means come from positions 0..N-2, ablation happens at position N-1.

Usage:
    uv run python scripts/load_bearing_mean_ablation.py \
        --model gemma --prompts-file data/prompts_50.json \
        --topk-source reports/load_bearing_pos10_gemma_50.json \
        --output reports/load_bearing_mean_ablation_gemma_50.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from neograph.config import PATHS
from neograph.util import get_logger

sys.path.insert(0, str(Path(__file__).parent))
from load_bearing_topk import MODEL_SPECS, _score  # noqa: E402

log = get_logger("neograph.mean_ablation")


def compute_corpus_means(model, sae, spec, prompts: list[dict]) -> torch.Tensor:
    """Per-feature mean activation across all (prompt, non-last-position) pairs."""
    log.info("Computing per-feature corpus means from %d prompts (non-last positions only)...", len(prompts))
    n_features = sae.cfg.d_sae
    sums = torch.zeros(n_features, dtype=torch.float64)
    counts = 0
    for p in prompts:
        tokens = model.to_tokens(p["prompt"], prepend_bos=True)
        with torch.no_grad():
            _logits, cache = model.run_with_cache_with_saes(tokens, saes=[sae])
        feat_key = next(k for k in cache.keys() if "sae" in k and "acts_post" in k)
        # Shape: [1, T, n_features]. Exclude last position (the prediction site).
        feat_acts = cache[feat_key][0, :-1, :].float().cpu().to(torch.float64)
        sums += feat_acts.sum(dim=0)
        counts += feat_acts.shape[0]
    means = (sums / counts).to(torch.float32)
    log.info("  total positions=%d, mean activation density (>1e-3): %.4f",
             counts, (means > 1e-3).float().mean().item())
    return means


def mean_ablate(model, sae, spec, prompt: str, target_id: int,
                feature_indices: list[int], means: torch.Tensor) -> dict:
    """Replace feature activations at the last position with their corpus mean."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    hook_name = f"{spec['hook_name']}.hook_sae_acts_post"
    fidxs = torch.tensor(feature_indices, dtype=torch.long)
    fmeans = means[fidxs].to(tokens.device)

    def ablate(act, _idxs=fidxs.to(tokens.device), _vals=fmeans, **kwargs):
        act = act.clone()
        act[..., -1, _idxs] = _vals
        return act

    with torch.no_grad():
        logits = model.run_with_hooks_with_saes(tokens, saes=[sae], fwd_hooks=[(hook_name, ablate)])
    return _score(logits[0, -1, :], target_id)


def zero_ablate(model, sae, spec, prompt: str, target_id: int, feature_indices: list[int]) -> dict:
    """Re-run zero-ablation for direct comparison in the same script."""
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


def baseline(model, sae, prompt: str, target_id: int) -> dict:
    tokens = model.to_tokens(prompt, prepend_bos=True)
    with torch.no_grad():
        logits = model.run_with_hooks_with_saes(tokens, saes=[sae], fwd_hooks=[])
    return _score(logits[0, -1, :], target_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_SPECS.keys()), default="gemma")
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--topk-source", required=True,
                        help="Path to existing load_bearing JSON to reuse the topk_supporting feature lists.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    spec = MODEL_SPECS[args.model]

    prompts_path = Path(args.prompts_file)
    if not prompts_path.is_absolute():
        prompts_path = PATHS.root / prompts_path
    prompts = json.loads(prompts_path.read_text())
    log.info("Loaded %d prompts from %s", len(prompts), prompts_path)

    topk_path = Path(args.topk_source)
    if not topk_path.is_absolute():
        topk_path = PATHS.root / topk_path
    topk_data = json.loads(topk_path.read_text())
    topk_by_id = {r["id"]: r for r in topk_data["results"]}
    log.info("Loaded %d topk results from %s", len(topk_by_id), topk_path)

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

    means = compute_corpus_means(model, sae, spec, prompts)

    results: list[dict] = []
    for i, p in enumerate(prompts):
        if p["id"] not in topk_by_id:
            log.warning("No topk for %s, skipping", p["id"])
            continue
        target_ids = model.tokenizer.encode(p["target"], add_special_tokens=False)
        if not target_ids:
            continue
        tid = target_ids[0]
        sup_ids = [e["feature_index"] for e in topk_by_id[p["id"]]["topk_supporting"][:10]]
        if not sup_ids:
            continue

        b = baseline(model, sae, p["prompt"], tid)
        z = zero_ablate(model, sae, spec, p["prompt"], tid, sup_ids)
        m = mean_ablate(model, sae, spec, p["prompt"], tid, sup_ids, means)

        row = {
            "id": p["id"], "category": p["category"], "prompt": p["prompt"], "target": p["target"],
            "supporting_top10_ids": sup_ids,
            "baseline":  {**b, "argmax_token_str": model.tokenizer.decode([b["argmax_token_id"]])},
            "zero_ablation": {**z, "argmax_token_str": model.tokenizer.decode([z["argmax_token_id"]]),
                              "log_p_drop": b["log_p_target"] - z["log_p_target"]},
            "mean_ablation": {**m, "argmax_token_str": model.tokenizer.decode([m["argmax_token_id"]]),
                              "log_p_drop": b["log_p_target"] - m["log_p_target"]},
        }
        results.append(row)
        log.info("[%d/%d] %s  base=%+.2f  zero=%+.2f (Δ%+.2f)  mean=%+.2f (Δ%+.2f)",
                 i + 1, len(prompts), p["id"], b["log_p_target"],
                 z["log_p_target"], row["zero_ablation"]["log_p_drop"],
                 m["log_p_target"], row["mean_ablation"]["log_p_drop"])

    # Aggregate
    baseline_hits = sum(r["baseline"]["hit"] for r in results)
    zero_hits = sum(r["zero_ablation"]["hit"] for r in results)
    mean_hits = sum(r["mean_ablation"]["hit"] for r in results)
    n = len(results)
    summary = {
        "n": n,
        "baseline_hit_rate": baseline_hits / n,
        "zero_ablation": {
            "hit_rate": zero_hits / n,
            "mean_log_p_drop": sum(r["zero_ablation"]["log_p_drop"] for r in results) / n,
        },
        "mean_ablation": {
            "hit_rate": mean_hits / n,
            "mean_log_p_drop": sum(r["mean_ablation"]["log_p_drop"] for r in results) / n,
        },
    }

    out = {
        "model": spec["nickname"],
        "prompts_file": str(prompts_path.relative_to(PATHS.root)),
        "topk_source": str(topk_path.relative_to(PATHS.root)),
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
    log.info("  baseline       hit=%.2f", summary["baseline_hit_rate"])
    log.info("  zero-ablation  hit=%.2f  Δlog P=%+.2f", summary["zero_ablation"]["hit_rate"], summary["zero_ablation"]["mean_log_p_drop"])
    log.info("  mean-ablation  hit=%.2f  Δlog P=%+.2f", summary["mean_ablation"]["hit_rate"], summary["mean_ablation"]["mean_log_p_drop"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
