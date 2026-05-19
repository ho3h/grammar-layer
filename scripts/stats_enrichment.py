"""Bootstrap confidence intervals + permutation tests on the grammar-enrichment claims.

Closes the "you ran one keyword classifier on 52 prompts and got 2.9× — what's the
sampling noise?" objection. Computes:

1. Bootstrap 95% CI on supporting → opposing grammar share (overall + per category)
   by resampling prompts with replacement.
2. Permutation test for fingerprint significance: how often would two specific features
   appear in top-5 opposing across 6/6 capital prompts by chance, given the random
   feature pool?
3. Per-feature consistency report — for each Gemma feature, on what fraction of
   capital prompts does it appear in top-5 opposing?

Reads existing load_bearing_pos10_{gemma,gpt2}_50.json + labels_cache files.
No compute, no model loading.

Usage:
    uv run python scripts/stats_enrichment.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GRAMMAR_PAT = re.compile(
    r"\b(is|to[- ]?be|copula|verb|tense|past tense|present tense|conjunction|"
    r"article|preposition|punctuation|grammar|function word|determiner|"
    r"pronoun|auxiliary)\b",
    re.I,
)


def _load_labels(path: str) -> dict[int, str]:
    raw = json.loads((ROOT / path).read_text())
    return {int(k): v.get("text", "") for k, v in raw.items() if isinstance(v, dict)}


def _is_grammar(label: str | None) -> bool:
    return bool(label and GRAMMAR_PAT.search(label))


def bootstrap_enrichment(results: list[dict], labels: dict[int, str],
                          n_boot: int = 5000, top_k: int = 10, seed: int = 0):
    """For each bootstrap resample of prompts, compute supporting %grammar and opposing %grammar.
    Return mean, 95% CI for sup, opp, and the ratio."""
    rng = random.Random(seed)
    n = len(results)
    sup_pcts = []
    opp_pcts = []
    ratios = []
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        sample = [results[i] for i in idxs]
        sup_n = sup_gram = opp_n = opp_gram = 0
        for r in sample:
            for e in r["topk_supporting"][:top_k]:
                sup_n += 1
                if _is_grammar(labels.get(e["feature_index"])):
                    sup_gram += 1
            for e in r["topk_opposing"][:top_k]:
                opp_n += 1
                if _is_grammar(labels.get(e["feature_index"])):
                    opp_gram += 1
        sup_pct = 100 * sup_gram / max(sup_n, 1)
        opp_pct = 100 * opp_gram / max(opp_n, 1)
        sup_pcts.append(sup_pct)
        opp_pcts.append(opp_pct)
        ratios.append(opp_pct / max(sup_pct, 0.01))

    def _ci(xs):
        s = sorted(xs)
        return s[int(0.025 * len(s))], s[int(0.975 * len(s))]

    return {
        "sup_pct": {"mean": sum(sup_pcts) / len(sup_pcts), "ci95": _ci(sup_pcts)},
        "opp_pct": {"mean": sum(opp_pcts) / len(opp_pcts), "ci95": _ci(opp_pcts)},
        "enrichment_ratio": {"mean": sum(ratios) / len(ratios), "ci95": _ci(ratios)},
        "n_boot": n_boot,
        "n_prompts": n,
    }


def feature_consistency(results: list[dict], category: str | None = None, top_k: int = 5):
    """For each feature, return the fraction of prompts in which it appears in the top-K opposing."""
    sel = [r for r in results if (category is None or r["category"] == category)]
    counts: dict[int, int] = defaultdict(int)
    for r in sel:
        for e in r["topk_opposing"][:top_k]:
            counts[e["feature_index"]] += 1
    n = len(sel)
    return {f: c / n for f, c in counts.items()}, n


def fingerprint_permutation(results: list[dict], target_features: list[int],
                             category: str, top_k: int = 5,
                             n_perm: int = 10000, seed: int = 0) -> dict:
    """Permutation test: how often would the given target features appear in top-K opposing
    of every prompt in `category`, if each prompt's opposing set were a uniform random
    sample of size top_k from the union of all opposing features observed across prompts?

    P-value: fraction of permutations where the target features appear in top-K on >=
    same number of prompts as observed.
    """
    sel = [r for r in results if r["category"] == category]
    n_sel = len(sel)

    # Pool of features that appeared in any prompt's top-K opposing across the category
    pool: list[int] = []
    for r in sel:
        for e in r["topk_opposing"][:top_k]:
            pool.append(e["feature_index"])
    pool_uniq = list(set(pool))

    # Observed: count prompts where ALL target features appear in their top-K opposing
    observed_all = 0
    for r in sel:
        feats = {e["feature_index"] for e in r["topk_opposing"][:top_k]}
        if all(t in feats for t in target_features):
            observed_all += 1
    observed_each = []
    for t in target_features:
        c = sum(1 for r in sel if t in {e["feature_index"] for e in r["topk_opposing"][:top_k]})
        observed_each.append(c)

    # Permutation null: for each prompt, sample top_k features uniformly without replacement
    # from pool_uniq. Count how often all target features appear on >= observed_all prompts.
    rng = random.Random(seed)
    null_all = 0
    null_each = [0] * len(target_features)
    for _ in range(n_perm):
        sim_all = 0
        sim_each = [0] * len(target_features)
        for _r in range(n_sel):
            samp = set(rng.sample(pool_uniq, min(top_k, len(pool_uniq))))
            if all(t in samp for t in target_features):
                sim_all += 1
            for i, t in enumerate(target_features):
                if t in samp:
                    sim_each[i] += 1
        if sim_all >= observed_all:
            null_all += 1
        for i in range(len(target_features)):
            if sim_each[i] >= observed_each[i]:
                null_each[i] += 1

    return {
        "category": category,
        "target_features": target_features,
        "n_prompts_in_category": n_sel,
        "top_k": top_k,
        "pool_size": len(pool_uniq),
        "observed_all_features_present": observed_all,
        "observed_per_feature": dict(zip(target_features, observed_each)),
        "n_perm": n_perm,
        "p_all_features": null_all / n_perm,
        "p_per_feature": {t: c / n_perm for t, c in zip(target_features, null_each)},
    }


def main() -> int:
    g = json.loads((ROOT / "reports/load_bearing_pos10_gemma_50.json").read_text())
    gp2 = json.loads((ROOT / "reports/load_bearing_pos10_gpt2_50.json").read_text())
    g_labels = _load_labels("data/labels_cache.json")
    gp2_labels = _load_labels("data/labels_cache_gpt2.json")

    print("=== Bootstrap enrichment (5000 resamples) ===\n")
    for name, results, labels in [
        ("Gemma 2 2B  (all 52)",      g["results"],   g_labels),
        ("GPT-2 small (all 52)",      gp2["results"], gp2_labels),
        ("Gemma 2 2B  (6 capitals)",  [r for r in g["results"]   if r["category"] == "capital"], g_labels),
        ("GPT-2 small (6 capitals)",  [r for r in gp2["results"] if r["category"] == "capital"], gp2_labels),
    ]:
        if not results:
            continue
        b = bootstrap_enrichment(results, labels, n_boot=5000, top_k=10)
        print(f"{name}:")
        print(f"  supporting %grammar:  mean {b['sup_pct']['mean']:5.2f}%  95% CI [{b['sup_pct']['ci95'][0]:.2f}, {b['sup_pct']['ci95'][1]:.2f}]")
        print(f"  opposing %grammar:    mean {b['opp_pct']['mean']:5.2f}%  95% CI [{b['opp_pct']['ci95'][0]:.2f}, {b['opp_pct']['ci95'][1]:.2f}]")
        print(f"  enrichment ratio:     mean {b['enrichment_ratio']['mean']:5.2f}×  95% CI [{b['enrichment_ratio']['ci95'][0]:.2f}, {b['enrichment_ratio']['ci95'][1]:.2f}]")
        print()

    print("=== Per-feature consistency in Gemma capital opposers (top-5) ===\n")
    consistency, n = feature_consistency(g["results"], category="capital", top_k=5)
    top = sorted(consistency.items(), key=lambda kv: -kv[1])[:10]
    for fidx, frac in top:
        lab = g_labels.get(fidx, "<no label>")[:80]
        gram = "[GRAMMAR]" if _is_grammar(g_labels.get(fidx)) else "         "
        print(f"  f{fidx:>5}  in {int(round(frac*n))}/{n} prompts  {gram}  {lab}")
    print()

    print("=== Fingerprint permutation test (features 15596 + 10142 across Gemma capitals) ===\n")
    perm = fingerprint_permutation(g["results"], target_features=[15596, 10142],
                                    category="capital", top_k=5, n_perm=10000)
    print(json.dumps({
        "p_both_features_present_on_all_prompts": perm["p_all_features"],
        "p_15596_alone_on_all_prompts":           perm["p_per_feature"][15596],
        "p_10142_alone_on_all_prompts":           perm["p_per_feature"][10142],
        "observed_all_features_present":           perm["observed_all_features_present"],
        "observed_15596":                          perm["observed_per_feature"][15596],
        "observed_10142":                          perm["observed_per_feature"][10142],
        "pool_size":                                perm["pool_size"],
        "n_prompts":                                perm["n_prompts_in_category"],
        "top_k":                                    perm["top_k"],
        "n_perm":                                   perm["n_perm"],
    }, indent=2))

    out = {
        "bootstrap_enrichment": {
            "gemma_all":       bootstrap_enrichment(g["results"], g_labels),
            "gpt2_all":        bootstrap_enrichment(gp2["results"], gp2_labels),
            "gemma_capitals":  bootstrap_enrichment([r for r in g["results"] if r["category"] == "capital"], g_labels),
            "gpt2_capitals":   bootstrap_enrichment([r for r in gp2["results"] if r["category"] == "capital"], gp2_labels),
        },
        "feature_consistency_gemma_capital_top5": dict(top),
        "fingerprint_permutation_gemma_capital_15596_10142": perm,
    }
    (ROOT / "reports" / "stats_enrichment.json").write_text(json.dumps(out, indent=2, default=lambda o: list(o) if isinstance(o, tuple) else o))
    print("\nWrote reports/stats_enrichment.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
