"""Run the grammar/content classifier across all 7 models with labels.

Reads each model's load_bearing_pos10_*.json + its label cache. For each prompt and
each rank position, classifies whether the feature is grammar-labelled. Outputs a
cross-model enrichment table.

Usage:
    uv run python scripts/cross_model_grammar_classify.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GRAMMAR_PAT = re.compile(
    r"\b(is|to[- ]?be|copula|verb|tense|past tense|present tense|conjunction|"
    r"article|preposition|punctuation|grammar|function word|determiner|"
    r"pronoun|auxiliary)\b",
    re.I,
)

MODELS = [
    ("Pythia 70M",        "pythia_70m",  "reports/load_bearing_pos10_pythia_70m_50.json",  "data/labels_cache_pythia_70m.json"),
    ("GPT-2 small",       "gpt2",        "reports/load_bearing_pos10_gpt2_50.json",         "data/labels_cache_gpt2.json"),
    ("Gemma 1 2B",        "gemma_1_2b",  "reports/load_bearing_pos10_gemma_1_2b_50.json",  "data/labels_cache_gemma_1_2b.json"),
    ("Gemma 2 2B (16k)",  "gemma_2b",    "reports/load_bearing_pos10_gemma_50.json",       "data/labels_cache.json"),
    ("Gemma 2 2B (65k)",  "gemma_w65k",  "reports/load_bearing_pos10_gemma_w65k_50.json",  "data/labels_cache_gemma_w65k.json"),
    ("Gemma 2 9B",        "gemma_9b",    "reports/load_bearing_pos10_gemma_9b_50.json",    "data/labels_cache_gemma_9b.json"),
]


def _load_labels(path: str) -> dict[int, str]:
    p = ROOT / path
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    out: dict[int, str] = {}
    for k, v in raw.items():
        if isinstance(v, dict) and "text" in v:
            out[int(k)] = v["text"]
        elif isinstance(v, str):
            out[int(k)] = v
    return out


def _is_grammar(label: str | None) -> bool:
    return bool(label and GRAMMAR_PAT.search(label))


def analyse(results: list[dict], labels: dict[int, str], top_k: int = 10) -> dict:
    sup_n = sup_gram = opp_n = opp_gram = 0
    per_cat: dict[str, dict] = defaultdict(lambda: {"sup_n": 0, "sup_gram": 0, "opp_n": 0, "opp_gram": 0})
    feat_in_opp_capitals: dict[int, int] = defaultdict(int)
    n_caps = sum(1 for r in results if r["category"] == "capital")
    n_with_labels = 0
    for r in results:
        for e in r["topk_supporting"][:top_k]:
            sup_n += 1
            per_cat[r["category"]]["sup_n"] += 1
            if e["feature_index"] in labels:
                n_with_labels += 1
            if _is_grammar(labels.get(e["feature_index"])):
                sup_gram += 1
                per_cat[r["category"]]["sup_gram"] += 1
        for e in r["topk_opposing"][:top_k]:
            opp_n += 1
            per_cat[r["category"]]["opp_n"] += 1
            if _is_grammar(labels.get(e["feature_index"])):
                opp_gram += 1
                per_cat[r["category"]]["opp_gram"] += 1
        # Fingerprint check on capitals
        if r["category"] == "capital":
            for e in r["topk_opposing"][:5]:
                feat_in_opp_capitals[e["feature_index"]] += 1

    label_coverage = n_with_labels / max(sup_n, 1)
    sup_pct = 100 * sup_gram / max(sup_n, 1)
    opp_pct = 100 * opp_gram / max(opp_n, 1)
    enrich = opp_pct / max(sup_pct, 0.01)

    per_cat_out = {}
    for cat, v in per_cat.items():
        per_cat_out[cat] = {
            "n": v["sup_n"] // top_k,  # n prompts in category
            "sup_pct_grammar": 100 * v["sup_gram"] / max(v["sup_n"], 1),
            "opp_pct_grammar": 100 * v["opp_gram"] / max(v["opp_n"], 1),
            "enrichment": (100 * v["opp_gram"] / max(v["opp_n"], 1)) / max(100 * v["sup_gram"] / max(v["sup_n"], 1), 0.01),
        }

    # Cross-prompt fingerprint: features appearing in opp5 on >= 5/6 capital prompts
    fingerprint_features = [(f, c) for f, c in feat_in_opp_capitals.items() if c >= max(5, n_caps - 1)]
    fingerprint_features.sort(key=lambda x: -x[1])
    fingerprint_labelled = [
        {"feature_index": f, "n_capital_prompts": c, "label": labels.get(f, "<no label>")[:80],
         "is_grammar": _is_grammar(labels.get(f))}
        for f, c in fingerprint_features
    ]

    return {
        "n_prompts": len(results),
        "n_capital_prompts": n_caps,
        "label_coverage_top_k_sup": label_coverage,
        "sup_pct_grammar": sup_pct,
        "opp_pct_grammar": opp_pct,
        "enrichment_ratio": enrich,
        "per_category": per_cat_out,
        "fingerprint_features": fingerprint_labelled,
    }


def main() -> None:
    summary = {}
    for display, nickname, results_path, labels_path in MODELS:
        rp = ROOT / results_path
        if not rp.exists():
            print(f"[skip] {display}: no results file")
            continue
        results = json.loads(rp.read_text())["results"]
        labels = _load_labels(labels_path)
        if not labels:
            print(f"[skip] {display}: no labels available at {labels_path}")
            continue
        a = analyse(results, labels)
        summary[nickname] = {"display": display, **a}
        print(f"\n{display}  ({len(labels)} labels available, coverage of top-10 sup={a['label_coverage_top_k_sup']:.0%})")
        print(f"  supporting %grammar: {a['sup_pct_grammar']:5.2f}%")
        print(f"  opposing  %grammar:  {a['opp_pct_grammar']:5.2f}%")
        print(f"  enrichment:          {a['enrichment_ratio']:5.2f}×")
        if a["fingerprint_features"]:
            print(f"  fingerprint features (in opp5 on ≥5/{a['n_capital_prompts']} capitals):")
            for fp in a["fingerprint_features"][:5]:
                tag = "[GRAMMAR]" if fp["is_grammar"] else "         "
                print(f"    f{fp['feature_index']:>5}  {fp['n_capital_prompts']}/{a['n_capital_prompts']}  {tag}  {fp['label']}")
        else:
            print(f"  fingerprint features: (none with ≥{max(5, a['n_capital_prompts']-1)}/{a['n_capital_prompts']} capital coverage)")
        print(f"  per-category capital opp%-grammar: {a['per_category'].get('capital', {}).get('opp_pct_grammar', 0):.1f}%")

    (ROOT / "reports" / "cross_model_grammar.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote reports/cross_model_grammar.json")


if __name__ == "__main__":
    main()
