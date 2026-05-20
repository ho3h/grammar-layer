"""Analyse cross-task generalization: does the Gemma fingerprint (15596, 10142) also
oppose specific completions on currencies / languages / compositions / continents?

For each new task category, count:
- how often f15596 and f10142 appear in top-5 opposing
- how often the broader Gemma grammar-labelled feature set appears in top-5 opposing
- enrichment ratio vs supporting side per category

This decides whether the fingerprint generalizes to the "X is Y" template broadly or is
specific to capital-completions.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GRAMMAR_PAT = re.compile(
    r"\b(is|to[- ]?be|copula|verb|tense|past tense|present tense|conjunction|"
    r"article|preposition|punctuation|grammar|function word|determiner|"
    r"pronoun|auxiliary|are|am|was|were|be|been|being)\b",
    re.I,
)

KNOWN_FINGERPRINT = {15596, 10142}


def _is_grammar(label: str | None) -> bool:
    return bool(label and GRAMMAR_PAT.search(label))


def load_labels(model: str) -> dict[int, str]:
    name_map = {
        "gemma": "data/labels_cache.json",
        "gpt2": "data/labels_cache_gpt2.json",
        "pythia_70m": "data/labels_cache_pythia_70m.json",
        "gemma_1_2b": "data/labels_cache_gemma_1_2b.json",
    }
    path = ROOT / name_map.get(model, "")
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    labels: dict[int, str] = {}
    for k, v in raw.items():
        try:
            fid = int(k)
        except ValueError:
            continue
        if isinstance(v, dict):
            labels[fid] = v.get("text", "")
        elif isinstance(v, str):
            labels[fid] = v
    return labels


def analyse_one(model: str, path: Path) -> dict:
    data = json.loads(path.read_text())
    labels = load_labels(model)

    by_category: dict[str, dict] = {}
    for r in data.get("results", []):
        cat = r["category"]
        bucket = by_category.setdefault(cat, {
            "n_prompts": 0,
            "n_fingerprint_in_opp5": 0,  # Gemma f15596/f10142 specifically
            "n_grammar_in_opp5": 0,
            "n_grammar_in_sup5": 0,
            "fingerprint_per_prompt": [],
            "baseline_hits": 0,
            "joint_ablated_hits": 0,
            "mean_log_p_drop": 0.0,
        })
        bucket["n_prompts"] += 1
        opp5 = r.get("topk_opposing", [])[:5]
        sup5 = r.get("topk_supporting", [])[:5]
        opp5_idxs = {e["feature_index"] for e in opp5}
        sup5_idxs = {e["feature_index"] for e in sup5}

        # Fingerprint check (Gemma-specific)
        if model == "gemma":
            in_fp = opp5_idxs & KNOWN_FINGERPRINT
            bucket["n_fingerprint_in_opp5"] += len(in_fp)
            bucket["fingerprint_per_prompt"].append({
                "prompt": r["id"], "fingerprint_in_opp5": sorted(in_fp),
            })

        # Grammar share on opposing side
        for e in opp5:
            lab = labels.get(e["feature_index"], "")
            if _is_grammar(lab):
                bucket["n_grammar_in_opp5"] += 1
        for e in sup5:
            lab = labels.get(e["feature_index"], "")
            if _is_grammar(lab):
                bucket["n_grammar_in_sup5"] += 1

        bucket["baseline_hits"] += int(r["baseline"].get("hit", False))
        bucket["joint_ablated_hits"] += int(r["joint_topk_ablated"].get("hit", False))
        bucket["mean_log_p_drop"] += r.get("log_p_drop_vs_baseline", 0.0)

    # Normalize
    for cat, b in by_category.items():
        n = b["n_prompts"]
        b["baseline_hit_rate"] = b["baseline_hits"] / n
        b["joint_ablated_hit_rate"] = b["joint_ablated_hits"] / n
        b["mean_log_p_drop"] = b["mean_log_p_drop"] / n
        b["grammar_share_opp_pct"] = b["n_grammar_in_opp5"] / (n * 5) * 100
        b["grammar_share_sup_pct"] = b["n_grammar_in_sup5"] / (n * 5) * 100
        b["enrichment_opp_vs_sup"] = (
            b["grammar_share_opp_pct"] / b["grammar_share_sup_pct"]
            if b["grammar_share_sup_pct"] > 0
            else float("inf") if b["grammar_share_opp_pct"] > 0 else 1.0
        )
        if model == "gemma":
            b["fingerprint_rate"] = b["n_fingerprint_in_opp5"] / (n * 2) * 100  # 2 features per prompt

    return by_category


def main() -> int:
    models_to_analyze = ["gemma", "gpt2", "pythia_70m", "gemma_1_2b"]
    out: dict[str, dict] = {}
    for model in models_to_analyze:
        path = ROOT / "reports" / f"cross_task_{model}.json"
        if not path.exists():
            print(f"  {model}: SKIP (no cross_task file yet)")
            continue
        out[model] = analyse_one(model, path)

    # Pretty print
    print("\n" + "=" * 90)
    print("Cross-task analysis: does the fingerprint generalize beyond capitals?")
    print("=" * 90)
    for model, cats in out.items():
        print(f"\n--- {model} ---")
        print(f"  {'category':<14} {'n':>3} {'baseline':>9} {'ablated':>9} {'Δlog P':>8} "
              f"{'sup%gram':>9} {'opp%gram':>9} {'enrich':>8}", end="")
        if model == "gemma":
            print(f" {'fingerprint%':>13}")
        else:
            print()
        for cat, b in sorted(cats.items()):
            print(
                f"  {cat:<14} {b['n_prompts']:>3} "
                f"{b['baseline_hit_rate']:>9.2f} "
                f"{b['joint_ablated_hit_rate']:>9.2f} "
                f"{b['mean_log_p_drop']:>+8.2f} "
                f"{b['grammar_share_sup_pct']:>9.1f} "
                f"{b['grammar_share_opp_pct']:>9.1f} "
                f"{b['enrichment_opp_vs_sup']:>8.2f}",
                end=""
            )
            if model == "gemma":
                print(f" {b.get('fingerprint_rate', 0):>13.1f}")
            else:
                print()

    out_path = ROOT / "reports" / "cross_task_analysis.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
