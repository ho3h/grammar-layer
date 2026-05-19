"""Pivot reports/load_bearing_*.json into a single web/data/summary.json that the
Three.js explainer site consumes.

The script picks the best-available result file per model (prefers 50-prompt over
12-prompt), classifies each load-bearing feature as grammar-flavored or
content-thematic using a strict label keyword list, and writes the summary blob.

Re-run whenever the underlying load-bearing results change:

    uv run python scripts/export_web_data.py
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sys
from pathlib import Path

from neograph.config import PATHS


# Strict grammar-feature keyword set. The rule of thumb: a feature counts as
# "grammar-flavored" only if its label describes a syntactic or function-word
# property, not a topical category. Topical labels containing grammar-like words
# (e.g. "interrogative questions about historical events") are CONTENT.
#
# Each alternative is self-anchored (uses \b internally where appropriate) so we
# don't wrap the whole group in \b…\b — quotes around function words break that.
_GRAMMAR_PATTERNS = re.compile(
    "|".join([
        # Predicate / verb-form
        r"\bforms of the verb\b",
        r"\bforms of (to ?be|to ?have|to ?do)\b",
        r"the verb [\"']to ?be[\"']",
        r"the verb [\"']to ?have[\"']",
        r"\bpast (and|or) present tense\b",
        r"\b(present|past|future) tense\b",
        r"\btenses?\b",
        r"\b(copula|auxiliar(y|ies)|modal verbs?|particles?)\b",
        r"\b(conjugat\w*|inflect\w*|declension)\b",
        r"\b(stative verb|verb form|verbs? in various contexts)\b",
        # Function words and "the word X"
        r"the word [\"'](is|are|was|were|be|been|being|am|to|the|a|an|of|in|with|by|at|on|that|this|those|these|it|its|and|or|but)[\"']",
        r"instances of (the word )?[\"'](is|are|was|were|be|the|a|an)[\"']",
        r"\b(definite|indefinite) article\b",
        r"\barticles in\b",
        # Predicate-of-being / existential
        r"\bstatements? of (existence|presence|being)\b",
        r"\b(existential|stative) statements?\b",
        # Pronouns / function categories
        r"\b(personal|relative|interrogative) pronouns?\b",
        r"\b(prepositions?|conjunctions?|determiners?)\b",
        # Punctuation / structural
        r"\bpunctuation marks?\b",
        r"\bsentence-final punctuation\b",
        r"\bword boundar\w+\b",
        r"\bsentence boundar\w+\b",
        r"\bgrammatical (function|role|category|structure)\b",
        r"\bsyntactic (structure|role|category|markers?)\b",
    ]),
    re.IGNORECASE,
)


MODEL_DISPLAY = {
    "pythia_70m": {"display_name": "Pythia 70M",  "color": "#7d4cbf"},
    "gpt2":       {"display_name": "GPT-2 small", "color": "#2471a3"},
    "qwen3_1_7b": {"display_name": "Qwen 3 1.7B", "color": "#c73e8a"},
    "gemma_1_2b": {"display_name": "Gemma 1 2B",  "color": "#e09142"},
    "gemma":      {"display_name": "Gemma 2 2B",  "color": "#d35400"},
    "mistral_7b": {"display_name": "Mistral 7B",  "color": "#1b8a64"},
    "gemma_9b":   {"display_name": "Gemma 2 9B",  "color": "#a04400"},
    "gemma_27b":  {"display_name": "Gemma 2 27B", "color": "#702c00"},
}


def is_grammar_label(text: str | None) -> bool:
    if not text:
        return False
    return bool(_GRAMMAR_PATTERNS.search(text))


def model_has_labels(model: str) -> bool:
    return load_labels(model) != {}


def load_labels(model: str) -> dict[int, str]:
    if model == "gemma":
        name = "labels_cache.json"
    elif model == "gpt2":
        name = "labels_cache_gpt2.json"
    else:
        # Other models don't have a curated label cache yet. The Three.js
        # tooltip will show "(no label)" until one is populated, and the
        # cross-model grammar % will be reported as "n/a" (not 0%) for them.
        name = f"labels_cache_{model}.json"
    p = PATHS.data / name
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    out: dict[int, str] = {}
    for k, v in raw.items():
        try:
            idx = int(k)
        except ValueError:
            continue
        if isinstance(v, dict) and "text" in v:
            out[idx] = v["text"]
        elif isinstance(v, str):
            out[idx] = v
    return out


def pick_results_file(model: str) -> Path | None:
    """Prefer the 50-prompt supporting result; fall back to 12-prompt."""
    candidates = [
        PATHS.reports / f"load_bearing_pos10_{model}_50.json",
        PATHS.reports / f"load_bearing_pos10_{model}_12.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _make_edges(results: list[dict], feat_key: str, labels: dict[int, str]) -> list[dict]:
    """Pivot one (supporting|opposing|by_abs) feature set per prompt into edge rows."""
    out: list[dict] = []
    for r in results:
        feats = r.get(feat_key) or []
        for f in feats:
            text = labels.get(f["feature_index"]) or f.get("label") or ""
            out.append({
                "prompt_id": r["id"],
                "prompt": r["prompt"],
                "prompt_category": r["category"],
                "target": r["target"],
                "feature_index": f["feature_index"],
                "feature_label": text,
                "feature_is_grammar": is_grammar_label(text),
                "single_log_p_drop": f["single_log_p_drop"],
            })
    return out


def summarise_model(model: str) -> dict | None:
    path = pick_results_file(model)
    if path is None:
        return None
    data = json.loads(path.read_text())
    labels = load_labels(model)
    results = data["results"]

    # Supporting & opposing edge sets. Older result files only have `topk_features`
    # (= the primary --sign result), so fall back to that for whichever sign matches
    # the run's `--sign` flag.
    sign = data.get("sign", "positive")
    supporting_key = "topk_supporting" if any("topk_supporting" in r for r in results) else \
                     ("topk_features" if sign in ("positive", "abs") else None)
    opposing_key = "topk_opposing" if any("topk_opposing" in r for r in results) else \
                   ("topk_features" if sign == "negative" else None)

    edges = _make_edges(results, supporting_key, labels) if supporting_key else []
    opposing_edges = _make_edges(results, opposing_key, labels) if opposing_key else []

    # Hit-rate and Δlog-P aggregates
    per_category: dict[str, dict] = {}
    cats = sorted({r["category"] for r in results})
    for cat in cats:
        rows = [r for r in results if r["category"] == cat]
        per_category[cat] = {
            "n": len(rows),
            "baseline_hit_rate": sum(r["baseline"]["hit"] for r in rows) / len(rows),
            "ablated_hit_rate":  sum(r["joint_topk_ablated"]["hit"] for r in rows) / len(rows),
            "mean_log_p_drop":   sum(r["log_p_drop_vs_baseline"] for r in rows) / len(rows),
        }

    baseline_hits = sum(r["baseline"]["hit"] for r in results)
    ablated_hits  = sum(r["joint_topk_ablated"]["hit"] for r in results)
    mean_drop = sum(r["log_p_drop_vs_baseline"] for r in results) / len(results)

    # Feature character composition — separate accounting for supporting vs opposing
    def _split(edge_rows: list[dict], rows: list[dict]) -> dict:
        prompt_ids = {r["id"] for r in rows}
        local = [e for e in edge_rows if e["prompt_id"] in prompt_ids]
        total = len(local)
        gram = sum(1 for e in local if e["feature_is_grammar"])
        return {
            "total": total,
            "grammar": gram,
            "content": total - gram,
            "grammar_pct": 100 * gram / total if total else 0,
            "content_pct": 100 * (total - gram) / total if total else 0,
        }

    character = _split(edges, results)
    character["per_category"] = {cat: _split(edges, [r for r in results if r["category"] == cat]) for cat in cats}

    opposing_character = _split(opposing_edges, results)
    opposing_character["per_category"] = {
        cat: _split(opposing_edges, [r for r in results if r["category"] == cat]) for cat in cats
    }

    display = MODEL_DISPLAY.get(model, {"display_name": model, "color": "#888"})
    return {
        "display_name": display["display_name"],
        "color": display["color"],
        "results_source": str(path.relative_to(PATHS.root)),
        "n_prompts": len(results),
        "has_labels": bool(labels),
        "baseline_hit_rate": baseline_hits / len(results),
        "ablated_hit_rate": ablated_hits / len(results),
        "mean_log_p_drop": mean_drop,
        "per_category": per_category,
        "feature_character": character,
        "opposing_feature_character": opposing_character,
        "backbone_edges": edges,
        "opposing_edges": opposing_edges,
    }


def main() -> int:
    out = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "models": {},
        "categories": [],
    }
    all_cats: set[str] = set()
    for model in ("pythia_70m", "gpt2", "qwen3_1_7b", "gemma_1_2b", "gemma", "mistral_7b", "gemma_9b", "gemma_27b"):
        m = summarise_model(model)
        if m is None:
            print(f"  skip {model}: no results file found", file=sys.stderr)
            continue
        out["models"][model] = m
        all_cats.update(m["per_category"].keys())
        print(f"  {model}: source={m['results_source']}  n={m['n_prompts']}  "
              f"baseline={m['baseline_hit_rate']:.2f}  ablated={m['ablated_hit_rate']:.2f}  "
              f"Δlog P={m['mean_log_p_drop']:+.2f}  "
              f"grammar={m['feature_character']['grammar_pct']:.1f}%",
              file=sys.stderr)

    out["categories"] = sorted(all_cats)

    target = PATHS.root / "web" / "data" / "summary.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2))
    print(f"Wrote {target}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
