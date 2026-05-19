"""Extract behavioral metrics from model continuations and compare distributions.

For each generation, compute four behavioral metrics that should be enriched by a
grammar-suppression apparatus (per the v3 finding):

1. **copula_per_100tok** — count of forms of 'to be' per 100 tokens. Direct expression
   of the v3 opposers (Gemma feat 15596 'forms of to-be' + 10142 'word is').
2. **hedge_per_100tok** — modals + epistemic adverbs per 100 tokens. Words like 'may',
   'might', 'perhaps', 'generally', 'typically' that the grammar layer would license.
3. **generic_np_per_100tok** — count of 'a/the/an + abstract noun' patterns per 100 tokens.
   The exact pattern the grammar layer pushes the model toward when answering 'X is _'.
4. **copula_opener_fraction** — fraction of sentences starting with 'This is', 'There is',
   'It is', 'These are', 'There are', 'It was'. Copula-led sentence openers.

Per-model summary + Welch's t-test comparing Gemma vs GPT-2 across all generations.

Usage:
    uv run python scripts/analyze_behavior.py \
        --inputs reports/generations_gemma.json reports/generations_gpt2.json \
        --output reports/behavior_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

from neograph.config import PATHS
from neograph.util import get_logger

log = get_logger("neograph.behavior")


COPULA_FORMS = {
    "is", "are", "was", "were", "be", "been", "being",
    "isn't", "aren't", "wasn't", "weren't",
    "'s", "'re",  # contracted forms
}

MODALS = {
    "can", "could", "may", "might", "must", "shall", "should",
    "will", "would", "ought",
}

EPISTEMIC_ADVERBS = {
    "perhaps", "possibly", "probably", "likely", "unlikely",
    "generally", "typically", "usually", "often", "sometimes",
    "rarely", "seldom", "certainly", "definitely", "presumably",
    "supposedly", "apparently", "evidently", "essentially",
    "fundamentally", "primarily", "mostly", "largely",
}

# Abstract nouns that commonly appear in generic "a X" / "the X" templates.
# Drawn from frequency lists of nouns that frequently follow copulas in generic completions.
ABSTRACT_NOUNS = {
    "thing", "things", "way", "ways", "place", "places", "time", "times",
    "person", "people", "kind", "kinds", "type", "types", "sort", "sorts",
    "area", "areas", "point", "points", "idea", "ideas", "matter", "matters",
    "case", "cases", "fact", "facts", "form", "forms", "part", "parts",
    "issue", "issues", "concept", "concepts", "process", "processes",
    "system", "systems", "world", "moment", "moments", "situation", "situations",
    "experience", "experiences", "topic", "topics",
}

OPENER_PATTERNS = [
    re.compile(r"^\s*(this|there|it|these|those)\s+(is|are|was|were)\b", re.I),
]

WORD_RE = re.compile(r"\b[\w']+\b")
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def tokenize_words(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


def split_sentences(text: str) -> list[str]:
    parts = SENT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def count_generic_np(words: list[str]) -> int:
    n = 0
    for i, w in enumerate(words[:-1]):
        if w in ("a", "an", "the"):
            nxt = words[i + 1]
            if nxt in ABSTRACT_NOUNS:
                n += 1
    return n


def metrics_for_text(text: str) -> dict:
    if not text.strip():
        return {"n_tokens": 0, "n_sentences": 0,
                "copula": 0, "hedge": 0, "generic_np": 0, "copula_openers": 0,
                "copula_per_100tok": 0.0, "hedge_per_100tok": 0.0,
                "generic_np_per_100tok": 0.0, "copula_opener_fraction": 0.0}

    words = tokenize_words(text)
    sentences = split_sentences(text)
    n_tokens = len(words)
    n_sentences = max(len(sentences), 1)

    copula = sum(1 for w in words if w in COPULA_FORMS)
    hedge = sum(1 for w in words if w in MODALS or w in EPISTEMIC_ADVERBS)
    generic_np = count_generic_np(words)
    copula_openers = sum(1 for s in sentences if any(p.search(s) for p in OPENER_PATTERNS))

    denom = max(n_tokens, 1)
    return {
        "n_tokens": n_tokens, "n_sentences": n_sentences,
        "copula": copula, "hedge": hedge,
        "generic_np": generic_np, "copula_openers": copula_openers,
        "copula_per_100tok": 100 * copula / denom,
        "hedge_per_100tok": 100 * hedge / denom,
        "generic_np_per_100tok": 100 * generic_np / denom,
        "copula_opener_fraction": copula_openers / n_sentences,
    }


def welch_t(a: list[float], b: list[float]) -> tuple[float, float]:
    """Returns (t-stat, two-sided p-value) for Welch's t-test. p-value via stdlib only."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan"), float("nan")
    ma = sum(a) / na
    mb = sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    se = math.sqrt(va / na + vb / nb) if (va > 0 or vb > 0) else float("nan")
    if not math.isfinite(se) or se == 0:
        return float("nan"), float("nan")
    t = (ma - mb) / se
    # Welch-Satterthwaite df
    num = (va / na + vb / nb) ** 2
    den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    df = num / den if den > 0 else float("nan")
    # Approximate two-sided p-value via normal approximation for moderate df
    # Good enough for n=75; df is large.
    z = abs(t)
    p = 2 * 0.5 * math.erfc(z / math.sqrt(2))
    return t, p


METRIC_KEYS = ["copula_per_100tok", "hedge_per_100tok", "generic_np_per_100tok", "copula_opener_fraction"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Paths to generations_*.json files (one per model).")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    by_model: dict[str, list[dict]] = {}
    for path in args.inputs:
        p = Path(path)
        if not p.is_absolute():
            p = PATHS.root / p
        data = json.loads(p.read_text())
        model = data["model"]
        rows = []
        for r in data["results"]:
            for g in r["generations"]:
                m = metrics_for_text(g["continuation"])
                rows.append({
                    "model": model,
                    "prompt_id": r["id"],
                    "category": r["category"],
                    "seed": g["seed"],
                    **m,
                })
        by_model[model] = rows
        log.info("Loaded %d generations from %s (model=%s)", len(rows), p.name, model)

    # Aggregate per model + per category
    summary: dict = {"per_model": {}, "per_model_per_category": {}, "tests": {}}
    for model, rows in by_model.items():
        bucket = {"n": len(rows)}
        for k in METRIC_KEYS:
            vals = [r[k] for r in rows]
            mean = sum(vals) / len(vals) if vals else 0
            var = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)
            bucket[k] = {"mean": mean, "std": math.sqrt(var), "n": len(vals)}
        summary["per_model"][model] = bucket

        cat_bucket = {}
        cats = sorted({r["category"] for r in rows})
        for cat in cats:
            crows = [r for r in rows if r["category"] == cat]
            cb = {"n": len(crows)}
            for k in METRIC_KEYS:
                vals = [r[k] for r in crows]
                cb[k] = sum(vals) / len(vals) if vals else 0
            cat_bucket[cat] = cb
        summary["per_model_per_category"][model] = cat_bucket

    # Pairwise t-tests if we have at least two models
    models = list(by_model.keys())
    if len(models) >= 2:
        # Compare each pair; the canonical comparison is models[0] vs models[1].
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a_rows = by_model[models[i]]
                b_rows = by_model[models[j]]
                test = {}
                for k in METRIC_KEYS:
                    a = [r[k] for r in a_rows]
                    b = [r[k] for r in b_rows]
                    t, p = welch_t(a, b)
                    test[k] = {"t": t, "p": p,
                               f"{models[i]}_mean": sum(a) / len(a) if a else 0,
                               f"{models[j]}_mean": sum(b) / len(b) if b else 0}
                summary["tests"][f"{models[i]}_vs_{models[j]}"] = test

    out = {
        "models": models,
        "summary": summary,
        "raw": {m: by_model[m] for m in models},
    }
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PATHS.root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", out_path)

    log.info("=== Summary ===")
    for model, b in summary["per_model"].items():
        log.info("  %s  (n=%d)", model, b["n"])
        for k in METRIC_KEYS:
            log.info("    %-26s mean=%6.3f  std=%6.3f", k, b[k]["mean"], b[k]["std"])
    for pair, test in summary["tests"].items():
        log.info("  Test: %s", pair)
        for k, t in test.items():
            log.info("    %-26s t=%+6.2f  p=%.4f", k, t["t"], t["p"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
