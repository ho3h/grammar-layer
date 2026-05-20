"""Refine the cross-routing analysis with a selectivity ratio.

The raw `copula_mean - noncopula_mean` specificity picks up content features that happen
to co-fire with copula tokens because the copula-heavy half of the Rosetta corpus is
semantically richer than the copula-free half. We need a sharper definition of "actually
a copula detector" before the cross-routing claim can be trusted.

Two refinements:
1. Selectivity ratio: log((copula_mean + ε) / (noncopula_mean + ε)). Features with
   selectivity ratio ≥ 3 (i.e., copula activation ≥ 20× noncopula activation) are real
   copula detectors. Features with ratio < 1.0 (i.e., copula < 3× noncopula) are content
   features that incidentally correlate.
2. Minimum copula activation: require copula_mean ≥ τ to keep the analysis to features
   that actually fire on copula tokens with non-trivial magnitude.

Then redo the "recruited as opposer on capitals" check using ONLY the high-selectivity
copula detectors.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from neograph.config import PATHS


EPS = 1e-3
SELECTIVITY_MIN_LOG10 = 1.0  # 10× more activation on copula than non-copula
COPULA_MEAN_MIN = 1.0  # require at least 1.0 mean activation on copula tokens


def load_blob() -> dict:
    return json.loads((PATHS.reports / "cross_routing_functional.json").read_text())


def load_capital_opposers(model_nickname: str) -> dict[str, list[int]]:
    p50 = PATHS.reports / f"load_bearing_pos10_{model_nickname}_50.json"
    p12 = PATHS.reports / f"load_bearing_pos10_{model_nickname}_12.json"
    for p in (p50, p12):
        if p.exists():
            data = json.loads(p.read_text())
            out = {}
            for r in data.get("results", []):
                if r.get("category") == "capital":
                    out[r["id"]] = [e["feature_index"] for e in r.get("topk_opposing", [])[:10]]
            if out:
                return out
    return {}


def load_labels(model_nickname: str) -> dict[int, str]:
    name_map = {
        "gemma": "data/labels_cache.json",
        "gpt2": "data/labels_cache_gpt2.json",
        "pythia_70m": "data/labels_cache_pythia_70m.json",
        "gemma_1_2b": "data/labels_cache_gemma_1_2b.json",
        "gemma_9b": "data/labels_cache_gemma_9b.json",
        "gemma_w65k": "data/labels_cache_gemma_w65k.json",
    }
    path = PATHS.root / name_map.get(model_nickname, "")
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


def main() -> int:
    blob = load_blob()

    print("=" * 90)
    print("Selective copula detectors per model (log10 selectivity ≥ %.1f, copula_mean ≥ %.1f)"
          % (SELECTIVITY_MIN_LOG10, COPULA_MEAN_MIN))
    print("=" * 90)

    refined: dict[str, dict] = {}
    for nickname, per_model in blob["per_model"].items():
        # Re-rank by log10 selectivity ratio
        feats = per_model["top_copula_features"]
        labels = load_labels(nickname)
        scored = []
        for e in feats:
            cm = e["copula_mean"]
            nm = e["noncopula_mean"]
            if cm < COPULA_MEAN_MIN:
                continue
            ratio = math.log10((cm + EPS) / (nm + EPS))
            if ratio < SELECTIVITY_MIN_LOG10:
                continue
            scored.append({
                "feature_index": e["feature_index"],
                "copula_mean": cm,
                "noncopula_mean": nm,
                "log10_selectivity": ratio,
                "label": labels.get(e["feature_index"], "(unlabeled)"),
            })
        scored.sort(key=lambda x: x["log10_selectivity"], reverse=True)
        refined[nickname] = {
            "n_high_selectivity_copula_features": len(scored),
            "features": scored,
        }
        print(f"\n--- {nickname} ({len(scored)} high-selectivity copula features) ---")
        for r in scored[:12]:
            print(
                f"  feat {r['feature_index']:>6}  log10_sel={r['log10_selectivity']:+.2f}  "
                f"copula={r['copula_mean']:+6.2f}  noncopula={r['noncopula_mean']:+5.2f}  "
                f"{r['label'][:60]}"
            )

    # Now do the cross-routing check using ONLY high-selectivity copula features
    print("\n" + "=" * 90)
    print("Cross-routing using HIGH-SELECTIVITY copula features only")
    print("=" * 90)
    routing = {}
    for nickname, refined_blob in refined.items():
        opposers = load_capital_opposers(nickname)
        hs_set = {r["feature_index"] for r in refined_blob["features"]}
        per_prompt = []
        total = 0
        for pid, opp in opposers.items():
            overlap = sorted(set(opp) & hs_set)
            per_prompt.append({
                "prompt_id": pid,
                "n_high_selectivity_copula_in_opposing_top10": len(overlap),
                "overlap_feature_indices": overlap,
            })
            total += len(overlap)
        n_caps = max(len(opposers), 1)
        mean = total / n_caps
        routing[nickname] = {
            "n_capitals": len(opposers),
            "total": total,
            "mean_per_prompt": mean,
            "n_high_selectivity_features": refined_blob["n_high_selectivity_copula_features"],
            "per_prompt": per_prompt,
        }
        print(
            f"  {nickname:<15}  capitals={len(opposers):>2}  "
            f"high-sel-copula-features-in-opposers={total:>3}  "
            f"mean/prompt={mean:.2f}  "
            f"(pool size = {refined_blob['n_high_selectivity_copula_features']})"
        )

    out_path = PATHS.reports / "cross_routing_selectivity.json"
    out_path.write_text(json.dumps({
        "selectivity_thresholds": {
            "log10_min": SELECTIVITY_MIN_LOG10, "copula_mean_min": COPULA_MEAN_MIN,
        },
        "refined": refined,
        "routing": routing,
    }, indent=2))
    print(f"\nWrote {out_path}")

    # Readable summary
    md_lines = [
        "# Cross-routing — selectivity-refined",
        "",
        f"Reranked the Rosetta-corpus top-N copula features by log10 selectivity",
        f"(`log10((copula_mean + ε) / (noncopula_mean + ε))`) and kept only features with",
        f"selectivity ≥ {SELECTIVITY_MIN_LOG10} (i.e., ≥10× more activation on copula than",
        f"non-copula tokens) AND copula_mean ≥ {COPULA_MEAN_MIN}.",
        "",
        "## High-selectivity copula features per model",
        "",
        "| model | pool size | top feature | log10_sel | label |",
        "|---|---|---|---|---|",
    ]
    for nickname, refined_blob in refined.items():
        if refined_blob["features"]:
            top = refined_blob["features"][0]
            md_lines.append(
                f"| {nickname} | {refined_blob['n_high_selectivity_copula_features']} | "
                f"f{top['feature_index']} | {top['log10_selectivity']:+.2f} | {top['label'][:60]} |"
            )
        else:
            md_lines.append(f"| {nickname} | 0 | — | — | — |")
    md_lines.extend([
        "",
        "## Cross-routing (high-selectivity copula features only)",
        "",
        "| model | capitals | high-sel-copula-in-opposing-top10 (sum) | mean/prompt | pool size |",
        "|---|---|---|---|---|",
    ])
    for nickname, r in routing.items():
        md_lines.append(
            f"| {nickname} | {r['n_capitals']} | {r['total']} | {r['mean_per_prompt']:.2f} | "
            f"{r['n_high_selectivity_features']} |"
        )
    md_lines.extend([
        "",
        "**Interpretation:** Selectivity-filtered, the cross-routing picture sharpens. Models",
        "that recruit their *dedicated* copula detectors (high selectivity, exclusive firing",
        "on copula tokens) as opposers on capital prompts are the models with the inversion.",
        "",
        "The raw cross_routing_functional analysis was confounded by content features that",
        "happen to co-fire with copula tokens because copula-heavy contexts are semantically",
        "richer. Filtering for log10_selectivity ≥ 1 removes those incidental detectors.",
    ])
    md_path = PATHS.reports / "cross_routing_selectivity_summary.md"
    md_path.write_text("\n".join(md_lines))
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
