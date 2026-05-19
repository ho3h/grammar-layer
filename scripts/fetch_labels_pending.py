"""Fetch Neuronpedia labels for the features that appear in top-K sets in pending models.

Only fetches labels for features that actually appear in the load_bearing_pos10 JSONs
(top supporting / opposing / by-abs / topk_features). That's ~500-600 features per
model instead of the full 16k-32k SAE, so we can label everything in a minute or two
per model.

Output: data/labels_cache_<nickname>.json — same format as labels_cache.json (Gemma 2 2B)
and labels_cache_gpt2.json (GPT-2 small).

Usage:
    uv run python scripts/fetch_labels_pending.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent

# Neuronpedia slug map (verified by HTTP probe; non-listed models 500'd at probe time)
SLUGS = {
    "gemma_9b":   ("gemma-2-9b",          "20-gemmascope-res-16k"),
    "gemma_1_2b": ("gemma-2b",            "12-res-jb"),
    "pythia_70m": ("pythia-70m-deduped",  "5-res-sm"),
    "gemma_w65k": ("gemma-2-2b",          "20-gemmascope-res-65k"),
}

LOAD_BEARING_PATHS = {
    "gemma_9b":   "reports/load_bearing_pos10_gemma_9b_50.json",
    "gemma_1_2b": "reports/load_bearing_pos10_gemma_1_2b_50.json",
    "pythia_70m": "reports/load_bearing_pos10_pythia_70m_50.json",
    "gemma_w65k": "reports/load_bearing_pos10_gemma_w65k_50.json",
}


def collect_features(path: str) -> set[int]:
    d = json.loads((ROOT / path).read_text())
    feats: set[int] = set()
    for r in d["results"]:
        for k in ("topk_features", "topk_supporting", "topk_opposing", "topk_by_abs"):
            if k in r:
                for e in r[k]:
                    feats.add(e["feature_index"])
    return feats


def fetch_label(http: httpx.Client, model_slug: str, sae_slug: str, idx: int) -> str | None:
    url = f"https://www.neuronpedia.org/api/feature/{model_slug}/{sae_slug}/{idx}"
    try:
        r = http.get(url, timeout=20.0)
        if r.status_code != 200:
            return None
        j = r.json()
        explanations = j.get("explanations") or []
        if not explanations:
            return ""
        # Take the first non-empty explanation
        for exp in explanations:
            text = exp.get("description") or ""
            if text:
                return text
        return ""
    except Exception:
        return None


def main() -> int:
    for nickname, (model_slug, sae_slug) in SLUGS.items():
        if nickname not in LOAD_BEARING_PATHS:
            continue
        feats = collect_features(LOAD_BEARING_PATHS[nickname])
        cache_path = ROOT / f"data/labels_cache_{nickname}.json"
        cache: dict[str, dict] = {}
        if cache_path.exists():
            cache = json.loads(cache_path.read_text())
        existing = {int(k) for k in cache.keys() if cache[k] is not None}
        todo = sorted(feats - existing)
        print(f"[{nickname}] {model_slug}/{sae_slug}: {len(feats)} target features, "
              f"{len(existing)} cached, {len(todo)} to fetch")
        if not todo:
            continue

        ok = miss = 0
        t0 = time.time()
        with httpx.Client(timeout=20.0) as http:
            for i, idx in enumerate(todo):
                text = fetch_label(http, model_slug, sae_slug, idx)
                if text is None:
                    cache[str(idx)] = None
                    miss += 1
                else:
                    cache[str(idx)] = {"text": text, "source": f"neuronpedia:{model_slug}/{sae_slug}"}
                    ok += 1
                if (i + 1) % 50 == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1) / max(elapsed, 0.01)
                    eta = (len(todo) - (i + 1)) / max(rate, 0.01)
                    print(f"  [{i+1}/{len(todo)}]  ok={ok} miss={miss}  rate={rate:.1f}/s  ETA {eta:.0f}s")
                    cache_path.write_text(json.dumps(cache, indent=0))
                time.sleep(0.02)  # be polite

        cache_path.write_text(json.dumps(cache, indent=0))
        print(f"[{nickname}] DONE: ok={ok} miss={miss}  wrote {cache_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
