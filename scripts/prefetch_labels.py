"""Prefetch all 16,384 Neuronpedia autointerp labels into the local cache,
so that `04_ingest_features.py` can ingest fast. Resumable — checks cache first.
"""

from __future__ import annotations

import httpx
from tqdm import tqdm

from neograph.config import SAE
from neograph.labels import LabelsCache, fetch_neuronpedia_label
from neograph.util import get_logger

log = get_logger("neograph.prefetch")


def main() -> int:
    cache = LabelsCache()
    have = {int(k) for k in cache._data.keys()}
    todo = [i for i in range(SAE.d_sae) if i not in have]
    log.info("Cache has %d/%d labels; fetching %d more", len(have), SAE.d_sae, len(todo))

    n_ok = 0
    n_miss = 0
    with httpx.Client(timeout=20.0) as http:
        for i, idx in enumerate(tqdm(todo, desc="neuronpedia")):
            lab = fetch_neuronpedia_label(idx, http, cache=cache, sleep_s=0.02)
            if lab is None:
                n_miss += 1
            else:
                n_ok += 1
            if i % 200 == 0:
                cache.flush()
    cache.flush()
    log.info("Done. ok=%d miss=%d total=%d", n_ok, n_miss, n_ok + n_miss)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
