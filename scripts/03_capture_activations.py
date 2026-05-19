"""P2: Capture model + SAE activations across the corpus."""

from __future__ import annotations

import sys

import pandas as pd
import torch

from neograph.activations import CaptureBudget, capture_all
from neograph.config import MODEL, PATHS, SAE
from neograph.util import exit_marker, get_logger

log = get_logger("neograph.capture")


def main() -> int:
    prompts_path = PATHS.staging / "prompts.parquet"
    if not prompts_path.exists():
        log.error("Run scripts/02_seed_corpus.py first — prompts.parquet missing")
        exit_marker("activations-captured", ok=False, stage="missing-prompts")
        return 1

    df = pd.read_parquet(prompts_path)
    log.info("Loaded %d prompts", len(df))

    log.info("Loading %s + SAE %s ...", MODEL.name, SAE.neograph_id)
    from sae_lens import SAE as SaeLensSAE
    from sae_lens import HookedSAETransformer as HookedTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(MODEL.name, device=device)
    model.eval()
    sae = SaeLensSAE.from_pretrained(release=SAE.release, sae_id=SAE.sae_id, device=device)

    budget = CaptureBudget(batch_size=4, max_seq_len=128, device=device)
    paths = capture_all(model, sae, df, budget)
    exit_marker(
        "activations-captured",
        ok=all(p.exists() for p in paths.values()),
        **{k: str(v) for k, v in paths.items()},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
