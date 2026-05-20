"""Cross-model fingerprint API — the substrate's user-facing surface.

Three primitives the writeup actually uses:

1. `identify_copula_opposers(model)` — returns the canonical fingerprint features
   for a model, from the cached per-prompt opposing-feature analysis.
2. `steer(model_nickname, feature_index, scale, prompts)` — apply an SAE
   feature-activation hook at the last position, multiply by scale, score the
   resulting logits. Bidirectional control: scale > 1 amplifies, scale = 0
   ablates, scale ∈ (0, 1) attenuates.
3. `cross_model_routing(prompt)` — for a single prompt, return the supporting
   and opposing top-K features in every model with a cached load-bearing run.

Designed to be the smallest API that reproduces the headline figures.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from neograph.config import PATHS


# Canonical fingerprint features per model — sourced from the per-prompt
# opposing-feature analysis in reports/load_bearing_pos10_<model>_50.json.
# These are the features that appear in top-5 opposing on ≥5/6 capital prompts.
CANONICAL_FINGERPRINT: dict[str, list[dict]] = {
    "gemma": [
        {"feature": 15596, "label": "past and present tense forms of the verb 'to be'", "role": "opposer"},
        {"feature": 10142, "label": "instances of the word 'is' in various contexts", "role": "opposer"},
    ],
    "pythia_70m": [
        {"feature": 23527, "label": "occurrences of the verb 'is' and its various forms", "role": "opposer"},
    ],
    "gemma_1_2b": [
        {"feature":  5541, "label": "instances of the verb 'is'", "role": "opposer"},
        {"feature": 16346, "label": "the verb 'is' and its variants", "role": "opposer"},
        {"feature":  5943, "label": "the verb 'is' in various contexts", "role": "opposer"},
    ],
    "gpt2": [],  # No grammar fingerprint — content opposers only
}


@dataclass(frozen=True)
class FeatureRecord:
    feature: int
    label: str
    role: str  # "opposer" or "supporter"


def identify_copula_opposers(model_nickname: str) -> list[FeatureRecord]:
    """Return the canonical opposing-side copula fingerprint for `model_nickname`."""
    entries = CANONICAL_FINGERPRINT.get(model_nickname, [])
    return [FeatureRecord(**e) for e in entries]


def cross_model_routing(prompt_id: str) -> dict[str, dict]:
    """For a single benchmark prompt id, return the top-K supporting/opposing features
    per model, from cached load_bearing_pos10_<model>_50.json files.

    Returns: {model: {supporting: [...], opposing: [...], baseline: {...}, joint: {...}}}
    """
    out: dict[str, dict] = {}
    for path in sorted(PATHS.reports.glob("load_bearing_pos10_*_50.json")):
        nick = path.stem.replace("load_bearing_pos10_", "").replace("_50", "")
        data = json.loads(path.read_text())
        for r in data.get("results", []):
            if r["id"] == prompt_id:
                out[nick] = {
                    "supporting": r.get("topk_supporting", [])[:10],
                    "opposing": r.get("topk_opposing", [])[:10],
                    "baseline": r.get("baseline", {}),
                    "joint_top10_ablated": r.get("joint_topk_ablated", {}),
                    "log_p_drop": r.get("log_p_drop_vs_baseline", None),
                }
                break
    return out


@contextmanager
def steer_feature(model, sae, feature_index: int, scale: float,
                  hook_name: str | None = None) -> Iterator[None]:
    """Context manager: multiply the SAE feature activation at the last position by `scale`.

    scale = 0.0 ablates; 1.0 is identity; > 1.0 amplifies; < 0.0 reverses.

    The hook is registered on entry and removed on exit, so the model returns to
    its baseline state.

    Usage:
        with steer_feature(model, sae, 15596, scale=10.0):
            logits = model(tokens)
            # logits reflect amplified f15596
    """
    sae_hook_name = hook_name or getattr(sae.cfg, "hook_name", None)
    if sae_hook_name is None:
        raise ValueError("Cannot determine SAE hook name; pass hook_name explicitly.")
    full_hook = f"{sae_hook_name}.hook_sae_acts_post"

    def steer_fn(act, **kwargs):
        act = act.clone()
        act[..., -1, feature_index] = act[..., -1, feature_index] * scale
        return act

    model.add_sae(sae)
    model.add_hook(full_hook, steer_fn)
    try:
        yield
    finally:
        model.reset_hooks()
        model.reset_saes()


def known_fingerprint_pairs() -> list[tuple[str, int, str]]:
    """Flat list of (model_nickname, feature_index, label) for all canonical fingerprints."""
    out: list[tuple[str, int, str]] = []
    for nick, feats in CANONICAL_FINGERPRINT.items():
        for f in feats:
            out.append((nick, f["feature"], f["label"]))
    return out
