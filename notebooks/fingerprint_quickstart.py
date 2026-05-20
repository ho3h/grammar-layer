"""Reproduce the three headline figures with a small API surface.

This file is a runnable script, not a notebook — it prints the headline numbers and
demonstrates how to drive each model from outside.

    uv run python notebooks/fingerprint_quickstart.py

Three things this script demonstrates:

1. Identify the canonical fingerprint features for each model (no model loading required —
   read from the cached load_bearing_pos10_<model>_50.json).
2. Side-by-side per-prompt: who opposes "The capital of Japan is Tokyo" across models.
3. Bidirectional steering: with Gemma 2 2B loaded, amplify f15596 by 10× and observe the
   argmax flip on a capital prompt.

The library code is in `src/neograph/fingerprint.py`.
"""

from __future__ import annotations

import json

from neograph.fingerprint import (
    CANONICAL_FINGERPRINT,
    cross_model_routing,
    identify_copula_opposers,
    known_fingerprint_pairs,
    steer_feature,
)


def demo_1_identify() -> None:
    print("=" * 80)
    print("1. Canonical opposing-side fingerprint per model")
    print("=" * 80)
    for nickname in ("gemma", "pythia_70m", "gemma_1_2b", "gpt2"):
        records = identify_copula_opposers(nickname)
        print(f"\n  {nickname}:")
        if not records:
            print("    (no grammar fingerprint — content opposers only)")
        for r in records:
            print(f"    feat {r.feature:>6}  {r.label}")


def demo_2_cross_model_routing(prompt_id: str = "capital-jp") -> None:
    print("\n" + "=" * 80)
    print(f"2. Cross-model routing on '{prompt_id}'")
    print("=" * 80)
    routing = cross_model_routing(prompt_id)
    for nickname, blob in routing.items():
        baseline = blob.get("baseline", {})
        joint = blob.get("joint_top10_ablated", {})
        drop = blob.get("log_p_drop")
        print(f"\n  {nickname}:")
        print(
            f"    baseline argmax: {baseline.get('argmax_token_str', '?')!r}  "
            f"log P(target) = {baseline.get('log_p_target', float('nan')):+.3f}"
        )
        print(
            f"    after joint top-10 ablation: argmax = "
            f"{joint.get('argmax_token_str', '?')!r}  Δlog P = {drop or 0:+.3f}"
        )
        if blob.get("opposing"):
            print("    top-3 opposers:")
            for e in blob["opposing"][:3]:
                lab = e.get("label") or ""
                print(
                    f"      feat {e['feature_index']:>6}  Δlog P = "
                    f"{e['single_log_p_drop']:+.3f}  {lab[:60]}"
                )


def demo_3_steer_gemma(prompt: str = "The capital of Japan is") -> None:
    print("\n" + "=" * 80)
    print(f"3. Bidirectional steering of Gemma 2 2B f15596 on prompt:\n   {prompt!r}")
    print("=" * 80)
    print("\n  (loads gemma-2-2b + Gemma Scope L20/16k canonical — ~90s on M-series MPS)")

    import torch
    from sae_lens import SAE as SaeLensSAE, HookedSAETransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = HookedSAETransformer.from_pretrained("gemma-2-2b", device=device)
    model.eval()
    sae = SaeLensSAE.from_pretrained(
        release="gemma-scope-2b-pt-res-canonical",
        sae_id="layer_20/width_16k/canonical",
        device=device,
    )

    tokens = model.to_tokens(prompt, prepend_bos=True)
    print(f"\n  {'scale':>6}  {'argmax':>14}  {'log P(Tokyo)':>14}  {'log P(a)':>10}")
    target_id = model.tokenizer.encode(" Tokyo", add_special_tokens=False)[0]
    a_id = model.tokenizer.encode(" a", add_special_tokens=False)[0]

    for scale in (0.0, 1.0, 2.0, 5.0, 10.0):
        with steer_feature(model, sae, feature_index=15596, scale=scale):
            with torch.no_grad():
                logits = model(tokens)
        last = logits[0, -1, :].float()
        log_probs = torch.log_softmax(last, dim=-1)
        argmax_id = int(log_probs.argmax().item())
        argmax_str = model.tokenizer.decode([argmax_id]).strip()
        print(
            f"  {scale:>6.1f}  {argmax_str!r:>14}  "
            f"{log_probs[target_id].item():>+14.3f}  {log_probs[a_id].item():>+10.3f}"
        )


def main() -> None:
    demo_1_identify()
    demo_2_cross_model_routing("capital-jp")
    print("\n(Demo 3 is disabled by default — it loads Gemma 2 2B. Uncomment to run.)")
    # demo_3_steer_gemma()


if __name__ == "__main__":
    main()
