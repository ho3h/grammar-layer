"""GPT-2 layer sweep — the load-bearing robustness experiment for the predicate-backbone finding.

Theo's note (2026-05-12 evening): the v2 finding compared Gemma L20 (~77% deep) with
GPT-2 small L8 (~67% deep). If predicate features start appearing in GPT-2 circuits at
L10 or L11, the headline inverts to "predicate-circuit recruitment is depth-dependent
in GPT-2 and consistent throughout in Gemma."

Approach (cheap version — does not re-ingest into Neo4j):
- For each candidate GPT-2 layer (L4, L8, L10, L11):
  - Load the matching gpt2-small-res-jb SAE.
  - Run the same 12 prompts from data/causal_prompts.json; for each, ablate every
    active feature at the last position and record per-feature attribution.
  - Identify features in ≥3 circuits with high mean |attribution|.
  - Fetch their Neuronpedia autointerp labels.
- Compare labels across layers: are predicate-style ("forms of to be", "statements
  of existence", "beginning of text") present at any deeper layer?

Output:
- reports/gpt2_layer_sweep.json — per-layer top-30 multi-circuit features + labels
- reports/gpt2_layer_sweep_summary.md — readable verdict
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

import httpx
import torch
from tenacity import retry, stop_after_attempt, wait_exponential

from neograph.config import PATHS
from neograph.util import get_logger

log = get_logger("neograph.gpt2_sweep")


LAYERS = [4, 8, 10, 11]
ACTIVATION_THRESHOLD = 1e-3


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=0.5, max=4))
def _np_label(client, layer: int, idx: int) -> str | None:
    url = f"https://www.neuronpedia.org/api/feature/gpt2-small/{layer}-res-jb/{idx}"
    r = client.get(url, timeout=15.0)
    if r.status_code != 200:
        return None
    data = r.json()
    exps = data.get("explanations") or []
    if not exps:
        return None
    text = (exps[0].get("description") or "").strip()
    return text or None


def run_layer(model, sae, layer: int, prompts: list[dict]) -> dict:
    """Run zero-ablation patching for `prompts` at this layer's SAE. Return per-feature stats."""
    hook_post = f"blocks.{layer}.hook_resid_pre.hook_sae_acts_post"

    # feature_index -> list of (prompt_id, effect_size, magnitude)
    by_feat: dict[int, list[tuple[str, float, float]]] = defaultdict(list)

    for spec in prompts:
        target_ids = model.tokenizer.encode(spec["target"], add_special_tokens=False)
        if not target_ids:
            continue
        target_id = target_ids[0]
        tokens = model.to_tokens(spec["prompt"], prepend_bos=True)

        with torch.no_grad():
            baseline_logits = model.run_with_hooks_with_saes(tokens, saes=[sae], fwd_hooks=[])
        baseline = float(baseline_logits[0, -1, target_id].item())

        with torch.no_grad():
            _, cache = model.run_with_cache_with_saes(tokens, saes=[sae])
        feat_key = next(k for k in cache.keys() if "sae" in k and "acts_post" in k)
        feat_acts = cache[feat_key][0, -1, :].float().cpu()
        active = (feat_acts > ACTIVATION_THRESHOLD).nonzero(as_tuple=True)[0]
        log.info("  L%d  %s: %d active features", layer, spec["id"], len(active))

        for fidx in active.tolist():
            def ablate(act, fidx=fidx, **kwargs):
                act = act.clone()
                act[..., -1, fidx] = 0.0
                return act

            with torch.no_grad():
                ablated = model.run_with_hooks_with_saes(
                    tokens, saes=[sae], fwd_hooks=[(hook_post, ablate)]
                )
            ablated_logit = float(ablated[0, -1, target_id].item())
            effect = baseline - ablated_logit
            by_feat[int(fidx)].append((spec["id"], effect, float(feat_acts[fidx].item())))

    # Aggregate to multi-circuit features
    multi_circuit = []
    for fidx, entries in by_feat.items():
        n = len({pid for pid, _, _ in entries})
        if n < 3:
            continue
        mean_abs = sum(abs(e) for _, e, _ in entries) / len(entries)
        total = sum(e for _, e, _ in entries)
        multi_circuit.append({
            "feature": fidx, "n_circuits": n, "total_attr": total,
            "mean_abs_attr": mean_abs,
            "circuits": [pid for pid, _, _ in entries],
            "max_magnitude": max(m for _, _, m in entries),
        })
    multi_circuit.sort(key=lambda r: (-r["n_circuits"], -r["mean_abs_attr"]))
    return {"multi_circuit": multi_circuit[:30]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    parser.add_argument("--label-top-k", type=int, default=20,
                        help="Fetch Neuronpedia labels for top-K multi-circuit features per layer")
    args = parser.parse_args()

    prompts = json.loads((PATHS.data / "causal_prompts.json").read_text())
    from sae_lens import SAE as SaeLensSAE, HookedSAETransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("Loading GPT-2 small on %s ...", device)
    model = HookedSAETransformer.from_pretrained("gpt2", device=device)
    model.eval()

    results = {}
    with httpx.Client(timeout=15.0) as http:
        for layer in args.layers:
            log.info("=== Layer %d ===", layer)
            sae_id = f"blocks.{layer}.hook_resid_pre"
            try:
                sae = SaeLensSAE.from_pretrained(release="gpt2-small-res-jb", sae_id=sae_id, device=device)
            except Exception as exc:  # noqa: BLE001
                log.error("Skipping L%d (load failed): %s", layer, exc)
                continue
            log.info("L%d SAE: d_in=%d d_sae=%d", layer, sae.cfg.d_in, sae.cfg.d_sae)
            layer_res = run_layer(model, sae, layer, prompts)
            # Fetch labels for top-K features
            top_k = layer_res["multi_circuit"][: args.label_top_k]
            log.info("Fetching labels for top-%d multi-circuit features ...", len(top_k))
            for entry in top_k:
                entry["label"] = _np_label(http, layer, entry["feature"]) or "(no Neuronpedia label)"
            results[f"L{layer}"] = layer_res
            log.info("L%d: %d multi-circuit features (≥3); top counts: %s", layer,
                     len(layer_res["multi_circuit"]),
                     [r["n_circuits"] for r in layer_res["multi_circuit"][:5]])
            # Show top-5 inline
            for e in layer_res["multi_circuit"][:5]:
                log.info("  feat %5d  in %d/%d circuits  Σ=%+.3f  %s",
                         e["feature"], e["n_circuits"], len(prompts),
                         e["total_attr"], (e.get("label") or "")[:60])

    out = PATHS.reports / "gpt2_layer_sweep.json"
    out.write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", out)

    # Render quick summary
    summary_lines = ["# GPT-2 layer sweep — predicate-feature recruitment across layers\n"]
    for layer_key, layer_res in results.items():
        summary_lines.append(f"\n## {layer_key}\n")
        summary_lines.append(f"_{len(layer_res['multi_circuit'])} features in ≥3 of 12 circuits_\n\n")
        summary_lines.append("| feature | # circuits | Σ attr | mean |attr| | label |")
        summary_lines.append("|---:|---:|---:|---:|---|")
        for e in layer_res["multi_circuit"][:15]:
            label = (e.get("label") or "").replace("|", "\\|")[:80]
            summary_lines.append(
                f"| {e['feature']} | {e['n_circuits']} | "
                f"{e['total_attr']:+.3f} | {e['mean_abs_attr']:.3f} | {label} |"
            )
    (PATHS.reports / "gpt2_layer_sweep_summary.md").write_text("\n".join(summary_lines))
    log.info("Wrote summary md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
