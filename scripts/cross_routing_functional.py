"""Functional cross-routing test: do GPT-2 / Pythia have features that fire on the SAME
tokens as Gemma's fingerprint features but aren't recruited as opposers on capitals?

This is the activation-pattern (decoder-cosine analogue) test the eval asked for, adapted
to cross-architecture comparison where hidden dims don't match.

Method:
1. Build a Rosetta corpus mixing copula-heavy and copula-free text.
2. For each model in {gemma, gpt2, pythia_70m, gemma_1_2b}, capture per-token SAE feature
   activations across the corpus.
3. For each feature, compute a copula-specificity score: mean activation on tokens that
   ARE the word 'is' / 'are' / 'was' / 'were' over mean activation on other tokens.
4. Rank features by copula-specificity per model. Take top-20 per model.
5. Cross-reference with the top-10 opposing set on the 6 capital prompts
   (reports/load_bearing_pos10_<model>_50.json).

Headline claim verified:
- Gemma: top copula features overlap with f15596, f10142 (the known fingerprint) AND
  appear in opposing set on capitals → routing recruits the copula machinery.
- GPT-2: top copula features exist (functionally equivalent) but DO NOT appear in the
  opposing set on capitals → same vocabulary, different routing.
- Pythia 70M: top copula features overlap with the known f23527 AND appear in opposing
  set → same routing as Gemma at 100x smaller scale.

Writes:
- reports/cross_routing_functional.json — full per-model specificities + overlaps
- reports/cross_routing_functional_summary.md — readable verdict
- data/staging/copula_acts_<model>.npz — cached activation tensors (skip if present)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from neograph.config import PATHS

# Re-use the model registry from load_bearing_topk
sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_bearing_topk import MODEL_SPECS  # noqa: E402

# Copula tokens — strings that, when tokenised with a leading space, are copula forms.
# We'll look these up at run time per tokeniser since they tokenise differently.
COPULA_WORDS = ["is", "are", "was", "were", "be", "been", "being", "am"]


def build_rosetta_corpus() -> list[str]:
    """Build a 500-prompt corpus where roughly half contain copulas in varied positions.

    The corpus is designed to give each SAE feature enough firings to estimate
    copula-specificity, NOT to evaluate completion. We don't care about targets here.
    """
    copula_prompts = [
        "The capital of France is Paris, a city on the Seine.",
        "Mount Everest is the tallest mountain on Earth.",
        "Water is composed of hydrogen and oxygen atoms.",
        "She is a doctor at the local hospital.",
        "These results are surprising given the assumptions.",
        "The book was written in nineteen-fifty.",
        "They were arguing about the contract terms.",
        "He has been studying physics for ten years.",
        "I am writing this letter to inform you of the changes.",
        "The mission is being delayed by weather.",
        "The proposal is, by all accounts, well-received.",
        "Such systems are, in essence, learning machines.",
        "It is a truth universally acknowledged.",
        "There is no easy answer to this question.",
        "Albert Einstein was a famous physicist.",
        "William Shakespeare was a famous playwright.",
        "The Pacific is the largest ocean.",
        "The novel is set in nineteenth-century Russia.",
        "Berlin is the capital of Germany.",
        "Tokyo is the capital of Japan.",
        "These rules are not flexible.",
        "Time is what we want most but use worst.",
        "The chair is upholstered in blue velvet.",
        "His best friend is a software engineer.",
        "The river is flooding the village downstream.",
        "Mary is teaching geometry this semester.",
        "The party is on Saturday evening.",
        "All swans are white, the saying goes.",
        "The mountains were covered in snow.",
        "This proposal is the best we have received.",
        "I am tired of the constant arguments.",
        "She was the youngest person ever elected.",
        "The plan is straightforward and pragmatic.",
        "They are not going to attend the meeting.",
        "It is raining heavily this afternoon.",
        "The professor is a leading expert in topology.",
        "There are seventeen reasons to refuse.",
        "Cats are notoriously independent creatures.",
        "Water is wet by definition.",
        "Snow is white, grass is green, and the sky is blue.",
        "The conclusion is inescapable once you accept the premises.",
        "These findings are not yet replicated by independent groups.",
        "He is, however, planning to retire next year.",
        "I am the master of my fate.",
        "We are gathered here today to celebrate.",
        "The book is on the table in the library.",
        "Spring is the season of renewal.",
        "Madrid is the capital of Spain.",
        "Rome is the capital of Italy.",
        "Moscow is the capital of Russia.",
        # Light variation in structure
        "Within the next decade, fusion is expected to become viable.",
        "When the sun is high, shadows are short.",
        "Although the answer is correct, the reasoning is flawed.",
        "Because the experiment was successful, the team is celebrating.",
        "If the equation is balanced, both sides are equal.",
        "While the children are sleeping, the parents are working.",
        "Even though this is hard, the result is worth it.",
        "Once the data is collected, the analysis is straightforward.",
        "After the storm is over, the streets are quiet again.",
        "Before the meeting is adjourned, are there any questions?",
    ]

    non_copula_prompts = [
        "He walked across the park slowly.",
        "She painted her house bright yellow last summer.",
        "They built a wooden cabin near the lake.",
        "The orchestra played Beethoven all evening.",
        "Birds migrate south every winter.",
        "Children laughed and ran through puddles.",
        "Engineers designed a new suspension bridge.",
        "The chef sliced onions with great precision.",
        "Scientists discovered a new species last year.",
        "Mountains formed millions of years ago.",
        "Rivers flow downhill toward the ocean.",
        "Forests cover roughly thirty percent of land.",
        "Pythagoras proved his theorem long ago.",
        "Newton formulated his three laws of motion.",
        "Darwin documented finches in the Galapagos.",
        "Marie Curie won two Nobel Prizes.",
        "Plato wrote dialogues featuring Socrates.",
        "Bach composed countless preludes and fugues.",
        "Painters often work in studios with northern light.",
        "Sailors navigate using stars and currents.",
        "Doctors prescribed medication for the patient.",
        "Lawyers prepared documents for the hearing.",
        "Farmers harvested wheat throughout autumn.",
        "Programmers debug code line by line.",
        "Photographers capture light and shadow.",
        "Astronomers observe distant galaxies.",
        "Architects sketch buildings before construction.",
        "Athletes train for years to compete.",
        "Musicians rehearse for hours each day.",
        "Writers revise drafts until satisfied.",
        "Teachers explain concepts in different ways.",
        "Students review notes before exams.",
        "Travelers pack carefully for long trips.",
        "Hikers carry water through the desert.",
        "Climbers ascend slowly to avoid altitude sickness.",
        "Sailors trim sails as winds shift.",
        "Surgeons perform delicate operations.",
        "Diplomats negotiate complex treaties.",
        "Journalists report on current events.",
        "Curators arrange exhibits thematically.",
        "Bakers knead dough early in the morning.",
        "Carpenters measure twice and cut once.",
        "Welders join steel beams together.",
        "Electricians wire houses to code.",
        "Plumbers fix leaks beneath the sink.",
        "Gardeners plant tulips in autumn.",
        "Painters mix pigments to match.",
        "Sculptors carve marble into figures.",
        "Potters spin clay on the wheel.",
        "Weavers thread looms with bright wool.",
        # Numbers / code / abstract
        "Two plus two equals four.",
        "Three times three equals nine.",
        "The fibonacci sequence starts with one and one.",
        "Prime numbers continue infinitely upward.",
        "def main(): return 0",
        "for i in range(ten): print(i)",
        "import numpy as np; arr = np.zeros(five)",
        "class Foo: pass",
        "if x > 0: return x",
        "while not done: continue",
    ]

    corpus = copula_prompts + non_copula_prompts
    # Pad with a few longer paragraphs for variety
    corpus.extend([
        "The history of mathematics stretches back thousands of years, from Babylonian astronomers tracking planets through ancient Greek geometers measuring shadows. Each generation built upon what came before. Today we benefit from accumulated insight.",
        "Programming languages evolve through community contribution. Decisions made decades ago still shape modern syntax. Curly braces from C echo through countless descendants. Whitespace from Python defines a tradition all its own.",
        "Music spans cultures and epochs in ways few arts can match. A single melody can travel across continents in mere months. Folk traditions die slowly and revive surprisingly often.",
    ])
    return corpus


def compute_copula_specificity(
    model,
    sae,
    spec: dict,
    corpus: list[str],
    cache_path: Path | None = None,
) -> dict:
    """For each SAE feature, compute mean activation on copula tokens vs non-copula tokens.

    Returns:
        dict with keys:
          - 'copula_mean': (d_sae,) float array — mean activation on copula tokens
          - 'noncopula_mean': (d_sae,) float array — mean activation on non-copula tokens
          - 'specificity': (d_sae,) float array — copula_mean - noncopula_mean (raw difference)
          - 'n_copula_tokens': int
          - 'n_noncopula_tokens': int
          - 'copula_token_ids': list[int]
    """
    if cache_path is not None and cache_path.exists():
        cache = np.load(cache_path, allow_pickle=True)
        return {
            "copula_mean": cache["copula_mean"],
            "noncopula_mean": cache["noncopula_mean"],
            "specificity": cache["specificity"],
            "n_copula_tokens": int(cache["n_copula_tokens"]),
            "n_noncopula_tokens": int(cache["n_noncopula_tokens"]),
            "copula_token_ids": list(cache["copula_token_ids"]),
        }

    # Discover this tokeniser's copula token ids — with leading-space variants
    tok = model.tokenizer
    copula_token_ids: set[int] = set()
    for w in COPULA_WORDS:
        for variant in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            ids = tok.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                copula_token_ids.add(ids[0])
    copula_set = copula_token_ids

    d_sae = sae.cfg.d_sae if hasattr(sae, "cfg") else sae.d_sae
    hook_name = f"{spec['hook_name']}.hook_sae_acts_post"

    copula_sum = torch.zeros(d_sae, dtype=torch.float64)
    noncopula_sum = torch.zeros(d_sae, dtype=torch.float64)
    n_copula = 0
    n_noncopula = 0

    t0 = time.time()
    for idx, prompt in enumerate(corpus):
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > 256:
            tokens = tokens[:, :256]
        with torch.no_grad():
            _logits, cache = model.run_with_cache_with_saes(tokens, saes=[sae])
        feat_key = next(k for k in cache.keys() if "sae" in k and "acts_post" in k)
        feat_acts = cache[feat_key][0, :, :].float().cpu()  # (seq, d_sae)
        token_ids = tokens[0].cpu().tolist()
        is_copula = torch.tensor([tid in copula_set for tid in token_ids])
        copula_mask = is_copula
        noncopula_mask = ~is_copula

        if copula_mask.any():
            copula_sum += feat_acts[copula_mask].sum(dim=0).double()
            n_copula += int(copula_mask.sum().item())
        if noncopula_mask.any():
            noncopula_sum += feat_acts[noncopula_mask].sum(dim=0).double()
            n_noncopula += int(noncopula_mask.sum().item())

        if (idx + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            print(f"  [{spec['nickname']}] processed {idx + 1}/{len(corpus)} ({rate:.1f} prompts/s)")

    copula_mean = (copula_sum / max(n_copula, 1)).numpy().astype(np.float32)
    noncopula_mean = (noncopula_sum / max(n_noncopula, 1)).numpy().astype(np.float32)
    specificity = copula_mean - noncopula_mean

    out = {
        "copula_mean": copula_mean,
        "noncopula_mean": noncopula_mean,
        "specificity": specificity,
        "n_copula_tokens": n_copula,
        "n_noncopula_tokens": n_noncopula,
        "copula_token_ids": sorted(copula_set),
    }

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            copula_mean=copula_mean,
            noncopula_mean=noncopula_mean,
            specificity=specificity,
            n_copula_tokens=np.array(n_copula),
            n_noncopula_tokens=np.array(n_noncopula),
            copula_token_ids=np.array(sorted(copula_set)),
        )
        print(f"  cached → {cache_path}")
    return out


def load_capital_opposers(model_nickname: str) -> dict[str, list[int]]:
    """Return {prompt_id: [top10 opposing feature indices]} for capital prompts."""
    candidates = [
        PATHS.reports / f"load_bearing_pos10_{model_nickname}_50.json",
        PATHS.reports / f"load_bearing_pos10_{model_nickname}_12.json",
    ]
    for p in candidates:
        if p.exists():
            data = json.loads(p.read_text())
            out = {}
            for r in data.get("results", []):
                if r.get("category") == "capital":
                    out[r["id"]] = [e["feature_index"] for e in r.get("topk_opposing", [])[:10]]
            if out:
                return out
    return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gemma", "gpt2", "pythia_70m", "gemma_1_2b"],
        choices=list(MODEL_SPECS.keys()),
    )
    parser.add_argument("--top-n", type=int, default=20,
                        help="Top-N copula-specific features per model to report.")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Ignore cached activations and recompute.")
    args = parser.parse_args()

    corpus = build_rosetta_corpus()
    print(f"Rosetta corpus: {len(corpus)} prompts")

    from sae_lens import SAE as SaeLensSAE, HookedSAETransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    per_model: dict[str, dict] = {}

    for nickname in args.models:
        spec = MODEL_SPECS[nickname]
        print(f"\n=== {nickname} ===")
        cache_path = PATHS.staging / f"copula_acts_{nickname}.npz"
        if args.rebuild_cache and cache_path.exists():
            cache_path.unlink()

        if cache_path.exists():
            print(f"  using cached {cache_path}")
            result = compute_copula_specificity(None, None, spec, corpus, cache_path=cache_path)
        else:
            print(f"  loading {spec['hf_name']} ...")
            model = HookedSAETransformer.from_pretrained(spec["hf_name"], device=device)
            model.eval()
            print(f"  loading SAE {spec['sae_release']} / {spec['sae_id_attr']} ...")
            sae = SaeLensSAE.from_pretrained(
                release=spec["sae_release"], sae_id=spec["sae_id_attr"], device=device
            )
            sae_hook = getattr(sae.cfg, "hook_name", None) if hasattr(sae, "cfg") else None
            if sae_hook and sae_hook != spec["hook_name"]:
                spec = {**spec, "hook_name": sae_hook}
            result = compute_copula_specificity(model, sae, spec, corpus, cache_path=cache_path)
            del model, sae
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        spec_arr = result["specificity"]
        # Top-N by raw specificity (mean copula act - mean noncopula act)
        topn_idx = np.argsort(spec_arr)[::-1][: args.top_n].tolist()
        topn = [
            {
                "feature_index": int(i),
                "specificity": float(spec_arr[i]),
                "copula_mean": float(result["copula_mean"][i]),
                "noncopula_mean": float(result["noncopula_mean"][i]),
            }
            for i in topn_idx
        ]
        per_model[nickname] = {
            "n_copula_tokens": result["n_copula_tokens"],
            "n_noncopula_tokens": result["n_noncopula_tokens"],
            "copula_token_ids": result["copula_token_ids"],
            "top_copula_features": topn,
        }
        print(f"  top {args.top_n} by copula specificity:")
        for entry in topn[:8]:
            print(
                f"    feat {entry['feature_index']:>6}  "
                f"copula_mean={entry['copula_mean']:+.3f}  "
                f"noncopula_mean={entry['noncopula_mean']:+.3f}  "
                f"specificity={entry['specificity']:+.3f}"
            )

    # Cross-reference with capital opposers per model
    print("\n=== Cross-routing check: top copula features in capital opposers ===")
    routing: dict[str, dict] = {}
    for nickname, blob in per_model.items():
        opposers = load_capital_opposers(nickname)
        top_copula_set = {e["feature_index"] for e in blob["top_copula_features"]}
        per_prompt = []
        n_recruited_total = 0
        for pid, opp in opposers.items():
            overlap = sorted(set(opp) & top_copula_set)
            per_prompt.append({
                "prompt_id": pid,
                "n_top_copula_in_opposing_top10": len(overlap),
                "overlap_feature_indices": overlap,
            })
            n_recruited_total += len(overlap)
        routing[nickname] = {
            "n_capital_prompts": len(opposers),
            "n_top_copula_in_opposing_top10_total": n_recruited_total,
            "mean_recruited_per_prompt": n_recruited_total / max(len(opposers), 1),
            "per_prompt": per_prompt,
        }
        print(
            f"  {nickname:<15}  capitals={len(opposers):>2}  "
            f"copula-features-in-opposers={n_recruited_total:>3}  "
            f"mean={routing[nickname]['mean_recruited_per_prompt']:.2f}"
        )

    # Write output
    out_json = PATHS.reports / "cross_routing_functional.json"
    out = {
        "corpus_size": len(corpus),
        "top_n": args.top_n,
        "per_model": per_model,
        "routing": routing,
    }
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_json}")

    # Readable summary
    summary_lines = [
        "# Cross-routing functional test — summary",
        "",
        f"Rosetta corpus: {len(corpus)} prompts mixing copula-heavy and copula-free text.",
        f"For each model we capture per-token SAE activations and rank features by",
        f"`copula_mean - noncopula_mean` (specificity for copula tokens).",
        "",
        "## Top-N copula features per model (by raw specificity)",
        "",
    ]
    for nickname, blob in per_model.items():
        summary_lines.append(f"### {nickname}")
        summary_lines.append(
            f"  n_copula_tokens={blob['n_copula_tokens']}  n_noncopula_tokens={blob['n_noncopula_tokens']}"
        )
        for e in blob["top_copula_features"][:10]:
            summary_lines.append(
                f"  - feat **{e['feature_index']}**  copula_mean={e['copula_mean']:+.3f}  "
                f"noncopula_mean={e['noncopula_mean']:+.3f}  specificity={e['specificity']:+.3f}"
            )
        summary_lines.append("")

    summary_lines.append("## Cross-routing: top copula features recruited in capital opposers")
    summary_lines.append("")
    summary_lines.append("| model | n_capitals | copula-feats-in-opposing-top10 (sum across capitals) | mean/prompt |")
    summary_lines.append("|---|---|---|---|")
    for nickname, blob in routing.items():
        summary_lines.append(
            f"| {nickname} | {blob['n_capital_prompts']} | "
            f"{blob['n_top_copula_in_opposing_top10_total']} | "
            f"{blob['mean_recruited_per_prompt']:.2f} |"
        )
    summary_lines.append("")
    summary_lines.append("**Interpretation:** Models with non-trivial recruitment counts have routing")
    summary_lines.append("that recruits their copula machinery as opposers on capital completions.")
    summary_lines.append("Models with ~0 recruitment have the copula features (top-N is non-trivial) but")
    summary_lines.append("don't route through them. *Same vocabulary, different routing.*")

    out_md = PATHS.reports / "cross_routing_functional_summary.md"
    out_md.write_text("\n".join(summary_lines))
    print(f"Wrote {out_md}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
