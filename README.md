# grammar-layer

**A causal-ablation study of next-token prediction across seven SAE-equipped language models.**

Across 52 next-token-prediction prompts spanning 12 task categories, we identify the per-prompt top-10 SAE features whose ablation most reduces log P(target), and joint-zero-ablate them. The effect is **41× stronger than random ablation** of the same size (control: 10 random active features → Δlog P +0.085 vs +3.45 for the targeted set on Gemma 2 2B), so it isn't a tautology of the selection. Mean-ablation reproduces the collapse (Δlog P +2.87), so it isn't a zero-projection artifact.

The smoking gun: across all six capital-completion prompts ("The capital of France / Germany / Italy / Spain / Russia / Japan is ___"), **the same two Gemma features** — #15596 ("forms of the verb 'to be'") and #10142 ("instances of the word 'is'") — appear as top opposers on every single prompt. Permutation test p = 0.0077 that this co-occurrence is chance. GPT-2 small has *zero* of its 652 grammar-labelled features in top-K opposing on any capital prompt despite having direct decoder/label-similar counterparts of the Gemma fingerprint pair. Same vocabulary, completely different routing.

**The grammar-suppression apparatus is not Gemma-specific.** With Neuronpedia labels populated for 5 of 7 models, the inversion appears in:
- **Pythia 70M (5.80×)** — EleutherAI, 70M params, *smaller than GPT-2 small*. Its top opposer on capitals is literally labelled "occurrences of the verb 'is' and its various forms".
- **Gemma 1 2B (3.40×)** — Google, older generation
- **Gemma 2 2B (2.80×)** — the original fingerprint

Absent in GPT-2 small (0.93×) and at mid-network L20/42 of Gemma 2 9B (1.31×, plausibly a layer-depth confound). It's not a scale signature, not a Google fingerprint, and not a recent-training-recipe phenomenon. The question reframes from "why does Gemma have this" to "why does GPT-2 lack this when its tiny EleutherAI peer at the same scale has it".

External behavioural correlate: all four predicted grammar-suppression metrics (copula density, hedge density, generic NP rate, copula-led sentence opener fraction) are higher in Gemma's generations than GPT-2's, with two of four reaching p ≤ 0.05 at n=75 per model.

→ **Non-technical walkthrough (~10 min read):** [`STORY.md`](STORY.md)
→ **Full technical writeup:** [`reports/writeup_v3_revised.md`](reports/writeup_v3_revised.md)
→ **Hero figure:** [`reports/viz_smoking_gun.png`](reports/viz_smoking_gun.png) — single-prompt case study, Gemma vs GPT-2
→ **Fingerprint figure:** [`reports/viz_capital_fingerprint.png`](reports/viz_capital_fingerprint.png) — two specific Gemma features (15596, 10142) act as top-opposers on every capital prompt; GPT-2 has no such consistency
→ **Interactive viewer:** [`apps/grammar_layer/index.html`](apps/grammar_layer/index.html) (load via local HTTP)

---

## What's in this repo

| Path | What |
|---|---|
| `STORY.md` | non-technical walkthrough of the finding |
| `reports/writeup_v3_revised.md` | the canonical writeup with the v3 inversion + controls |
| `reports/findings.md` | accumulated end-to-end findings log |
| `reports/load_bearing_pos10_*.json` | per-prompt causal-ablation results, 7 models × 52 prompts |
| `reports/load_bearing_control_gemma_50.json` | targeting-control: top-10 vs bottom-10 vs random-10 vs all-supporting |
| `reports/load_bearing_mean_ablation_gemma_50.json` | mean-ablation vs zero-ablation comparison |
| `reports/viz_smoking_gun.png` | single-prompt case study (capital-jp, Gemma vs GPT-2) |
| `reports/viz_capital_fingerprint.png` | feature fingerprint across 6 capital prompts |
| `reports/viz_control.png` | targeting control: targeted vs random vs bottom ablation |
| `reports/viz_*.png` | other figures embedded in the writeups |
| `apps/grammar_layer/` | Three.js interactive viewer (UMAP + circuits) |
| `web/` | publication-ready static site |
| `scripts/load_bearing_topk.py` | the script that produced the v3 ablation tables |
| `scripts/load_bearing_control.py` | targeting-control runner (top-10 vs bottom vs random) |
| `scripts/load_bearing_mean_ablation.py` | mean-ablation replication runner |
| `scripts/viz_smoking_gun.py`, `viz_capital_fingerprint.py`, `viz_control.py` | figure generators |
| `scripts/causal_attribution_v2.py` | per-feature attribution (predecessor of load-bearing) |
| `src/neograph/` | Neo4j ingestion library (manifold-fit + relations + steering) |
| `cypher/` | Neo4j schema (constraints + vector indexes) |
| `data/prompts_50.json` | the 50-prompt × 12-category evaluation set |
| `data/causal_prompts.json` | the original 12-prompt set used in v2 |

The substrate name in the code is **neograph** — a multi-relation feature graph stored in Neo4j with GDS. The headline finding came out of running ablation experiments on top of it; the graph is reusable for other cross-model interpretability work.

## The v2 → v3 correction

The v2-era framing — "Gemma routes through a *predicate backbone*, GPT-2 doesn't" — was based on counting how many attribution circuits each feature participated in. Joint-ablating the top-3 of those v2 backbone features produced no behavioral change. The right test is per-prompt: rank features by *signed* Δlog P(target), take the top-10 supporting set, ablate jointly. That set is uniformly load-bearing across every category, and the grammar features that v2 named as "the backbone" turn out to live on the *opposing* side — features that, when ablated, make the target *more* likely. They're load-bearing for what the model **doesn't** say.

See [`reports/writeup_v3_revised.md`](reports/writeup_v3_revised.md) for the full correction.

## Quick reproduce (Gemma 2 2B)

```bash
# 1. Install. Requires uv (https://docs.astral.sh/uv/) and Python 3.12.
uv sync

# 2. Set HF_TOKEN in .env (Gemma 2 is gated on Hugging Face).
echo "HF_TOKEN=hf_..." > .env

# 3. The 50-prompt causal-ablation analysis (~13 min on Apple Silicon MPS).
uv run python scripts/load_bearing_topk.py \
  --model gemma --prompts-file data/prompts_50.json \
  --top-k 10 --sign positive \
  --output reports/load_bearing_pos10_gemma_50.json
```

Other models: pass `--model gpt2 | pythia_70m | gemma_1_2b | qwen3_1_7b | mistral_7b | gemma_9b`. The model specs (HF names, SAE releases, layer picks) are in [`scripts/load_bearing_topk.py`](scripts/load_bearing_topk.py).

## Full substrate setup (optional — only needed for graph queries)

The full Neograph substrate ingests SAE features into Neo4j with co-activation, decoder, and label cosine edges, runs Leiden community detection via GDS, and fits principal-curve manifolds in PCA space. The ablation analysis does *not* require any of this — `scripts/load_bearing_topk.py` is self-contained on top of `sae-lens`.

If you want the graph layer:
```bash
bash scripts/00_bootstrap_neo4j.sh    # local Neo4j 2026.03 + GDS 2026.04 + APOC
uv run python scripts/migrate.py      # constraints + vector indexes
bash scripts/run_all.sh               # full P1→P6 pipeline (~70 min)
```

The local DB ends up around 8 GB and is gitignored.

## Environment

Development happened on an Apple Silicon M5 Max (128 GB unified memory) under macOS. All SAE forward passes use MPS; CPU parity verified at smoke time (max |Δ| ≈ 9.8e-04 on Gemma 2 2B L20 vs CPU). Pythia 70M, GPT-2 small, Gemma 1 2B, Gemma 2 2B, and Qwen 3 1.7B all run comfortably in 128 GB; Mistral 7B and Gemma 2 9B are tight — close other apps. Pythia ≥ 6.9B is untested on this machine.

## Notes for forks / collaborators

- The v2 framing in [`reports/writeup_v2_draft.md`](reports/writeup_v2_draft.md) is preserved as a historical artifact. The v3 revision is the live narrative.
- The dev handover in [`HANDOVER.md`](HANDOVER.md) is from an earlier agent-driven sprint; its priority queue (P1 ablation → P2 50-prompt scale → P3 Pythia scale curve) has since been worked through, with the cross-model breadth expanded to seven model families. Items 2–7 in the writeup's "open follow-ups" section are still open.
- The Neo4j substrate is local-first by design (port 7693, password `neograph_local_dev`). Nothing in the code reaches out to a hosted Neo4j; if you want one, replace `bolt://localhost:7693` in `src/neograph/config.py`.

## License

MIT — see [LICENSE](LICENSE).

## Citation

If this work is useful to you:

```bibtex
@misc{hopkinson2026grammarlayer,
  author = {Hopkinson, Theo},
  title  = {The Grammar Suppression Layer: A causal-ablation study across seven SAE-equipped language models},
  year   = {2026},
  url    = {https://github.com/ho3h/grammar-layer},
  note   = {The writeup at reports/writeup_v3_revised.md is the canonical reference.}
}
```
