# grammar-layer

**A causal-ablation study of next-token prediction across seven SAE-equipped language models.**

Across 52 next-token-prediction prompts spanning 12 task categories, we identify the per-prompt top-10 SAE features whose ablation most reduces log P(target), and joint-zero-ablate them. Every model tested collapses — from Pythia 70M to Gemma 2 9B, Δlog P(target) of 1.7 to 7.3 nats. The most striking single number: in Gemma 2 2B, the load-bearing features that *suppress* the target (the "opposing" set) are **2.9× more grammar-flavored** than the features that *support* it — rising to **16× on capital-completion prompts**.

Gemma 2 2B routes its next-token prediction through a coordinated grammar-suppression apparatus. GPT-2 small does not.

→ **Read the full writeup:** [`reports/writeup_v3_revised.md`](reports/writeup_v3_revised.md)
→ **Interactive viewer:** [`apps/grammar_layer/index.html`](apps/grammar_layer/index.html) (load via local HTTP)

---

## What's in this repo

| Path | What |
|---|---|
| `reports/writeup_v3_revised.md` | the canonical writeup with the v3 inversion |
| `reports/findings.md` | accumulated end-to-end findings log |
| `reports/load_bearing_pos10_*.json` | per-prompt causal-ablation results, 7 models × 52 prompts |
| `reports/causal_circuits_{gemma,gpt2}.json` | attribution circuits for the v2-era analysis |
| `reports/viz_*.png` | figures embedded in the writeups |
| `apps/grammar_layer/` | Three.js interactive viewer (UMAP + circuits) |
| `web/` | publication-ready static site |
| `scripts/load_bearing_topk.py` | the script that produced the v3 ablation tables |
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
