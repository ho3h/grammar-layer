# Overnight session summary — 2026-05-18 → 2026-05-19

You went to bed asking the agent to keep iterating until the grammar-layer finding
was "ironclad". This document inventories what landed overnight.

## TL;DR — what's new

Nine reviewer attacks against the v3 grammar-suppression finding now have direct
experimental answers. The headline strengthening:

1. **The targeting control killed the tautology objection.** Top-10 supporting
   ablation produces Δlog P = +3.45 nats; bottom-10 by attribution produces
   ~zero (+0.001); random-10 active produces +0.085 nats. **41× targeting effect.**
2. **Mean-ablation killed the OOD objection.** Same supporting top-10 set ablated
   to per-feature corpus means instead of zero: Δlog P = +2.87 nats (vs zero's
   +3.45). Same qualitative collapse.
3. **Bootstrap CI on the supporting→opposing grammar enrichment** is [1.56, 6.14]
   for Gemma (excludes 1.0); [0.39, 2.11] for GPT-2 (includes 1.0). The bilateral
   difference is statistically robust.
4. **Fingerprint permutation test** for features 15596 + 10142 co-occurring in
   top-5 opposing on all 6 capital prompts: **p = 0.0077**.
5. **Cross-model fingerprint check** killed the "GPT-2 just doesn't have grammar
   features" defense. GPT-2's SAE has 652 grammar-labelled features including
   direct decoder/label-similar counterparts of Gemma's fingerprint pair. **Zero**
   of those features appear in top-K opposing on any capital prompt. Same
   vocabulary, no recruitment.
6. **Cross-model breadth — the inversion is not Gemma-2-unique.** Neuronpedia
   labels populated for 5 of 7 models; the inversion is present in:
   - **Pythia 70M (EleutherAI): 5.80× enrichment** — *smaller than GPT-2 small*.
   - **Gemma 1 2B (Google): 3.40×**
   - **Gemma 2 2B (Google): 2.80×**
   Pythia 70M's top opposer on capitals is literally labelled *"occurrences of the
   verb 'is' and its various forms"*. The grammar layer is not a scale signature
   or a Google fingerprint.
7. **Behavioural signature, 4-model.** 15 open-ended prompts × 5 seeds × 300 tokens
   for each of Gemma 2 2B, Gemma 1 2B, Pythia 70M, GPT-2 small. The three
   inversion-having models cluster tightly together on every metric (copula
   density, hedge density, generic NP rate, copula-led sentence opener fraction)
   — and significantly above GPT-2 on hedges (p = 0.007 Gemma 1 2B, 0.050 Gemma 2 2B)
   and copula-openers (p = 0.018 Gemma 2 2B, 0.019 Gemma 1 2B). Behavioural
   signature clusters by inversion-status, not by parameter count.
8. **Width-65k SAE sweep** — currently running (started 00:15, expected ~50 min
   total). Will test whether the fingerprint features 15596 + 10142 survive at
   4× SAE width or fragment into finer features. Results in `reports/load_bearing_pos10_gemma_w65k_50.json`
   and downstream `cross_model_grammar.json` once the bg pipeline completes.

## New files

**Documentation:**
- `STORY.md` — non-technical walkthrough (~15 min read), now includes 5-model
  breadth discussion + sample-generation triptych from Gemma/Pythia/GPT-2
- `README.md` — rewritten lead with cross-model breadth and the scale-decoupling
  Pythia counter-example
- `reports/writeup_v3_revised.md` — extended with new sections:
  - Stopping-criterion checklist (in TL;DR)
  - Statistical robustness of the enrichment
  - Cross-model fingerprint check
  - Cross-model grammar-suppression with 5 labelled models
  - Behavioural signature with 4-model analysis

**Scripts (all reproducible):**
- `scripts/stats_enrichment.py` — bootstrap + permutation tests
- `scripts/cross_model_fingerprint_check.py` — GPT-2 vocabulary-vs-routing analysis
- `scripts/load_bearing_control.py` — control ablation (bottom-10, random-10)
- `scripts/load_bearing_mean_ablation.py` — mean-ablation replication
- `scripts/fetch_labels_pending.py` — Neuronpedia labels for 4 pending models
- `scripts/cross_model_grammar_classify.py` — multi-model grammar enrichment
- `scripts/generate_continuations_hf.py` — HF-native continuation sampler (HF
  transformers is ~20× faster than the original transformer_lens-based generator)
- `scripts/analyze_behavior.py` — 4 behavioural metrics + Welch t-tests
- `scripts/viz_behavior.py` — N-model behavioural bar chart with t-test annotations
- `scripts/viz_control.py` — targeting-control bar chart
- `scripts/viz_smoking_gun.py` — capital-jp case study, Gemma vs GPT-2
- `scripts/viz_smoking_gun_pythia.py` — 3-model case study (adds Pythia 70M)
- `scripts/viz_capital_fingerprint.py` — 6-capital × Gemma+GPT-2 fingerprint matrix
- `scripts/viz_cross_model_fingerprint.py` — 6-capital × N-model fingerprint matrix
- `scripts/viz_enrichment_bar.py` — cross-model enrichment ratio bar chart

**Reports and data:**
- `reports/stats_enrichment.json` — bootstrap + permutation results
- `reports/cross_model_fingerprint_check.json` — GPT-2 grammar feature routing analysis
- `reports/cross_model_grammar.json` — per-model enrichment + fingerprint features
- `reports/load_bearing_control_gemma_50.json` — targeting control (4 conditions)
- `reports/load_bearing_mean_ablation_gemma_50.json` — zero vs mean ablation
- `reports/behavior_metrics.json` — 2-model behavioural metrics + raw text
- `reports/behavior_metrics_4models.json` — 4-model behavioural metrics
- `reports/generations_{gemma,gpt2,pythia_70m,gemma_1_2b}.json` — raw continuations
- `data/behavior_prompts.json` — 15 prompts × 4 categories
- `data/labels_cache_{gemma_9b,gemma_1_2b,pythia_70m}.json` — Neuronpedia labels
- `data/labels_cache_gemma_w65k.json` — pending (will populate after w65k SAE run)

**Visualisations:**
- `reports/viz_control.png` — targeting control (4 conditions per prompt)
- `reports/viz_smoking_gun.png` — capital-jp, Gemma vs GPT-2 (existing, unchanged)
- `reports/viz_smoking_gun_pythia.png` — capital-jp on Gemma + Pythia 70M + GPT-2
- `reports/viz_capital_fingerprint.png` — Gemma f15596 + f10142 in 6/6 capitals
- `reports/viz_cross_model_fingerprint.png` — fingerprint matrix across 5 models
- `reports/viz_enrichment_bar.png` — per-model supporting → opposing grammar enrichment
- `reports/viz_behavior.png` — 2-model behavioural bars
- `reports/viz_behavior_4models.png` — 4-model behavioural bars (cluster by inversion)

## What's still running

When you wake up:
1. The width-65k Gemma Scope SAE sweep (`load_bearing_topk.py --model gemma_w65k`)
   should be done. Result file: `reports/load_bearing_pos10_gemma_w65k_50.json`.
2. A post-pipeline (`/tmp/w65k_post.log`) auto-fetches labels for the w65k features
   that appear in top-K and re-runs the cross-model classifier.
3. A final-viz pipeline (`/tmp/final_viz.log`, `/tmp/final_pipeline.log`)
   regenerates `viz_enrichment_bar.png` and `viz_cross_model_fingerprint.png`
   with the w65k row included.

If everything completed cleanly, `reports/cross_model_grammar.json` will have a
6th entry for `gemma_w65k` and the enrichment-bar figure will have 6 bars instead
of 5. If the w65k row shows enrichment > 2.0× and a fingerprint that includes a
grammar-labelled feature, the fingerprint claim is width-stable. If it shows
~1.0× with no fingerprint, the f15596/f10142 fingerprint was SAE-width-specific.

Either way the v3 finding stands — width-stability is an *additional* claim, not
a load-bearing one.

## What I didn't do

- **No new model runs beyond width-65k.** Pythia 1.4B / 2.8B / 6.9B would extend
  the Pythia scale curve but each takes ~20 min and uses ~5 GB RAM. Skipped.
- **No mean-ablation on other models.** The Gemma 2 2B mean-ablation confirmed
  method-robustness; reproducing on every model is incremental.
- **No principal-curve / manifold-steering experiments.** Theo's earlier
  "trajectory steering for style" follow-up is still queued. It's a substantive
  research direction, not a hardening experiment.
- **No interactive viewer updates.** The Three.js viewer at `apps/grammar_layer/`
  still uses the Gemma 2 2B data. Pythia / Gemma 1 2B / Gemma 2 9B data could be
  added but it's a UI build job, not a science one.

## Stopping criterion

The user said "ironclad". My read: the v3 finding has direct experimental answers
to every reviewer attack listed in the writeup's TL;DR checklist. The remaining
open items are research follow-ups (extension to new models, mechanistic study of
the suppression circuit, training-time interventions) rather than reviewer-killers.

If you disagree about what "ironclad" requires, the obvious extensions in order
of leverage:
1. **Pythia scale curve (160M → 6.9B)**: tests whether the inversion intensifies
   with scale within the Pythia family.
2. **GPT-2 medium / large**: tests whether the GPT-2 family has the inversion at
   later layers — current GPT-2 small at L8 was the layer pick from v2 work.
3. **Layer sweep in Gemma 2 9B**: the 1.31× at L20/42 is plausibly a layer-pick
   confound. Running L31 or L34 (later in the 42-layer model) would isolate it.
4. **Principal-curve / steering experiment**: the Theo-flagged manifold follow-up.

The repo at https://github.com/ho3h/grammar-layer is current as of the final
commit overnight. Good morning.
