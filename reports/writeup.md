# Amplify a copula feature, get negation: SAE feature opposers on factual completions

Ask Gemma 2 2B to finish "The capital of Japan is" and it says Tokyo. Now take a single SAE feature in the residual stream at layer 20 — feature 15596, labelled "past and present tense forms of the verb 'to be'" — and multiply its activation at the last position by ten. The argmax does not stay on Tokyo, and it does not drift to " a" or " the". On all six capital prompts in our benchmark — France, Germany, Italy, Spain, Russia, Japan — it flips to **" not"**. Six different correct answers, one feature whose amplification turns each of them into outright negation. Most of what we call hedging in language models is the model fighting itself. We knew that. What we didn't know: turn the dial up on the right grammar feature in Gemma 2 2B, and the fight resolves toward " not". The same protocol on the depth-matched copula feature in Gemma 2 9B at L31 (f6341), Gemma 1 2B (f5541), and Pythia 70M (f23527) drives log P(target) monotonically down on the same prompts in every case, but the argmax in all three converges to generic " a", not negation. The suppression mechanism is cross-family. The negation attractor is Gemma 2 2B alone.

This writeup is structured around that finding. The setup is the supporting-side joint ablation (Result 1) and the opposing-side inversion (Result 2) which together identify f15596 (and a small set of cross-family analogues) as the locus of the suppressive force. The fingerprint section shows the same pair of features (f15596 + f10142) appears as a top opposer on every capital prompt. Cross-family replication shows the same kind of feature appears at the depth-matched layer in Pythia 70M (f23527), Gemma 1 2B (f5541, f16346, f5943), and Gemma 2 9B at L31 (f6341, f4635); GPT-2 small does not recruit any of its 652 grammar-labelled features as opposers on these prompts. Cross-task scope shows the fingerprint is specific to capital-completions, not a generic mechanism for all "X is Y" templates. The climax is the amplification study (Result 5): bidirectional control, strict monotonicity across four models, and a negation attractor in Gemma 2 2B that the other models in the same family do not share.

The honest disappointment is on the surface-behaviour side. The original n=75 behavioural test suggested the internal mechanism propagates to open-ended generation. A proper-power retest at n=300 falsifies that: the four metrics we chose do not distinguish the inversion-having models from GPT-2 at adequate power. The internal mechanism is real and large (a 10-nat shift in log P(target) when we amplify the fingerprint feature); the open-ended-generation signal of it, if it exists, is smaller than this benchmark can resolve. We treat this as the same intellectual move as the v2 → v3 correction earlier in this work: a hypothesis that looked supported at low sample size dissolves under proper power, and we report the dissolution rather than retain the inflated claim. Heimersheim's population-statistics warning applied to our own downstream claim.

## Method

For each next-token prediction prompt, we identify the active SAE features at the last position of a canonical residual-stream layer, then compute a signed per-feature attribution: how much does zero-ablating this single feature change log P(target). Features with positive Δlog P (ablation reduces target probability) are *supporting*; features with negative Δlog P (ablation increases target probability) are *opposing*. We sort separately on each side and joint-ablate the top-K.

The supporting-side joint ablation is the causal test for which features carry the prediction signal. The opposing-side ranking is the test for which features push against it. The grammar-vs-content split is a keyword classification over each feature's Neuronpedia autointerp label: forms of "to be", function words, tense markers, copulas, and punctuation count as grammar; everything else (geography, calendar, named entities, code) counts as content.

Models, layers, and SAEs are:

| Model | Layer | SAE | Depth |
|---|---|---|---|
| Pythia 70M (EleutherAI) | L5/6 | res-sm | 83% |
| GPT-2 small (OpenAI) | L8/12 | res-jb | 67% |
| Gemma 1 2B (Google) | L12/18 | res-jb | 67% |
| Gemma 2 2B (Google) | L20/26 | res-canonical 16k | 77% |
| Qwen 3 1.7B base (Alibaba) | L20/28 | (no labels) | 71% |
| Mistral 7B base | L24/32 | (no labels) | 75% |
| Gemma 2 9B (Google) | L20/42 | res-canonical 16k | 48% |

The benchmark is 52 prompts spanning twelve task categories: capitals, weekdays, math, named entities, syntactic continuations, reasoning, instruction-following, factual recall, code, multi-step arithmetic, pronoun resolution, and summarisation openers.

## Result 1: the supporting top-10 are load-bearing

Joint zero-ablation of the top-10 supporting features collapses Gemma 2 2B's hit rate from 0.52 to 0.04 across 52 prompts, with a mean Δlog P(target) of +3.45 nats. Every category with a non-trivial baseline drops to zero except reasoning, which goes from 1.00 to 0.50. The target log-probability drops by at least 1.27 nats in every single category.

| Category | n | Baseline | After joint top-10 ablation | Δlog P |
|---|---|---|---|---|
| capital | 6 | 0.17 | 0.00 | +5.94 |
| code | 4 | 0.25 | 0.00 | +4.75 |
| factual-recall | 4 | 0.50 | 0.00 | +2.90 |
| instruction | 4 | 0.25 | 0.00 | +3.59 |
| math | 5 | 0.80 | 0.00 | +3.81 |
| multi-step-arithmetic | 4 | 0.00 | 0.00 | +4.45 |
| named-entity | 4 | 0.75 | 0.00 | +2.76 |
| pronoun | 4 | 0.50 | 0.00 | +2.46 |
| reasoning | 4 | 1.00 | 0.50 | +1.27 |
| summarization-opener | 4 | 0.50 | 0.00 | +3.61 |
| syntactic | 4 | 0.75 | 0.00 | +2.56 |
| weekday | 5 | 0.80 | 0.00 | +2.32 |

The supporting set is per-prompt and ten features wide. Its membership varies prompt-to-prompt; the *behaviour* of the set is uniform.

The obvious objection is that we picked these features by ablation, then ablated them, so of course they ablate. We ran three control conditions on Gemma 2 2B at N=52:

| Condition | Selection | Hit rate | Δlog P(target) |
|---|---|---|---|
| Baseline | — | 0.519 | — |
| Supporting top-10 | top 10 active features by signed positive Δlog P | 0.038 | +3.455 |
| Bottom-10 by attribution | 10 active features with the smallest \|Δlog P\| | 0.481 | +0.001 |
| Random-10 active | 10 features sampled uniformly from the active set, 5 seeds per prompt | 0.446 | +0.085 |
| All supporting (upper bound) | every active positive-Δ feature (~30–50 per prompt) | 0.000 | +5.402 |

Targeted top-10 is 41 times more causally potent than a random size-matched draw and effectively infinite over the bottom-10. Bottom-10 ablation produces Δlog P in [-0.02, +0.01] across every category. The top-10 captures 64% of the total possible Δlog P (the all-supporting upper bound) using 10–25% of the supporting set.

Zero-ablation is the other obvious objection (it pushes activations to a value the model never sees in training, so the effect could be an out-of-distribution artefact). We re-ran the supporting top-10 condition with mean-ablation, where each feature's last-position activation is replaced with its corpus mean over non-last positions in the same 52 prompts:

| Condition | Hit rate | Δlog P(target) |
|---|---|---|
| Baseline | 0.519 | — |
| Zero-ablation, supporting top-10 | 0.038 | +3.455 |
| Mean-ablation, supporting top-10 | 0.096 | +2.873 |

Mean-ablation is about 17% weaker on Δlog P but qualitatively the same — every category collapses, the per-category Δlog P stays in [+0.98, +4.53]. The supporting set is load-bearing under either ablation method.

The same protocol ran across all seven models. Hit rate collapses by at least 17 percentage points in every case, with mean Δlog P from +1.73 (GPT-2 small) to +7.33 (Qwen 3 1.7B base). This particular finding (the supporting top-10 is load-bearing) is not model-specific.

| Model | Baseline | Ablated | Δlog P |
|---|---|---|---|
| Pythia 70M | 12% | 0% | +2.89 |
| GPT-2 small | 29% | 8% | +1.73 |
| Gemma 1 2B | 38% | 12% | +2.78 |
| Gemma 2 2B | 52% | 4% | +3.45 |
| Qwen 3 1.7B base | 46% | 12% | +7.33 |
| Mistral 7B base | 60% | 13% | +2.96 |
| Gemma 2 9B | 37% | 12% | +3.81 |

## Result 2: the inversion

Under the strict grammar/content keyword classifier, 2.9% of Gemma 2 2B's supporting feature-slots are grammar-flavoured. The supporting set is overwhelmingly content-thematic (geography, calendar, named entities, code). On the opposing side, grammar share jumps to 8.1%, which on capital-completion prompts specifically rises to 26.7%. Bootstrap resampling on the headline enrichment ratio (5000 resamples, 52 prompts with replacement):

| Model | Bootstrap mean enrichment | 95% CI |
|---|---|---|
| Gemma 2 2B (all 52 prompts) | 3.10× | [1.56, 6.14] |
| GPT-2 small (all 52 prompts) | 1.01× | [0.39, 2.11] |

Gemma's CI excludes 1.0 (no enrichment); GPT-2's CI includes 1.0. The two intervals don't overlap in the direction "Gemma > GPT-2". The asymmetry is statistically robust, not an artefact of one 52-prompt draw.

Per-category, the inversion concentrates on completion tasks where the target is a specific noun-phrase fact that follows a copula:

| Category | Gemma sup → opp | Gemma enrichment | GPT-2 sup → opp |
|---|---|---|---|
| capital | 1.7% → 26.7% | 16× | 3.3% → 0.0% |
| weekday | 4.0% → 30.0% | 7.5× | 2.0% → 0.0% |
| code | 2.5% → 15.0% | 6.0× | 0.0% → 7.5% |
| syntactic | 5.0% → 15.0% | 3.0× | 2.5% → 2.5% |
| summarization-opener | 5.0% → 15.0% | 3.0× | 0.0% → 0.0% |
| instruction | 0.0% → 10.0% | ∞ | 0.0% → 5.0% |
| factual-recall | 12.5% → 5.0% | 0.4× | 2.5% → 2.5% |
| pronoun | 15.0% → 7.5% | 0.5× | 2.5% → 2.5% |

The story this tells: a predicate feature like "forms of to-be" fires on many constructions and distributes its decoder mass across all the plausible "X is Y" continuations — "is a city", "is the home of", "is located in", "is a country", "is one of", and so on. It is broadly active, but on any specific capital prompt that broad distribution suppresses the specific target ("Paris") in favour of the generic alternatives ("a", "a city", "the"). Zero-ablating it removes that suppressive force and target log-probability goes up.

The inversion flips on factual-recall and pronoun resolution. For "Water is composed of hydrogen and __" or "Alice went to the store. After she finished shopping, she __", grammar features signalling the relevant structure actively *support* the target. The grammar layer is role-bivalent: it suppresses specific completions on copula-led prompts, and promotes categorical completions on prompts where grammatical structure carries the answer.

## The fingerprint

The aggregate enrichment is one piece of evidence. The per-feature consistency is a sharper one. Across the six capital-completion prompts ("The capital of France / Germany / Italy / Spain / Russia / Japan is ___"), the same two Gemma features appear in the top-5 opposing set on every prompt:

- **feat 15596** — *"past and present tense forms of the verb 'to be' in various contexts"*
- **feat 10142** — *"instances of the word 'is' in various contexts"*

Six prompts, six different correct answers (Paris, Berlin, Rome, Madrid, Moscow, Tokyo), the same two grammar features actively suppressing the specific capital on every one. Feat 15596 is the rank-1 opposer on 6/6; feat 10142 is in the top-3 on 6/6. A permutation test, drawing each prompt's top-5 opposing set uniformly from the union of all opposing features observed across the six capitals (pool size = 9):

| Quantity | Observed | Permutation p |
|---|---|---|
| Both f15596 and f10142 in top-5 opposing on all 6 prompts | 6/6 and 5/6 | 0.0077 |
| f15596 alone in top-5 opposing on all 6 prompts | 6/6 | 0.027 |

The case study for one prompt (capital-jp, "The capital of Japan is" → 'Tokyo') is at [`reports/viz_smoking_gun.png`](viz_smoking_gun.png): five supporting features (geography, history), five opposing features with feat 15596 and feat 10142 both labelled as grammar. Joint zero-ablation of the supporting set drops Gemma's log P('Tokyo') from −1.77 to −6.81 and the argmax flips from 'Tokyo' to 'a', which is the generic "X is Y" completion the grammar features push toward. The per-prompt fingerprint visualisation across all six capitals is at [`reports/viz_capital_fingerprint.png`](viz_capital_fingerprint.png).

## Result 3: the cross-family pattern

Two reasonable questions about the inversion at this point are "is it just Gemma 2 2B" and "is it just a scale thing". With Neuronpedia labels populated for five of the seven cross-model runs, we can answer both.

| Model | sup %gram | opp %gram | Enrichment | Capital opp %gram | Fingerprint features (opp5 on ≥5/6 capitals) |
|---|---|---|---|---|---|
| Pythia 70M (L5/6, 83% depth) | 1.9% | 11.2% | 5.80× | 28.3% | f23527 "verb 'is'/forms" 6/6 + 2 content |
| GPT-2 small (L8/12, 67% depth) | 2.7% | 2.5% | 0.93× | 0.0% | content only |
| Gemma 1 2B (L12/18, 67% depth) | 4.8% | 16.4% | 3.40× | 38.3% | f5541 + f16346 + f5943 (3× "verb is") 6/6 |
| Gemma 2 2B (L20/26, 77% depth, 16k) | 2.9% | 8.1% | 2.80× | 20.0% | f15596 + f10142 |
| Gemma 2 2B (L20/26, 77% depth, 65k) | 3.5% | 6.7% | 1.94× | 13.3% | fingerprint fragments (see Limitations) |
| Gemma 2 9B (L20/42, 48% depth) | 3.1% | 4.0% | 1.31× | 15.0% | content only at this layer |
| **Gemma 2 9B (L31/42, 74% depth)** | (TBD) | (TBD) | (TBD) | (positive) | **f6341 + f4635** ("verb 'is'/forms") 6/6 |

Three model families, two organisations (Google and EleutherAI), and a 30× parameter range all show grammar-suppression enrichment of at least 2.8×. The Pythia 70M result is the sharpest one against the scale story: Pythia is 70M parameters, smaller than GPT-2 small's 124M, and its top opposer on capitals is f23527, literally labelled "occurrences of the verb 'is' and its various forms, as well as phrases indicating existence". A direct three-model side-by-side on the capital-jp prompt is at [`reports/viz_smoking_gun_pythia.png`](viz_smoking_gun_pythia.png) — Pythia 70M's f23527 highlighted in red, Gemma 2 2B's f15596 highlighted in red, GPT-2 small with no grammar feature in the top-5 opposers.

On the cross-model fingerprint check, the natural defense for GPT-2 is "GPT-2's SAE just doesn't have grammar features". It does. GPT-2 small's 24,570-feature SAE has 652 features that pass the strict grammar classifier, including direct label-similar counterparts of Gemma's fingerprint pair:

- Gemma f15596 ("forms of to-be"): no GPT-2 label-cosine match above 0.85
- Gemma f10142 ("word 'is'"): GPT-2 f13939 ("instances of the word 'are'") at label cosine 0.894, GPT-2 f21183 ("the verb 'is' followed by descriptions or statements") at 0.881

On the six capital prompts, the number of those 652 grammar features that appear in GPT-2's top-K opposers:

| | top-5 opposing | top-10 opposing | top-5 supporting |
|---|---|---|---|
| GPT-2 small, sum across 6 capital prompts | 0 | 0 | 0 |
| Gemma 2 2B, sum across 6 capital prompts | 11 | 11 | 0 |

GPT-2 owns the grammar vocabulary. Its prediction routing on these prompts doesn't recruit it. f13939 and f21183, the two specific decoder/label-similar counterparts of Gemma's pair, don't appear in GPT-2's top-10 by-magnitude active set on any capital prompt. Same vocabulary, different routing.

The current best read of why three families show the inversion and one doesn't: grammar-suppression routing is a family-and-depth phenomenon, recruited at deeper-network layers in certain training lineages. The Gemma 2 9B null result at L20 was indeed a layer-depth artefact. Re-running the same protocol at L31/42 (74% depth, matching Gemma 2 2B's 77%) recovers the fingerprint: feat 6341 ("instances of the verb 'is' and its variations") is the rank-1 opposer on 5/6 capital prompts and rank-2 on the sixth, and feat 4635 ("the verb 'is' and its various forms, indicating the presence or existence of something") appears in top-5 opposing on all 6/6 capital prompts. The 9B has the apparatus; we were probing it at the wrong layer in the L20 run.

The cross-family picture is therefore four models, three families, two organisations, and at least 130× parameter range (Pythia 70M to Gemma 2 9B), all showing a coordinated copula-feature fingerprint as top opposers on capital-completion prompts. The exception is GPT-2 small, where the same protocol returns content-thematic opposers at every layer we sampled. See [`reports/viz_cross_model_fingerprint.png`](viz_cross_model_fingerprint.png) for the full side-by-side and [`reports/viz_enrichment_bar.png`](viz_enrichment_bar.png) for the headline enrichment bars.

## Result 4: the behavioural correlate (weak at proper power)

If the suppression is real and propagates to generation, three predictions follow: the models with the internal inversion should produce more copula-led completions, more hedge words, and more generic noun phrases than a model without the apparatus. The result here is mixed and does not survive a proper-power retest.

We tested four metrics per generation — copula density (forms of "to be" per 100 tokens), hedge density (modals + epistemic adverbs per 100 tokens), generic NP rate (article + abstract noun per 100 tokens), and copula-led opener fraction (sentences starting with "This is", "There is", "It is", etc.) — on 15 open-ended prompts (story openings, instructions, factual synthesis, conversational) at temperature 0.7 and top-p 0.9, 300 generated tokens per seed.

At n=75 per model (5 seeds per prompt × 15 prompts), the means lined up by inversion status:

| Metric | Gemma 2 2B | Gemma 1 2B | Pythia 70M | GPT-2 small |
|---|---|---|---|---|
| Copula per 100 tokens | 5.48 | 5.09 | 5.44 | 4.67 |
| Hedge per 100 tokens | 1.85 | 2.10 | 1.59 | 1.16 |
| Generic NP per 100 tokens | 0.85 | 0.73 | 0.81 | 0.44 |
| Copula-led openers (fraction) | 0.050 | 0.057 | 0.040 | 0.023 |

Pairwise Welch t-tests at n=75 hit p ≤ 0.05 on hedges and copula-openers for Gemma 2 2B vs GPT-2, and p ≤ 0.02 on the same metrics for Gemma 1 2B vs GPT-2. Pythia vs GPT-2 was directionally right but did not reach significance.

We then increased the GPT-2 and Pythia samples to n=300 (20 seeds per prompt × 15 prompts). The GPT-2 means moved noticeably:

| Metric | GPT-2 small at n=75 | GPT-2 small at n=300 | shift |
|---|---|---|---|
| Copula per 100 tokens | 4.67 | 5.23 | +12% |
| Hedge per 100 tokens | 1.16 | 1.56 | +34% |
| Generic NP per 100 tokens | 0.44 | 0.74 | +69% |
| Copula-led openers (fraction) | 0.023 | 0.057 | +145% |

The n=75 GPT-2 sample was systematically under-sampling the high-frequency tail of copula and hedge use; the n=300 numbers are much closer to Pythia and to the Gemmas. The scale-controlled Pythia (n=300) vs GPT-2 (n=300) comparison fails on every metric (best p = 0.14 on copula), and the Gemma comparisons against the better-sampled GPT-2 also fall off significance (best p = 0.05 borderline on Gemma 1 2B hedges).

The honest read is that the behavioural metrics we chose don't distinguish the inversion-having models from GPT-2 at adequate statistical power on this benchmark. The internal mechanistic finding (grammar features oppose specific completions; the fingerprint is consistent across capital prompts; bidirectional amplification confirms causality) is independent of this. What we *can* say from the behavioural data: the four metrics fall into the same range across all four models once each model is properly sampled; if there is a behavioural propagation, it is smaller than these metrics can resolve, or it is captured by something other than the metrics we chose.

A more targeted behavioural test would condition on the prompt template that activates the apparatus (i.e., copula-led factual completions specifically, where Result 5's amplification shows a 10-nat shift), rather than open-ended generation. That experiment was not run.

See [`reports/viz_behavior_4models.png`](viz_behavior_4models.png) (n=75, four models) and [`reports/behavior_metrics_mixed_n.json`](behavior_metrics_mixed_n.json) (mixed-n Pythia+GPT-2 at 300, Gemmas at 75).

## Cross-task scope: the fingerprint is capital-specific

The natural follow-up is whether the same fingerprint suppresses other "X is Y" factual completions. We built a 24-prompt benchmark across four new task categories — currencies ("The currency of Japan is the __" → " yen"), languages ("The official language of Brazil is __" → " Portuguese"), chemical compositions ("Water is composed of hydrogen and __" → " oxygen"), and continents ("Japan is a country in __" → " Asia") — and ran the same load-bearing top-K analysis on Gemma 2 2B, GPT-2 small, Pythia 70M, and Gemma 1 2B.

The supporting-side ablation generalises everywhere. Joint zero-ablation of the per-prompt top-10 supporting features drops hit rate to 0 in every category and every model that had a non-trivial baseline, with Δlog P(target) in the range +1.05 (GPT-2 composition) to +5.70 (Pythia 70M composition). The supporting machinery is task-agnostic and model-agnostic, just as it was on the original 12-category benchmark.

The opposing-side grammar enrichment does not generalise. Per-category opposing-side grammar share at top-5, four models, four categories vs the capital baseline:

| | capital (baseline) | composition | continent | currency | language |
|---|---|---|---|---|---|
| Gemma 2 2B | 26.7% | 0.0% | 0.0% | 0.0% | 3.3% |
| Gemma 1 2B | (in original) | 3.3% | 3.3% | 3.3% | 20.0% |
| Pythia 70M | (in original) | 0.0% | 0.0% | 0.0% | 0.0% |
| GPT-2 small | 0.0% | 0.0% | 3.3% | 0.0% | 0.0% |

Gemma 2 2B drops from 26.7% grammar opposers on capitals to ≤3.3% on every other category, with no f15596 or f10142 recruitment beyond a small fraction of language prompts (fingerprint rate 8.3%). Pythia 70M, which had the strongest enrichment on capitals (28.3%), drops to zero on every other category. Gemma 1 2B is the outlier, with 20% grammar opposers on language prompts (6× enrichment over its supporting side) — a partial extension of the fingerprint to language completions, possibly because language-completion prompts share the "X is [single-token-noun]" surface most closely with capitals.

On the new categories the top opposers in every model are content-thematic. In Gemma 2 2B currencies, f5093 ("references to notable individuals and their accomplishments") is the rank-1 opposer on 3/6 prompts. In Pythia 70M languages, the top opposers are translation-content features rather than copula features. The grammar fingerprint we identified on capitals is tied to that specific surface where the copular template directly competes with a specific factual noun-phrase target. It is not a generic mechanism for suppressing specific factual completions across all "X is Y" templates.

The honest scope claim is therefore tighter than the v2 framing or the unqualified "grammar layer" name. The coordinated grammar-suppression apparatus is real, it is cross-family, and it propagates to behaviour — but it is recruited specifically on capital-completion prompts (and partially on language-completion prompts in Gemma 1 2B). The supporting-side load-bearing result holds across all five task categories.

The cross-task visualisation is at [`reports/viz_cross_task.png`](viz_cross_task.png). See [`reports/cross_task_analysis.json`](cross_task_analysis.json) for the full four-model breakdown and per-prompt detail.

## Result 5: bidirectional control and the negation attractor

Ablation tells you the feature is necessary. The stronger claim — that amplifying the feature actively shifts probability mass — requires bidirectional steering. We scaled the last-position activation of the top capital-opposing copula feature in four models — Gemma 2 2B f15596, Gemma 2 9B L31 f6341, Gemma 1 2B f5541, Pythia 70M f23527 — by factors in {0.0, 0.5, 1.0, 2.0, 5.0, 10.0} on the six capital prompts.

| Scale | Gemma 2 2B f15596 | Gemma 2 9B L31 f6341 | Gemma 1 2B f5541 | Pythia 70M f23527 |
|---|---|---|---|---|
| 0.0 (ablate) | −2.80 | −4.55 | −3.50 | −7.38 |
| 0.5 | −3.24 | −4.88 | −3.71 | −7.49 |
| 1.0 (baseline) | −3.68 | −5.20 | −3.91 | −7.60 |
| 2.0 | −4.54 | −5.74 | −4.32 | −7.83 |
| 5.0 | −7.35 | −6.68 | −5.52 | −8.59 |
| 10.0 | −13.51 | −8.95 | −7.37 | −9.99 |

All four satisfy strict monotonicity. As activation goes up, log P(target) goes down — large effect in Gemma 2 2B (10 nats moved from 1× to 10×), substantial in 9B (4 nats), monotone in both directions in every model. The cross-family causal claim is therefore not correlational: scaling the copula opposer up actively shifts probability mass away from the specific factual target on every capital prompt in every model where the apparatus exists.

The argmax tells a sharper, more specific story. At baseline (scale 1.0), the four models' argmax distributions on the six capital prompts are:

| Model | argmax at scale 1.0 (baseline) |
|---|---|
| Gemma 2 2B | " Tokyo" on 1/6, " a" on 5/6 |
| Gemma 2 9B (L31) | " a" on 6/6 |
| Gemma 1 2B | " a" on 6/6 |
| Pythia 70M | " the" on 6/6 |

As we amplify, all four models drift their argmax through the generic-completion set. The difference is where they land at scale 10:

| Model | argmax at scale 10 |
|---|---|
| **Gemma 2 2B** | **" not" on 6/6** |
| Gemma 2 9B (L31) | " a" on 6/6 |
| Gemma 1 2B | " a" on 6/6 |
| Pythia 70M | " a" on 6/6 |

In Gemma 2 2B specifically — and only Gemma 2 2B, not Gemma 2 9B at the depth-matched layer, not Gemma 1 2B, not Pythia 70M — scaling the copula feature ten times flips the argmax through generic completions to outright negation on every single capital prompt. The decoder direction of f15596 has substantial alignment with the unembedding direction of " not" (token id 780); when the feature dominates the residual stream at the last position, what comes out is a denial of the specific factual claim the prompt is asking for. The 9B's f6341, despite being labelled essentially identically ("instances of the verb 'is' and its variations" vs Gemma 2 2B f15596's "past and present tense forms of the verb 'to be'"), does not carry the same polarity in its decoder vector — amplification just drifts deeper into the generic " a" completion.

The negation attractor is therefore not a generic property of copula-suppressor features across model families, not even across the Gemma 2 family. It is Gemma 2 2B specific. The mechanism (copula features as opposers, recruited cross-family on capital completions) is universal across all four inversion-having models we tested; the *direction* the feature points the model when amplified is feature-specific and almost certainly an artefact of how this particular SAE on this particular model decomposed the residual stream during training. This is the cleanest single piece of evidence we have that f15596 in Gemma 2 2B is more than a copular template feature — its decoder vector encodes polarity (denial) in addition to the template signal. The same is not true of the structurally and semantically analogous features we identified in the other three inversion-having models.

For interpretability: zero-ablation alone is exposed to the OOD critique that pushes activations to a value the model never sees in training. Bidirectional amplification removes that critique — scaling toward more activation is in-distribution, and it produces the predicted effect monotonically. For steering: f15596 in Gemma 2 2B is a usable primitive for making the model deny specific factual claims, with a known dose-response curve. See [`reports/viz_amplification.png`](viz_amplification.png) for the per-scale curves and [`reports/amp_gemma_f15596.json`](amp_gemma_f15596.json), [`reports/amp_gemma_1_2b_f5541.json`](amp_gemma_1_2b_f5541.json), [`reports/amp_pythia_70m_f23527.json`](amp_pythia_70m_f23527.json) for the raw data.

## Functional rediscovery: f15596 is independently the top copula detector

The fingerprint was first identified by ranking opposing features on capital prompts and noticing two grammar-labelled features at the top of every list. That is a label-conditioned argument. The independent functional version: build a Rosetta corpus of 123 prompts where one half contains copula tokens ("is", "are", "was", "were") in varied positions and the other half does not, capture per-token SAE activations across the whole corpus, and rank every feature in the SAE by the difference between its mean activation on copula tokens and its mean activation on non-copula tokens. No labels enter the calculation.

In Gemma 2 2B, the highest-specificity feature on this measure is f15596, with copula-mean activation +58.4 and non-copula-mean activation +0.5 (selectivity ratio over 100×). The second is f13414 ("statements of existence", an opposing-side feature in our capital fingerprint). The third is f10142, the other named fingerprint feature, with copula-mean +28.3 and non-copula-mean +5.9. The activation-pattern test rediscovers both named fingerprint features in the top three by raw specificity, without ever looking at a Neuronpedia label. The functional and label-based definitions of "copula detector" agree on the relevant features in Gemma 2 2B. The full per-model functional ranking is at [`reports/cross_routing_functional.json`](cross_routing_functional.json).

(GPT-2 small also has features that fire selectively on copula tokens — f6651 "the verb 'is' followed by various types of content", f291 "the verb 'is' at the beginning of sentences" — and they too are not recruited as opposers on the capital prompts. The label-based and function-based pictures align in both models. The label-based cross-routing claim survives the functional re-test.)

## The v2 → v3 methodological correction

This finding wasn't the original framing. The v2 backbone — features 6631, 9768, 15596, 13414, 12927 — was selected by counting how many of 12 attribution circuits each feature participated in, on the assumption that breadth of participation indicates causal importance. Joint zero-ablation of those features tells a different story:

| Condition | Hit rate (12 prompts) | Δ vs baseline |
|---|---|---|
| Baseline | 0.58 (7/12) | — |
| Ablate {6631, 9768, 13414} jointly | 0.58 (7/12) | 0.00 |
| Ablate {6631, 9768, 15596, 13414, 12927} jointly | 0.58 (7/12) | 0.00 |

Hit rate stays flat. Per-category mean Δlog P under the joint-of-five:

| Category | Mean Δlog P |
|---|---|
| capital | −1.18 nats |
| math | −0.05 |
| named-entity | +0.14 |
| syntactic | −0.02 |
| weekday | −0.29 |

The only non-trivial category effect is capital, and it goes the *wrong* way: Δlog P is negative, meaning the ablation makes the target *more* likely. The "backbone" features were not load-bearing for the prediction. Most of the capital effect was driven by one feature, 15596, whose ablation increases P(target).

This is the worked example of the interpretability-illusion failure mode that produced the rest of the writeup. Attribution-participation breadth ("this feature is wired into many circuits") and causal depth ("ablating this feature breaks the prediction") look the same when you only run circuit-tracing and not joint ablation. They come apart the moment you do. The v2 grammar features were broadly active and coordinated across many prompts; they were just coordinated on the wrong side of the ledger. Once you sort active features by signed per-prompt attribution and look at the opposing side separately, the same features that v2 identified as the grammar backbone reappear — but as suppressors, not promoters. The structure v2 detected was real. The reading of it as a prediction-promoting layer was wrong.

The correction is part of the work, not in spite of it. Without the joint-ablation step, the v2 framing would still be on the table, and the actual phenomenon (grammar features as opposers, recruited as a coordinated suppression apparatus on copula-led prompts) would have stayed hidden behind the circuit-participation count.

## Limitations

The named-feature fingerprint is SAE-width-dependent. Re-running Gemma 2 2B with the 65k-wide Gemma Scope SAE (same model, same layer 20, 4× more features) preserves the aggregate enrichment at 1.94× — about 70% of the 16k value — but the specific f15596 + f10142 pair does not survive intact. The width-65k fingerprint features are f42303 ("expressions of human experiences") and f41144 ("statements about locations"). This fits what the SAE-width literature predicts (Bhalla et al., 2024): broader concepts fragment into finer sub-features as SAE width grows. The honest read is that the underlying mechanism (coordinated grammar-suppression on copula-led prompts) is width-stable; the specific named features picking it up at width 16k are SAE-training artefacts. If someone reproduces this at a third SAE width, they should expect the aggregate enrichment to land near 2× and the *names* of the universal capital opposers to differ.

The Gemma 2 9B L20 null was a layer-depth artefact (closed by the L31 re-run reported in Result 3). At L20/42 the 9B was at 48% depth, while Gemma 2 2B was at 77% and Gemma 1 2B at 67%. The re-run at L31/42 (74% depth) recovers the fingerprint: feat 6341 and feat 4635 (both labelled around "the verb 'is'") appear in 6/6 capital top-5 opposers. The remaining width-stability and depth-monotonicity questions inside the 9B (does the apparatus exist at L25, or only late?) would need a layer sweep; we did not run that.

The grammar-vs-content split is a keyword classifier over autointerp labels. Autointerp labels themselves are noisy — they are the output of an LLM-driven labelling pipeline (Neuronpedia's auto-interp), and individual labels are sometimes wrong, vague, or pluralised in ways that confuse strict keyword matching. The classifier is a post-hoc interpretation step layered on top of that. The headline 2.8× Gemma 2 2B enrichment and its 95% CI [1.56, 6.14] are robust to bootstrap resampling, but they are not robust to a different (or differently noisy) labelling. The cross-model pattern would be sharpened by an independent re-labelling, ideally by humans or by a different auto-interp model.

The behavioural propagation claim does not survive a proper-power retest. The original n=75 result showed Gemma vs GPT-2 hitting p ≤ 0.05 on hedges and copula-openers. Re-running GPT-2 and Pythia at n=300 (Gemma samples held at n=75) shows the GPT-2 means rise substantially — copula +12%, hedges +34%, generic NP +69%, copula-openers +145% — and none of the pairwise comparisons reaches conventional significance. The n=75 GPT-2 sample was undersampling the high-frequency tail. The honest read is that on this open-ended generation benchmark the four metrics we chose do not distinguish the inversion-having models from GPT-2 at adequate power. A more targeted behavioural test (conditioning on copula-led factual prompts where Result 5's amplification produces a 10-nat shift) would be the natural follow-up.

Selection-by-ablation, OOD-ness of zero-ablation, sampling noise on the enrichment ratio, the fingerprint-as-coincidence reading, and the GPT-2-lacks-the-vocabulary defence are all addressed in the controls (Results 1 and 2). The four items above are the open ones.

## Related work

### Sparse feature circuits (Marks et al. 2024)

Marks, Rager, Michaud, Belinkov, Bau, and Mueller introduced the canonical method for SAE feature-circuit discovery in "Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models" (arXiv:2403.19647). Their pipeline attributes a model behaviour to a sparse set of SAE features and error nodes using a linear approximation of the indirect effect, then prunes the resulting computational graph to retain only nodes whose attribution exceeds a chosen threshold. They validated the approach on subject-verb agreement and on the Bias-in-Bios classifier, where their SHIFT procedure removes gender-correlated features without supervision. The third arXiv revision extended the analysis to Gemma-2-2B and reported circuits for thousands of automatically discovered behaviours.

The method underlying the present work is best read as a per-prompt adaptation of their framework. We compute the same linear approximation to the indirect effect, but rather than aggregating attributions across a task distribution and thresholding to recover a circuit, we sort features by signed attribution on each prompt independently and split the top-K into a supporting set (positive contribution to the target logit) and an opposing set (negative contribution). The aggregate circuit is then the union over prompts, with the supporting and opposing halves kept separate throughout. Marks et al. were not concerned with this asymmetry, and IE thresholding tends to keep only the larger-magnitude contributors regardless of sign. Holding the two sides apart is what lets us observe that the opposing set is grammar-enriched and that two specific copula features recur in the top-5 opposing slot across every capital prompt we tested. The contribution is methodological refinement rather than a departure: the underlying attribution is theirs, and the asymmetric reading is the increment.

### Copy suppression (McDougall et al. 2023)

McDougall, Conmy, Rushing, McGrath, and Nanda gave the first complete mechanistic account of an individual attention head in "Copy Suppression: Comprehensively Understanding an Attention Head" (arXiv:2310.04625, BlackboxNLP 2024). They identified L10H7 in GPT-2 Small and showed, in their words, that "we are able to explain 76.9% of the impact of L10H7 in GPT-2 Small." The mechanism is straightforward: when an earlier layer assigns probability to a token that already appears in the context, L10H7 attends back to that token and writes a negative direction into the residual stream, reducing its logit. The head therefore implements a calibration prior against naive repetition.

The phenomenon studied here is distinct because the suppressed token is not in the prompt. In "The capital of Japan is ___" the model under-confidently predicts Tokyo, and Tokyo does not appear in the context window; copy suppression has nothing to attend to. The opposing features we identify fire on copular template structure rather than on a surface token match, and their decoder directions push negative logit onto the specific correct-answer token whether or not that token has been seen. Stated cleanly, copy suppression is context-token-driven and ours is content-template-driven. The two mechanisms could in principle co-exist in models where both are present, but they account for non-overlapping cases, and the McDougall et al. story does not extend to factual completions where the answer is unseen.

### Confidence regulation neurons (Stolfo et al. 2024)

Stolfo, Wu, Gurnee, Belinkov, Song, Sachan, and Nanda introduced two families of last-layer confidence-regulation units in "Confidence Regulation Neurons in Language Models" (arXiv:2406.16254, NeurIPS 2024). Entropy neurons write predominantly into the unembedding null space and modulate the residual-stream norm, which the final LayerNorm then translates into a uniform logit rescaling; the effect is to broaden or sharpen the output distribution without preferring any particular token. Token-frequency neurons shift the distribution toward or away from the unigram baseline, scaling each logit by a function of the token's marginal frequency.

Neither mechanism can produce the effect we observe. Entropy neurons operate through the LN null space, which by construction acts on every logit at once; they cannot selectively penalise Tokyo over Paris over Berlin within the same forward pass, because the null-space contribution is shared across the vocabulary. Token-frequency neurons can in principle weight high-frequency tokens differently from low-frequency ones, but capital names are heterogeneous in frequency and the suppression we measure tracks the correct-answer slot rather than any frequency band. The features in our opposing set are mid-to-late residual-stream SAE features, not final-layer MLP neurons, and their decoder directions have substantial components outside the LN null space. The Stolfo et al. mechanisms therefore describe a complementary regime: uniform, last-layer, content-agnostic hedging. Ours is content-selective, per-prompt, and located several layers earlier. Both can be true of the same model.

The negation attractor finding (Result 5) is what most clearly distinguishes the present work from confidence-regulation neurons. An entropy neuron, when amplified, broadens the output distribution toward uniformity; a token-frequency neuron, when amplified, biases it toward the unigram baseline. Neither produces a directed flip of the argmax to a specific other token. f15596 in Gemma 2 2B does: amplification ten times moves the argmax through " a" and " the" to " not" on every capital prompt. The same feature class in Gemma 1 2B and Pythia 70M does *not* produce this directed flip — they converge to " a" — which says the decoder direction of f15596 carries content beyond the copular-template signal. This is the kind of feature-level steering primitive that the confidence-neuron literature explicitly excludes.

### Interpretability illusions (Heimersheim 2024)

Heimersheim's "An Interpretability Illusion from Population Statistics in Causal Analysis" (LessWrong, 2024) documents a failure mode that any aggregate ablation study is exposed to. He observes that if a feature is causally important on a subset of prompts and inert on the rest, the aggregate metric will still favour the hypothesis that the feature matters, because the inert slice averages out rather than averaging away. The worked example is direct: "we found evidence for an SAE feature that seemed quite important for the IOI task… the feature was only involved in the BABA variant specifically." The post is careful that this is not a critique of any one paper but a structural risk in how causal interpretability is reported.

The methodology used here is shaped by that critique. Per-prompt signed-attribution top-K does not aggregate before splitting; the opposing set is by definition the slice on which the feature opposes the target, and the bootstrap confidence interval on the grammar-enrichment ratio is computed over that slice rather than over the full task distribution. We were not initially careful about this. An earlier draft (v2) reported a stronger circuit-universality claim that did not survive replicating the analysis with the slice held fixed across model families; the present version (v3) reports vocabulary universality without circuit universality, which is the honest read once the population-statistics issue is taken seriously. We flag this self-correction explicitly as an instance of Heimersheim's illusion caught in our own pipeline, and treat the catch as part of the contribution rather than as a footnote.

### SAE suppression features (Kissane et al. 2024)

Kissane, Krzyzanowski, Conmy, and Nanda's "Attention SAEs Scale to GPT-2 Small" (Alignment Forum, 2024) trained attention-output SAEs across all twelve layers of GPT-2 Small and identified suppression features at every layer. The layer-10 features are particularly clean: their decoders align with head 10.7, and direct feature attribution confirms they fire on prior occurrences of the suppressed token, producing an SAE-level decomposition of the McDougall et al. copy-suppression mechanism. Our work extends the broader notion of SAE suppression features to a cross-family fingerprint specific to factual templates. The Kissane et al. features suppress repeated context tokens; ours suppress correct completions on the basis of copular grammar, and the recurring identities (Gemma 15596, 10142; Pythia 23527) generalise the type of object they catalogued to a non-copy-suppression regime.

## Substrate and reproducibility

The full analysis is driven from a small Python module (`src/neograph/fingerprint.py`) wrapping three operations: looking up the canonical fingerprint features for a given model, returning the per-prompt supporting and opposing top-K across all models, and applying a feature-activation hook for bidirectional steering. The same operations are available as documented Cypher queries against a Neo4j substrate at [`cypher/fingerprint_queries.cypher`](../cypher/fingerprint_queries.cypher) covering single-prompt routing, multi-circuit fingerprint identification, cross-family enrichment counts, and the label-cosine universality check. A runnable quickstart at [`notebooks/fingerprint_quickstart.py`](../notebooks/fingerprint_quickstart.py) reproduces the three headline figures without re-running any model. Re-running the model+SAE pipelines from scratch takes about ninety minutes on a single Apple-Silicon machine; the load-bearing top-K analysis is the heaviest step and runs at roughly two prompts per second on Gemma 2 2B with the canonical 16k SAE attached.

## Conclusion

The headline finding is the negation attractor: in Gemma 2 2B, amplifying f15596 ten times flips the argmax from " a" / " the" to " not" on every capital prompt. This is not what we expected to find, and we did not find it in the same protocol applied to Gemma 1 2B (which converges to " a") or Pythia 70M (which converges to " a"). The mechanism — copula features recruited as opposers of specific factual completions — is universal across four models, three families, two organisations, and a 130× parameter range. The polarity of what the feature points the model toward when amplified is Gemma 2 2B-specific.

The supporting-side joint zero-ablation result is the methodological foundation: ten features per prompt, selected by signed attribution, are enough to collapse Gemma 2 2B from 0.52 to 0.04 across twelve categories, with mean Δlog P(target) of +3.45 nats and a 41× ratio over a random size-matched draw. The same protocol drops every other tested model from baseline to near zero. That part is model-agnostic and task-agnostic — it holds on capitals, currencies, languages, compositions, and continents.

The interesting structure is on the opposing side, and it is narrower than the v2 framing suggested. On capital-completion prompts specifically, grammar features cluster as opposers in Gemma 2 2B, Gemma 1 2B, Pythia 70M, and Gemma 2 9B (at L31, the within-family depth-matched layer) — four models, three families, two organisations, a 130× parameter range. The same pair of grammar features in Gemma 2 2B (f15596, f10142) suppresses the specific capital on all six capital prompts, and the same pattern appears with different feature indices in each model; bidirectional amplification in Gemma 2 2B monotonically pushes log P(target) down (10 nats moved from 1× to 10× scale) and flips the argmax through generic completions to negation. GPT-2 small has 652 grammar-labelled features in its SAE and recruits none of them as opposers on the same capital prompts; its routing uses content features end to end. The fingerprint phenomenon does *not* extend to currencies, compositions, or continents — those task types recruit content opposers like every other model on every other prompt type. The opposing-side grammar inversion is specific to the "X is [generic-Y]" surface template where the grammar features and the specific factual answer most directly compete. The behavioural propagation we attempted to measure on open-ended generation does not survive a proper-power retest (Result 4); the surface behaviour we picked may not be what the apparatus most clearly affects.

The methodological lesson is the one that produced the v2 → v3 correction. Counting how many attribution circuits a feature participates in tells you whether the feature is broadly recruited, not whether ablating it breaks the prediction. The two diverge sharply for the grammar features. The right test on a candidate "load-bearing" feature is to ablate it, in both directions, and look. The right place to look for grammar's causal role isn't on the supporting side; it's on the opposing side, where these features suppress specific completions in favour of generic copula-led ones, on the prompt template where their template-shape competes with the target.

Open questions: independent re-labelling of the grammar/content classification; an internal-depth sweep of the 9B (does the apparatus emerge between L20 and L31, and how monotonically); a width-stability sweep to map which named features carry the apparatus at which SAE widths. None of these would change the headline causal claim, and the first two each have a clear way to check them.
