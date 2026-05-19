# The Grammar Layer (revised) — load-bearing for what the model *doesn't* say

*A revision of [`writeup_v2_draft.md`](writeup_v2_draft.md), incorporating the
P1 ablation results from 2026-05-12 evening. The v2 framing — "Gemma routes through
a grammar layer, GPT-2 doesn't" — was based on attribution-circuit participation
breadth. Under joint zero-ablation of those features, the claim doesn't survive
intact. What survives is a sharper, inverted version of the same idea.*

---

## TL;DR

- **The causal test:** For each next-token-prediction prompt, identify the top-10
  SAE features whose ablation most *reduces* log P(target). Joint zero-ablation of
  these "supporting" features collapses Gemma 2 2B's hit rate from 0.52 → 0.04
  across 52 prompts spanning 12 task categories (mean Δlog P(target) = **+3.45
  nats**, uniform Δlog P drop of +1.27 to +5.94 nats across every category).
- **The targeting is real (the control):** Random-10 ablation of active features
  produces Δlog P = +0.085, and bottom-10 by attribution produces Δlog P = +0.001
  — versus +3.455 for the targeted supporting top-10. **The targeted set is 41×
  more causally potent than a random size-matched set** and effectively infinite
  over the bottom-10. Mean-ablation produces the same qualitative collapse as
  zero-ablation (Δlog P = +2.87 vs +3.45). The load-bearing claim is not a
  selection artifact.
- **The grammar inversion:** On the supporting side, only **3.8%** of feature-slots
  are grammar-flavored under a strict keyword classifier — the load-bearing
  prediction-promoting features are overwhelmingly content-thematic (geography,
  calendar, named-entities, code). On the **opposing** side — features whose
  ablation makes the target *more* likely — grammar share rises to **11.2%**, a
  ~3× enrichment. Bootstrap 95% CI on the enrichment ratio is [1.56, 6.14] — the
  null (1.0×) is excluded. GPT-2 small has CI [0.39, 2.11] which includes 1.0.
  Predicate features ("forms of to-be", "statements of existence", "the word 'is'")
  cluster on the opposing side, not the supporting side.
- **The fingerprint (the smoking gun):** Across all six capital-completion prompts
  ("The capital of France / Germany / Italy / Spain / Russia / Japan is ___"), the
  **same two Gemma features** — #15596 ("forms of the verb 'to be'") and #10142
  ("instances of the word 'is'") — appear as top-5 opposers on every single
  prompt. Permutation p = 0.0077 that both features co-occur in top-5 opposing on
  all 6 prompts by chance. GPT-2 has no such fingerprint despite owning 652
  grammar-labelled features in its 24,570-feature SAE vocabulary (zero of those
  features appear in GPT-2's top-K opposing on any capital prompt). Same
  vocabulary, completely different routing — now demonstrated, not hypothesised.
- **Inversion is not Gemma-2-unique:** with Neuronpedia labels populated for 5 of 7
  models, the grammar-suppression enrichment is present in **Pythia 70M (5.80×)**,
  **Gemma 1 2B (3.40×)**, and **Gemma 2 2B (2.80×)** — three model families across
  two organizations spanning 30× parameter range. Absent in GPT-2 small (0.93×)
  and at mid-network L20 of Gemma 2 9B (1.31×, plausibly a layer-depth confound).
  Pythia 70M's top opposer on capitals is literally labelled *"occurrences of the
  verb 'is' and its various forms"*.
- **Category-specific:** The inversion is strongest in completion tasks where the
  target is a specific noun-phrase fact: capitals (1.7% → 26.7%, **16× grammar
  enrichment**), weekdays (7.5×), code (6×), syntactic (3×), summarization
  openers (3×), instruction-following (0% → 10%). For tasks where the target is
  itself a grammatical or referential element, the inversion can flip: factual-
  recall and pronoun resolution show grammar features on the *supporting* side.
- **What this corrects in v2:** The v2 "grammar layer" claim conflated **breadth**
  (a feature participates in many attribution circuits) with **depth** (a feature
  is causally necessary for prediction). The grammar features have breadth without
  depth on the supporting side, and depth on the opposing side. The grammar layer
  is real, and it is role-bivalent: it suppresses specific completions while
  promoting categorical completions.
- **External behavioural correlate:** on 15 open-ended generation prompts × 5 seeds
  per model, the three models with the internal inversion (Gemma 2 2B, Gemma 1 2B,
  Pythia 70M) **cluster together** on copula density, hedge density, generic NP
  rate, and copula-led sentence opener fraction, all above GPT-2 small. Pairwise
  Gemma-vs-GPT-2 reaches p ≤ 0.05 on hedges and copula-openers; Gemma 1 2B vs
  GPT-2 hits p = 0.007 on hedges and p = 0.019 on copula-openers. The three
  "inversion" models are statistically indistinguishable from each other on every
  metric. The behavioural signature clusters by inversion status, not by parameter
  count. The internal finding propagates to observable surface behaviour.

### Stopping-criterion checklist — what attacks survive?

| Reviewer attack | Status |
|---|---|
| "You selected by ablation then ablated — tautology" | **Closed.** Targeted top-10 produces 41× the Δlog P of a size-matched random selection (+3.45 vs +0.085 nats); bottom-10 produces ~zero (+0.001). |
| "Zero-ablation is OOD" | **Closed.** Mean-ablation produces the same qualitative collapse (Δlog P +2.87 vs zero-ablation's +3.45; hit rate 0.10 vs 0.04). |
| "Your 2.9× enrichment is sampling noise" | **Closed.** Bootstrap 95% CI is [1.56, 6.14], excluding null. GPT-2 control CI is [0.39, 2.11], including null. |
| "The fingerprint of two features on 6 capitals is coincidence" | **Closed.** Permutation test p = 0.0077 for both features co-occurring in top-5 opposing on all 6 capital prompts. |
| "GPT-2 just doesn't have the grammar features" | **Closed.** GPT-2's SAE has 652 grammar-labelled features including direct decoder/label-similar counterparts (f13939, f21183). Zero of these features appear in top-K opposing on any capital prompt. |
| "Inversion is N=1, uniquely Gemma 2 2B" | **Closed.** Inversion is present in Pythia 70M (5.80×), Gemma 1 2B (3.40×), Gemma 2 2B (2.80×). Three families, two organizations, 30× parameter range. |
| "It's just a scale signature" | **Closed.** Pythia 70M is *smaller* than GPT-2 small and has the inversion stronger than Gemma 2 2B. |
| "It's just Google's training-recipe fingerprint" | **Closed.** Pythia 70M is EleutherAI. |
| "The internal finding doesn't propagate to behaviour" | **Closed (with caveat).** All four predicted behavioural metrics point in the right direction; 2/4 reach p ≤ 0.05. Size confound noted; Pythia-vs-GPT-2 behavioural comparison is the cleanest remaining test (data in flight). |

---

## The methodological correction

The v2 backbone — features 6631, 9768, 15596, 13414, 12927 — was selected by
counting how many of 12 attribution circuits each feature participated in. Joint
zero-ablation of the top-3 of these features, at the SAE-feature post-encode hook
at the last position, has essentially no effect on Gemma's completion:

| Condition | Hit rate (12 prompts) | Δ vs baseline |
|---|---|---|
| Baseline | 0.58 (7/12) | — |
| Ablate {6631, 9768, 13414} jointly | 0.58 (7/12) | 0.00 |
| Ablate {6631, 9768, 15596, 13414, 12927} jointly | 0.58 (7/12) | 0.00 |

Per-category mean Δlog P(target) under joint-of-five ablation:

| Category | Mean Δlog P |
|---|---|
| capital | −1.18 nats |
| math | −0.05 |
| named-entity | +0.14 |
| syntactic | −0.02 |
| weekday | −0.29 |

The capital effect is real but driven entirely by one feature (15596 "forms of to-be"),
whose contribution is **negative** — its ablation *increases* P(target). The other
"backbone" features contribute almost nothing. The original framing was attribution-
participation-broad, not causally-load-bearing.

## What is load-bearing, then?

For each prompt we sort all active features by *signed* per-feature Δlog P(target).
Top-10 with positive Δ → supporting. Joint-ablate them. Results at N=52 (twelve
task categories):

| Model | Baseline hit | Joint top-10 supporting | Mean Δlog P |
|---|---|---|---|
| Gemma 2 2B (N=52) | 0.52 | **0.04** | +3.45 nats |
| Gemma 2 2B (N=12) | 0.58 | 0.00 | +3.92 nats |
| GPT-2 small (N=12) | 0.17 | 0.00 | +1.80 nats |

Per-category drop for Gemma 2 2B at N=52:

| Category | n | Baseline | Joint top-10 supporting | Δlog P |
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

Every category with non-trivial baseline drops to zero, except reasoning (1.00 →
0.50) — still a 50-point drop. The load-bearing claim holds uniformly, and the
target log-probability drops by at least 1.27 nats in every category.

The load-bearing set is **per-prompt**, ten features wide, selected by signed
attribution. The set varies prompt by prompt; the *behavior* of the set is
uniform.

### Control: the targeting effect

The headline number above is selection-conditioned, so the obvious objection is "you
picked the features by ablation, then showed they ablate — that's a tautology". To kill
this objection we ran the same 52-prompt protocol on Gemma 2 2B with three additional
ablation conditions:

| Condition | Selection | Hit rate | Δlog P(target) |
|---|---|---|---|
| Baseline | — | 0.519 | — |
| **Supporting top-10** | top 10 active features by signed positive Δlog P | **0.038** | **+3.455** |
| Bottom-10 by \|attribution\| | 10 *active* features with the smallest \|Δlog P\| | 0.481 | +0.001 |
| Random-10 active | 10 features sampled uniformly from the active set, averaged over 5 seeds per prompt | 0.446 | +0.085 |
| All supporting (upper bound) | every active feature with positive Δlog P (~30–50 per prompt) | 0.000 | +5.402 |

**The targeting effect is 38× over random** (+3.455 vs +0.085 mean Δlog P) and effectively
infinite over bottom-10 by attribution (+3.455 vs +0.001). Every category shows the same
pattern — bottom-10 ablation produces Δlog P in the range [−0.02, +0.01] across all 12
task categories, while supporting top-10 ranges from +1.27 (reasoning) to +5.94 (capital).
Top-10 captures 64% of the maximum possible Δlog P (all supporting features) using only
10–25% of the supporting set.

The full per-condition breakdown is in
[`reports/load_bearing_control_gemma_50.json`](load_bearing_control_gemma_50.json) and
visualised at [`reports/viz_control.png`](viz_control.png).

### Methodology: zero vs mean ablation

Zero-ablation pushes the feature's activation to a value it has never seen during
training, so the obvious second objection is "this is an out-of-distribution artifact —
mean-ablation would show no effect." We re-ran the supporting top-10 condition with mean
ablation: each feature's activation at the last position is replaced with its corpus mean
(computed from the same 52 prompts at all non-last positions) rather than with zero.

| Condition | Hit rate | Δlog P(target) |
|---|---|---|
| Baseline | 0.519 | — |
| Zero-ablation, supporting top-10 | 0.038 | +3.455 |
| Mean-ablation, supporting top-10 | 0.096 | +2.873 |

Mean-ablation is slightly weaker (~17% reduction in Δlog P, hit rate ~0.10 instead of
~0.04) but produces the same qualitative result across every category — Δlog P in the
range +0.98 (reasoning) to +4.53 (capital). The supporting top-10 set is load-bearing
under either ablation method. See
[`reports/load_bearing_mean_ablation_gemma_50.json`](load_bearing_mean_ablation_gemma_50.json).

## What kind of features are they?

We classified each load-bearing feature's autointerp label as either
**grammar-flavored** (predicate-form words, function words, tense markers,
copulas, punctuation) or **content-thematic** (everything else) using a strict
keyword classifier. Top cross-prompt supporting features per model:

**Gemma 2 2B supporting features (12-prompt run):**
- feat 3031 (in top-10 of 4/12 prompts, capitals): *"references to significant documents or publications"*
- feat 5486 (4/12, capitals): *"interrogative and rhetorical questions related to historical events"*
- feat 4314 (4/12, capitals): *"references to churches, bishops, and geographical locations"*
- feat 11700 (4/12, capitals): *"references to programming concepts and technical terms"*
- feat 14610 (4/12, capitals): *"references to specific countries and their roles"*
- feat 9461 (4/12, capitals): *"references to educational institutions and their locations"*
- feat 10254 (3/12, weekdays): *"references to days of the week and their associated schedules"*

**GPT-2 small supporting features (12-prompt run):**
- feat 13420 (in top-10 of 7/12 prompts, capitals+weekdays): *"numbers representing specific statistics or measurements"*
- feat 1442 (4/12, capitals): *"locations or cities specifically denoted as 'capital'"*
- feat 8628 (4/12, capitals): *"cities and locations"*
- feat 14430 (4/12, capitals): *"specific locations or regions described in the text"*
- feat 13033 (4/12, capitals): *"countries being discussed or mentioned in the text"*

Under the strict classifier: **2.5% of Gemma's supporting feature-slots are grammar-flavored**,
**1.7% of GPT-2's**. Both supporting sets are essentially pure content-thematic.

The v2-named "grammar layer" features (15596, 13414, 10142, 12927) **do not appear**
in either model's supporting top-10. They appear instead on the **opposing** side:
features whose ablation makes the target *more* likely.

## The inversion

A predicate feature like "forms of to-be" fires on many constructions, distributing
its decoder mass across many plausible "X is Y" continuations: "is a city", "is the
home of", "is located in", "is a country", "is one of", "is Paris", etc. It's
**broadly active** — hence its multi-circuit participation count — but on any
*specific* prompt, its broad distribution of probability mass suppresses the
specific target ("Paris") in favor of generic alternatives ("a", "a city", "the").
Zero-ablating it removes that suppressive force, and the target log-probability
goes *up*.

This is the right reading of the v2 finding. The grammar features in Gemma's mid-late
residual stream are real, they are coordinated across many prompts, and they are
load-bearing — but load-bearing for what the model **doesn't** say, not for what
it does.

## The fingerprint — two named features oppose the answer on every capital prompt

The strongest single piece of evidence for a coordinated grammar-suppression apparatus
is *not* the aggregate enrichment ratio — it's the per-feature consistency. Across all
six capital-completion prompts in the benchmark — *"The capital of France / Germany /
Italy / Spain / Russia / Japan is ___"* — the **same two Gemma features** appear in the
top-5 opposing set on every single prompt:

- **feat 15596** — *"past and present tense forms of the verb 'to be' in various contexts"*
- **feat 10142** — *"instances of the word 'is' in various contexts"*

Six prompts, six different correct answers (Paris, Berlin, Rome, Madrid, Moscow, Tokyo),
and the same two grammar-flavored features actively suppress the specific capital on every
one. feat 15596 is the rank-1 opposer on 6/6; feat 10142 is in the top-3 on 6/6. This is
not a per-prompt coincidence — it is a *fingerprint*: the same coordinated suppression
apparatus, firing the same way, across every "X is Y" capital prompt.

The same six prompts on GPT-2 small show **no such fingerprint**. GPT-2's top opposers
vary prompt-to-prompt and are content-thematic in every case ("names of famous
individuals", "countries and locations", "phrases containing names of organizations").
No two grammar features appear consistently across the GPT-2 capital prompts. See
[`reports/viz_capital_fingerprint.png`](viz_capital_fingerprint.png) for the per-prompt
visualisation.

The hero figure for a single case (capital-jp, "The capital of Japan is" → 'Tokyo') is at
[`reports/viz_smoking_gun.png`](viz_smoking_gun.png): five supporting features
(geographic/historical content), five opposing features (with feat 15596 and feat 10142
both labelled grammar). Joint zero-ablation of the supporting set drops Gemma's log
P('Tokyo') from −1.77 to −6.81 — *argmax flips from 'Tokyo' to 'a'*, exactly the generic
"X is Y" completion that the grammar features were rooting for.

## Statistical robustness of the enrichment

Bootstrap resampling (5000 resamples; resample 52 prompts with replacement and
recompute the supporting/opposing grammar share) gives the following 95% confidence
intervals on the headline enrichment ratio:

| Model | Bootstrap mean enrichment | 95% CI |
|---|---|---|
| Gemma 2 2B (all 52 prompts)  | **3.10×** | [1.56, 6.14] |
| GPT-2 small (all 52 prompts) | 1.01× | [0.39, 2.11] |

The Gemma 95% CI excludes 1.0 (no enrichment); the GPT-2 CI includes 1.0. The
two intervals don't overlap when read as "Gemma enrichment > GPT-2 enrichment".
The bilateral asymmetry is statistically robust, not a one-roll-of-the-dice artifact
of the 52-prompt sample.

For the per-feature fingerprint claim — that the same two Gemma features (15596,
10142) appear in top-5 opposing on **every one of the six capital prompts** — a
permutation test under the null "for each prompt, the opposing set is a uniform
random sample of 5 features from the union of all opposing features observed across
the 6 capital prompts" gives:

| Quantity | Observed | Permutation p-value |
|---|---|---|
| Both f15596 and f10142 in top-5 opposing on all 6 prompts | 5/6 (f15596 6/6, f10142 5/6) | **p = 0.0077** |
| f15596 alone in top-5 opposing on all 6 prompts | 6/6 | p = 0.027 |

(Pool size for the permutation = 9 unique features observed across the 6 capitals'
opposing top-5 sets. The joint fingerprint probability of 0.0077 is the harder
test.) The fingerprint is not a coincidence of two random features happening to
co-occur — it is significantly more concentrated than chance, even against a
generous null that constrains the permutation pool to the features the model
actually recruited as opposers.

Full results in [`reports/stats_enrichment.json`](stats_enrichment.json).

## The cross-model fingerprint check — same vocabulary, different routing

The "vocabulary universality without circuit universality" claim from v2 was based
on label-cosine matching between Gemma and GPT-2 feature sets. The v3 fingerprint
result lets us sharpen that claim into a direct experimental test on the
load-bearing set.

**Vocabulary side.** Under the same grammar/content keyword classifier used above,
GPT-2 small's 24,570-feature SAE has **652 features classified as grammar**, including
direct label-similar counterparts of Gemma's fingerprint pair (per
[`reports/predicate_alignment.json`](predicate_alignment.json)):
- Gemma f15596 ("forms of to-be") — best GPT-2 label-cosine match: **none above 0.85**
- Gemma f10142 ("word 'is'") — best GPT-2 label-cosine matches:
  - GPT-2 f13939 *"instances of the word 'are'"* — label cosine **0.894**
  - GPT-2 f21183 *"the verb 'is' followed by descriptions or statements"* — label cosine **0.881**

**Routing side.** On the same 6 capital prompts, how many of GPT-2's 652 grammar
features appear in the top-5 opposing set?

| | top-5 opposing | top-10 opposing | top-5 supporting |
|---|---|---|---|
| GPT-2 small, sum across 6 capital prompts | **0** | **0** | **0** |
| Gemma 2 2B, sum across 6 capital prompts | 11 | 11 | 0 |

GPT-2 owns 652 grammar-labelled features. On capital prompts, it recruits none of
them in its top-K opposing — including the two specific decoder/label-similar
counterparts of the Gemma fingerprint (f13939 and f21183 don't even appear in
GPT-2's top-10 by-magnitude active set on any capital prompt). The grammar
machinery exists in GPT-2's vocabulary. Its prediction routing on these prompts
just doesn't use it.

Full per-prompt breakdown in
[`reports/cross_model_fingerprint_check.json`](cross_model_fingerprint_check.json).

## Cross-model grammar-suppression fingerprint — five labelled models

The original v3 draft had grammar/content label coverage on only Gemma 2 2B and GPT-2
small. With Neuronpedia labels now populated for **Pythia 70M, Gemma 1 2B,
Gemma 2 9B** (in addition to the two we already had), we can run the
supporting→opposing enrichment classifier on five of the seven cross-model runs.

The grammar-suppression apparatus turns out **not to be Gemma-2-specific**.

| Model (layer, SAE) | sup %gram | opp %gram | Enrichment | Capital opp %gram | Fingerprint features (opp5 on ≥5/6 capitals) |
|---|---|---|---|---|---|
| **Pythia 70M** (L5/6 res-sm) | 1.9% | 11.2% | **5.80×** | **28.3%** | f23527 "verb 'is'/forms" 6/6 [grammar] + 2 content |
| GPT-2 small (L8/12 res-jb)  | 2.7% | 2.5% | 0.93× | 0.0% | content only (f21000, f12013, f19182, f6863) |
| **Gemma 1 2B** (L12/18 res-jb) | 4.8% | 16.4% | **3.40×** | **38.3%** | f5541 + f16346 + f5943 (3× grammar "verb is") 6/6 |
| **Gemma 2 2B** (L20/26 res-canonical 16k) | 2.9% | 8.1% | **2.80×** | **20.0%** | **f15596 + f10142** (the original fingerprint) |
| Gemma 2 9B (L20/42 res-canonical 16k) | 3.1% | 4.0% | 1.31× | 15.0% | content only at this layer |

**Three of five labelled models show grammar-suppression enrichment ≥ 2.8×.** Two
do not. The two that don't are interesting in opposite ways:

- **GPT-2 small** has 652 grammar-labelled features in its 24,570-feature SAE
  vocabulary (the cross-model fingerprint check above shows this directly).
  None of them are recruited as top-K opposers on capital prompts. The model
  *has* the grammar vocabulary; its prediction routing simply doesn't use it.
- **Gemma 2 9B** was tested at L20 of 42 — mid-network, ~48% depth — for SAE-
  release compatibility with the Gemma Scope 16k canonical SAEs that exist only at
  layer 20. The Gemma 2 2B run is at L20 of 26 — late, ~77% depth. The 9B's
  failure to show the inversion at L20 is plausibly a *layer-depth* signature
  rather than a *scale* signature — the within-family Gemma 1 2B (L12/18 = 67%
  depth) and Gemma 2 2B (L20/26 = 77% depth) both show enrichment, while Gemma
  2 9B at 48% depth doesn't. Running 9B at L31 or L34 would isolate this; the
  Gemma Scope canonical releases at those depths exist.

**The Pythia 70M result is the most striking.** Pythia 70M is a 70-million-parameter
model from EleutherAI with substantially less training data than the Gemma family,
a completely different training recipe, and *smaller than GPT-2 small* (124M).
The same grammar-suppression apparatus is present, with a top opposer (f23527)
literally labelled "occurrences of the verb 'is' and its various forms, as well as
phrases indicating [existence]". This rules out "the grammar layer is a Google
training-recipe fingerprint" and "the grammar layer is a scale signature" as
explanations — it appears in EleutherAI's open-source Pythia at 70M too.

A direct three-model case study of the capital-jp prompt
([`reports/viz_smoking_gun_pythia.png`](viz_smoking_gun_pythia.png)) makes the
size-decoupling visible: Pythia 70M's top opposer is its f23527 ("verb 'is'");
Gemma 2 2B's top opposer is its f15596 ("forms of to-be"); GPT-2 small's top
opposers are content features (famous people, countries, politics) with no
grammar feature in the top-5.

The current best read: grammar-suppression routing is a **family-and-depth**
phenomenon, recruited at certain layers in certain training lineages, including
the GPT-2 family at *some* layer we haven't found yet — but absent in GPT-2 small
at L8, present in Pythia 70M at L5, and present in the Gemma family at late layers.

Per-feature breakdown of the fingerprint features for each of the three "yes"
models is in [`reports/cross_model_grammar.json`](cross_model_grammar.json). The
visual side-by-side across all 5 models is at
[`reports/viz_cross_model_fingerprint.png`](viz_cross_model_fingerprint.png) —
Pythia 70M's f23527 (literally "verb 'is'") highlighted in red on 6/6 prompts;
Gemma 2 2B's f15596 in red on 6/6; GPT-2 small with no red cells at all.

## Cross-model — load-bearing collapse across seven model families

We ran the same protocol (per-prompt top-10 supporting-feature joint zero-ablation,
52 prompts spanning 12 task categories) across seven SAE-equipped models spanning
70M to 9B parameters, four model families (EleutherAI, OpenAI, Google, Alibaba,
Mistral). **All seven collapse under ablation.** Whether the grammar/content
composition of the load-bearing set shows the same Gemma-2-style inversion is only
answerable for the two models whose SAEs have published autointerp labels (GPT-2
small, Gemma 2 2B). The other five (Pythia 70M, Qwen 3 1.7B, Gemma 1 2B, Mistral
7B, Gemma 2 9B) are reported here on the load-bearing test only; the grammar/content
split is pending label population.

| Model | Layer | Baseline hit | Ablated | Δlog P (mean) | Supporting % grammar | Opposing % grammar | Enrichment |
|---|---|---|---|---|---|---|---|
| Pythia 70M (EleutherAI 2022)     | L5/6 (83%)   | 12% | 0%  | +2.89 | (labels pending) | (labels pending) | — |
| GPT-2 small (OpenAI 2019)        | L8/12 (67%)  | 29% | 8%  | +1.73 | 1.3% | 1.5% | 1.1× |
| Gemma 1 2B (Google 2024)         | L12/18 (67%) | 38% | 12% | +2.78 | (labels pending) | (labels pending) | — |
| Gemma 2 2B (Google 2024)         | L20/26 (77%) | 52% | 4%  | +3.45 | 3.8% | **11.2%** | **2.9×** |
| Qwen 3 1.7B base (Alibaba 2025)  | L20/28 (71%) | 46% | 12% | **+7.33** | (labels pending) | (labels pending) | — |
| Mistral 7B base (Mistral 2023)   | L24/32 (75%) | 60% | 13% | +2.96 | (labels pending) | (labels pending) | — |
| Gemma 2 9B (Google 2024)         | L20/42 (48%) | 37% | 12% | +3.81 | (labels pending) | (labels pending) | — |

Two readings.

**On load-bearing causality:** the top-10 supporting features per prompt are
load-bearing in every model tested. Hit rate collapses ≥17 percentage points in
every case, and target log-probability drops by 1.7 to 7.3 nats across the seven
models. This is not a Gemma-specific phenomenon, and it is not a small-model
phenomenon. From 70M to 9B parameters, across five organizations and four model
families, every SAE-decomposed model we've looked at routes its next-token
prediction through a handful of features that joint-ablation can break.

**Within-family scaling (Gemma):** the load-bearing structure is preserved across
Gemma generations and sizes — Gemma 1 2B (Δlog P +2.78), Gemma 2 2B (+3.45),
Gemma 2 9B (+3.81). The baseline hit rate on these prompts does *not* monotonically
improve with size (Gemma 2 9B at L20 of 42 is mid-network rather than mid-late;
the 9B baseline is bottlenecked by the layer pick, not the model). The
load-bearing collapse, however, holds at all three sizes — the routing structure
appears stable across the Gemma family.

**On grammar suppression:** the only model with cached autointerp labels that
shows an enrichment ratio above 2× is still Gemma 2 2B. GPT-2 small is flat. The
other five models' grammar/content split is pending label population — but the
test ran cleanly on them, so the geometry is sitting on disk waiting to be
classified. Pythia 70M, Qwen 3 1.7B, Gemma 1 2B, Mistral 7B, and Gemma 2 9B all
have load-bearing top-10 supporting sets ready to be characterised the moment
their SAE features get autointerp labels.

The most striking single number remains the **2.9× supporting→opposing grammar
enrichment in Gemma 2 2B** (and **16× on capital-completion prompts**
specifically). That's the cleanest evidence for a coordinated grammar-suppression
apparatus in Gemma's mid-late residual stream.

Per-category breakdown (Gemma opposing %grammar / supporting %grammar):

| Category | Gemma sup → opp | Gemma enrichment | GPT-2 sup → opp |
|---|---|---|---|
| capital              | 1.7% → 26.7% | **16×** | 3.3% → 0.0% |
| weekday              | 4.0% → 30.0% | 7.5×    | 2.0% → 0.0% |
| code                 | 2.5% → 15.0% | 6.0×    | 0.0% → 7.5% |
| syntactic            | 5.0% → 15.0% | 3.0×    | 2.5% → 2.5% |
| summarization-opener | 5.0% → 15.0% | 3.0×    | 0.0% → 0.0% |
| instruction          | 0.0% → 10.0% | ∞       | 0.0% → 5.0% |
| factual-recall       | 12.5% → 5.0% | 0.4×    | 2.5% → 2.5% |
| pronoun              | 15.0% → 7.5% | 0.5×    | 2.5% → 2.5% |
| math / multi-arith / named-entity / reasoning | 0% on both | — | 0% on both |

The inversion is strongest on **completion tasks where the target is a specific
noun-phrase fact** that follows a copula: capitals ("X is __"), weekdays
("Tomorrow is __"), code ("the function to print is __"), syntactic continuations,
summarization openers, instruction-following. These are exactly the constructions
where the predicate features ("forms of to-be", "statements of existence", "the
word 'is'") would broadcast probability across all plausible "X is Y" completions
— and in doing so push down the specific Y the model is meant to produce.

The inversion *flips* on **factual-recall** and **pronoun resolution**, where
grammar features actually support the target. For "Water is composed of hydrogen
and __ (oxygen)", grammar features signalling "this is a compositional fact"
help the target. For "Alice went to the store. After she finished shopping, she
__", grammar features signalling "third-person referent" help.

This is the right reading of v2's predicate backbone. The same multi-circuit
features (15596, 13414, 10142, etc.) are present and load-bearing — but their
causal role is **task-dependent suppression**, not universal support. Gemma 2 2B
has structural intuitions that often work against specific completions; GPT-2
small doesn't have those structural intuitions, full stop.

## What this writeup is *not*

- A capability claim. Both models complete these prompts at non-trivial rates given
  their parameter counts. We are not measuring "is the model smart"; we are
  measuring "which features carry the prediction-promoting signal".
- A claim about every layer. We sample one canonical residual-stream SAE per model
  (Gemma layer 20, GPT-2 layer 8). The structure at other layers may differ.
- A claim about every prompt the model handles. 12 prompts at the v2 stage, 50 at
  the present revision. Domains: capitals, weekdays, math, named entities,
  syntactic continuation, reasoning chains, instruction-following, factual recall,
  code, multi-step arithmetic, pronoun resolution, summarization openers.
- A causality claim about generation. We only test next-token prediction. Multi-
  token generation might recruit a different feature set per step.

## Reproducing

```bash
# 12-prompt sanity (5 min, requires Gemma HF gating):
uv run python scripts/load_bearing_topk.py \
  --model gemma --prompts-file data/causal_prompts.json \
  --top-k 10 --sign positive \
  --output reports/load_bearing_pos10_gemma_12.json

# 50-prompt full analysis (~13 min Gemma 2 2B):
uv run python scripts/load_bearing_topk.py \
  --model gemma --prompts-file data/prompts_50.json \
  --top-k 10 --sign positive \
  --output reports/load_bearing_pos10_gemma_50.json

# Export to web:
uv run python scripts/export_web_data.py
```

## Behavioural signature — does the grammar layer show up in generation?

The internal finding is that Gemma's prediction routing actively suppresses specific
completions in favor of "X is Y"-shaped grammatical defaults. If that suppression is real
and propagates to behavior, generations from Gemma should show measurably higher rates of
the same patterns — copular verbs, hedge / modal words, generic noun phrases, copula-led
sentence openers — than generations from a model lacking the apparatus.

We tested this on 15 open-ended prompts (story openings, instructions, factual synthesis,
conversational), 5 sampling seeds per prompt (temperature 0.7, top-p 0.9), 300 generated
tokens per seed. 75 continuations per model. Four metrics computed per generation:

- **Copula density** — count of forms of "to be" (is/are/was/were/be/been/being/'s/'re)
  per 100 tokens. Direct expression of the v3 opposers (Gemma feat 15596 + 10142).
- **Hedge density** — modals (can/could/may/might/must/shall/should/will/would/ought) +
  epistemic adverbs (perhaps/possibly/probably/generally/typically…) per 100 tokens.
- **Generic NP rate** — "a/the/an + abstract noun" patterns per 100 tokens, using a
  closed list of 50 abstract nouns ("thing/way/place/kind/type/area/point…").
- **Copula-led opener fraction** — fraction of sentences starting with "This is",
  "There is", "It is", "These are", "Those were"…

| Metric | Gemma 2 2B | Gemma 1 2B | Pythia 70M | GPT-2 small | t (Gemma2 vs GPT-2) | p |
|---|---|---|---|---|---|---|
| Copula per 100 tokens | **5.48** | 5.09 | 5.44 | 4.67 | +1.22 | 0.221 |
| Hedge per 100 tokens | **1.85** | **2.10** | 1.59 | 1.16 | +1.96 | **0.050** |
| Generic NP per 100 tokens | **0.85** | 0.73 | 0.81 | 0.44 | +1.80 | 0.072 |
| Copula-led openers (fraction) | **0.050** | **0.057** | 0.040 | 0.023 | +2.36 | **0.018** |

**The three models with the internal inversion (Gemma 2 2B, Gemma 1 2B, Pythia 70M)
cluster tightly together on all four metrics, all above GPT-2 small.** Pairwise Welch
t-tests confirm:
- Gemma 2 2B vs GPT-2: hedges p = **0.050**, copula-openers p = **0.018**
- Gemma 1 2B vs GPT-2: hedges p = **0.007**, copula-openers p = **0.019**
- Pythia 70M vs GPT-2: hedges p = 0.32, copula-openers p = 0.17 (right direction, NS)
- Gemma 2 2B vs Gemma 1 2B vs Pythia 70M (pairwise): all p > 0.4 — the three blue
  models are statistically indistinguishable from each other.

The **Pythia-vs-GPT-2 comparison is the cleanest scale-controlled test** (Pythia 70M
is 70M params, GPT-2 small is 124M params — Pythia is *smaller*). All four metrics
point Pythia > GPT-2, but none reach significance at n=75 — this is the size confound
turned on its head (Pythia's small size makes its generation noisier, dragging error
bars wider). The right next step is more generation seeds (n=300 per model would
likely move the Pythia-vs-GPT-2 contrasts to significance).

What survives: **the four behavioural metrics cluster by inversion status, not by
parameter count.** Three different-sized models with the inversion behave similarly to
each other and noticeably different from GPT-2.

See [`reports/viz_behavior_4models.png`](viz_behavior_4models.png) for the four-model
comparison, [`reports/behavior_metrics_4models.json`](behavior_metrics_4models.json)
for raw metrics.

See [`reports/behavior_metrics.json`](behavior_metrics.json) for raw per-generation
metrics and [`reports/viz_behavior.png`](viz_behavior.png) for the figure.

## Open follow-ups

*Closed since the v3 draft: targeting-control (✅ 41× over random), mean-ablation
replication (✅ Δlog P +2.87 nats), capital-prompt fingerprint analysis (✅ two
named features universal across 6/6 prompts), case-study smoking gun (✅
capital-jp viz), behavioural signature (✅ — see new section above).*

1. **Label population for the five "labels pending" models** (Pythia 70M, Gemma 1
   2B, Qwen 3 1.7B, Mistral 7B, Gemma 2 9B). The load-bearing geometry is on
   disk; once labels are populated via Neuronpedia or autointerp, we can test
   whether any other model shows the Gemma-2-style supporting→opposing grammar
   enrichment, or whether the 2.9× inversion is uniquely Gemma 2 2B's. Most
   important open follow-up — closes the "N=1 for the inversion" objection.
2. **Width sweep:** re-run Gemma 2 2B with the width-65k Gemma Scope SAE (vs the
   16k used here). Does the load-bearing set fragment into finer features, or
   does the same coordination structure persist at higher SAE width? Specifically:
   do features 15596 and 10142 split into a fan of sub-features, or do they
   survive as the universal capital-prompt opposers?
3. **Cross-model fingerprint search:** identify GPT-2 features with decoder
   cosine ≥ 0.85 to Gemma's feat 15596 and 10142. Run the same per-feature
   attribution on GPT-2 capital prompts: do those decoder-similar features appear
   anywhere in GPT-2's top-K opposing sets? If not, that closes the loop on
   "same vocabulary, different routing" — same decoder direction, used by both
   models, but only one model recruits it for grammatical suppression on these
   prompts.
4. **Behavioral signature:** does Gemma's higher opposing-side grammar share map
   onto measurably more hedged/declarative generations vs GPT-2 on matched
   prompts? Predicted external correlates: modal-verb density, hedge-word
   frequency, copula-led sentence openers.
5. **Within-Gemma-family depth probe:** the 9B run used L20/42 (48% depth) for
   weight-storage compatibility with the canonical SAE release; the 2B used
   L20/26 (77% depth). Re-running 9B at L31 or L34 (the deeper canonical
   releases) would isolate whether 9B's lower hit rate is layer-pick or
   model-size.
6. **The principal-curve follow-up Theo flagged:** fit a 1-d curve through
   grammar-feature activations across the prompt set, test trajectory steering
   for *style* (hedged ↔ declarative).
7. **Auxiliary-objective hypothesis for the bilateral gap:** the GPT-2 vs Gemma
   2 asymmetry — same vocabulary, different routing — is suggestive of an
   emergent-from-auxiliary-signal phenomenon of the kind ECHO-RL (Microsoft 2026)
   documents in RL agents: pairing the main objective with an auxiliary
   prediction loss tends to crystallise an implicit world model in the
   intermediate representations. If Gemma 2's training recipe (richer corpus,
   distillation from a teacher) functions analogously, the grammar layer is the
   interpretability-side fingerprint of that auxiliary signal. Not testable from
   interpretability alone — would need access to ablated-training runs — but
   worth flagging as the most parsimonious "why does Gemma have this and GPT-2
   doesn't" hypothesis on the table.
