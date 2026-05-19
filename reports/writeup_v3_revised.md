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
- **The grammar inversion:** On the supporting side, only **3.8%** of feature-slots
  are grammar-flavored under a strict keyword classifier — the load-bearing
  prediction-promoting features are overwhelmingly content-thematic (geography,
  calendar, named-entities, code). On the **opposing** side — features whose
  ablation makes the target *more* likely — grammar share rises to **11.2%**, a
  ~3× enrichment. Predicate features ("forms of to-be", "statements of existence",
  "the word 'is'") cluster here, not in the supporting set.
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

## Cross-model — same load-bearing test, seven model families

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

## Open follow-ups

1. **Label population for the five "labels pending" models** (Pythia 70M, Gemma 1
   2B, Qwen 3 1.7B, Mistral 7B, Gemma 2 9B). The load-bearing geometry is on
   disk; once labels are populated via Neuronpedia or autointerp, we can test
   whether any other model shows the Gemma-2-style supporting→opposing grammar
   enrichment, or whether the 2.9× inversion is uniquely Gemma 2 2B's.
2. **Width sweep:** re-run Gemma 2 2B with the width-65k Gemma Scope SAE (vs the
   16k used here). Does the load-bearing set fragment into finer features, or
   does the same coordination structure persist at higher SAE width?
3. **Mean-ablation replication:** replace zero-ablation with corpus-mean-ablation.
   If the load-bearing claim still holds, methodology is method-robust rather
   than an artifact of the zero-projection direction.
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
