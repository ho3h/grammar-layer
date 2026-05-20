# The Grammar Layer

*Two AI systems answer the same 12 questions. They get the same 12 answers. But they
think in completely different ways to get there. Gemma reaches for grammar first —
"this is a statement of existence", "this is a beginning of text", "this is an
authoritative claim" — before it reaches for the facts. GPT-2 goes straight to the facts
and skips the grammar layer entirely. Both arrive at "Paris". Only one of them seems to
know it's making a claim.*

![The Grammar Layer — hero](reports/viz_grammar_layer.png)

*Each point is one SAE feature (a learned "concept" inside the model), UMAP-projected
from its decoder direction. Brightness = how many of 12 next-token-prediction circuits
recruit that feature. Gemma's hot zone (orange labels) is structural — forms of "to be",
statements of existence, beginnings of text. GPT-2's hot zone (blue labels) is
content-thematic — politics, statistics, capital cities, URLs. The vocabulary is the
same in both SAEs. The routing isn't.*

## A brain-scan analogy that holds up

Cognitive neuroscience has been arguing for fifteen years that the brain has two
broad classes of network. The **multi-demand network** (Duncan and colleagues, 2010+)
is a set of frontoparietal regions that activates across diverse cognitive tasks —
arithmetic, language, working memory, reasoning. Same regions light up regardless
of content. It's been called the brain's "executive control" backbone. Distinct from
this are the **task-positive networks**: language network for language, fusiform
face area for faces, spatial network for navigation. Different content, different
regions.

Run the same 12 questions through two AI systems and look at the analogous "scan",
and you see the same dichotomy. **Gemma 2 2B has something that looks like a
multi-demand network in its mid-late residual stream.** A handful of features fire
across capitals *and* weekdays *and* arithmetic *and* named entities *and* syntactic
completion — content-invariant, task-general. **GPT-2 small does not.** Its
features partition by content type: politics features for political prompts, capital-city
features for capital prompts, statistics features for math.

The analogy isn't perfect — our attribution is interventional (zero-ablation patching,
not BOLD signal), our "features" are post-hoc SAE decomposition, and a forward pass
isn't sustained cognition. But the structural finding maps cleanly onto the cognitive-
neuroscience frame, and the frame makes the finding legible: *we ran a functional scan
on two AI brains, and one of them has a multi-demand network.*

We call the structural-feature core **the grammar layer** in this post. In code and
in some figures it's also called the "predicate backbone" — same thing, more
technical term.

## Headline

Across 12 next-token-prediction tasks spanning capitals, weekdays, arithmetic,
named-entity completion, and syntactic continuation, **Gemma 2 2B recruits a small
grammar layer of SAE features** — features for "beginning of text",
"control and authority", "forms of to-be", "statements of existence" — in
**5–12 out of 12 circuits each**. These are not content features; they are
structural features for grammatical and propositional roles.

When we run the same 12 prompts on **GPT-2 small + Joseph Bloom's RES-JB SAEs** and
ask whether GPT-2 has analogous predicate features, the answer is **yes at the
vocabulary level** — GPT-2 has features whose autointerp labels match Gemma's
predicates at cosine 0.87–0.93 — **but no at the circuit level**. None of those
GPT-2 features participate in multiple GPT-2 circuits in our test set.

**Vocabulary universality without circuit universality**: the two SAEs decompose the
same predicate concepts into features, but only one model *routes through* those
features when completing. The other reaches for content-specific features (capital
cities, famous individuals, numerical statistics) instead. That distinction is what
the graph schema made naturally askable; it's the kind of question that requires
multi-circuit, multi-model joins to even formulate.

**We don't read this as GPT-2 failing to use predicate features.** Both models complete
these prompts successfully. We read it as two different routing strategies arriving at
similar outputs from different intermediate representations. The interpretability
finding isn't a capability claim; it's a *cognitive style* claim.

![Vocabulary links across models](reports/viz_vocab_links.png)

*Thin lines connect SAE features whose autointerp labels match across models at cosine
≥ 0.88 — same concepts, both sides. The lit features (bright on each side) cluster
differently. Same words; different sentences.*

## 1. The Gemma grammar layer

Method: zero-ablation patching. For each prompt-target pair, we forward-pass with
Gemma 2 2B + Gemma Scope width-16k SAE attached at L20-resid-post, identify the active
SAE features at the last position (typically 60–90 features), ablate each one, and
measure the change in next-token logit. Positive attribution = feature supports
target; negative = opposes.

12 prompts (`data/causal_prompts.json`):

| Category | Prompts |
|---|---|
| Capitals | France→Paris, Japan→Tokyo, Germany→Berlin, Italy→Rome |
| Weekdays | Mon→Tue, Wed→Thu, Fri→Sat |
| Math | 2+2=4, 3×3=9 |
| Syntactic | "The dog chased the ___" |
| Named entities | Einstein→physicist, Shakespeare→play |

Then this query against the populated `Circuit -[:INCLUDES]-> SAEFeature` schema:

```cypher
MATCH (cir:Circuit)-[inc:INCLUDES]->(f:SAEFeature)
WHERE cir.model = 'gemma'
WITH f, count(DISTINCT cir) AS n_circuits,
     sum(inc.attribution) AS total, avg(abs(inc.attribution)) AS mean_abs
WHERE n_circuits >= 4
OPTIONAL MATCH (f)-[:LABELED_AS]->(a:AutoInterpLabel)
RETURN f.index, n_circuits, total, mean_abs, a.text
ORDER BY n_circuits DESC, mean_abs DESC
```

Result (top 12 by circuit participation):

| Gemma feat | # circuits / 12 | Σ attribution | mean \|attr\| | autointerp label |
|---:|---:|---:|---:|---|
| **6631** | **12** | +2.23 | 0.19 | "the beginning of a text or important markers in a document" |
| **9768** | **11** | +1.21 | 0.12 | "terms related to control and authority, particularly in political or systemic contexts" |
| 15596 | 7 | **−1.86** | 0.35 | "past and present tense forms of the verb 'to be'" |
| 13414 | 7 | +1.29 | 0.22 | "statements of existence or presence" |
| 12927 | 7 | +1.14 | 0.16 | "statements of existence or presence" |
| 1692 | 7 | −0.79 | 0.11 | "legal and technical terminology related to statutes and inventions" |
| 6143 | 6 | +2.13 | 0.36 | "phrases related to medical conditions and treatments" |
| 2021 | 6 | −1.68 | 0.28 | "topics related to political events and elections" |
| 15149 | 6 | −0.82 | 0.16 | "sentences that describe the characteristics or functionalities of structures or systems" |
| 1583 | 6 | +0.32 | 0.08 | "words related to administrative or technical tasks and issues" |
| 6793 | 5 | +1.03 | 0.21 | "elements that resemble structured data or identifiers, likely in JSON format" |
| 10419 | 5 | +0.04 | 0.03 | "technical terms related to scientific research and modeling" |

Read this column by column. **Feat 6631 fires in literally every one of 12 circuits.**
A "beginning of a text" feature is recruited by every completion-style prompt,
regardless of content. **Feat 9768 (control / authority) in 11/12** is similar —
Gemma reaches for it almost universally. **Feat 15596 (forms of "to-be") in 7/12 with
Σattr=−1.86** suggests it's a *competitor* feature: forms of "to be" are themselves
likely completions, and ablating it frees logit mass for the actual content target.
The pattern is consistent across category: capitals, weekdays, math, named entities
all recruit roughly the same backbone.

## 2. The structural cross-model test

For each Gemma predicate feature, we query the `AutoInterpLabel` vector index for
the top-K nearest GPT-2 features (Joseph Bloom's RES-JB SAE at layer 8, 24,576
features). Then for each candidate, ask: *does this GPT-2 feature also participate
in multiple GPT-2 circuits on the same 12 prompts?*

The result is the clean structural distinction:

| Gemma feat | label | best GPT-2 match (cos) | in how many GPT-2 circuits |
|---:|---|---|---:|
| 6631 | "beginning of text or important markers" | *(no GPT-2 feature within k=10 nearest neighbours has an autointerp label about beginnings/markers)* | — |
| 9768 | "control and authority" | gpt2 17182 cos +0.93 "political terms related to control and power" | **0** |
| 9768 | same | gpt2 22079 cos +0.92 "words related to control and authority" | **0** |
| 15596 | "forms of to-be" | gpt2 14993 cos +0.92 "forms of the verb 'to be'" | **0** |
| 13414 | "statements of existence" | gpt2 18741 cos +0.93 "discussing existence or presence" | **0** |
| 13414 | same | gpt2 21721 cos +0.92 "references to existence or presence" | **0** |
| 10142 | "instances of the word 'is'" | gpt2 13939 cos +0.89 "instances of the word 'are'" | **0** |
| 10142 | same | gpt2 21183 cos +0.88 "the verb 'is' followed by statements" | **0** |

**The GPT-2 features encoding Gemma's predicate concepts exist, are easy to find,
and never participate in GPT-2's 12 circuits.**

For comparison, GPT-2's own multi-circuit backbone (≥3 circuits):

| GPT-2 feat | # circuits / 12 | Σ attr | label |
|---:|---:|---:|---|
| 6863 | 9 | −1.49 | "politics and government institutions" |
| 13420 | 7 | +1.82 | "numbers representing specific statistics or measurements" |
| 5858 | 7 | +0.15 | "information related to events or releases" |
| 18220 | 6 | +2.34 | "phrases related to personal preferences or favorites" |
| 22852 | 6 | −0.06 | "URLs" |
| 21000 | 4 | **−4.47** | "names of famous individuals" |
| 1442 | 4 | +3.77 | "locations or cities specifically denoted as 'capital' in the text" |

GPT-2's backbone is **content-thematic** (politics, statistics, preferences, URLs,
named individuals, capital cities) — not the structural / propositional roles that
make up Gemma's backbone. The 1442 ("capital cities") feature is striking: it fires
in exactly the 4 capital-completion circuits, with a +3.77 total attribution. It's a
content-specific feature recruited for content-specific tasks. Gemma has nothing
analogous in its multi-circuit list — instead, capital completion in Gemma is
distributed across the structural backbone plus a few prompt-specific content features.

## 3. Why this is graph-shaped, concretely

The single query that produced the cross-model finding (`scripts/predicate_alignment.py`):

```cypher
MATCH (cir_g:Circuit {model: 'gemma'})-[inc_g:INCLUDES]->(g:SAEFeature)
WITH g, count(DISTINCT cir_g) AS gemma_n
WHERE gemma_n >= 3
MATCH (g)-[:LABELED_AS]->(a:AutoInterpLabel)
CALL db.index.vector.queryNodes('label_emb', 10, a.embedding) YIELD node, score
MATCH (p:SAEFeature)-[:LABELED_AS]->(node)
WHERE p.sae_id = 'gpt2-small-res-jb/L8' AND score >= 0.7
OPTIONAL MATCH (cir_p:Circuit {model: 'gpt2'})-[:INCLUDES]->(p)
WITH g, gemma_n, p, score, count(DISTINCT cir_p) AS gpt2_n
RETURN g.index AS gemma_feat, p.index AS gpt2_feat, score, gemma_n, gpt2_n
ORDER BY gemma_n DESC, score DESC
```

That joins:
- circuits → features (model A)
- features → labels (model A)
- labels → labels (vector index, cross-model)
- labels → features (model B)
- features → circuits (model B)

…in a single declarative query. The same operation in pandas decomposes into five
separate dataframe joins with careful index management. The graph paradigm doesn't
make this answer *possible* — pandas can do it — it makes it *trivial to formulate*,
which is the difference between "asked it" and "didn't think to ask it".

The result that matters isn't the universality verdict per se; it's the *granularity*
of the verdict. Standard cross-model universality work asks "do similar features exist
across models?" (yes, by label cosine). Circuit-level work asks "do features participate
in causally homologous structures?" (no, on this 12-prompt set). The conjunction —
*similar features exist but aren't routed through* — is a distinction that requires
both the feature-graph and the circuit-graph to coexist in one queryable substrate.

## 3a. The layer-asymmetry robustness check

A reasonable critic will object that Gemma L20 is ~77% deep through 26 layers while
GPT-2 RES-JB L8 is ~67% deep through 12 layers. If GPT-2 only develops predicate-feature
circuit recruitment at deeper layers, the bilateral comparison is unfair.

We checked. `scripts/gpt2_layer_sweep.py` runs the same 12-prompt zero-ablation
attribution at GPT-2 layers **4, 8, 10, and 11** (33%, 67%, 83%, 92% depth), pulls
Neuronpedia autointerp for every feature recruited by ≥3 circuits, and compares.
**No GPT-2 layer in the sweep develops a structural-predicate backbone**. The
content-thematic pattern is consistent across depth:

| Layer | Top multi-circuit feature (n_circuits) | Closest predicate-adjacent feature |
|---:|---|---|
| L4  | 20731 — "numerical information / dates / statistics" (11/12) | 19673 — "phrases indicating certainty or assurance" (7/12) |
| L8  | 6863 — "politics and government institutions" (9/12) | *(none in top-30)* |
| L10 | 23638 — "brand names and technical terms" (9/12) | 4198 — "phrases indicating factual information or discussions" (5/12) |
| L11 | 7055 — "coding error messaging" (11/12) | *(none in top-30)* |

The closest GPT-2 ever gets to a predicate backbone is L4's feat 19673 ("certainty or
assurance") in 7/12 circuits. That's the same circuit-count as Gemma's *median* backbone
feature; it's nowhere near Gemma's max of 12/12. And it's a *modal* feature, not a
*propositional* one — "certainty" rather than "to be" or "exists".

The robustness verdict: depth doesn't rescue GPT-2's predicate backbone because there
isn't one to rescue. GPT-2 small's circuits on these 12 prompts are content-thematic
all the way down.

Raw data: `reports/gpt2_layer_sweep.json` + `reports/gpt2_layer_sweep_summary.md`.

## 4. What this writeup is honest about

**The manifold story has gone quiet.** The PRD specced principal-curve fitting and
trajectory steering. The implementation does the math: PCA per community, polygonal-line
fitting, cyclic-spline fallback, waypoint sampling. But the trajectory-steering
experiment on Mon→Tue showed both linear and manifold steers hit 100% on single-step
day prediction — which is the right experimental answer for *clusters*; it doesn't tell
us anything about trajectories the model actually traverses during multi-token
generation. The `LIES_ON` edges in our graph represent feature-to-closest-waypoint
membership; that is more honestly described as cluster membership with geometric
scaffolding, not manifold reconstruction in the Engels-et-al sense. We are not
publishing a manifold result.

**The community-level cross-model alignment was overstated.** A Hungarian assignment
on per-community label-embedding centroids scored cos=0.95 on pairs like
"numerical sports stats" ↔ "sports references", which is MiniLM common-noun overlap,
not structural homology. We replaced it with the structural test above. The 2-of-4
concept-alignment number from the earlier draft (`reports/cross_model_universality.json`)
should be read as "embedding-centroid noise was answering a different question".

**12 circuits is the minimum credible N for the predicate-backbone claim.** It's not
publication-quality; it's enough to know N=2 wasn't coincidence. Scaling to 50 circuits
plus a third model (Pythia 1.4B) is the obvious next step. We didn't run those yet
because the marginal value of running the structural detector on bilateral first is
high — if predicate features had failed to align *even at the label-vocabulary level*,
no amount of amplification would have rescued it.

## 5. Reproduction

Code: `scripts/causal_attribution_v2.py --model {gemma|gpt2}` (~2 min per model on M5
Max MPS for 12 prompts × ~80 features each). Cross-model alignment:
`scripts/predicate_alignment.py` (~10 seconds, vector-index k-NN). Substrate (Neo4j
+ GDS + APOC, schema, capture, ingest, Leiden) takes ~3 hours on first run; cached
parquet + idempotent ingest makes subsequent runs ~30 minutes. Hardware: Apple M5 Max
128 GB. Full pipeline runs locally.

## 6. Next experiments

1. **Pythia 1.4B + EleutherAI SAEs**: triangulate. If Pythia's circuits look like
   Gemma's (predicate-backbone style) or like GPT-2's (content-thematic), that's
   diagnostic. ~30 min ingest, ~5 min causal attribution.
2. **Scale to 50 circuits**. The 12-prompt N rules out coincidence but doesn't have
   the power to say "feat 6631 is universal to Gemma 2 2B's completion-style behaviour".
   50 prompts across more varied domains (instruction-following, reasoning chains,
   factual recall) tests robustness.
3. **GPT-2 layer sweep**. Maybe layer 8 is too early. Run the same 12-prompt
   attribution at GPT-2 layers 0, 4, 8, 10, 11 and see whether predicate features
   start participating in circuits at deeper layers.
4. **Within-model manifold steering on multi-step generation** to test whether the
   geometric scaffolding earns its name. Score sentence-level embedding smoothness
   across 5-token continuations, not single-step targets.

## 7. What this writeup is not

- Not a manifold paper. We have community clusters with waypoint scaffolding.
- Not a definitive cross-model universality claim. Bilateral evidence for vocabulary
  universality + bilateral evidence against circuit universality at this N.
- Not a graph-database advocacy piece. The pandas-equivalent of every query exists;
  the argument is about which questions get asked when the data shape invites them.

## 8. So what?

Three implications worth a screenshot:

- **For safety**: Same answer doesn't mean same process. Models that route through
  different intermediate representations can fail differently under distribution
  shift. Output benchmarks miss this; routing benchmarks would catch it.
- **For evaluation**: We've been comparing model behaviours by their outputs. The
  graph view lets us compare their *internal reasoning strategies* directly,
  feature by feature, circuit by circuit, model against model.
- **For trust**: Auditing what a model *says* is the easy half. Auditing how it
  *thinks* requires substrate that holds features, circuits, and concepts as
  first-class connected objects. We just built one and used it.

GPT-2 has the same words in its dictionary. It just doesn't use those words to answer
questions. That's the one-sentence version. It fits in a tweet, and it survives the
methodology.
