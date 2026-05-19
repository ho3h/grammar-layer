# Multi-relation graphs over SAE features recover sub-concept structure that single-relation views miss

*Draft writeup for LessWrong / Alignment Forum. The graph framing is load-bearing: the
result depends on combining co-activation, decoder cosine, and label cosine as
distinct edge types — any single similarity collapses the distinction we find.*

## Claim

Sparse autoencoder (SAE) features for a concept like "words starting with X" are
"shattered" across many individual features ([Goodfire 2026][goodfire]). A multi-relation
feature graph + Leiden community detection at default resolution recovers not just the
shattered concept as a single cluster, but **structurally meaningful sub-concepts** that
a single similarity measure would collapse.

Concretely, on Gemma 2 2B layer 20 with the canonical width-16k Gemma Scope SAE: 24
features whose Neuronpedia autointerp labels match "starting with X" / "beginning with X"
split into two Leiden communities that we can characterise post-hoc:

- **Community 1 (general)**: 11 features for ordinary words and letters.
  *"Words starting with 'ru'"*, *"Words beginning with 'cra'"*, *"Words or names
  beginning with 'Z'"*, *"Mentions of the letter 'F'"*, *"Slavic po- prefixes"*.
- **Community 11 (proper nouns)**: 6 features for proper-noun prefixes.
  *"Names starting with 'Ste'"*, *"Starting with 'El'"*, *"Names starting with 'Hal'"*,
  *"Initials starting with 'Sh'"*, *"Names starting with 'Mc'"*, *"Names beginning
  with 'Wil'"*.

The model never knew these were two distinct things. Leiden on a graph weighted across
co-activation PMI + decoder direction cosine + label embedding cosine did.

## Why the multi-relation framing is doing work

Each relation alone is weaker:

- **Decoder cosine alone** clusters features by where their direction lives in residual
  space. For most "starting-with-X" features this is a thin shared subspace; the general /
  proper-noun distinction barely registers.
- **Co-activation alone** is sparse — for low-density features that fire on rare prefixes,
  there isn't enough corpus overlap to discriminate sub-clusters.
- **Label cosine alone** is the strongest single signal, but MiniLM doesn't natively know
  that "names starting with Mc" and "words starting with cra" should sit far apart.

Combining all three in Leiden at γ=1.0 — `0.5 · PMI/10 + 0.3 · decoder + 0.2 · label`,
all edges undirected — produces the split. We verified on a γ sweep that resolution
isn't the active ingredient: the split appears at γ=1.0 and persists at γ=2,3,4,6.

We think the mechanism is: decoder cosine puts both groups in the same neighbourhood,
co-activation distinguishes the "fires on proper nouns" feature subset because they share
contexts (names tend to cluster in formal text), and label cosine sharpens the partition
by separating "name" / "title" / "individual" tokens from "word" / "letter" / "prefix"
tokens. None of the three relations alone would do this.

## Why this isn't a Neo4j marketing pitch

A determined researcher could reproduce this in numpy. Three sparse similarity matrices,
weighted-summed, fed to `leidenalg`. ~500 lines of pandas and scipy. The graph database
is not the contribution. The **combination of relations** is.

What the graph database does buy: when you want to ask "give me features on this manifold,
participating in *that* attribution graph, with a decoder-similar neighbour in a second
model" — that's one Cypher query against a populated multi-model store, and unboundedly
ugly in pandas. We haven't fully exercised that yet. *[Section to fill in once we have
GPT-2 + RES-JB ingested in the same schema, plus an attribution graph from circuit-tracer
written as `:CAUSES` edges.]*

## Reproduction

Code: <https://github.com/[…]/graphgeometry>. Schema in `cypher/00_constraints.cypher` and
`cypher/01_indexes.cypher`. Pipeline P1–P6 in `scripts/`. The headline subgraph is rendered
by `scripts/viz_for_theo.py`.

Hardware budget: M5 Max, 128 GB, ~3 hours wall time for activation capture (Gemma 2 2B
across 10,526 prompts on MPS), the rest in seconds-to-minutes. Total disk footprint
~7 GB including the Neo4j store.

## Open questions

1. **Does the split survive at width 65k**? Goodfire's anchor features are from width-32k+
   training runs; the same prefix-concept may shatter further in a higher-width SAE,
   revealing more sub-concepts.
2. **Is this universal across models**? Run the same pipeline on GPT-2 small + RES-JB and
   check whether the "proper-noun prefix" sub-cluster appears as its own community.
   *[Section to fill in.]*
3. **What's the steering correlate**? The two communities should respond differently to
   trajectory steering — if we move along community 11's manifold during generation, do
   we get name-like completions while moving along community 1 gets ordinary words?

## Acknowledgements

This is built on Gemma Scope (Google DeepMind), Neuronpedia, SAELens, TransformerLens.
The original *World Inside Neural Networks* framing is from Goodfire's Geiger, Lubana,
Fel, Merullo, Byun, Lewis, McGrath. The infrastructure (Neo4j + GDS) does not deserve
top billing — the experimental result is what does.

[goodfire]: https://goodfire.ai/research/world-inside-neural-networks
