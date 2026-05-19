# Neograph — End-to-end findings

*Generated 2026-05-12 after a complete P1→P6 run + γ sweep + label-pattern eval.*

## 1. The framework works

A multi-relation feature graph (co-activation PMI + decoder cosine + autointerp-label cosine) clustered via Leiden in Neo4j+GDS does recover SAE feature manifolds that match by-eye semantic categories. Concrete evidence from `reports/label_clustering.json`:

| Concept (by Neuronpedia label pattern) | n features | top community | concentration |
|---|---:|---:|---:|
| Days of the week | 7 | community 19 | **100%** |
| Money / financial | 281 | community 6 | **85%** |
| Programming / code | 3,092 | community 8 | **79%** |
| Word/letter prefix | 24 | communities 1 + 11 | **67%** (split into two sub-clusters — see below) |

These are not curated; they come from a single Leiden run at γ=1.0 on the multi-relation graph (PRD §7.1 weighting).

## 2. Goodfire's specific feature indices don't apply to our SAE

The PRD §9.1 evaluation anchored on Goodfire's 23 width-32k+ feature indices was a **categorically wrong test** for our setup. Different SAE training runs produce different feature indices — index 2478 in our width-16k canonical SAE is not the same concept as index 2478 in Goodfire's width-32k SAE. We measured this directly:

- Of all 66 possible pairs among the 12 in-range Goodfire indices (those < 16,384), only **one pair (3596, 4806) co-activates** in our 11k-prompt corpus. No DECODER_SIMILAR edges exist between any pair.
- This is consistent with: the indices simply don't correspond.

The label-pattern evaluation (above) is the correct test for "did Leiden recover semantically coherent communities," and it passes well.

## 3. Leiden γ sweep — resolution doesn't matter for concept-cluster recovery

From `reports/leiden_gamma_sweep.json`:

| γ | communities | modularity | Goodfire-index NMI |
|---:|---:|---:|---:|
| 1.0 | 17 | 0.469 | 0.000 |
| 2.0 | 50 | 0.424 | 0.001 |
| 3.0 | 83 | 0.401 | 0.001 |
| 4.0 | 103 | 0.383 | 0.001 |
| 6.0 | 136 | 0.355 | 0.001 |

Higher γ produces finer communities but no better Goodfire-index NMI (because the indices don't correspond). Modularity drops as expected. **Verdict**: γ=1.0 is fine for concept-level clustering; turn it up if you want sub-concept granularity.

## 4. The "shattered prefix manifold" — Leiden found two sub-clusters

24 features in our SAE have autointerp labels matching "beginning with X" / "starting with X" / "starts with X". Leiden splits them into:

**Community 1 (11 features)** — general word/letter prefixes:
- "Words or names beginning with the letter 'Z'"
- "Words starting with 'ru'"
- "Words beginning with 'cra'"
- "Words beginning with 'fl'"
- "Words related to 'pri', indicating a focus on terms starting with or containing this prefix"
- "Prefixes and prepositions in Slavic languages, particularly those starting with 'po-'"

**Community 11 (6 features)** — proper-noun prefixes:
- "References starting with 'Ste'"
- "Names or titles starting with 'El'"
- "Names starting with 'Hal'"
- "Initials starting with 'Sh'"
- "Names starting with 'Mc'"
- "Names beginning with 'Wil'"

This is the kind of distinction a human researcher might draw by hand. Leiden + our multi-relation graph **automatically recovered it** at γ=1.0. The communities-have-coherent-semantics property the PRD claimed (PRD §7, P3 exit criterion) holds.

## 5. Trajectory steering ≈ linear steering on Gemma 2 2B days-of-week single-step

From `reports/p6_steering.json` (Mon→Tue / Tue→Wed / etc. completions, α=4 for linear delta, α=0.7 for manifold interp):

| Method | target hit rate | log P(target) | log P(any-day) |
|---|---:|---:|---:|
| Baseline (no steer) | 43% | −2.50 | −0.51 |
| Linear (target − source mean) | **100%** | **−0.43** | −0.40 |
| Manifold (interp toward target waypoint) | **100%** | −0.66 | −0.39 |

Both methods steer perfectly on single-step. Linear narrowly beats manifold on log-probability. **The PRD's "1.5× ratio + 0.2 nat entropy" target is missed** because at this resolution Gemma's days-of-week representation is already linear enough that either method works. The Engels-style manifold advantage probably needs:
- multi-step generation (current eval is one token)
- prompts with longer cyclic chains ("Mon → ? → ? → ? → Fri")
- a coherence/smoothness metric across sliding-window embeddings (PRD H7)

## 6. Engineering notes

- `gds.graph.project.cypher` is deprecated in GDS 2026.04 and rejects `undirectedRelationshipTypes`. Use the **aggregating** `gds.graph.project` function (`WITH gds.graph.project($name, a, b, dataMap, configMap) AS g RETURN g…`) with `{undirectedRelationshipTypes: ['*']}`.
- SAELens 4.x: `SAE.from_pretrained` returns only the SAE (no tuple). `model.run_with_cache_with_saes` exists on `HookedSAETransformer`, not vanilla `HookedTransformer`.
- Manifold steering hook needs waypoints at the SAME residual-stream position as the hook (last token before generation), not at content-token positions. Position-mismatched centroids destroy generation coherence.
- Pile streaming via HF Datasets needs `zstandard` (not auto-installed).

## 7. Cross-model universality (added 2026-05-12)

After Theo flagged this as priority #1 — the test of the meta-question "does graph tech help at the scale and heterogeneity interpretability is moving toward?" — we ingested **GPT-2 small + Joseph Bloom's RES-JB SAE layer 8** (24,576 features, d_model 768) into the same Neograph schema, under `sae_id='gpt2-small-res-jb/L8'`. Then we ran the same Leiden pipeline and asked: **for each Leiden community in one model, which community in the other model has the closest label-embedding centroid?**

GPT-2 Leiden produces **14 communities** at modularity **0.482** — strikingly close to Gemma's 18 / 0.469. So both models partition into roughly the same number of natural concept clusters at the default resolution.

**Hungarian one-to-one assignment** between Gemma and GPT-2 communities, ranked by label-centroid cosine (`reports/cross_model_universality.json`, top 5):

| Gemma cID | GPT-2 cID | centroid cos | Gemma top label | GPT-2 top label |
|---:|---:|---:|---|---|
| 15 | 12 | **0.951** | "numerical statistics and data points" | "geographical locations, proper nouns" |
| 19 | 4 | **0.946** | "references to specific days of the week" | "dates and locations" |
| 16 | 8 | 0.931 | "legal and technical terminology" | "baking instructions or recipes" (false positive) |
| 12 | 13 | 0.925 | "references to user instructions" | "phrases related to physical actions" |
| 11 | 14 | 0.826 | "proper nouns and specific terms" | "proper nouns related to public figures" |

The high-cosine pairs are doing real work — `gemma:19 ↔ gpt2:4` is the weekday community in both models, `gemma:11 ↔ gpt2:14` is the proper-noun community in both. But there's noise too — `gemma:16` (legal/technical) paired with `gpt2:8` (recipes) at cos=0.93 is mostly MiniLM-embedding overlap on common nouns.

**Concept-level alignment** (does each labelled concept land in the Hungarian-paired community pair?):

| Concept | Gemma community | GPT-2 community | Hungarian-says | Aligned |
|---|---:|---:|---:|---|
| weekday | 19 | 4 | 4 | ✅ |
| money | 6 | 3 | 3 | ✅ |
| programming | 8 | 0 | not paired with 0 | ❌ |
| word_prefix | 1 | 7 | 6 | ❌ |

**2 of 4 concepts align cross-model.** That's a real positive result — the weekday manifold lands in the same Hungarian-paired community pair in both models, and so does money/financial. The programming and word-prefix concepts disagree, which is also informative: in both Gemma and GPT-2, programming features form a giant community (the "default residual stream" cluster), and the centroid is dominated by general programming-token features rather than by the conceptual hook.

![Concept alignment across models](reports/viz_cross_model_concepts.png)

The verdict on Theo's meta-question: **the graph tech earns its keep, just barely**. This specific test could have been done in pandas — three calls to scipy + scikit-learn's linear_sum_assignment + careful per-model indexing. The win for Neo4j here is that the schema let us run the same Leiden + concept-pattern queries on both models without rewriting any analysis code; the `sae_id` partition is the only thing that changed. That's *competent* graph tech, not *paradigm-shifting* graph tech. The paradigm shift comes when we add `:CAUSES` edges from circuit-tracer (next experiment) and ask 4-hop questions like "features on the same manifold in two models that participate in causally homologous circuits" — that question is unimplementable in pandas without writing a graph traversal anyway.

## 8. Causal attribution (`:CAUSES` edges via zero-ablation patching)

After Theo's note ("if those Cypher queries feel like superpowers, you have your answer"), we ran zero-ablation patching to populate `:CAUSES` (actually `:INCLUDES` from `Circuit` to `SAEFeature`, attribution-weighted) for two prompts:

- `"Today is Monday. Tomorrow is " → " Tuesday"` (baseline logit 27.20, 82 active features)
- `"The capital of France is " → " Paris"`

For each active feature at the last position, we zero its activation post-encode and measure the change in target-token logit. Positive effect = feature supports target; negative = feature opposes.

**The 5 Cypher queries that motivate the schema** (`scripts/causal_attribution.py`):

`Q-CAUSE-1` — Communities recruited by a circuit:

```cypher
MATCH (cir:Circuit {id: $cid})-[inc:INCLUDES]->(f:SAEFeature)
WHERE f.communityId IS NOT NULL
RETURN f.communityId AS community, count(f) AS n, sum(inc.attribution) AS total
ORDER BY total DESC
```

For Mon→Tue: community 19 (the weekday Leiden community) contributes +0.30 total attribution across 5 features. Community 16 (legal/technical) contributes +0.39 across 5 features — which is suspicious until you read the labels and realise it's actually "structural/declarative" patterns the model leans on.

`Q-CAUSE-2` — Top-attribution features that also lie on a fitted manifold:

```cypher
MATCH (cir:Circuit {id: $cid})-[inc:INCLUDES]->(f:SAEFeature)-[lo:LIES_ON]->(m:Manifold)
OPTIONAL MATCH (m)-[:DESCRIBES]->(co:Concept)
RETURN f.index, inc.attribution, m.id, co.name, lo.arc_position
ORDER BY abs(inc.attribution) DESC
```

For France→Paris: feature 3031 (+0.83 effect) lies on `community-22/L20`. Feature 15935 (−0.33 effect, "references to countries") lies on `community-14/L20`. The negative attribution is genuinely interpretable: ablating the "countries" feature *increases* P(Paris), suggesting Gemma is using the countries-feature to bias toward country-name completions, and Paris is a city.

`Q-CAUSE-3` — Top features by absolute attribution + their autointerp label (the "what story does the circuit tell" query). For Mon→Tue, the top causal supporter is **feat 10254 (+0.107, "references to days of the week and their associated schedules")** — and it sits in community 19, the weekday community. The schema joined that automatically.

`Q-CAUSE-5` — Cross-circuit overlap (the genuinely graph-paradigm query):

```cypher
MATCH (cir:Circuit)-[inc:INCLUDES]->(f:SAEFeature)
WITH f, count(DISTINCT cir) AS n_circuits,
     collect(DISTINCT cir.id) AS circuits, sum(inc.attribution) AS total
WHERE n_circuits > 1
OPTIONAL MATCH (f)-[:LABELED_AS]->(a:AutoInterpLabel)
RETURN f.index, n_circuits, total, a.text
ORDER BY n_circuits DESC, abs(total) DESC
```

The features that appear in **both** circuits (Mon→Tue *and* France→Paris) are the universal predicate-style features:

| feat | n_circuits | Σ effect | label |
|---:|---:|---:|---|
| 15596 | 2 | −0.526 | past/present forms of "to be" |
| 13414 | 2 | +0.392 | statements of existence/presence |
| 12927 | 2 | +0.315 | statements of existence/presence |
| 6631 | 2 | +0.230 | beginning of a text or important markers |
| 9768 | 2 | +0.186 | control and authority terms |

That answer takes ~5 lines of Cypher. In pandas it's a nested groupby with collect-then-filter, easily 30 lines, that nobody writes because the question doesn't surface naturally. **This is where the graph paradigm earns its keep — not because pandas can't do it, but because the data shape invites the question.**

### Verdict on Theo's meta-question

The schema composes. `Circuit -[:INCLUDES]-> SAEFeature -[:LIES_ON]-> Manifold -[:DESCRIBES]-> Concept` is a 3-hop traversal that surfaces a sentence like "in the Mon→Tuesday circuit, the top supporting feature on the weekday manifold is feat 10254" in one query. The same composition in pandas requires four separate frames + careful merge-on-index. Multiply by cross-model and the asymmetry compounds.

**Where the answer is still hedged**: we have *one* circuit per task, on *one* model. The decisive test is whether the same query patterns scale to 50 circuits across 3 models without rewriting any analysis code. The schema is *ready* for that. Whether it pays off is the next experiment.

## 9. What would move the needle from here

In rough order of expected impact:

1. **`:CAUSES` edges from causal patching** for one prompt in each model. Then ask: "Do the features that causally support 'Tuesday' on Mon→Tue completion in Gemma have decoder-similar counterparts in GPT-2 that also support their model's day-of-week predictions?" That's the test of graph-tech-as-paradigm vs graph-tech-as-feature-store. Script staged at `scripts/causal_attribution.py`.
2. **Width-65k canonical Gemma Scope SAE**. PRD §11 stretch. More features → finer Leiden → likely splits the prefix community further into linguistic / ethnic / corporate sub-groups.
3. **Multi-step manifold steering** during generation (PRD H7) — score sentence-level coherence across token positions, not just one-step target hit.
4. **A third model** (Pythia 1.4B or Llama 3.1 8B) to make cross-model the rule, not just the bilateral case.
