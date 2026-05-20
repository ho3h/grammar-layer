// Fingerprint queries — the substrate's user-facing Cypher surface.
//
// Each query is a documented, reproducible operation against the Neograph store:
// SAEFeature nodes per model, Circuit nodes per (prompt × model), and INCLUDES
// edges with signed attribution. The substrate is populated by
// scripts/causal_attribution_v2.py for each model.
//
// All queries run against bolt://localhost:7693 (see .env / src/neograph/config.py).

// -----------------------------------------------------------------------------
// Q1: Top opposing features on a single prompt, in a single model.
// Used by: reports/cross_routing_functional_summary.md, the writeup's Result 2.
// -----------------------------------------------------------------------------
MATCH (cir:Circuit {prompt_id: 'capital-jp'})-[inc:INCLUDES]->(f:SAEFeature)
WHERE cir.model = 'gemma' AND inc.role = 'oppose'
OPTIONAL MATCH (f)-[:LABELED_AS]->(a:AutoInterpLabel)
RETURN f.index AS feature, inc.attribution AS attribution, a.text AS label
ORDER BY inc.attribution ASC LIMIT 10;

// -----------------------------------------------------------------------------
// Q2: Cross-model side-by-side — same prompt, every available model.
// Drives the "same vocabulary, different routing" figure.
// -----------------------------------------------------------------------------
MATCH (cir:Circuit {prompt_id: 'capital-jp'})-[inc:INCLUDES]->(f:SAEFeature)
WHERE inc.role = 'oppose' AND inc.rank < 5
OPTIONAL MATCH (f)-[:LABELED_AS]->(a:AutoInterpLabel)
RETURN cir.model AS model, f.index AS feature, inc.attribution AS attribution, a.text AS label
ORDER BY cir.model, inc.attribution ASC;

// -----------------------------------------------------------------------------
// Q3: Multi-circuit features — the "fingerprint" features, the ones recruited
// as opposers across multiple capital prompts. ≥4/6 capitals = fingerprint.
// -----------------------------------------------------------------------------
MATCH (cir:Circuit)-[inc:INCLUDES]->(f:SAEFeature)
WHERE cir.model = 'gemma' AND cir.category = 'capital' AND inc.role = 'oppose'
WITH f, count(DISTINCT cir) AS n_capital_circuits,
     collect(DISTINCT cir.prompt_id) AS prompts,
     avg(inc.attribution) AS mean_attribution
WHERE n_capital_circuits >= 4
OPTIONAL MATCH (f)-[:LABELED_AS]->(a:AutoInterpLabel)
RETURN f.index AS feature, n_capital_circuits, mean_attribution, a.text AS label, prompts
ORDER BY n_capital_circuits DESC, mean_attribution ASC;

// -----------------------------------------------------------------------------
// Q4: Vocabulary universality check — for a Gemma fingerprint feature, find
// label-similar features in GPT-2 (via the label_emb vector index) and check
// whether THEY are recruited as opposers anywhere.
// -----------------------------------------------------------------------------
MATCH (gemma_f:SAEFeature {index: 15596})-[:LABELED_AS]->(g_label:AutoInterpLabel)
WHERE gemma_f.sae_id CONTAINS 'gemma-scope-2b'
CALL db.index.vector.queryNodes('label_emb', 10, g_label.embedding) YIELD node, score
MATCH (gpt2_f:SAEFeature)-[:LABELED_AS]->(node)
WHERE gpt2_f.sae_id = 'gpt2-small-res-jb/L8' AND score >= 0.80
OPTIONAL MATCH (gpt2_cir:Circuit)-[gi:INCLUDES]->(gpt2_f)
WHERE gpt2_cir.category = 'capital' AND gi.role = 'oppose'
WITH gpt2_f, node, score, count(DISTINCT gpt2_cir) AS n_capital_opposings
RETURN gpt2_f.index AS gpt2_feature, node.text AS gpt2_label, score AS label_cosine,
       n_capital_opposings AS recruited_as_opposer_in_n_capitals
ORDER BY score DESC;

// -----------------------------------------------------------------------------
// Q5: Cross-family enrichment of grammar features on the opposing side.
// Counts grammar-labelled vs content-labelled opposers per model, sums across
// capital prompts. Used to populate the enrichment table in the writeup.
// -----------------------------------------------------------------------------
WITH ['is','to be','copula','verb','tense','grammar','function word'] AS grammar_kw
MATCH (cir:Circuit)-[inc:INCLUDES]->(f:SAEFeature)
WHERE cir.category = 'capital' AND inc.role = 'oppose' AND inc.rank < 5
OPTIONAL MATCH (f)-[:LABELED_AS]->(a:AutoInterpLabel)
WITH cir.model AS model, f, a,
     CASE WHEN any(kw IN grammar_kw WHERE toLower(coalesce(a.text, '')) CONTAINS kw)
          THEN 'grammar' ELSE 'content' END AS kind
RETURN model, kind, count(*) AS n_opposer_slots,
       count(DISTINCT f) AS n_distinct_features
ORDER BY model, kind;

// -----------------------------------------------------------------------------
// Q6: Find a model's content-only opposers — the contrast for the inversion
// claim. Used in writing the "GPT-2 routes on content, not grammar" sentence.
// -----------------------------------------------------------------------------
MATCH (cir:Circuit {model: 'gpt2', category: 'capital'})-[inc:INCLUDES]->(f:SAEFeature)
WHERE inc.role = 'oppose' AND inc.rank < 10
OPTIONAL MATCH (f)-[:LABELED_AS]->(a:AutoInterpLabel)
RETURN cir.prompt_id AS prompt, f.index AS feature,
       inc.attribution AS attribution, a.text AS label
ORDER BY cir.prompt_id, inc.attribution ASC;
