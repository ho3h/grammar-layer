# Neograph Bloom perspective

A minimal Bloom perspective for the Neograph graph. Drop into Neo4j Desktop's Bloom UI to explore the rhyme + weekday manifolds.

## Load

1. Open Neo4j Desktop → Bloom.
2. Connect to `bolt://localhost:7693` with `neo4j / neograph_local_dev`.
3. **Perspectives → Import** → select `bloom/neograph-perspective.json`.

## Scenes

- **Rhyme manifold** — shows the manifold whose Concept name contains "rhyme", its waypoints, and member features with their autointerp labels.
- **Weekday manifold** — same for the weekday manifold (cyclic).

## Reading the graph

- **SAEFeature nodes** are coloured by `communityId` (Leiden) and sized by `activation_density` (log scale). Sparse features are smaller; dense features (e.g. punctuation features) are larger.
- **Manifold** (red) **→ Waypoint** (orange) chain shows the principal-curve waypoints in arc order.
- **LIES_ON** edges (red) connect features to the manifold they belong to. Edge style encodes `perp_distance`.
- **NEXT** edges between waypoints encode arc-length delta.
- **DESCRIBES** edges (purple) tie a manifold to its Concept (auto-named via Claude on community labels).

## Useful manual queries

```cypher
// What manifolds exist?
MATCH (m:Manifold)-[:DESCRIBES]->(c:Concept)
RETURN m.id, c.name, m.n_waypoints, m.is_cyclic, m.fit_residual
ORDER BY m.fit_residual ASC LIMIT 50;

// Features on the rhyme manifold, in arc order
MATCH (f:SAEFeature)-[r:LIES_ON]->(m:Manifold)
WHERE m.id CONTAINS 'rhyme' OR toLower(m.id) STARTS WITH 'community-'
RETURN f.index, f.communityId, r.arc_position, r.perp_distance,
       [(f)-[:LABELED_AS]->(a:AutoInterpLabel) | a.text][0] AS label
ORDER BY r.arc_position ASC LIMIT 50;

// Shattered manifolds (Q3 — features clustered by structure but not by label)
MATCH (f:SAEFeature)-[:LABELED_AS]->(a:AutoInterpLabel)
WHERE f.communityId IS NOT NULL
WITH f.communityId AS cid, count(f) AS n, collect(a.embedding)[..10] AS embs
WHERE n >= 5
WITH cid, n, embs,
     reduce(s = 0.0, i IN range(0, size(embs)-2) |
       s + reduce(s2 = 0.0, j IN range(i+1, size(embs)-1) |
         s2 + vector.similarity.cosine(embs[i], embs[j]))) AS labelSum
WITH cid, n, labelSum / (size(embs) * (size(embs)-1) / 2.0) AS meanLabelCos
WHERE meanLabelCos < 0.25
RETURN cid, n, meanLabelCos ORDER BY meanLabelCos ASC LIMIT 20;
```
