"""Neo4j driver wrapper + Q1-Q6 query helpers (PRD §8)."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from neo4j import Driver, GraphDatabase, Session

from neograph.config import NEO4J, PATHS
from neograph.util import get_logger

logger = get_logger("neograph.cypher")


class NeographClient:
    """Thin sync wrapper over neo4j-driver with Q1-Q6 helpers."""

    def __init__(
        self,
        uri: str = NEO4J.uri,
        user: str = NEO4J.user,
        password: str = NEO4J.password,
        database: str = NEO4J.database,
    ) -> None:
        self.uri = uri
        self.database = database
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "NeographClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self._driver.session(database=self.database) as s:
            yield s

    # === Low-level helpers ===

    def run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        with self.session() as s:
            return [r.data() for r in s.run(cypher, **params)]

    def run_df(self, cypher: str, **params: Any) -> pd.DataFrame:
        return pd.DataFrame(self.run(cypher, **params))

    def execute_script(self, script_path: Path) -> None:
        """Run a `.cypher` file by splitting on ';' (semicolon-terminated stmts)."""
        text = Path(script_path).read_text()
        with self.session() as s:
            for raw in text.split(";"):
                meaningful = "\n".join(
                    line for line in raw.splitlines() if not line.strip().startswith("//")
                ).strip()
                if not meaningful:
                    continue
                logger.info("Running: %s", meaningful.splitlines()[0][:90])
                s.run(meaningful + ";")

    def apply_schema(self) -> None:
        """Apply constraints + vector indexes from cypher/00_*.cypher and 01_*.cypher."""
        for f in sorted(PATHS.cypher_dir.glob("0*_*.cypher")):
            self.execute_script(f)

    # === Q1-Q6 (PRD §8) ===

    def q1_features_on_same_manifold_diff_community(self, feature_index: int) -> pd.DataFrame:
        cy = """
        MATCH (f:SAEFeature {index: $idx})-[:LIES_ON]->(m:Manifold)
        MATCH (g:SAEFeature)-[:LIES_ON]->(m)
        WHERE g.communityId <> f.communityId
        RETURN g.index AS index, g.communityId AS community,
               g.activation_density AS density
        ORDER BY g.activation_density DESC LIMIT 50
        """
        return self.run_df(cy, idx=feature_index)

    def q2_shortest_causal_path(
        self, a_idx: int, b_idx: int, max_hops: int = 6
    ) -> pd.DataFrame:
        cy = f"""
        MATCH (a:SAEFeature {{index: $a}}), (b:SAEFeature {{index: $b}})
        MATCH p = shortestPath((a)-[:CAUSES*..{max_hops}]->(b))
        RETURN [n IN nodes(p) | n.index] AS path,
               [r IN relationships(p) | r.effect_size] AS effects
        """
        return self.run_df(cy, a=a_idx, b=b_idx)

    def q3_shattered_manifolds(
        self,
        min_community_size: int = 5,
        max_label_cos: float = 0.25,
        min_pmi: float = 2.0,
        min_decoder_cos: float = 0.4,
    ) -> pd.DataFrame:
        cy = """
        MATCH (f:SAEFeature)-[:LABELED_AS {primary: true}]->(a:AutoInterpLabel)
        WHERE f.communityId IS NOT NULL
        WITH f.communityId AS cid, collect(f) AS feats, collect(a.embedding) AS embs
        WHERE size(feats) >= $minSize
        WITH cid, feats, embs,
             reduce(s = 0.0, i IN range(0, size(embs)-2) |
               s + reduce(s2 = 0.0, j IN range(i+1, size(embs)-1) |
                 s2 + vector.similarity.cosine(embs[i], embs[j]))) AS labelSum,
             toFloat(size(embs) * (size(embs)-1) / 2) AS nPairs
        WITH cid, feats, labelSum / nPairs AS meanLabelCos
        WHERE meanLabelCos < $maxLabelCos
        MATCH (a:SAEFeature)-[r:CO_ACTIVATES_WITH]->(b:SAEFeature)
        WHERE a.communityId = cid AND b.communityId = cid
        WITH cid, meanLabelCos, avg(r.pmi) AS meanPmi, avg(r.cosine_decoder) AS meanDec
        WHERE meanPmi > $minPmi AND meanDec > $minDecoder
        RETURN cid AS community, meanLabelCos, meanPmi, meanDec
        ORDER BY (meanPmi + meanDec) - meanLabelCos DESC LIMIT 20
        """
        return self.run_df(
            cy,
            minSize=min_community_size,
            maxLabelCos=max_label_cos,
            minPmi=min_pmi,
            minDecoder=min_decoder_cos,
        )

    def q4_manifold_waypoints(self, manifold_id: str) -> pd.DataFrame:
        cy = """
        MATCH (m:Manifold {id: $id})-[:HAS_WAYPOINT]->(w:Waypoint)
        RETURN w.index AS index, w.arc_position AS arc,
               w.centroid AS centroid, w.tangent AS tangent
        ORDER BY w.index
        """
        return self.run_df(cy, id=manifold_id)

    def q5_features_for_token(
        self, surface: str, taxonomy: str | None = None
    ) -> pd.DataFrame:
        cy = """
        MATCH (t:Token {surface: $surface})-[a:ACTIVATES]->(f:SAEFeature)
              -[:LIES_ON]->(m:Manifold)-[:DESCRIBES]->(c:Concept)
        WHERE $taxonomy IS NULL OR c.taxonomy = $taxonomy
        RETURN c.name AS concept, m.id AS manifold,
               f.index AS feature, a.magnitude AS magnitude
        ORDER BY a.magnitude DESC LIMIT 20
        """
        return self.run_df(cy, surface=surface, taxonomy=taxonomy)

    def q6_candidate_absorptions(
        self, decoder_threshold: float = 0.6, jaccard_threshold: float = 0.05
    ) -> pd.DataFrame:
        cy = """
        MATCH (parent:SAEFeature)-[d:DECODER_SIMILAR]->(child:SAEFeature)
        WHERE d.cosine > $dec
        OPTIONAL MATCH (parent)-[c:CO_ACTIVATES_WITH]->(child)
        WITH parent, child, d.cosine AS decSim, coalesce(c.jaccard, 0.0) AS jac
        WHERE decSim > $dec AND jac < $jac
        RETURN parent.index AS parent, child.index AS child,
               decSim, jac, decSim - jac AS evidence
        ORDER BY evidence DESC LIMIT 200
        """
        return self.run_df(cy, dec=decoder_threshold, jac=jaccard_threshold)

    def q6_write_absorptions(
        self, decoder_threshold: float = 0.6, jaccard_threshold: float = 0.05
    ) -> int:
        cy = """
        MATCH (parent:SAEFeature)-[d:DECODER_SIMILAR]->(child:SAEFeature)
        WHERE d.cosine > $dec
        OPTIONAL MATCH (parent)-[c:CO_ACTIVATES_WITH]->(child)
        WITH parent, child, d.cosine AS decSim, coalesce(c.jaccard, 0.0) AS jac
        WHERE decSim > $dec AND jac < $jac
        MERGE (parent)-[r:ABSORBED_BY]->(child)
          SET r.evidence = decSim - jac
        RETURN count(r) AS n
        """
        rows = self.run(cy, dec=decoder_threshold, jac=jaccard_threshold)
        return rows[0]["n"] if rows else 0
