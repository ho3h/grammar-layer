"""Label-pattern-based community evaluation — the right experiment for our SAE.

Instead of using Goodfire's anchor indices (which come from a *different* SAE training and
don't correspond to our features), we define ground-truth label patterns and ask:
'how concentrated are features with this label pattern in Leiden communities?'

Patterns we test:
- 'word-prefix' features (labels containing "beginning with"/"starting with")
- 'temporal: day/week' features
- 'money/financial' features (positive control — should obviously cluster)
- 'programming/code' features (positive control)

Reports NMI, max community concentration, and the dominant community.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
from sklearn.metrics import normalized_mutual_info_score

from neograph.config import PATHS
from neograph.cypher import NeographClient
from neograph.util import get_logger

log = get_logger("neograph.label_eval")


PATTERNS = {
    "word_prefix": {
        "any": ["beginning with", "starting with", "starts with"],
        "not": ["sentence", "phrase", "section"],
        "expected": "Goodfire's shattered-rhyme analog: 'words beginning with Hor/Por/Marg' style",
    },
    "weekday_time": {
        "any": ["day of the week", "days of the week", "weekday", "weekend",
                "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
                "two days", "three days", "five days"],
        "not": [],
        "expected": "Engels-style day-of-week manifold",
    },
    "money_financial": {
        "any": ["monetary", "financial", "currency", "dollar", "salary"],
        "not": [],
        "expected": "Positive control — strongly clustered concept",
    },
    "programming_code": {
        "any": ["programming", "code", "function", "variable", "syntax", "import"],
        "not": ["import duty", "import tax"],
        "expected": "Positive control — strongly clustered concept",
    },
}


def find_matching(c: NeographClient, any_patterns: list[str], not_patterns: list[str]) -> list[dict]:
    or_clauses = " OR ".join(f"toLower(a.text) CONTAINS '{p}'" for p in any_patterns)
    not_clauses = " AND ".join(f"NOT toLower(a.text) CONTAINS '{p}'" for p in not_patterns) or "TRUE"
    cy = f"""
    MATCH (f:SAEFeature)-[:LABELED_AS]->(a:AutoInterpLabel)
    WHERE ({or_clauses}) AND ({not_clauses})
    RETURN f.index AS idx, f.communityId AS cid, a.text AS label
    """
    return c.run(cy)


def analyse_pattern(c: NeographClient, name: str, spec: dict) -> dict:
    rows = find_matching(c, spec["any"], spec["not"])
    if not rows:
        return {"name": name, "n_features": 0, "n_communities": 0, "nmi": 0.0,
                "expected": spec["expected"]}
    feature_set = {int(r["idx"]) for r in rows}
    counts = Counter(int(r["cid"]) for r in rows if r["cid"] is not None)
    top = counts.most_common(5)

    # Full partition for NMI
    all_rows = c.run(
        "MATCH (f:SAEFeature) WHERE f.communityId IS NOT NULL "
        "RETURN f.index AS idx, f.communityId AS cid"
    )
    indices = np.array([r["idx"] for r in all_rows])
    communities = np.array([r["cid"] for r in all_rows])
    label_mask = np.array([idx in feature_set for idx in indices])
    nmi = float(normalized_mutual_info_score(communities, label_mask.astype(int)))

    return {
        "name": name,
        "expected": spec["expected"],
        "n_features": len(rows),
        "n_distinct_communities": len(counts),
        "top_communities": top,
        "concentration_top1": top[0][1] / len(rows) if top else 0.0,
        "concentration_top2": sum(c for _, c in top[:2]) / len(rows) if top else 0.0,
        "nmi": nmi,
        "example_labels": [r["label"][:90] for r in rows[:5]],
    }


def main() -> int:
    results: dict[str, dict] = {}
    with NeographClient() as c:
        for name, spec in PATTERNS.items():
            res = analyse_pattern(c, name, spec)
            results[name] = res
            log.info(
                "%s: n=%d distinct_communities=%d top1=%.0f%% top2=%.0f%% NMI=%.3f",
                name, res["n_features"], res.get("n_distinct_communities", 0),
                100 * res.get("concentration_top1", 0),
                100 * res.get("concentration_top2", 0),
                res["nmi"],
            )
            if res.get("top_communities"):
                log.info("  top community %s has %d features", res["top_communities"][0][0], res["top_communities"][0][1])
    out = PATHS.reports / "label_clustering.json"
    out.write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
