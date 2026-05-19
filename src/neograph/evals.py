"""Phase exit-criterion evaluators.

- nmi_vs_goodfire: NMI between Leiden communities and Goodfire's 23 rhyme features (PRD §9.1).
- rhyme_community_recovery: report whether the Goodfire features land in the same community.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import normalized_mutual_info_score

from neograph.cypher import NeographClient

# PRD §9.1: Goodfire's anchor features (width-32k+). With our width-16k SAE the indices may
# not line up; we'll still try to find them and compute NMI on the intersection.
GOODFIRE_RHYME_FEATURES = [
    2478, 3596, 4583, 4596, 4806, 5316, 6440, 7471, 9514, 10637,
    12145, 12714, 17398, 20283, 21241, 22084, 23104, 23118, 24140,
    25233, 28555, 31648, 31747,
]


def nmi_vs_goodfire(client: NeographClient) -> dict:
    """Compute NMI(communityId, is_in_goodfire_set) over features whose index falls in our SAE.

    Treats Goodfire's 23 features as one cluster, all other live features as another.
    """
    # In width-16k we have indices 0..16383; some Goodfire indices (e.g. 28555) won't exist.
    avail = [i for i in GOODFIRE_RHYME_FEATURES if i < 16384]
    empty = {
        "nmi": 0.0,
        "matched": 0,
        "total_in_sae": len(avail),
        "n_features_scored": 0,
        "goodfire_community_counts": {},
    }
    if not avail:
        return empty

    rows = client.run(
        """
        MATCH (f:SAEFeature)
        WHERE f.communityId IS NOT NULL AND coalesce(f.is_dead, false) = false
        RETURN f.index AS idx, f.communityId AS cid
        """
    )
    if not rows:
        return empty

    indices = np.array([r["idx"] for r in rows])
    communities = np.array([r["cid"] for r in rows])
    goodfire_mask = np.isin(indices, avail)

    nmi = float(normalized_mutual_info_score(communities, goodfire_mask.astype(int)))

    # Dominant community of Goodfire features
    goodfire_communities = communities[goodfire_mask]
    counts = Counter(int(c) for c in goodfire_communities)

    return {
        "nmi": nmi,
        "matched": int(goodfire_mask.sum()),
        "total_in_sae": len(avail),
        "n_features_scored": int(len(rows)),
        "goodfire_community_counts": dict(counts.most_common(5)),
    }


def rhyme_community_summary(client: NeographClient) -> dict:
    """Returns the top community by Goodfire-feature concentration and its autointerp labels."""
    info = nmi_vs_goodfire(client)
    if not info["goodfire_community_counts"]:
        return info
    top_cid = int(next(iter(info["goodfire_community_counts"])))
    labels = client.run(
        """
        MATCH (f:SAEFeature)-[:LABELED_AS]->(a:AutoInterpLabel)
        WHERE f.communityId = $cid
        RETURN a.text AS text, f.index AS idx
        ORDER BY f.activation_density DESC
        LIMIT 30
        """,
        cid=top_cid,
    )
    info["top_community_id"] = top_cid
    info["top_community_labels"] = [r["text"] for r in labels]
    return info


def steering_summary(reports_path) -> dict:
    """Read reports/p6_steering.json and compute summary stats."""
    import json
    from pathlib import Path

    p = Path(reports_path)
    if not p.exists():
        return {"error": "no report"}
    rows = json.loads(p.read_text())
    by_method = {}
    for r in rows:
        m = r["method"]
        by_method.setdefault(m, []).append(r)
    summary = {}
    for m, items in by_method.items():
        hits = sum(1 for x in items if x["target_hit"]) / max(len(items), 1)
        avg_lp_target = float(np.mean([x["log_p_target"] for x in items]))
        avg_lp_day = float(np.mean([x["log_p_day_total"] for x in items]))
        summary[m] = {
            "target_hit_rate": hits,
            "avg_log_p_target": avg_lp_target,
            "avg_log_p_day_total": avg_lp_day,
            "n": len(items),
        }
    # Manifold vs linear ratio
    if "linear" in summary and "manifold" in summary:
        lin = summary["linear"]["target_hit_rate"]
        man = summary["manifold"]["target_hit_rate"]
        summary["manifold_vs_linear_hit_ratio"] = (man / lin) if lin > 0 else float("inf")
        summary["entropy_delta_nat"] = (
            summary["manifold"]["avg_log_p_day_total"] - summary["linear"]["avg_log_p_day_total"]
        )
    return summary
