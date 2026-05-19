"""Rerun only the Leiden + matching tail of cross_model_gpt2."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cross_model_gpt2 import project_and_leiden, cross_model_matching, GPT2  # noqa: E402

from neograph.config import PATHS  # noqa: E402
from neograph.cypher import NeographClient  # noqa: E402
from neograph.util import exit_marker, get_logger  # noqa: E402

log = get_logger("neograph.gpt2.finish")


def main() -> int:
    with NeographClient() as c:
        log.info("Projecting + Leiden ...")
        stats = project_and_leiden(c)
        log.info("GPT-2 Leiden: %d communities, modularity=%.3f",
                 int(stats.get("communityCount", 0)), float(stats.get("modularity", 0.0)))
        match = cross_model_matching(c)
    print("\n=== Cross-model community matching (shallow) ===")
    for name, rec in match.items():
        gemma_top = rec["gemma"][0] if rec["gemma"] else None
        gpt2_top = rec["gpt2"][0] if rec["gpt2"] else None
        print(f"  {name:14s}  Gemma: {gemma_top}     GPT-2: {gpt2_top}")
    out = PATHS.reports / "cross_model_matching.json"
    out.write_text(json.dumps({"matching": match, "gpt2_leiden": stats}, indent=2, default=str))
    log.info("Wrote %s", out)
    exit_marker("cross-model-finish", ok=stats.get("modularity", 0) > 0.3,
                gpt2_communities=stats.get("communityCount"),
                gpt2_modularity=stats.get("modularity"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
