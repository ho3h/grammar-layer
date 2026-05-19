"""Cross-model grammar enrichment ratio bar chart.

Reads reports/cross_model_grammar.json. For each model with labels, shows the
supporting%grammar → opposing%grammar enrichment ratio with annotation. Models
sorted by enrichment ratio.

Usage:
    uv run python scripts/viz_enrichment_bar.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    data = json.loads((ROOT / "reports/cross_model_grammar.json").read_text())
    rows = []
    for nick, v in data.items():
        rows.append({
            "display": v["display"],
            "nick": nick,
            "sup_pct": v["sup_pct_grammar"],
            "opp_pct": v["opp_pct_grammar"],
            "enrichment": v["enrichment_ratio"],
        })
    # Sort by enrichment (descending)
    rows.sort(key=lambda r: -r["enrichment"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    # Panel 1: enrichment ratio per model
    names = [r["display"] for r in rows]
    enrich = [r["enrichment"] for r in rows]
    colors = ["#2a7fff" if e >= 2.0 else "#d6324b" for e in enrich]
    bars = ax1.barh(range(len(rows)), enrich, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)
    ax1.set_yticks(range(len(rows)))
    ax1.set_yticklabels(names, fontsize=11)
    ax1.invert_yaxis()
    ax1.set_xlabel("Supporting → Opposing grammar enrichment ratio", fontsize=10)
    ax1.set_title("Cross-model grammar-suppression enrichment (Neuronpedia labels)",
                  fontsize=11, weight="bold")
    ax1.axvline(1.0, color="grey", linestyle="--", linewidth=1, label="null (no enrichment)")
    ax1.axvline(2.0, color="#2a7fff", linestyle=":", linewidth=1, alpha=0.5)
    ax1.legend(loc="lower right", fontsize=9, frameon=False)
    for bar, e in zip(bars, enrich):
        ax1.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                 f"{e:.2f}×", va="center", fontsize=10, weight="bold")
    ax1.set_xlim(0, max(enrich) * 1.15)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    # Panel 2: supporting vs opposing %grammar paired bars
    x = np.arange(len(rows))
    w = 0.4
    sup = [r["sup_pct"] for r in rows]
    opp = [r["opp_pct"] for r in rows]
    ax2.bar(x - w/2, sup, w, label="supporting %grammar", color="#7ab0ff", edgecolor="black", linewidth=0.4)
    ax2.bar(x + w/2, opp, w, label="opposing %grammar",  color="#d6324b", edgecolor="black", linewidth=0.4)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    ax2.set_ylabel("% of top-10 feature-slots classified as grammar", fontsize=10)
    ax2.set_title("Per-model supporting vs opposing grammar share",
                  fontsize=11, weight="bold")
    ax2.legend(loc="upper right", fontsize=9, frameon=False)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    fig.suptitle(
        "Grammar features cluster on the opposing side in three model families across two organisations.\n"
        "Pythia 70M (EleutherAI), Gemma 1 2B and Gemma 2 2B (Google) all show ≥ 2.8× enrichment. "
        "GPT-2 small (OpenAI) and mid-network Gemma 2 9B do not.",
        fontsize=11, y=1.04,
    )

    out = ROOT / "reports" / "viz_enrichment_bar.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
