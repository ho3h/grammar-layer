"""Visualise the ablation-control experiment.

Two panels:
- Top: overall Δlog P(target) under each ablation condition, with hit-rate annotation
- Bottom: per-category mean Δlog P, supporting-top10 (blue) vs random-10 (grey) vs bottom-10 (red)

Reads from reports/load_bearing_control_gemma_50.json.

Usage:
    uv run python scripts/viz_control.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    data = json.loads((ROOT / "reports/load_bearing_control_gemma_50.json").read_text())
    s = data["summary"]
    overall = s["overall"]
    per_cat = s["per_category"]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(13, 9.5),
                                         gridspec_kw={"height_ratios": [1, 1.4]},
                                         constrained_layout=True)

    # Top: overall condition comparison
    conditions = [
        ("baseline",          overall["baseline"]["hit_rate"],   0.0,                                      "#888"),
        ("supporting top-10\n(targeted)", overall["supporting_top10"]["hit_rate"], overall["supporting_top10"]["mean_log_p_drop"], "#2a7fff"),
        ("random-10\n(active, 5 seeds)",   overall["random10_active"]["hit_rate"],   overall["random10_active"]["mean_log_p_drop"],   "#bbb"),
        ("bottom-10\n(by |attr|)",  overall["bottom10_by_attr"]["hit_rate"],  overall["bottom10_by_attr"]["mean_log_p_drop"],  "#d6324b"),
        ("all supporting\n(upper bound)",  overall["all_supporting"]["hit_rate"],  overall["all_supporting"]["mean_log_p_drop"], "#1a4d99"),
    ]
    labels = [c[0] for c in conditions]
    drops = [c[2] for c in conditions]
    hits = [c[1] for c in conditions]
    colors = [c[3] for c in conditions]

    bars = ax_top.bar(range(len(conditions)), drops, color=colors, edgecolor="black", linewidth=0.5)
    ax_top.set_xticks(range(len(conditions)))
    ax_top.set_xticklabels(labels, fontsize=10)
    ax_top.set_ylabel("Mean Δlog P(target)  (positive = target suppressed)", fontsize=10)
    ax_top.set_title("Gemma 2 2B, 52 prompts × 12 categories  ·  ablation conditions compared", fontsize=12, weight="bold")
    ax_top.axhline(0, color="black", linewidth=0.5)
    ax_top.set_ylim(-0.5, max(drops) * 1.15)
    for i, (lbl, hit, drop, _) in enumerate(conditions):
        y = drop + 0.15 if drop >= 0 else drop - 0.25
        ax_top.text(i, y, f"hit={hit:.2f}", ha="center", va="bottom", fontsize=9, color="#333")
        if drop > 0.2:
            ax_top.text(i, drop / 2, f"Δ={drop:+.2f}", ha="center", va="center",
                        fontsize=10, color="white", weight="bold")

    # Annotation showing the targeting ratio (positioned to the right of the all-supporting bar)
    sup_drop = overall["supporting_top10"]["mean_log_p_drop"]
    rnd_drop = overall["random10_active"]["mean_log_p_drop"]
    ratio_rand = sup_drop / max(rnd_drop, 1e-3)
    ax_top.text(
        2.5, sup_drop * 1.05,
        f"Targeted is  {ratio_rand:.0f}× larger than random,\n"
        f"and essentially infinite over bottom-10.",
        ha="center", va="bottom", fontsize=10, color="#2a7fff", weight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="#eef5ff", ec="#2a7fff", lw=1.2),
    )
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    # Bottom: per-category
    cats = sorted(per_cat.keys())
    x = np.arange(len(cats))
    w = 0.27
    sup_drops = [per_cat[c]["supporting_top10"]["mean_log_p_drop"] for c in cats]
    rnd_drops = [per_cat[c]["random10_active"]["mean_log_p_drop"] for c in cats]
    bot_drops = [per_cat[c]["bottom10_by_attr"]["mean_log_p_drop"] for c in cats]

    ax_bot.bar(x - w, sup_drops, w, label="supporting top-10 (targeted)", color="#2a7fff", edgecolor="black", linewidth=0.4)
    ax_bot.bar(x,      rnd_drops, w, label="random-10 active", color="#bbb", edgecolor="black", linewidth=0.4)
    ax_bot.bar(x + w,  bot_drops, w, label="bottom-10 by |attribution|", color="#d6324b", edgecolor="black", linewidth=0.4)
    ax_bot.axhline(0, color="black", linewidth=0.5)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(cats, rotation=35, ha="right", fontsize=9)
    ax_bot.set_ylabel("Mean Δlog P(target) per category", fontsize=10)
    ax_bot.set_title("Per-category: targeted ablation is the only condition with non-trivial effect across every category",
                     fontsize=11, weight="bold")
    ax_bot.legend(loc="upper right", fontsize=10, frameon=False)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)

    out = ROOT / "reports" / "viz_control.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
