"""Visualise behavioral-metric distributions: Gemma 2 2B vs GPT-2 small.

Four-panel figure, one per metric. Each panel shows a paired bar chart (Gemma vs GPT-2)
with error bars (1 SD) and the Welch's t-test p-value annotation.

Usage:
    uv run python scripts/viz_behavior.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

METRIC_LABELS = {
    "copula_per_100tok":     "Copula density\n(forms of 'to be' / 100 tokens)",
    "hedge_per_100tok":      "Hedge + modal density\n(per 100 tokens)",
    "generic_np_per_100tok": "Generic noun-phrase rate\n('a/the + abstract noun' / 100 tokens)",
    "copula_opener_fraction": "Copula-led sentence openers\n(fraction of sentences)",
}


def main() -> None:
    data = json.loads((ROOT / "reports" / "behavior_metrics.json").read_text())
    summary = data["summary"]
    models = data["models"]
    if len(models) < 2:
        raise SystemExit("Need at least two models in behavior_metrics.json")

    # Use the first two-model comparison for the test annotations
    test_key = next(iter(summary["tests"].keys()))
    tests = summary["tests"][test_key]
    a_model, b_model = test_key.split("_vs_")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    metrics = list(METRIC_LABELS.keys())

    palette = {a_model: "#2a7fff", b_model: "#d6324b"}

    for ax, mk in zip(axes.flat, metrics):
        per_model = summary["per_model"]
        means = [per_model[m][mk]["mean"] for m in (a_model, b_model)]
        stds = [per_model[m][mk]["std"] for m in (a_model, b_model)]
        ns = [per_model[m][mk]["n"] for m in (a_model, b_model)]
        # Standard error of the mean (1 SE bars are more honest than 1 SD for the means)
        sems = [s / max(n ** 0.5, 1) for s, n in zip(stds, ns)]

        x = np.arange(2)
        bars = ax.bar(x, means, yerr=sems, color=[palette[a_model], palette[b_model]],
                      edgecolor="black", linewidth=0.6, capsize=8, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{a_model}\n(n={ns[0]})", f"{b_model}\n(n={ns[1]})"], fontsize=10)
        ax.set_title(METRIC_LABELS[mk], fontsize=11, weight="bold")
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02 * max(means),
                    f"{m:.2f}", ha="center", va="bottom", fontsize=10)
        # p-value annotation
        t = tests[mk]
        p = t["p"]
        p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        y_top = max(means) + max(sems) * 1.6 + 0.06 * max(means)
        ax.plot([0, 1], [y_top, y_top], color="black", linewidth=0.8)
        ax.text(0.5, y_top + 0.01 * max(means), p_str,
                ha="center", va="bottom", fontsize=10,
                color="black" if p < 0.05 else "grey",
                weight="bold" if p < 0.05 else "normal")
        ax.set_ylim(0, y_top * 1.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Behavioural signature  —  15 open-ended prompts × 5 sampling seeds × 300 tokens / model\n"
        "If Gemma's grammar-suppression apparatus is doing what the v3 finding says it does, all four metrics should be higher in Gemma.",
        fontsize=12, y=1.04,
    )

    out = ROOT / "reports" / "viz_behavior.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
