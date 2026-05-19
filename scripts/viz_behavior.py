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
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "reports/behavior_metrics.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "reports/viz_behavior.png"
    data = json.loads((ROOT / src).read_text())
    summary = data["summary"]
    models = data["models"]
    if len(models) < 2:
        raise SystemExit("Need at least two models in metrics file")

    # Sort models: "yes-inversion" first (Gemma family, Pythia), then GPT-2
    PRIORITY = {"gemma": 0, "gemma_1_2b": 1, "pythia_70m": 2, "gpt2": 3, "gemma_9b": 4, "mistral_7b": 5, "qwen3_1_7b": 6}
    models = sorted(models, key=lambda m: PRIORITY.get(m, 99))

    fig, axes = plt.subplots(2, 2, figsize=(2.5 + 2.4 * len(models), 9), constrained_layout=True)
    metrics = list(METRIC_LABELS.keys())

    # Color: blue for "yes inversion" (gemma family, pythia), red for "no" (gpt2, gemma_9b)
    color_inv_yes = "#2a7fff"
    color_inv_no = "#d6324b"
    YES_INVERSION = {"gemma", "gemma_1_2b", "pythia_70m"}
    palette = {m: (color_inv_yes if m in YES_INVERSION else color_inv_no) for m in models}

    for ax, mk in zip(axes.flat, metrics):
        per_model = summary["per_model"]
        means = [per_model[m][mk]["mean"] for m in models]
        stds = [per_model[m][mk]["std"] for m in models]
        ns = [per_model[m][mk]["n"] for m in models]
        sems = [s / max(n ** 0.5, 1) for s, n in zip(stds, ns)]
        colors = [palette[m] for m in models]

        x = np.arange(len(models))
        bars = ax.bar(x, means, yerr=sems, color=colors,
                      edgecolor="black", linewidth=0.6, capsize=8, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{m}\n(n={n})" for m, n in zip(models, ns)], fontsize=9.5)
        ax.set_title(METRIC_LABELS[mk], fontsize=11, weight="bold")
        for bar, mv in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02 * max(means),
                    f"{mv:.2f}", ha="center", va="bottom", fontsize=9)

        # Annotate p-values for each pairwise test against gpt2 (if available)
        if "gpt2" in models:
            ann_y = max(means) + max(sems) * 1.6 + 0.06 * max(means)
            ax.set_ylim(0, ann_y * 1.25)
            for i, m in enumerate(models):
                if m == "gpt2":
                    continue
                test_key = f"{m}_vs_gpt2" if f"{m}_vs_gpt2" in summary["tests"] else f"gpt2_vs_{m}"
                if test_key in summary["tests"]:
                    t = summary["tests"][test_key][mk]
                    p = t["p"]
                    p_str = "p<.001" if p < 0.001 else f"p={p:.3f}"
                    ax.text(i, means[i] + sems[i] * 1.6 + 0.05 * max(means), p_str,
                            ha="center", va="bottom", fontsize=8,
                            color="black" if p < 0.05 else "grey",
                            weight="bold" if p < 0.05 else "normal")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Behavioural signature  —  15 open-ended prompts × 5 sampling seeds × 300 tokens / model\n"
        "Blue: models with the internal grammar-suppression apparatus (per the v3 finding). Red: models without it. "
        "p-values are Welch's t-test against GPT-2 small.",
        fontsize=12, y=1.04,
    )

    out = ROOT / out_path
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
