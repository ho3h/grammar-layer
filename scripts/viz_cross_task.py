"""Visualise cross-task generalization: grammar opposer share per (model, category).

Reads reports/cross_task_analysis.json. Writes reports/viz_cross_task.png.

Shows that the fingerprint is capital-specific: high enrichment on capitals (from
the original benchmark, hard-coded here for comparison) and near-zero on the four
new categories — currency, language, composition, continent.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Capital baseline from the original 50-prompt run, reports/load_bearing_pos10_*_50.json
# These are the percentages from the writeup's enrichment table.
CAPITAL_BASELINE = {
    "gemma":      26.7,
    "gemma_1_2b": 38.3,  # Gemma 1 2B's capital opp%grammar
    "pythia_70m": 28.3,
    "gpt2":        0.0,
}

CATEGORIES = ["capital", "composition", "continent", "currency", "language"]
MODELS = ["gemma", "gemma_1_2b", "pythia_70m", "gpt2"]
MODEL_LABEL = {
    "gemma":      "Gemma 2 2B",
    "gemma_1_2b": "Gemma 1 2B",
    "pythia_70m": "Pythia 70M",
    "gpt2":       "GPT-2 small",
}
MODEL_COLOR = {
    "gemma":      "#1f4e7a",
    "gemma_1_2b": "#5e8bb4",
    "pythia_70m": "#c44d2e",
    "gpt2":       "#9ea1a3",
}


def main() -> None:
    data = json.loads((ROOT / "reports/cross_task_analysis.json").read_text())

    # Build matrix: rows = models, cols = categories
    matrix = np.zeros((len(MODELS), len(CATEGORIES)))
    for r, model in enumerate(MODELS):
        # Capital baseline first
        matrix[r, 0] = CAPITAL_BASELINE.get(model, 0.0)
        for c, cat in enumerate(CATEGORIES[1:], start=1):
            v = data.get(model, {}).get(cat, {}).get("grammar_share_opp_pct", 0.0)
            matrix[r, c] = v

    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.2
    x = np.arange(len(CATEGORIES))
    for r, model in enumerate(MODELS):
        offsets = (r - 1.5) * width
        ax.bar(x + offsets, matrix[r], width, color=MODEL_COLOR[model], label=MODEL_LABEL[model])

    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES, fontsize=11)
    ax.set_ylabel("Grammar share in top-5 opposing features (%)")
    ax.set_title(
        "Cross-task scope: the grammar fingerprint is capital-specific\n"
        "Opposing-side grammar percentage per (model, task category)",
        fontsize=11,
    )
    ax.axhline(y=0, color="grey", lw=0.5)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 45)

    fig.tight_layout()
    out = ROOT / "reports/viz_cross_task.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
