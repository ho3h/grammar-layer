"""Pythia 70M smoking-gun viz — capital-jp, 3 models on one page.

Three-panel figure showing per-feature supporting + opposing decomposition for the
"capital of Japan is" prompt across:
- Gemma 2 2B (the original): grammar features dominate the opposing side
- Pythia 70M (the size-decoupling control): grammar features also dominate the opposing side
- GPT-2 small (the bilateral comparison): content features only

The point: Pythia 70M is half the size of GPT-2 small but shows the same grammar-
suppression structure as Gemma 2 2B. The grammar layer is not a scale signature.

Usage:
    uv run python scripts/viz_smoking_gun_pythia.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _load_labels(p):
    raw = json.loads((ROOT / p).read_text())
    return {int(k): v.get("text", "") for k, v in raw.items() if isinstance(v, dict)}


def _trim(s, n=45):
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _panel(ax, results_path, labels_path, title, prompt_id="capital-jp"):
    r = next(row for row in json.loads((ROOT / results_path).read_text())["results"] if row["id"] == prompt_id)
    labels = _load_labels(labels_path)
    sup = r["topk_supporting"][:5]
    opp = r["topk_opposing"][:5]

    y_sup = np.arange(len(sup))[::-1] + 1
    y_opp = -(np.arange(len(opp))) - 1

    for i, e in enumerate(sup):
        ax.barh(y_sup[i], e["single_log_p_drop"], color="#2a7fff", alpha=0.85,
                edgecolor="black", linewidth=0.5, height=0.7)
        ax.text(e["single_log_p_drop"] + 0.03, y_sup[i],
                f" f{e['feature_index']}: {_trim(labels.get(e['feature_index'], '<no label>'))}",
                va="center", fontsize=8)
    for i, e in enumerate(opp):
        ax.barh(y_opp[i], e["single_log_p_drop"], color="#d6324b", alpha=0.85,
                edgecolor="black", linewidth=0.5, height=0.7)
        ax.text(e["single_log_p_drop"] - 0.03, y_opp[i],
                f"f{e['feature_index']}: {_trim(labels.get(e['feature_index'], '<no label>'))} ",
                va="center", ha="right", fontsize=8)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=12, weight="bold", pad=12)
    ax.set_xlabel("single-feature Δ log P(target)", fontsize=8.5)
    ax.set_yticks([])
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-len(opp) - 0.7, len(sup) + 1.0)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.spines["left"].set_visible(False)
    ax.text(1.75, len(sup) + 0.6, "supporting →", ha="right", va="center", fontsize=8, color="#2a7fff", weight="bold")
    ax.text(-1.75, -len(opp) - 0.4, "← opposing", ha="left", va="center", fontsize=8, color="#d6324b", weight="bold")
    return r


def main() -> None:
    fig = plt.figure(figsize=(20, 7.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.5, 4.5], hspace=0.05, wspace=0.25)
    ax_title = fig.add_subplot(gs[0, :])
    axes = [fig.add_subplot(gs[1, i]) for i in range(3)]

    g  = _panel(axes[0], "reports/load_bearing_pos10_gemma_50.json",       "data/labels_cache.json",            "Gemma 2 2B  ·  2.6B params  ·  L20/26")
    p  = _panel(axes[1], "reports/load_bearing_pos10_pythia_70m_50.json", "data/labels_cache_pythia_70m.json", "Pythia 70M  ·  70M params  ·  L5/6")
    gp = _panel(axes[2], "reports/load_bearing_pos10_gpt2_50.json",       "data/labels_cache_gpt2.json",        "GPT-2 small  ·  124M params  ·  L8/12")

    ax_title.axis("off")
    ax_title.text(0.5, 0.85, '"The capital of Japan is"  →  target  \'Tokyo\'',
                  ha="center", va="top", fontsize=16, weight="bold")
    ax_title.text(0.5, 0.35,
                  "Same prompt, three models spanning a 37× parameter range. Pythia 70M (middle, smaller than GPT-2) "
                  "recruits the same grammar-suppression apparatus as Gemma 2 2B — its top opposer is feature 23527 "
                  "(literally labelled 'occurrences of the verb is and its various forms'). GPT-2 small recruits only content features.",
                  ha="center", va="top", fontsize=10, style="italic", color="#555")

    out = ROOT / "reports" / "viz_smoking_gun_pythia.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
