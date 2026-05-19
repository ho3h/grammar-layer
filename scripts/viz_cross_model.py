"""Render the cross-model universality results from reports/cross_model_universality.json.

Two images:
- reports/viz_cross_model_heatmap.png — cosine-sim heatmap, Gemma communities × GPT-2 communities
  (rows/cols ordered by Hungarian assignment, so the diagonal is the best match per row).
- reports/viz_cross_model_concepts.png — for each labelled concept, side-by-side bars
  showing top-2 communities in Gemma vs GPT-2, with the centroid-cosine of the best pair.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from neograph.config import PATHS

plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"


def heatmap(data: dict, out_path):
    assignment = data["hungarian_assignment"]
    g_cids = sorted({a["gemma_cid"] for a in assignment})
    p_cids = sorted({a["gpt2_cid"] for a in assignment})
    g_idx = {c: i for i, c in enumerate(g_cids)}
    p_idx = {c: i for i, c in enumerate(p_cids)}
    M = np.zeros((len(g_cids), len(p_cids)))
    for a in assignment:
        M[g_idx[a["gemma_cid"]], p_idx[a["gpt2_cid"]]] = a["centroid_cosine"]
    # Reorder rows/cols by Hungarian-assigned diagonal
    order_rows = [a["gemma_cid"] for a in assignment]
    order_cols = [a["gpt2_cid"] for a in assignment]
    # Build the full pairwise similarity from the assignment (only diagonal known here —
    # rerun a full pairwise calc if you need off-diagonal; this viz just shows the
    # best-match scores per row.)
    fig, ax = plt.subplots(figsize=(max(6, len(p_cids) * 0.4), max(4, len(g_cids) * 0.4)))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=-0.2, vmax=1.0)
    ax.set_xticks(range(len(p_cids)))
    ax.set_xticklabels(p_cids, fontsize=7, rotation=90)
    ax.set_yticks(range(len(g_cids)))
    ax.set_yticklabels(g_cids, fontsize=7)
    ax.set_xlabel("GPT-2 community id")
    ax.set_ylabel("Gemma community id")
    ax.set_title("Cross-model label-centroid cosine (Hungarian-matched pairs)")
    plt.colorbar(im, ax=ax, label="cosine")
    plt.savefig(out_path)
    plt.close()
    print(f"wrote {out_path}")


def concept_bars(data: dict, out_path):
    concepts = data["concept_alignment"]
    aligns = data["concept_match"]
    fig, axes = plt.subplots(1, len(aligns), figsize=(4 * len(aligns), 4), sharey=False)
    if len(aligns) == 1:
        axes = [axes]
    for ax, concept in zip(axes, aligns):
        name = concept["concept"]
        g_top = concept["gemma_top"][:3]
        p_top = concept["gpt2_top"][:3]
        x = np.arange(max(len(g_top), len(p_top)))
        width = 0.4
        if g_top:
            ax.bar(x - width / 2, [n for _, n in g_top], width, label="Gemma 2 2B", color="#1f77b4",
                   tick_label=[f"cid {c}" for c, _ in g_top])
        if p_top:
            xp = np.arange(len(p_top))
            ax.bar(xp + width / 2, [n for _, n in p_top], width, label="GPT-2 small", color="#ff7f0e")
        ax.set_title(name)
        ax.set_xlabel("Top communities by concept-feature count")
        ax.set_ylabel("# features")
        align = next((c for c in concepts if c["concept"] == name), None)
        if align:
            symbol = "✓" if align["aligned"] else "✗"
            cos = align["hungarian_centroid_cos"]
            ax.text(
                0.5, 0.95,
                f"Hungarian: {symbol}  cos={cos:+.2f}" if cos is not None else f"Hungarian: {symbol}",
                transform=ax.transAxes, ha="center", va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
            )
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Concept-level community alignment across models", y=1.04)
    plt.savefig(out_path)
    plt.close()
    print(f"wrote {out_path}")


def main():
    data_path = PATHS.reports / "cross_model_universality.json"
    if not data_path.exists():
        print(f"Run scripts/cross_model_universality.py first ({data_path} missing)")
        return 1
    data = json.loads(data_path.read_text())
    heatmap(data, PATHS.reports / "viz_cross_model_heatmap.png")
    concept_bars(data, PATHS.reports / "viz_cross_model_concepts.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
