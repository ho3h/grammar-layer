"""Generate the smoking-gun case-study figure for the `capital-jp` prompt.

Two panels:
- Panel A (Gemma 2 2B): top-5 supporting features (bars going right) + top-5 opposing
  features (bars going left), labelled with their autointerp text. The visual story is
  "geographical/historical features promote Tokyo; grammar/predicate features suppress it."
- Panel B (GPT-2 small): same prompt, same chart structure. The story is "content features
  promote, content features oppose; no grammar layer."

A third box on the figure prints the before/after argmax: Gemma "Tokyo" → "a" under joint
ablation; GPT-2 doesn't predict Tokyo at all.

Usage:
    uv run python scripts/viz_smoking_gun.py
    # → reports/viz_smoking_gun.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _load_labels(path: str) -> dict[int, str]:
    raw = json.loads((ROOT / path).read_text())
    return {int(k): v.get("text", "") for k, v in raw.items() if isinstance(v, dict)}


def _trim_label(s: str, n: int = 50) -> str:
    s = s.strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _model_panel(ax, results_path: str, labels_path: str, prompt_id: str, title: str, color_sup: str, color_opp: str) -> dict:
    r = next(row for row in json.loads((ROOT / results_path).read_text())["results"] if row["id"] == prompt_id)
    labels = _load_labels(labels_path)
    sup = r["topk_supporting"][:5]
    opp = r["topk_opposing"][:5]

    y_sup = np.arange(len(sup))[::-1] + 1  # stack top → bottom
    y_opp = -(np.arange(len(opp))) - 1

    for i, e in enumerate(sup):
        ax.barh(y_sup[i], e["single_log_p_drop"], color=color_sup, alpha=0.85, edgecolor="black", linewidth=0.5, height=0.7)
        ax.text(e["single_log_p_drop"] + 0.03, y_sup[i],
                f" f{e['feature_index']}: {_trim_label(labels.get(e['feature_index'], '<no label>'))}",
                va="center", fontsize=8)
    for i, e in enumerate(opp):
        ax.barh(y_opp[i], e["single_log_p_drop"], color=color_opp, alpha=0.85, edgecolor="black", linewidth=0.5, height=0.7)
        ax.text(e["single_log_p_drop"] - 0.03, y_opp[i],
                f"f{e['feature_index']}: {_trim_label(labels.get(e['feature_index'], '<no label>'))} ",
                va="center", ha="right", fontsize=8)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=13, weight="bold", pad=14)
    ax.set_xlabel("single-feature Δ log P(target)   →   positive = supporting answer", fontsize=9)
    ax.set_yticks([])
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-len(opp) - 0.8, len(sup) + 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.text(1.75, len(sup) + 0.7, "supporting target →", ha="right", va="center", fontsize=9, color=color_sup, weight="bold")
    ax.text(-1.75, -len(opp) - 0.5, "← opposing target", ha="left", va="center", fontsize=9, color=color_opp, weight="bold")

    return {
        "id": r["id"],
        "prompt": r["prompt"],
        "target": r["target"],
        "baseline_argmax": r["baseline"]["argmax_token_str"],
        "baseline_log_p_target": r["baseline"]["log_p_target"],
        "joint_argmax": r["joint_topk_ablated"]["argmax_token_str"],
        "joint_log_p_target": r["joint_topk_ablated"]["log_p_target"],
        "drop": r["log_p_drop_vs_baseline"],
    }


def main() -> None:
    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.6, 0.5, 4.5], hspace=0.15, wspace=0.30)
    ax_prompt = fig.add_subplot(gs[0, :])
    ax_box_g = fig.add_subplot(gs[1, 0])
    ax_box_p = fig.add_subplot(gs[1, 1])
    ax_g = fig.add_subplot(gs[2, 0])
    ax_p = fig.add_subplot(gs[2, 1])

    g_info = _model_panel(
        ax_g, "reports/load_bearing_pos10_gemma_50.json", "data/labels_cache.json",
        "capital-jp", "Gemma 2 2B  ·  layer 20 of 26",
        color_sup="#2a7fff", color_opp="#d6324b",
    )
    gp_info = _model_panel(
        ax_p, "reports/load_bearing_pos10_gpt2_50.json", "data/labels_cache_gpt2.json",
        "capital-jp", "GPT-2 small  ·  layer 8 of 12",
        color_sup="#2a7fff", color_opp="#d6324b",
    )

    prompt = g_info["prompt"]
    target = g_info["target"].strip()

    ax_prompt.axis("off")
    ax_prompt.text(0.5, 0.55, f'"{prompt}"   →   target  {target!r}',
                   ha="center", va="center", fontsize=18, weight="bold")
    ax_prompt.text(0.5, 0.05,
                   "Ablating the top-10 supporting features in each model — same protocol, same prompt, same target.",
                   ha="center", va="center", fontsize=10, style="italic", color="#555")

    for ax, info, ec, fc in [(ax_box_g, g_info, "#2a7fff", "#eef2f7"),
                             (ax_box_p, gp_info, "#d6324b", "#fbeef0")]:
        ax.axis("off")
        ax.text(0.5, 0.5,
                f'argmax: {info["baseline_argmax"]!r}  →  {info["joint_argmax"]!r}\n'
                f'log P({target}): {info["baseline_log_p_target"]:+.2f}  →  {info["joint_log_p_target"]:+.2f}        '
                f'Δ {info["drop"]:+.2f} nats',
                ha="center", va="center", fontsize=11, family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", fc=fc, ec=ec, lw=1.5),
                transform=ax.transAxes)

    out = ROOT / "reports" / "viz_smoking_gun.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
