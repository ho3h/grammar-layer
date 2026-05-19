"""Visualise the 'feature fingerprint' across capital-completion prompts.

For each of the 6 capital prompts, show the top-5 opposing features as labelled cells.
Cells are coloured by whether the feature is one of Gemma's two universal grammar opposers
(15596 'forms of to-be', 10142 'word is'). The cross-prompt consistency in Gemma — those
two features appearing in every capital prompt's top-5 opposing — is the headline.

Usage:
    uv run python scripts/viz_capital_fingerprint.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent

GEMMA_FINGERPRINT = {15596, 10142}  # 'to-be' + 'is' features in Gemma


def _load_labels(p):
    return {int(k): v.get("text", "") for k, v in json.loads((ROOT / p).read_text()).items() if isinstance(v, dict)}


def _capital_results(path):
    return [r for r in json.loads((ROOT / path).read_text())["results"] if r["category"] == "capital"]


def _trim(s, n=44):
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def main() -> None:
    order = {"capital-fr": 0, "capital-de": 1, "capital-it": 2, "capital-es": 3, "capital-ru": 4, "capital-jp": 5}
    g_caps = sorted(_capital_results("reports/load_bearing_pos10_gemma_50.json"), key=lambda r: order.get(r["id"], 99))
    p_caps = sorted(_capital_results("reports/load_bearing_pos10_gpt2_50.json"), key=lambda r: order.get(r["id"], 99))
    g_labels = _load_labels("data/labels_cache.json")
    p_labels = _load_labels("data/labels_cache_gpt2.json")

    n_caps = len(g_caps)
    n_cols = 5  # top-5 opposing per prompt
    fig, (ax_g, ax_p) = plt.subplots(1, 2, figsize=(20, 7), constrained_layout=True)

    def _draw(ax, caps, labels, model_name, fingerprint):
        ax.set_xlim(-0.5, n_cols - 0.5)
        ax.set_ylim(-0.5, n_caps - 0.5)
        ax.invert_yaxis()
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels([f"opp #{i+1}" for i in range(n_cols)], fontsize=10)
        ax.set_yticks(range(n_caps))
        ax.set_yticklabels(
            [f"{caps[i]['prompt'].replace('The capital of ', '').replace(' is','')} → {caps[i]['target'].strip()}"
             for i in range(n_caps)],
            fontsize=10, weight="bold",
        )
        ax.set_title(model_name, fontsize=13, weight="bold", pad=12)
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for r_i, r in enumerate(caps):
            opp = r["topk_opposing"][:n_cols]
            for c_i, e in enumerate(opp):
                fidx = e["feature_index"]
                lbl = labels.get(fidx, "<no label>")
                is_fp = fidx in fingerprint
                bg = "#d6324b" if is_fp else "#fce5e9"
                fg = "white" if is_fp else "#444"
                rect = mpatches.FancyBboxPatch(
                    (c_i - 0.45, r_i - 0.42), 0.9, 0.84,
                    boxstyle="round,pad=0.02",
                    facecolor=bg, edgecolor="#d6324b" if is_fp else "#e0a8b0",
                    linewidth=1.5 if is_fp else 0.6,
                )
                ax.add_patch(rect)
                weight = "bold" if is_fp else "normal"
                ax.text(c_i, r_i - 0.18, f"f{fidx}",
                        ha="center", va="center", fontsize=9, weight=weight, color=fg)
                ax.text(c_i, r_i + 0.18, _trim(lbl, 38),
                        ha="center", va="center", fontsize=7, color=fg, wrap=True)

    _draw(ax_g, g_caps, g_labels, "Gemma 2 2B  ·  layer 20  ·  top-5 opposing features per prompt", GEMMA_FINGERPRINT)
    _draw(ax_p, p_caps, p_labels, "GPT-2 small  ·  layer 8  ·  top-5 opposing features per prompt", set())

    fig.suptitle(
        "Two features in Gemma act as 'top-2 opposers' on every capital prompt:\n"
        "f15596 ('past and present tense forms of the verb to-be') and f10142 ('instances of the word is').\n"
        "GPT-2 has no comparable cross-prompt grammar fingerprint — its opposers are content-thematic.",
        fontsize=11.5,
    )

    out = ROOT / "reports" / "viz_capital_fingerprint.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
