"""5-model cross-model capital fingerprint viz.

For each of the 5 models with Neuronpedia labels (Pythia 70M, GPT-2 small, Gemma 1 2B,
Gemma 2 2B, Gemma 2 9B), shows top-5 opposing features across the 6 capital prompts.
Grammar-flavored features are highlighted dark; content features faded.

The point of the figure: visually, three models (Pythia 70M, Gemma 1 2B, Gemma 2 2B)
have the same coordinated pattern — a small set of grammar features appearing as top
opposers on every capital prompt. Two (GPT-2 small, Gemma 2 9B at L20/42) don't.

Usage:
    uv run python scripts/viz_cross_model_fingerprint.py
    # → reports/viz_cross_model_fingerprint.png
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent

GRAMMAR_PAT = re.compile(
    r"\b(is|to[- ]?be|copula|verb|tense|past tense|present tense|conjunction|"
    r"article|preposition|punctuation|grammar|function word|determiner|"
    r"pronoun|auxiliary)\b",
    re.I,
)

MODELS = [
    ("Pythia 70M",       "reports/load_bearing_pos10_pythia_70m_50.json",  "data/labels_cache_pythia_70m.json"),
    ("GPT-2 small",      "reports/load_bearing_pos10_gpt2_50.json",         "data/labels_cache_gpt2.json"),
    ("Gemma 1 2B",       "reports/load_bearing_pos10_gemma_1_2b_50.json",  "data/labels_cache_gemma_1_2b.json"),
    ("Gemma 2 2B",       "reports/load_bearing_pos10_gemma_50.json",       "data/labels_cache.json"),
    ("Gemma 2 9B",       "reports/load_bearing_pos10_gemma_9b_50.json",    "data/labels_cache_gemma_9b.json"),
]

CAP_ORDER = {"capital-fr": 0, "capital-de": 1, "capital-it": 2,
             "capital-es": 3, "capital-ru": 4, "capital-jp": 5}


def _load_labels(path: str) -> dict[int, str]:
    p = ROOT / path
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    out: dict[int, str] = {}
    for k, v in raw.items():
        if isinstance(v, dict) and "text" in v:
            out[int(k)] = v["text"]
    return out


def _is_grammar(label: str | None) -> bool:
    return bool(label and GRAMMAR_PAT.search(label))


def _capital_results(path: str) -> list[dict]:
    p = ROOT / path
    if not p.exists():
        return []
    return [r for r in json.loads(p.read_text())["results"] if r["category"] == "capital"]


def _trim(s: str, n: int = 36) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def main() -> None:
    rows_per_model = {}
    for display, results_path, labels_path in MODELS:
        labels = _load_labels(labels_path)
        caps = _capital_results(results_path)
        if not caps:
            continue
        caps.sort(key=lambda r: CAP_ORDER.get(r["id"], 99))
        rows_per_model[display] = (caps, labels)

    n_models = len(rows_per_model)
    n_cols = 3  # top-3 opposing - cleaner than top-5
    n_caps = 6

    fig, axes = plt.subplots(n_caps, n_models,
                             figsize=(3.4 * n_models, 1.4 * n_caps + 1.2),
                             constrained_layout=True)

    fig.suptitle(
        "Capital-completion fingerprint  ·  top-5 opposing features per (model × prompt)\n"
        "Grammar-labelled features highlighted dark; content features faded. "
        "Pythia 70M, Gemma 1 2B, and Gemma 2 2B show a coordinated grammar fingerprint; "
        "GPT-2 small and Gemma 2 9B (at L20/42) don't.",
        fontsize=11.5, y=1.005,
    )

    for col, (display, (caps, labels)) in enumerate(rows_per_model.items()):
        for row_idx, r in enumerate(caps):
            ax = axes[row_idx, col]
            ax.set_xlim(-0.5, n_cols - 0.5)
            ax.set_ylim(0, 1)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            opp = r["topk_opposing"][:n_cols]
            for ci, e in enumerate(opp):
                fidx = e["feature_index"]
                lbl = labels.get(fidx, "<no label>")
                is_g = _is_grammar(lbl)
                fc = "#d6324b" if is_g else "#f4e7e8"
                ec = "#d6324b" if is_g else "#e0c0c5"
                tc = "white" if is_g else "#444"
                rect = mpatches.FancyBboxPatch(
                    (ci - 0.46, 0.05), 0.92, 0.92,
                    boxstyle="round,pad=0.02",
                    facecolor=fc, edgecolor=ec, linewidth=1.6 if is_g else 0.5,
                )
                ax.add_patch(rect)
                weight = "bold" if is_g else "normal"
                ax.text(ci, 0.78, f"f{fidx}", ha="center", va="center",
                        fontsize=9.5, weight=weight, color=tc)
                ax.text(ci, 0.38, _trim(lbl, 30), ha="center", va="center",
                        fontsize=6.5, color=tc, wrap=True)
            if col == 0:
                prompt_short = r["prompt"].replace("The capital of ", "").replace(" is", "")
                ax.text(-0.65, 0.5, f"{prompt_short}\n→ {r['target'].strip()}",
                        ha="right", va="center", fontsize=10, weight="bold")
            if row_idx == 0:
                ax.set_title(display, fontsize=12, weight="bold", pad=14)

    out = ROOT / "reports" / "viz_cross_model_fingerprint.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
