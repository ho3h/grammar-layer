"""Visualise the bidirectional steering result.

Two panels (Gemma f15596 and Pythia f23527), each showing:
- log P(target) and log P(generic max) as a function of activation scale
- Markers for argmax-flip events

Reads reports/amp_<model>_f<feat>.json. Writes reports/viz_amplification.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_run(model: str, feat: int) -> dict:
    return json.loads((ROOT / "reports" / f"amp_{model}_f{feat}.json").read_text())


def plot_panel(ax: plt.Axes, run: dict, title: str) -> None:
    summary = run["summary"]
    scales = [float(s) for s in summary["scales"]]
    logp_target = [summary["mean_log_p_target_by_scale"][str(s)] for s in scales]
    logp_generic = [summary["mean_log_p_generic_max_by_scale"][str(s)] for s in scales]
    hit_rate = [summary["hit_rate_target_by_scale"][str(s)] for s in scales]
    generic_rate = [summary["generic_rate_by_scale"][str(s)] for s in scales]

    ax.plot(scales, logp_target, "o-", color="#1f4e7a", label="log P(target)", lw=2)
    ax.plot(scales, logp_generic, "s--", color="#c44d2e", label="log P(generic max)", lw=2)
    ax.set_xscale("symlog", linthresh=0.5)
    ax.set_xlabel("Activation scale on the fingerprint feature")
    ax.set_ylabel("Mean log probability (6 capital prompts)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")

    # Annotate argmax-flip events
    for i, s in enumerate(scales):
        if hit_rate[i] == 0 and i > 0 and hit_rate[i - 1] > 0:
            ax.axvline(s, color="grey", linestyle=":", alpha=0.7)
            ax.text(s, logp_target[i], "  argmax flips off target", fontsize=8, color="grey", va="bottom")


def main() -> None:
    panels = [
        ("gemma",      15596, "Gemma 2 2B  •  f15596  •  'forms of the verb \"to be\"'  →  argmax flips to \" not\""),
        ("gemma_1_2b",  5541, "Gemma 1 2B  •  f5541  •  'instances of the verb \"is\"'  →  argmax stays on \" a\""),
        ("pythia_70m", 23527, "Pythia 70M  •  f23527  •  'occurrences of the verb \"is\" and its forms'  →  argmax stays on \" a\""),
    ]
    # Optional 9B panel if available
    try:
        load_run("gemma_9b_l31", 6341)
        panels.append(("gemma_9b_l31", 6341, "Gemma 2 9B (L31)  •  f6341  •  'instances of the verb \"is\" and its variations'"))
    except FileNotFoundError:
        pass

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, (model, feat, title) in zip(axes, panels):
        run = load_run(model, feat)
        plot_panel(ax, run, title)

    fig.suptitle(
        "Bidirectional steering of the copula opposer: monotone log-P collapse in every "
        "inversion model;\nargmax flips to ' not' in Gemma 2 2B only (the negation attractor)",
        fontsize=12,
    )
    fig.tight_layout()
    out = ROOT / "reports" / "viz_amplification.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
