"""The hero visualisation — Goodfire visual grammar applied to the routing finding.

UMAP every SAE feature in each model to 3D. Render as a gray point cloud with a
projection plane below (Goodfire aesthetic). Overlay the features that participate in
the 12 circuits, coloured by how many circuits they appear in. Gemma should show a
bright glowing core (backbone features hit by all 12 prompts). GPT-2 should show
scattered embers (different features lit for different prompts).

Caption: "Same 12 questions. Same 12 answers. Two different routes."

Outputs:
- reports/viz_grammar_layer.png — the hero static image (side-by-side)
- reports/viz_grammar_layer_gemma.png — Gemma panel alone, 1200×1200
- reports/viz_grammar_layer_gpt2.png  — GPT-2 panel alone, 1200×1200
- reports/viz_vocab_links.png — bonus third panel: cross-model label-similarity arcs
- reports/viz_grammar_layer.gif — animated cycle through the 12 prompts (Twitter ready)
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed for 3D projection
import umap

from neograph.config import PATHS, SAE as GEMMA_SAE
from neograph.cypher import NeographClient
from neograph.util import get_logger

log = get_logger("neograph.viz.grammar")

GPT2_SAE_ID = "gpt2-small-res-jb/L8"
GPT2_SAE_RELEASE = "gpt2-small-res-jb"
GPT2_SAE_ID_ATTR = "blocks.8.hook_resid_pre"

# Goodfire-style aesthetic
BG_GRAY = "#bdbdbd"
BG_GRAY_ALPHA = 0.05
SHADOW_GRAY = "#888888"
SHADOW_ALPHA = 0.035
FIGSIZE = (16, 8.5)


@dataclass
class ModelData:
    nickname: str
    pretty: str
    n_features: int
    embedding_3d: np.ndarray  # (n_features, 3)
    n_circuits_per_feature: np.ndarray  # (n_features,) int
    backbone_idx: np.ndarray  # indices of top-K backbone features
    circuit_per_prompt: dict[str, set[int]]  # prompt_id -> set of feature indices


def load_decoder_weights():
    """Load decoder weights via sae_lens for both models."""
    from sae_lens import SAE as SaeLensSAE

    log.info("Loading Gemma SAE weights ...")
    gemma_sae = SaeLensSAE.from_pretrained(
        release=GEMMA_SAE.release, sae_id=GEMMA_SAE.sae_id, device="cpu"
    )
    W_dec_gemma = gemma_sae.W_dec.detach().float().cpu().numpy()
    log.info("Gemma W_dec shape: %s", W_dec_gemma.shape)

    log.info("Loading GPT-2 SAE weights ...")
    gpt2_sae = SaeLensSAE.from_pretrained(
        release=GPT2_SAE_RELEASE, sae_id=GPT2_SAE_ID_ATTR, device="cpu"
    )
    W_dec_gpt2 = gpt2_sae.W_dec.detach().float().cpu().numpy()
    log.info("GPT-2 W_dec shape: %s", W_dec_gpt2.shape)

    return W_dec_gemma, W_dec_gpt2


def fit_umap_3d(W: np.ndarray, label: str, random_state: int = 42) -> np.ndarray:
    log.info("Running UMAP on %s (%d × %d) → 3D ...", label, W.shape[0], W.shape[1])
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=random_state,
        n_jobs=1,
    )
    Z = reducer.fit_transform(W)
    log.info("%s UMAP done; range: x=[%.2f,%.2f] y=[%.2f,%.2f] z=[%.2f,%.2f]",
             label, Z[:, 0].min(), Z[:, 0].max(), Z[:, 1].min(), Z[:, 1].max(), Z[:, 2].min(), Z[:, 2].max())
    return Z


def pull_circuits(c: NeographClient, model_nickname: str, sae_id: str) -> dict[str, set[int]]:
    """For each Circuit of this model, return the set of SAEFeature indices in its INCLUDES edges."""
    rows = c.run(
        """
        MATCH (cir:Circuit {model: $model})-[inc:INCLUDES]->(f:SAEFeature)
        WHERE f.sae_id = $sae_id
        RETURN cir.id AS circuit, cir.prompt_id AS prompt_id, collect(f.index) AS features
        """,
        model=model_nickname, sae_id=sae_id,
    )
    out = {}
    for r in rows:
        out[r["prompt_id"]] = set(int(i) for i in r["features"])
    return out


def render_panel(ax, data: ModelData, view_init=(22, 30), n_max_global: int | None = None):
    """One panel: gray base cloud + colored overlay + projection plane.

    Sizes and colours are normalised against `n_max_global` so the two panels are
    visually comparable (Gemma's max-12 feature looks the same intensity as a
    GPT-2 feature also recruited 12 times).
    """
    Z = data.embedding_3d
    counts = data.n_circuits_per_feature
    n_max = n_max_global if n_max_global else max(int(counts.max()), 1)

    z_floor = float(Z[:, 2].min()) - 0.5 * (float(Z[:, 2].max()) - float(Z[:, 2].min()))

    # Base cloud (all features, faint gray)
    ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=BG_GRAY, alpha=BG_GRAY_ALPHA, s=2.4, linewidth=0)
    ax.scatter(Z[:, 0], Z[:, 1], np.full(Z.shape[0], z_floor),
               c=SHADOW_GRAY, alpha=SHADOW_ALPHA, s=2.4, linewidth=0)

    cmap = plt.cm.inferno

    # Tier 1: faint, single-circuit lit features (content-specific)
    tier1 = (counts == 1)
    if tier1.any():
        ax.scatter(Z[tier1, 0], Z[tier1, 1], Z[tier1, 2],
                   c="#5a3a8a", alpha=0.55, s=14, linewidth=0)

    # Tier 2: 2-3 circuits — purple/red
    tier2 = (counts >= 2) & (counts <= 3)
    if tier2.any():
        c_norm = counts[tier2] / n_max
        ax.scatter(Z[tier2, 0], Z[tier2, 1], Z[tier2, 2],
                   c=c_norm, cmap=cmap, vmin=0, vmax=1,
                   s=40, alpha=0.85, linewidth=0)

    # Tier 3: 4-6 circuits — orange
    tier3 = (counts >= 4) & (counts <= 6)
    if tier3.any():
        c_norm = counts[tier3] / n_max
        # Draw halo first (larger, low alpha), then bright core
        ax.scatter(Z[tier3, 0], Z[tier3, 1], Z[tier3, 2],
                   c=c_norm, cmap=cmap, vmin=0, vmax=1,
                   s=180, alpha=0.18, linewidth=0)
        ax.scatter(Z[tier3, 0], Z[tier3, 1], Z[tier3, 2],
                   c=c_norm, cmap=cmap, vmin=0, vmax=1,
                   s=80, alpha=0.95, linewidth=0)

    # Tier 4: 7-9 circuits — yellow
    tier4 = (counts >= 7) & (counts <= 9)
    if tier4.any():
        c_norm = counts[tier4] / n_max
        ax.scatter(Z[tier4, 0], Z[tier4, 1], Z[tier4, 2],
                   c=c_norm, cmap=cmap, vmin=0, vmax=1,
                   s=380, alpha=0.15, linewidth=0)
        ax.scatter(Z[tier4, 0], Z[tier4, 1], Z[tier4, 2],
                   c=c_norm, cmap=cmap, vmin=0, vmax=1,
                   s=160, alpha=0.95, linewidth=0)

    # Tier 5: 10+ circuits — burning white-yellow
    tier5 = counts >= 10
    if tier5.any():
        # Multi-layer halo for that glowing core
        for radius, alpha in [(900, 0.06), (550, 0.12), (300, 0.30)]:
            ax.scatter(Z[tier5, 0], Z[tier5, 1], Z[tier5, 2],
                       c="#ffd866", s=radius, alpha=alpha, linewidth=0)
        # Bright core
        ax.scatter(Z[tier5, 0], Z[tier5, 1], Z[tier5, 2],
                   c="#ffffff", s=120, alpha=1.0, linewidth=0)

    # Lit shadow projection on plane (subtle)
    lit_any = counts >= 1
    if lit_any.any():
        ax.scatter(Z[lit_any, 0], Z[lit_any, 1], np.full(lit_any.sum(), z_floor),
                   c=counts[lit_any] / n_max, cmap=cmap, vmin=0, vmax=1,
                   s=8, alpha=0.30, linewidth=0)

    # Annotate the brightest feature with a halo
    if data.backbone_idx.size and counts[data.backbone_idx[0]] >= 7:
        top = int(data.backbone_idx[0])
        x, y, z = Z[top]
        ax.scatter([x], [y], [z], facecolors="none", edgecolors="white",
                   s=420, linewidth=1.6, zorder=20)

    ax.set_title(data.pretty, color="#111", fontsize=13, pad=8)
    ax.set_axis_off()
    ax.view_init(*view_init)
    ax.set_box_aspect((1, 1, 0.9))
    return z_floor


def annotate_backbone(ax, data: ModelData, label_map: dict[int, tuple[str, str]], counts_min: int = 6):
    """Draw text labels next to the top backbone features.

    label_map: {feature_index: (short_label, color)}
    """
    Z = data.embedding_3d
    counts = data.n_circuits_per_feature
    for fidx, (label, color) in label_map.items():
        if fidx >= data.n_features or counts[fidx] < counts_min:
            continue
        x, y, z = Z[fidx]
        ax.text(x, y, z + 0.4, label, color=color, fontsize=8.5, ha="left",
                weight="bold", zorder=30,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=color, alpha=0.85, linewidth=0.8))


def build_model_data(c: NeographClient, nickname: str, pretty: str, sae_id: str,
                      W_dec: np.ndarray, embed_3d: np.ndarray) -> ModelData:
    circuits = pull_circuits(c, nickname, sae_id)
    n_features = W_dec.shape[0]
    counts = np.zeros(n_features, dtype=np.int32)
    for feats in circuits.values():
        for f in feats:
            if 0 <= f < n_features:
                counts[f] += 1
    # Backbone: top-15 by n_circuits
    backbone_idx = np.argsort(-counts)[:15]
    return ModelData(nickname=nickname, pretty=pretty, n_features=n_features,
                     embedding_3d=embed_3d, n_circuits_per_feature=counts,
                     backbone_idx=backbone_idx, circuit_per_prompt=circuits)


def render_hero(data_g: ModelData, data_p: ModelData, out_path):
    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.16, wspace=0.0)
    gs = fig.add_gridspec(1, 2, wspace=0.0)
    ax_g = fig.add_subplot(gs[0, 0], projection="3d")
    ax_p = fig.add_subplot(gs[0, 1], projection="3d")
    ax_g.set_facecolor("white")
    ax_p.set_facecolor("white")
    n_max_global = max(int(data_g.n_circuits_per_feature.max()),
                       int(data_p.n_circuits_per_feature.max()), 1)
    render_panel(ax_g, data_g, n_max_global=n_max_global)
    render_panel(ax_p, data_p, n_max_global=n_max_global)

    # Hand-curated backbone annotations — colour-coded by structural (orange) vs content (blue)
    gemma_labels = {
        6631:  ("beginning of text",            "#d35400"),
        9768:  ("control & authority",          "#d35400"),
        15596: ("forms of \"to be\"",         "#d35400"),
        13414: ("statements of existence",      "#d35400"),
        1692:  ("legal / technical terms",      "#d35400"),
    }
    gpt2_labels = {
        6863:  ("politics & government",        "#2471a3"),
        13420: ("numerical statistics",         "#2471a3"),
        18220: ("personal preferences",         "#2471a3"),
        22852: ("URLs",                         "#2471a3"),
        1442:  ("capital cities",               "#2471a3"),
    }
    annotate_backbone(ax_g, data_g, gemma_labels, counts_min=6)
    annotate_backbone(ax_p, data_p, gpt2_labels, counts_min=4)
    # Headline + caption with breathing room
    fig.text(0.5, 0.945,
             "The Grammar Layer", ha="center", fontsize=20, color="#111", weight="bold")
    fig.text(0.5, 0.905,
             "Same 12 questions.  Same 12 answers.  Two different routes.",
             ha="center", fontsize=13, color="#444", style="italic")
    # Sub-caption naming the variable
    fig.text(0.5, 0.085,
             "Each point is one SAE feature, UMAP-projected from its decoder direction. "
             "Brightness = how many of 12 next-token-prediction circuits recruit that feature.\n"
             "Gemma's backbone (orange labels) is structural: forms of \"to be\", statements of existence, beginnings of text.   "
             "GPT-2's backbone (blue labels) is content-thematic: politics, statistics, capital cities, URLs.\n"
             "Same vocabulary in both SAEs — Gemma reaches for grammar; GPT-2 reaches for facts.",
             ha="center", fontsize=10, color="#333")
    # Colorbar (compact, gridded into the bottom)
    cax = fig.add_axes([0.34, 0.035, 0.32, 0.018])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.inferno, norm=plt.Normalize(vmin=0, vmax=n_max_global))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[0, 3, 6, 9, 12])
    cb.set_label("# of 12 circuits recruiting this feature", fontsize=9, color="#333")
    cb.ax.tick_params(labelsize=8, colors="#333")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info("wrote %s", out_path)


def render_single(data: ModelData, out_path, with_subtitle=True):
    fig = plt.figure(figsize=(7, 7), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    render_panel(ax, data)
    if with_subtitle:
        fig.text(0.5, 0.02, data.pretty, ha="center", fontsize=11, color="#333")
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info("wrote %s", out_path)


def render_vocab_links(c: NeographClient, data_g: ModelData, data_p: ModelData, out_path):
    """Third-panel bonus: features matched across models by label cosine."""
    log.info("Building cross-model vocabulary links ...")
    # Pull pairs of (gemma_feat, gpt2_feat) where label cosine ≥ 0.85, both labelled
    pairs = c.run(
        """
        MATCH (g:SAEFeature)-[:LABELED_AS]->(a:AutoInterpLabel)
        WHERE g.sae_id CONTAINS 'gemma'
        CALL db.index.vector.queryNodes('label_emb', 3, a.embedding) YIELD node, score
        MATCH (p:SAEFeature)-[:LABELED_AS]->(node)
        WHERE p.sae_id = $sid AND score >= 0.88 AND p.index <> g.index
        RETURN g.index AS gemma_idx, p.index AS gpt2_idx, score
        ORDER BY score DESC LIMIT 400
        """,
        sid=GPT2_SAE_ID,
    )
    if not pairs:
        log.warning("no cross-model pairs above threshold — skipping vocab links")
        return

    fig = plt.figure(figsize=(14, 6.5), facecolor="white")
    gs = fig.add_gridspec(1, 2, wspace=0.02)
    ax_g = fig.add_subplot(gs[0, 0], projection="3d")
    ax_p = fig.add_subplot(gs[0, 1], projection="3d")
    ax_g.set_facecolor("white")
    ax_p.set_facecolor("white")
    render_panel(ax_g, data_g)
    render_panel(ax_p, data_p)

    # Draw curved lines connecting matched feature pairs.
    # Project each 3D point to display coords using proj_transform.
    from mpl_toolkits.mplot3d import proj3d
    fig.canvas.draw()
    n_drawn = 0
    for r in pairs:
        gi = int(r["gemma_idx"])
        pi = int(r["gpt2_idx"])
        if gi >= data_g.n_features or pi >= data_p.n_features:
            continue
        x1, y1, z1 = data_g.embedding_3d[gi]
        x2, y2, z2 = data_p.embedding_3d[pi]
        # 3D → 2D data-space via proj_transform, then data → display
        xd1, yd1, _ = proj3d.proj_transform(float(x1), float(y1), float(z1), ax_g.get_proj())
        xd2, yd2, _ = proj3d.proj_transform(float(x2), float(y2), float(z2), ax_p.get_proj())
        disp1 = ax_g.transData.transform((xd1, yd1))
        disp2 = ax_p.transData.transform((xd2, yd2))
        fig_disp1 = fig.transFigure.inverted().transform(disp1)
        fig_disp2 = fig.transFigure.inverted().transform(disp2)
        fig.lines.append(
            plt.Line2D([fig_disp1[0], fig_disp2[0]], [fig_disp1[1], fig_disp2[1]],
                       transform=fig.transFigure, color="#2471a3", alpha=0.14, linewidth=0.45)
        )
        n_drawn += 1
        if n_drawn >= 200:
            break
    log.info("drew %d vocab-link arcs", n_drawn)
    fig.text(0.5, 0.97,
             "The vocabulary maps across models. The routing doesn't.",
             ha="center", fontsize=12, color="#111")
    fig.text(0.5, 0.04,
             "Lines connect SAE features whose autointerp labels match (cosine ≥ 0.88)\n"
             "— same concept on both sides — but lit features cluster differently in each model.",
             ha="center", fontsize=9, color="#444")
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info("wrote %s", out_path)


def render_gif(data_g: ModelData, data_p: ModelData, out_path):
    """Cycle through the 12 prompts, lighting only that prompt's features on each frame."""
    try:
        from matplotlib.animation import PillowWriter, FuncAnimation
    except ImportError:
        log.warning("matplotlib animation not available — skipping GIF")
        return

    prompt_ids = sorted(set(data_g.circuit_per_prompt) | set(data_p.circuit_per_prompt))
    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    gs = fig.add_gridspec(1, 2, wspace=0.02)
    ax_g = fig.add_subplot(gs[0, 0], projection="3d")
    ax_p = fig.add_subplot(gs[0, 1], projection="3d")
    ax_g.set_facecolor("white")
    ax_p.set_facecolor("white")
    fig.text(0.5, 0.965,
             "Same 12 questions, same 12 answers. Cycling through each prompt.",
             ha="center", fontsize=11, color="#111")

    def draw_frame(pid: str, ax: Axes3D, data: ModelData, title: str):
        ax.clear()
        ax.set_facecolor("white")
        Z = data.embedding_3d
        z_floor = float(Z[:, 2].min()) - 0.6 * (float(Z[:, 2].max()) - float(Z[:, 2].min()))
        ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=BG_GRAY, alpha=BG_GRAY_ALPHA, s=2, linewidth=0)
        ax.scatter(Z[:, 0], Z[:, 1], np.full(Z.shape[0], z_floor),
                   c=SHADOW_GRAY, alpha=SHADOW_ALPHA, s=2, linewidth=0)
        lit = data.circuit_per_prompt.get(pid, set())
        lit_arr = np.array([f for f in lit if 0 <= f < data.n_features])
        if lit_arr.size:
            ax.scatter(Z[lit_arr, 0], Z[lit_arr, 1], Z[lit_arr, 2],
                       c="#ffcc33", s=24, alpha=0.9, linewidth=0)
            ax.scatter(Z[lit_arr, 0], Z[lit_arr, 1], np.full(lit_arr.size, z_floor),
                       c="#ffaa33", s=6, alpha=0.4, linewidth=0)
        # Persistent backbone glow
        bb = data.backbone_idx[:8]
        ax.scatter(Z[bb, 0], Z[bb, 1], Z[bb, 2],
                   facecolors="none", edgecolors="#ffffff", s=160, linewidth=1.4)
        ax.set_axis_off()
        ax.view_init(20, 35)
        ax.set_box_aspect((1, 1, 0.9))
        ax.set_title(title, color="#111", fontsize=12, pad=10)

    cap = fig.text(0.5, 0.04, "", ha="center", fontsize=11, color="#333", style="italic")

    def animate(frame_idx):
        pid = prompt_ids[frame_idx % len(prompt_ids)]
        draw_frame(pid, ax_g, data_g, f"Gemma 2 2B  ·  {pid}")
        draw_frame(pid, ax_p, data_p, f"GPT-2 small  ·  {pid}")
        cap.set_text(f"Prompt: {pid}")
        return [ax_g, ax_p, cap]

    anim = FuncAnimation(fig, animate, frames=len(prompt_ids), interval=900, repeat=True)
    writer = PillowWriter(fps=1)
    anim.save(out_path, writer=writer, dpi=120)
    plt.close()
    log.info("wrote %s", out_path)


def main() -> int:
    PATHS.reports.mkdir(parents=True, exist_ok=True)
    W_dec_gemma, W_dec_gpt2 = load_decoder_weights()
    Z_gemma = fit_umap_3d(W_dec_gemma, "Gemma")
    Z_gpt2 = fit_umap_3d(W_dec_gpt2, "GPT-2")

    with NeographClient() as c:
        data_g = build_model_data(c, "gemma", "Gemma 2 2B  ·  Gemma Scope L20  ·  16,384 features",
                                  GEMMA_SAE.neograph_id, W_dec_gemma, Z_gemma)
        data_p = build_model_data(c, "gpt2", "GPT-2 small  ·  RES-JB L8  ·  24,576 features",
                                  GPT2_SAE_ID, W_dec_gpt2, Z_gpt2)
        log.info("Gemma: %d / %d features participate in ≥1 circuit; backbone hits up to %d circuits",
                 int((data_g.n_circuits_per_feature >= 1).sum()), data_g.n_features,
                 int(data_g.n_circuits_per_feature.max()))
        log.info("GPT-2: %d / %d features participate in ≥1 circuit; backbone hits up to %d circuits",
                 int((data_p.n_circuits_per_feature >= 1).sum()), data_p.n_features,
                 int(data_p.n_circuits_per_feature.max()))

        # Save UMAP coords for later reuse
        np.save(PATHS.reports / "umap_gemma_3d.npy", Z_gemma)
        np.save(PATHS.reports / "umap_gpt2_3d.npy", Z_gpt2)
        np.save(PATHS.reports / "circuits_per_feature_gemma.npy", data_g.n_circuits_per_feature)
        np.save(PATHS.reports / "circuits_per_feature_gpt2.npy", data_p.n_circuits_per_feature)

        # Hero image: side-by-side
        render_hero(data_g, data_p, PATHS.reports / "viz_grammar_layer.png")

        # Individual panels for Twitter / blog reuse
        render_single(data_g, PATHS.reports / "viz_grammar_layer_gemma.png")
        render_single(data_p, PATHS.reports / "viz_grammar_layer_gpt2.png")

        # Bonus 1: cross-model vocabulary links
        render_vocab_links(c, data_g, data_p, PATHS.reports / "viz_vocab_links.png")

        # Bonus 2: animated GIF cycling through 12 prompts
        render_gif(data_g, data_p, PATHS.reports / "viz_grammar_layer.gif")

    summary = {
        "gemma": {
            "n_features": int(data_g.n_features),
            "n_features_in_any_circuit": int((data_g.n_circuits_per_feature >= 1).sum()),
            "max_circuits_per_feature": int(data_g.n_circuits_per_feature.max()),
            "top_backbone": [int(i) for i in data_g.backbone_idx[:15]],
            "top_backbone_counts": [int(data_g.n_circuits_per_feature[i]) for i in data_g.backbone_idx[:15]],
        },
        "gpt2": {
            "n_features": int(data_p.n_features),
            "n_features_in_any_circuit": int((data_p.n_circuits_per_feature >= 1).sum()),
            "max_circuits_per_feature": int(data_p.n_circuits_per_feature.max()),
            "top_backbone": [int(i) for i in data_p.backbone_idx[:15]],
            "top_backbone_counts": [int(data_p.n_circuits_per_feature[i]) for i in data_p.backbone_idx[:15]],
        },
    }
    (PATHS.reports / "viz_grammar_layer_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
