"""Visualize the Neograph database from a graph-first perspective.

Produces three images in reports/:
- reports/viz_community_sizes.png — bar chart of the 18 Leiden communities, coloured by
  whether autointerp labels match a known concept pattern.
- reports/viz_prefix_subgraph.png — a 2D networkx layout of the word-prefix sub-cluster:
  features (nodes) coloured by community 1 (general) vs community 11 (proper-noun),
  with edges showing CO_ACTIVATES_WITH / DECODER_SIMILAR / LABEL_SIMILAR.
- reports/viz_concept_concentration.png — for each tested concept, bar chart of how
  features in that concept distribute across the 5 most populous Leiden communities.
"""

from __future__ import annotations

import json
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from neograph.config import PATHS
from neograph.cypher import NeographClient

plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"


def chart_community_sizes(c: NeographClient, out_path):
    rows = c.run(
        """
        MATCH (f:SAEFeature)
        WHERE f.communityId IS NOT NULL
        RETURN f.communityId AS cid, count(f) AS n
        ORDER BY n DESC
        """
    )
    cids = [str(r["cid"]) for r in rows]
    sizes = [r["n"] for r in rows]

    # Mark the "interesting" communities I named in findings.md
    highlights = {
        "19": ("Weekdays", "#2ca02c"),
        "6":  ("Money/financial", "#ff7f0e"),
        "8":  ("Programming/code", "#1f77b4"),
        "1":  ("Word prefixes\n(general)", "#9467bd"),
        "11": ("Word prefixes\n(proper nouns)", "#e377c2"),
    }
    colors = [highlights.get(c, ("", "#888"))[1] for c in cids]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(range(len(sizes)), sizes, color=colors, edgecolor="black", linewidth=0.5)
    for i, (cid, size) in enumerate(zip(cids, sizes)):
        if cid in highlights:
            ax.text(i, size + 60, highlights[cid][0], ha="center", fontsize=8, color=highlights[cid][1])
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(cids, fontsize=8)
    ax.set_xlabel("Leiden community ID")
    ax.set_ylabel("# SAE features")
    ax.set_title(f"18 Leiden communities of 16,384 SAE features (Gemma 2 2B, L20, width-16k)\n"
                 f"highlighted communities matched a labelled concept")
    plt.savefig(out_path)
    plt.close()
    print(f"wrote {out_path}")


def chart_concept_concentration(out_path):
    """Read label_clustering.json and show top communities per concept as stacked bars."""
    data = json.loads((PATHS.reports / "label_clustering.json").read_text())
    fig, axes = plt.subplots(1, len(data), figsize=(14, 4), sharey=True)
    palette = ["#2ca02c", "#ff7f0e", "#1f77b4", "#9467bd", "#e377c2", "#888", "#aaa", "#bbb"]
    for ax, (concept, rec) in zip(axes, data.items()):
        if rec["n_features"] == 0:
            ax.set_title(f"{concept}\n(no features)")
            continue
        top = rec.get("top_communities", [])[:5]
        n = rec["n_features"]
        labels = [f"cid {c}" for c, _ in top]
        sizes = [count / n * 100 for _, count in top]
        ax.barh(labels[::-1], sizes[::-1], color=palette[: len(top)][::-1])
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of concept features")
        ax.set_title(f"{concept}\nn={n}")
        for j, s in enumerate(sizes[::-1]):
            ax.text(s + 1, j, f"{s:.0f}%", va="center", fontsize=9)
    plt.suptitle("Where each labelled concept lives in the 18 Leiden communities", y=1.02)
    plt.savefig(out_path)
    plt.close()
    print(f"wrote {out_path}")


def chart_prefix_subgraph(c: NeographClient, out_path):
    rows = c.run(
        """
        MATCH (f:SAEFeature)-[:LABELED_AS]->(a:AutoInterpLabel)
        WHERE f.communityId IN [1, 11]
          AND (toLower(a.text) CONTAINS 'beginning with'
               OR toLower(a.text) CONTAINS 'starting with'
               OR toLower(a.text) CONTAINS 'starts with')
        RETURN f.communityId AS cid, f.index AS idx, a.text AS label
        """
    )
    if not rows:
        print("no prefix features found — skipping")
        return
    nodes = [(int(r["idx"]), int(r["cid"]), r["label"]) for r in rows]
    idx_set = {n[0] for n in nodes}

    # Edges between these features
    edges = c.run(
        """
        UNWIND $ids AS i
        MATCH (a:SAEFeature {index: i})-[r:CO_ACTIVATES_WITH|DECODER_SIMILAR|LABEL_SIMILAR]-(b:SAEFeature)
        WHERE b.index IN $ids AND a.index < b.index
        RETURN a.index AS a, b.index AS b, type(r) AS t,
               coalesce(r.cosine, r.pmi, 0.0) AS w
        """,
        ids=list(idx_set),
    )

    G = nx.Graph()
    for idx, cid, label in nodes:
        G.add_node(idx, community=cid, label=label)
    edge_colors = {"CO_ACTIVATES_WITH": "#888", "DECODER_SIMILAR": "#1f77b4", "LABEL_SIMILAR": "#ff7f0e"}
    for e in edges:
        G.add_edge(int(e["a"]), int(e["b"]), kind=e["t"], weight=float(e["w"]))

    fig, ax = plt.subplots(figsize=(11, 8))
    pos = nx.spring_layout(G, seed=42, k=1.2 / np.sqrt(max(len(G), 1)))
    colors = ["#9467bd" if G.nodes[n]["community"] == 1 else "#e377c2" for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_size=520, node_color=colors, edgecolors="black", linewidths=1, ax=ax)
    for kind, color in edge_colors.items():
        eds = [(u, v) for u, v, d in G.edges(data=True) if d["kind"] == kind]
        nx.draw_networkx_edges(G, pos, edgelist=eds, edge_color=color, width=1.4, alpha=0.7, ax=ax)
    # Short labels: first 12 chars of autointerp
    labels_short = {n: G.nodes[n]["label"][:34].rstrip(".") for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels_short, font_size=7, ax=ax)

    legend = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#9467bd", markersize=12, label="Community 1 (general)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#e377c2", markersize=12, label="Community 11 (proper nouns)"),
        plt.Line2D([0], [0], color="#888", linewidth=2, label="CO_ACTIVATES_WITH"),
        plt.Line2D([0], [0], color="#1f77b4", linewidth=2, label="DECODER_SIMILAR"),
        plt.Line2D([0], [0], color="#ff7f0e", linewidth=2, label="LABEL_SIMILAR"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=8)
    ax.set_title("Word-prefix subgraph: Leiden split into 'general words' vs 'proper-noun prefixes'")
    ax.set_axis_off()
    plt.savefig(out_path)
    plt.close()
    print(f"wrote {out_path}")


def main():
    PATHS.reports.mkdir(parents=True, exist_ok=True)
    with NeographClient() as c:
        chart_community_sizes(c, PATHS.reports / "viz_community_sizes.png")
        chart_prefix_subgraph(c, PATHS.reports / "viz_prefix_subgraph.png")
    chart_concept_concentration(PATHS.reports / "viz_concept_concentration.png")


if __name__ == "__main__":
    main()
