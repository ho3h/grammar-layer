"""Assign a Concept to each manifold via existing-Concept search + optional Claude summary."""

from __future__ import annotations

import os
from collections.abc import Iterable

import numpy as np

from neograph.cypher import NeographClient
from neograph.util import get_logger, sha1_short

log = get_logger("neograph.manifold.concept")


def summarise_with_claude(autointerp_labels: list[str], representative_tokens: list[str]) -> str:
    """One-paragraph community description via Claude Haiku. Requires ANTHROPIC_API_KEY."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return ""
    try:
        import anthropic
    except ImportError:
        return ""
    client = anthropic.Anthropic()
    rows_lab = "\n".join(f"- {x}" for x in autointerp_labels[:30])
    rows_tok = ", ".join(representative_tokens[:30])
    prompt = (
        "Given these autointerp labels for a community of SAE features in Gemma 2 2B layer 20, "
        "summarise the underlying concept in ONE short sentence (≤ 20 words). "
        "Style: 'Words containing X', 'Cyclic structure for days of the week', etc.\n\n"
        f"Labels:\n{rows_lab}\n\nTop tokens: {rows_tok}\n\nConcept summary:"
    )
    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=60,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
    return text.splitlines()[0].strip().strip(".")


def attach_concept(
    client: NeographClient,
    manifold_id: str,
    autointerp_labels: list[str],
    representative_tokens: Iterable[str],
    taxonomy_hint: str = "unspecified",
    name_seed: str = "",
) -> str:
    """Find or create a Concept for this manifold, then write (manifold)-[:DESCRIBES]->(concept).

    Returns the concept id.
    """
    rep = list(representative_tokens)
    summary = summarise_with_claude(autointerp_labels, rep)
    name = summary or name_seed or "Unnamed manifold"
    concept_id = f"concept:{sha1_short(name)}"
    with client.session() as s:
        s.run(
            """
            MERGE (c:Concept {id: $cid})
              SET c.name = $name,
                  c.description = $desc,
                  c.taxonomy = $tax
            WITH c
            MATCH (m:Manifold {id: $mid})
            MERGE (m)-[d:DESCRIBES]->(c)
              SET d.confidence = $conf
            """,
            cid=concept_id,
            name=name,
            desc=name,
            tax=taxonomy_hint,
            mid=manifold_id,
            conf=1.0 if summary else 0.5,
        )
    return concept_id
