"""Autointerp label sourcing: Neuronpedia → Anthropic fallback, plus MiniLM embedding."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from neograph.config import EMBEDDING_MODEL, PATHS, neuronpedia_feature_url
from neograph.util import get_logger

log = get_logger("neograph.labels")

# Prefer np_max-act-logits, then any other np_*, then any explanation.
NP_PREFERRED_TYPES = (
    "np_max-act-logits",
    "np_max-act",
    "oai_token-act-pair",
    "oai_attention",
    "default",
)


@dataclass
class Label:
    feature_index: int
    text: str
    source: str
    score: float | None = None


class LabelsCache:
    """JSON-backed dict cache to avoid re-hitting Neuronpedia / Anthropic on resumes."""

    def __init__(self, path: Path = PATHS.labels_cache) -> None:
        self.path = path
        self._data: dict[str, dict[str, Any]] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text())
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to read labels cache: %s — starting empty", exc)
                self._data = {}

    def get(self, idx: int) -> dict[str, Any] | None:
        return self._data.get(str(idx))

    def put(self, idx: int, payload: dict[str, Any]) -> None:
        self._data[str(idx)] = payload

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data))
        tmp.replace(self.path)


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=10),
)
def _http_get(client: httpx.Client, url: str) -> httpx.Response:
    r = client.get(url, timeout=20.0)
    if r.status_code >= 500:
        raise httpx.HTTPError(f"server {r.status_code}")
    return r


def _pick_explanation(explanations: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not explanations:
        return None
    by_type = {e.get("typeName", ""): e for e in explanations}
    for t in NP_PREFERRED_TYPES:
        if t in by_type:
            return by_type[t]
    return explanations[0]


def fetch_neuronpedia_label(
    feature_index: int,
    client: httpx.Client,
    cache: LabelsCache | None = None,
    sleep_s: float = 0.0,
) -> Label | None:
    """Fetch autointerp from Neuronpedia. Returns None if missing/empty."""
    cached = cache.get(feature_index) if cache else None
    if cached and cached.get("source", "").startswith("neuronpedia"):
        return Label(
            feature_index=feature_index,
            text=cached["text"],
            source=cached["source"],
            score=cached.get("score"),
        )
    url = neuronpedia_feature_url(feature_index)
    try:
        r = _http_get(client, url)
    except Exception as exc:  # noqa: BLE001
        log.warning("Neuronpedia fetch failed for feature %d: %s", feature_index, exc)
        return None
    if r.status_code != 200:
        return None
    data = r.json()
    exp = _pick_explanation(data.get("explanations") or [])
    if not exp:
        return None
    text = (exp.get("description") or "").strip()
    if not text:
        return None
    src = f"neuronpedia:{exp.get('typeName', 'unknown')}"
    score = None
    sv2 = exp.get("scoreV2")
    if isinstance(sv2, (int, float)):
        score = float(sv2)
    label = Label(feature_index=feature_index, text=text, source=src, score=score)
    if cache:
        cache.put(
            feature_index,
            {"text": label.text, "source": label.source, "score": label.score},
        )
    if sleep_s:
        time.sleep(sleep_s)
    return label


CLAUDE_PROMPT = """You will be shown the top-activating tokens for a single feature of a sparse autoencoder applied to Gemma 2 2B layer 20 residual stream. Each row shows (token | surrounding context). Produce a one-sentence summary (≤ 20 words) of what concept/feature this neuron-like feature represents. Use the style of Neuronpedia autointerp labels (e.g. "Words beginning with 'Hor'", "Tokens related to API authentication and identity verification"). Do not pad with hedging.

Top activations:
{rows}

Label:"""


def fetch_claude_label(
    feature_index: int,
    top_token_rows: list[str],
    cache: LabelsCache | None = None,
) -> Label | None:
    """Fallback when Neuronpedia is empty. Requires ANTHROPIC_API_KEY in env."""
    cached = cache.get(feature_index) if cache else None
    if cached and cached.get("source", "").startswith("claude"):
        return Label(
            feature_index=feature_index,
            text=cached["text"],
            source=cached["source"],
            score=cached.get("score"),
        )
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.warning("ANTHROPIC_API_KEY not set — cannot generate fallback label for %d", feature_index)
        return None
    try:
        import anthropic
    except ImportError:
        log.warning("anthropic SDK not installed — skipping Claude label for %d", feature_index)
        return None
    client = anthropic.Anthropic()
    rows = "\n".join(f"- {r}" for r in top_token_rows[:32])
    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=80,
        messages=[{"role": "user", "content": CLAUDE_PROMPT.format(rows=rows)}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
    text = text.splitlines()[0].strip().strip(".").strip()
    if not text:
        return None
    label = Label(feature_index=feature_index, text=text, source="claude-haiku-4-5", score=None)
    if cache:
        cache.put(
            feature_index,
            {"text": label.text, "source": label.source, "score": None},
        )
    return label


# === Embedding ===


class LabelEmbedder:
    """Wraps sentence-transformers MiniLM (CPU, 384-dim)."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")

    def embed(self, texts: list[str]) -> np.ndarray:
        out = self.model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(out, dtype=np.float32)
