"""Capture model + SAE activations and stage to parquet.

For each prompt:
- Run model + SAE → feature activations (batch, seq, d_sae)
- For each feature: collect top-K (token_id, prompt_id, position, magnitude)
- Aggregate stats: activation_density, max_act

Synthetic corpora additionally store full Activation rows (sparse — only where act > threshold)
for later Activation-node ingestion + manifold reconstruction.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from neograph.config import PATHS, SAE, TOP_K_TOKENS
from neograph.util import get_logger

log = get_logger("neograph.activations")


@dataclasses.dataclass
class CaptureBudget:
    batch_size: int = 4
    max_seq_len: int = 128
    device: str = "mps"
    activation_threshold: float = SAE.activation_threshold


def _topk_features(
    feat_acts: torch.Tensor, prompt_ids: list[str], k: int = TOP_K_TOKENS
) -> dict[int, list[tuple[float, str, int, int]]]:
    """For each feature, return its top-k (magnitude, prompt_id, position, token_id) entries.

    feat_acts: (batch, seq, d_sae)
    """
    b, seq, d_sae = feat_acts.shape
    flat = feat_acts.reshape(-1, d_sae).float().cpu()
    # For each feature, find top-k across all (b, seq) positions
    top_per_feat: dict[int, list[tuple[float, str, int, int]]] = {}
    # We do this in chunks to manage memory
    for f_start in range(0, d_sae, 512):
        f_end = min(f_start + 512, d_sae)
        chunk = flat[:, f_start:f_end]  # (b*seq, chunk_dim)
        # For each feature in chunk, get top-k indices
        vals, idx = torch.topk(chunk, k=min(k, chunk.shape[0]), dim=0)
        for ci, fidx in enumerate(range(f_start, f_end)):
            entries: list[tuple[float, str, int, int]] = []
            for kk in range(vals.shape[0]):
                mag = float(vals[kk, ci].item())
                if mag <= 0:
                    continue
                flat_pos = int(idx[kk, ci].item())
                b_idx = flat_pos // seq
                p_idx = flat_pos % seq
                if b_idx >= len(prompt_ids):
                    continue
                entries.append((mag, prompt_ids[b_idx], p_idx, -1))  # token_id filled later
            if entries:
                top_per_feat[fidx] = entries
    return top_per_feat


def capture_all(
    model,
    sae,
    prompts_df: pd.DataFrame,
    budget: CaptureBudget,
    *,
    synthetic_sources: tuple[str, ...] = ("rhyme-ore", "weekday"),
) -> dict[str, Path]:
    """Run all prompts through model+SAE, stage everything to parquet.

    Outputs (data/staging/):
    - feature_topk.parquet: (feature_index, prompt_id, position, magnitude)
    - feature_stats.parquet: (feature_index, max_act, activation_density, is_dead)
    - activations_synth.parquet: full Activation rows for synthetic prompts
    - prompts_meta.parquet: (id, source, n_tokens, token_ids)
    """
    PATHS.staging.mkdir(parents=True, exist_ok=True)

    d_sae = SAE.d_sae

    # Aggregators
    max_act = np.zeros(d_sae, dtype=np.float32)
    n_active = np.zeros(d_sae, dtype=np.int64)
    n_positions_total = 0
    topk_per_feat: dict[int, list[tuple[float, str, int, int]]] = {}
    activations_synth_rows: list[dict[str, Any]] = []
    prompt_meta_rows: list[dict[str, Any]] = []

    device = torch.device(budget.device if torch.backends.mps.is_available() else "cpu")

    for batch_start in tqdm(range(0, len(prompts_df), budget.batch_size), desc="capture"):
        batch = prompts_df.iloc[batch_start : batch_start + budget.batch_size]
        texts = batch["text"].tolist()
        ids = batch["id"].tolist()
        sources = batch["source"].tolist()

        with torch.no_grad():
            tokens = model.to_tokens(texts, prepend_bos=True)
            tokens = tokens[:, : budget.max_seq_len]
            _logits, cache = model.run_with_cache_with_saes(tokens, saes=[sae])

        feat_key = next(k for k in cache.keys() if "sae" in k and "acts_post" in k)
        feat_acts = cache[feat_key]  # (b, seq, d_sae)
        b, seq, _ = feat_acts.shape
        feat_acts_cpu = feat_acts.float().cpu().numpy()

        # Aggregates
        thr = budget.activation_threshold
        active_mask = feat_acts_cpu > thr  # (b, seq, d_sae)
        max_act = np.maximum(max_act, feat_acts_cpu.reshape(-1, d_sae).max(axis=0))
        n_active += active_mask.reshape(-1, d_sae).sum(axis=0).astype(np.int64)
        n_positions_total += b * seq

        # Top-K per feature
        batch_topk = _topk_features(feat_acts, ids, k=TOP_K_TOKENS)
        for fidx, entries in batch_topk.items():
            heap = topk_per_feat.setdefault(fidx, [])
            heap.extend(entries)
            heap.sort(key=lambda x: x[0], reverse=True)
            del heap[TOP_K_TOKENS:]

        # Per-prompt token metadata
        token_ids_np = tokens.cpu().numpy()
        for bi, (pid, src) in enumerate(zip(ids, sources)):
            n_tok = int((token_ids_np[bi] != model.tokenizer.pad_token_id).sum()) if hasattr(model.tokenizer, "pad_token_id") and model.tokenizer.pad_token_id is not None else token_ids_np.shape[1]
            prompt_meta_rows.append(
                {
                    "id": pid,
                    "source": src,
                    "n_tokens": n_tok,
                    "token_ids": token_ids_np[bi].tolist(),
                }
            )

        # Full Activation rows for synthetic sources
        for bi, (pid, src) in enumerate(zip(ids, sources)):
            if src not in synthetic_sources:
                continue
            positions = np.where(active_mask[bi].any(axis=-1))[0]
            for p in positions:
                feat_indices = np.where(active_mask[bi, p])[0]
                for f in feat_indices:
                    mag = float(feat_acts_cpu[bi, p, f])
                    activations_synth_rows.append(
                        {
                            "prompt_id": pid,
                            "position": int(p),
                            "feature_index": int(f),
                            "magnitude": mag,
                        }
                    )

    # Write aggregates
    density = n_active.astype(np.float64) / max(n_positions_total, 1)
    stats_df = pd.DataFrame(
        {
            "feature_index": np.arange(d_sae),
            "max_act": max_act,
            "activation_density": density.astype(np.float32),
            "is_dead": max_act < budget.activation_threshold,
        }
    )
    stats_path = PATHS.staging / "feature_stats.parquet"
    pq.write_table(pa.Table.from_pandas(stats_df), stats_path)
    log.info("Wrote %s (%d features, %d dead)", stats_path, len(stats_df), int(stats_df["is_dead"].sum()))

    # Top-K
    topk_rows = []
    for fidx, entries in topk_per_feat.items():
        for mag, pid, pos, tid in entries:
            topk_rows.append(
                {"feature_index": fidx, "prompt_id": pid, "position": pos, "magnitude": mag, "token_id": tid}
            )
    topk_df = pd.DataFrame(topk_rows)
    topk_path = PATHS.staging / "feature_topk.parquet"
    pq.write_table(pa.Table.from_pandas(topk_df), topk_path)
    log.info("Wrote %s (%d rows)", topk_path, len(topk_df))

    # Synthetic Activation rows
    synth_df = pd.DataFrame(activations_synth_rows)
    synth_path = PATHS.staging / "activations_synth.parquet"
    if len(synth_df):
        pq.write_table(pa.Table.from_pandas(synth_df), synth_path)
        log.info("Wrote %s (%d activation rows)", synth_path, len(synth_df))

    # Prompt meta
    meta_df = pd.DataFrame(prompt_meta_rows)
    meta_path = PATHS.staging / "prompts_meta.parquet"
    pq.write_table(pa.Table.from_pandas(meta_df), meta_path)

    return {
        "stats": stats_path,
        "topk": topk_path,
        "synth": synth_path,
        "meta": meta_path,
    }
