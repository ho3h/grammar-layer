"""P2: Build the three corpora and stage to parquet.

- Pile slice (HF datasets, no auth needed)
- Rhyme synth
- Weekday synth

Output: data/staging/prompts.parquet (id, text, source, n_tokens placeholder).

Idempotent — re-running uses cached parquet if present.
"""

from __future__ import annotations

import json
import random
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from neograph.config import PATHS, PILE_PROMPTS, PILE_TOKENS_PER_PROMPT, RHYME_PROMPTS, WEEKDAY_PROMPTS
from neograph.util import exit_marker, get_logger, sha1_short

log = get_logger("neograph.seed")

OUT = PATHS.staging / "prompts.parquet"


def _synth(spec_path, n: int, template_key: str, seed_key: str, source: str, seed: int) -> list[dict]:
    spec = json.loads(spec_path.read_text())
    items = spec[seed_key]
    templates = spec[template_key]
    rng = random.Random(seed)
    rows: list[dict] = []
    while len(rows) < n:
        tmpl = rng.choice(templates)
        item = rng.choice(items)
        text = tmpl.format(**{seed_key.rstrip("s"): item, "seed": item, "day": item})
        rid = f"{source}-{sha1_short(text)}"
        rows.append({"id": rid, "text": text, "source": source})
    return rows


def build_synth() -> list[dict]:
    rows: list[dict] = []
    rows.extend(_synth(PATHS.synthetic / "rhymes.json", RHYME_PROMPTS, "templates", "seeds", "rhyme-ore", seed=11))
    rows.extend(_synth(PATHS.synthetic / "weekdays.json", WEEKDAY_PROMPTS, "templates", "days", "weekday", seed=22))
    return rows


def build_pile(n_prompts: int, n_tokens: int) -> list[dict]:
    """Pull `n_prompts` passages from monology/pile-uncopyrighted, truncated to ~n_tokens."""
    log.info("Streaming Pile slice (%d prompts × ~%d tokens) ...", n_prompts, n_tokens)
    try:
        from datasets import load_dataset
    except ImportError as exc:  # noqa: BLE001
        raise SystemExit("datasets not installed") from exc

    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    rows: list[dict] = []
    # rough char→token: 4 chars per token
    char_cap = n_tokens * 5
    for item in ds:
        text = item.get("text") or item.get("content") or ""
        if not text:
            continue
        text = text.strip().replace("\x00", "")
        if len(text) < 80:
            continue
        text = text[:char_cap]
        rid = f"pile-{sha1_short(text)}"
        rows.append({"id": rid, "text": text, "source": "pile"})
        if len(rows) >= n_prompts:
            break
    log.info("Collected %d Pile rows", len(rows))
    return rows


def main() -> int:
    PATHS.ensure()
    if OUT.exists():
        existing = pq.read_table(OUT).to_pandas()
        log.info("Found cached %s (%d rows)", OUT.name, len(existing))
        if len(existing) >= (RHYME_PROMPTS + WEEKDAY_PROMPTS + PILE_PROMPTS - 100):
            exit_marker("corpus-built", ok=True, source="cache", rows=len(existing))
            return 0

    synth = build_synth()
    log.info("Synthetic prompts: %d (rhyme=%d, weekday=%d)", len(synth), RHYME_PROMPTS, WEEKDAY_PROMPTS)

    pile = build_pile(PILE_PROMPTS, PILE_TOKENS_PER_PROMPT)
    rows = synth + pile

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    log.info("Total unique prompts: %d", len(df))
    pq.write_table(pa.Table.from_pandas(df), OUT)
    log.info("Wrote %s", OUT)

    # Accept ≥ 10k rows after dedup (synthetic 1.5k + pile 10k − duplicates).
    ok = len(df) >= 10_000
    exit_marker("corpus-built", ok=ok, rows=len(df), file=str(OUT))
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
