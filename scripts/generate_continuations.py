"""Generate text continuations from a model + SAE for the behavioral-signature experiment.

For each prompt in data/behavior_prompts.json, sample N continuations of `--max-new-tokens`
length, with matched sampling parameters across all models. Save raw text + token IDs to
JSON.

Usage:
    uv run python scripts/generate_continuations.py \
        --model gemma --n-seeds 5 --max-new-tokens 300 \
        --output reports/generations_gemma.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from neograph.config import PATHS
from neograph.util import get_logger

sys.path.insert(0, str(Path(__file__).parent))
from load_bearing_topk import MODEL_SPECS  # noqa: E402

log = get_logger("neograph.generate")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_SPECS.keys()), default="gemma")
    parser.add_argument("--prompts-file", default="data/behavior_prompts.json")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    spec = MODEL_SPECS[args.model]

    prompts_path = Path(args.prompts_file)
    if not prompts_path.is_absolute():
        prompts_path = PATHS.root / prompts_path
    prompts = json.loads(prompts_path.read_text())
    log.info("Loaded %d prompts from %s", len(prompts), prompts_path)

    from transformer_lens import HookedTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("Loading %s on %s ...", spec["hf_name"], device)
    model = HookedTransformer.from_pretrained(spec["hf_name"], device=device)
    model.eval()

    results: list[dict] = []
    for i, p in enumerate(prompts):
        log.info("[%d/%d] %s  prompt=%r", i + 1, len(prompts), p["id"], p["prompt"][:60])
        gens = []
        for seed in range(args.n_seeds):
            torch.manual_seed(1000 * i + seed)
            tokens = model.to_tokens(p["prompt"], prepend_bos=True)
            with torch.no_grad():
                out = model.generate(
                    tokens,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=True,
                    verbose=False,
                )
            new_tokens = out[0, tokens.shape[-1]:]
            text = model.tokenizer.decode(new_tokens, skip_special_tokens=True)
            gens.append({
                "seed": seed,
                "n_new_tokens": int(new_tokens.shape[-1]),
                "continuation": text,
            })
        results.append({**p, "generations": gens})

    out = {
        "model": spec["nickname"],
        "prompts_file": str(prompts_path.relative_to(PATHS.root)),
        "sampling": {
            "n_seeds": args.n_seeds,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "results": results,
    }
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PATHS.root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
