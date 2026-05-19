"""HuggingFace-native continuation sampler (faster than transformer_lens.generate).

Drops back to the underlying HF API: AutoModelForCausalLM + tokenizer + KV caching.
Same I/O shape as scripts/generate_continuations.py.

Usage:
    uv run python scripts/generate_continuations_hf.py \
        --hf-name google/gemma-2-2b --nickname gemma \
        --output reports/generations_gemma.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

from neograph.config import PATHS
from neograph.util import get_logger

log = get_logger("neograph.generate_hf")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-name", required=True, help="Hugging Face repo, e.g. google/gemma-2-2b")
    parser.add_argument("--nickname", required=True, help="Short model id for output")
    parser.add_argument("--prompts-file", default="data/behavior_prompts.json")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    prompts_path = Path(args.prompts_file)
    if not prompts_path.is_absolute():
        prompts_path = PATHS.root / prompts_path
    prompts = json.loads(prompts_path.read_text())
    log.info("Loaded %d prompts from %s", len(prompts), prompts_path)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("Loading %s on %s ...", args.hf_name, device)
    tok = AutoTokenizer.from_pretrained(args.hf_name)
    model_kwargs = {"torch_dtype": torch.float16 if device == "mps" else torch.float32}
    # gemma-2-2b is gated — needs HF_TOKEN from .env
    if "gemma" in args.hf_name.lower():
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            model_kwargs["token"] = hf_token
    model = AutoModelForCausalLM.from_pretrained(args.hf_name, **model_kwargs).to(device)
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    log.info("Model loaded. n_params=%d, device=%s", sum(p.numel() for p in model.parameters()), device)

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PATHS.root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for i, p in enumerate(prompts):
        log.info("[%d/%d] %s  prompt=%r", i + 1, len(prompts), p["id"], p["prompt"][:60])
        gens = []
        # Tokenise once per prompt
        enc = tok(p["prompt"], return_tensors="pt").to(device)
        prompt_len = enc["input_ids"].shape[-1]
        for seed in range(args.n_seeds):
            torch.manual_seed(1000 * i + seed)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=True,
                    pad_token_id=tok.pad_token_id,
                    use_cache=True,
                )
            new_tokens = out[0, prompt_len:]
            text = tok.decode(new_tokens, skip_special_tokens=True)
            gens.append({
                "seed": seed,
                "n_new_tokens": int(new_tokens.shape[-1]),
                "continuation": text,
            })
        results.append({**p, "generations": gens})
        # Incremental save after each prompt so a crash doesn't lose everything
        out = {
            "model": args.nickname,
            "prompts_file": str(prompts_path.relative_to(PATHS.root)),
            "sampling": {
                "n_seeds": args.n_seeds,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
            },
            "results": results,
            "partial": i + 1 < len(prompts),
        }
        out_path.write_text(json.dumps(out, indent=2))

    out["partial"] = False
    out_path.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
