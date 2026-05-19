# The Grammar Layer — interactive viewer

Single-file Three.js viewer for the cross-model routing finding. Drop it on any
static host (Cloudflare Pages, GitHub Pages, S3, your own server) and it works.

## Run locally

```bash
cd apps/grammar_layer
python3 -m http.server 8765
# open http://localhost:8765
```

The viewer loads `data.json` (~2.5 MB; 41k features' UMAP coords + per-prompt
recruitment + cross-model label-similarity pairs). Re-export with:

```bash
uv run python scripts/export_viz_data.py
cp reports/viz_data.json apps/grammar_layer/data.json
```

## What's in it

- **Two side-by-side WebGL scenes** — Gemma's 16,384 features and GPT-2's 24,576 features
  rendered as 3D point clouds, UMAP-projected from their decoder directions. Both clouds
  rotate slowly when idle (Cartographer-style).
- **Per-prompt cycle** — switch which of the 12 prompts is "lit", or watch them auto-cycle.
  Backbone features stay visible as dim halos; per-prompt features glow bright.
- **Hover-to-label** — over any backbone feature shows its Neuronpedia autointerp label and
  per-feature circuit count.
- **Vocabulary links toggle** — marks each Gemma feature whose label has a high-cosine GPT-2
  counterpart with a coloured ring (and vice versa). Visualises "the words map across; the
  routing doesn't".
- **Reset view** — re-centres both cameras.

## Embedding in a LessWrong / Substack / blog post

Iframe it:

```html
<iframe src="https://your-host/grammar_layer/" width="100%" height="640"
        style="border:0;border-radius:8px;background:white"></iframe>
```

Mobile gracefully degrades to a single-panel stack.

## Why three.js and not a static PNG

A static 3D scatter collapses to flatland on a phone screen. The finding *is* spatial:
the glowing backbone *concentration* in Gemma vs the per-prompt *scattering* in GPT-2 only
reads when you can rotate and inspect. The slider also reveals which features participate
in which prompts — information that disappears in a single composite image.
