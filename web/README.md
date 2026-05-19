# Neograph explainer site

Static, single-page Three.js explainer that renders the live state of the
load-bearing-feature analysis. No build step.

## Serve locally

```bash
cd web
python3 -m http.server 8770
# open http://localhost:8770
```

## Update the data

Whenever the underlying ablation results change, re-export `web/data/summary.json`:

```bash
uv run python scripts/export_web_data.py
```

`summary.json` is built from the most recent `reports/load_bearing_pos10_*_{12,50}.json`
files. The exporter prefers 50-prompt files over 12-prompt files when both exist.

## Where each panel's data comes from

| Panel               | Source field in summary.json                              | Driven by                          |
|---------------------|-----------------------------------------------------------|------------------------------------|
| Ablation bars       | `models[*].per_category[*].{baseline,ablated}_hit_rate`   | `load_bearing_topk.py`             |
| Character bars      | `models[*].feature_character` (supporting)                | `export_web_data.py` (regex split) |
|                     | `models[*].opposing_feature_character` (opposing)         |                                    |
| Backbone graph      | `models[*].backbone_edges` (supporting)                   | `export_web_data.py`               |
| Takeaway list       | hard-coded in `index.html`                                | edit prose directly                |

## Hosting

Drop the whole `web/` directory on any static host (Cloudflare Pages, GitHub
Pages, S3, Vercel static). Three.js loads from a CDN via the import map in
`index.html` — no local install.
