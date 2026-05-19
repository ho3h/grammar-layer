// Entry point — loads the latest summary.json (or its inline fallback), then
// dispatches each section to its panel module.
//
// Three.js is loaded LAZILY via dynamic import so that a CDN block doesn't take
// down the rest of the page.

import { renderHeadlineStats } from "./panels/headline.js?v=7";
import { renderCharacterInversion, renderCrossModelInversion } from "./panels/inversion.js?v=7";
import { renderCategoryEnrichment } from "./panels/enrichment.js?v=7";
import { renderAblationBars } from "./panels/ablation.js?v=7";

const DATA_URL = "data/summary.json";

const statusBar = document.getElementById("status-bar");
const setStatus = (status, text) => {
  if (statusBar) {
    statusBar.innerHTML = `<span class="status-pill" data-status="${status}">${text}</span>`;
  }
};

window.addEventListener("error", (e) => {
  setStatus("stale", `JS error: ${e.message}`);
  const errEl = document.createElement("div");
  errEl.style.cssText = "max-width:760px;margin:0 auto;padding:16px 36px;font-family:var(--sans);color:#9c2a2a;font-size:13px;background:#fff5f0;border-top:1px solid var(--hairline)";
  errEl.textContent = `JS error: ${e.message} (${e.filename}:${e.lineno})`;
  document.body.insertBefore(errEl, document.body.firstChild.nextSibling);
});

async function boot() {
  setStatus("loading", "Loading data…");
  let summary = null;
  let source = "fetch";

  try {
    const r = await fetch(DATA_URL, { cache: "no-cache" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    summary = await r.json();
  } catch (e) {
    console.warn("fetch failed, falling back to inline data island:", e.message);
  }

  if (!summary) {
    const island = document.getElementById("summary-island");
    if (island && island.textContent.trim()) {
      try {
        summary = JSON.parse(island.textContent);
        source = "inline";
      } catch (e) {
        console.error("inline data island parse failed:", e);
      }
    }
  }

  if (!summary) {
    setStatus("stale", `Couldn't load ${DATA_URL} (and no inline fallback). Run scripts/export_web_data.py + scripts/embed_web_data.py.`);
    return;
  }

  const generatedAt = summary.generated_at
    ? new Date(summary.generated_at).toLocaleString()
    : "unknown";
  const models = Object.keys(summary.models);
  const nPrompts = summary.models[models[0]]?.n_prompts ?? 0;
  setStatus("ready", `${models.length} models · ${nPrompts} prompts · generated ${generatedAt}${source === "inline" ? " · inline" : ""}`);

  const ds = document.getElementById("data-source");
  if (ds) ds.textContent = source === "inline" ? "inline data island" : DATA_URL;

  // Synchronous panels.
  safeRender(() => renderHeadlineStats(document.getElementById("headline-stats"), summary), "headline");
  safeRender(() => renderCharacterInversion(document.getElementById("character-inversion"), summary), "character-inversion");
  safeRender(() => renderCrossModelInversion(document.getElementById("cross-model-inversion"), summary), "cross-model-inversion");
  safeRender(() => renderCategoryEnrichment(document.getElementById("category-enrichment"), summary), "category-enrichment");
  safeRender(() => renderAblationBars(document.getElementById("ablation-bars"), summary), "ablation");

  // Three.js panel — dynamic, soft-fail.
  try {
    const { renderBackboneGraph } = await import("./panels/backbone.js");
    renderBackboneGraph(document.getElementById("backbone-graph"), summary);
  } catch (e) {
    console.error("backbone graph failed:", e);
    const host = document.getElementById("backbone-graph");
    if (host) {
      host.innerHTML = `
        <div style="padding:32px;color:#666;font-family:var(--sans);font-size:13px">
          <strong style="color:#9c2a2a">3D backbone graph unavailable in this preview.</strong><br/>
          Three.js failed to load (probably a CDN block in this sandbox).
          Serve the page locally with <code>cd web &amp;&amp; python3 -m http.server 8770</code>.<br/>
          <span style="color:#999">${e.message || e}</span>
        </div>
      `;
    }
  }
}

function safeRender(fn, name) {
  try { fn(); } catch (e) { console.error(`${name} render failed:`, e); }
}

boot();
