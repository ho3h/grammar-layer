// Category enrichment table — for Gemma 2 2B, show per-category supporting vs
// opposing grammar share + the enrichment ratio. Highlight where the inversion
// is strongest, and where it flips into support.

const fmtPct = (v) => `${v.toFixed(1)}%`;

const CATEGORY_BLURB = {
  capital:              "target after copula (specific noun)",
  weekday:              "target after copula (specific noun)",
  code:                 "Python keyword after copula",
  syntactic:            "noun completing transitive verb",
  "summarization-opener": "target word after summary-introducer",
  instruction:          "single-word answer after directive",
  "factual-recall":     "grammatical/conjunctive completion",
  pronoun:              "referential element",
  reasoning:            "logical conclusion",
  math:                 "numerical word",
  "multi-step-arithmetic": "numerical word",
  "named-entity":       "category noun completing an entity",
};

export function renderCategoryEnrichment(root, summary) {
  if (!root) return;
  root.innerHTML = "";

  const gemma = summary.models.gemma ?? Object.values(summary.models)[0];
  if (!gemma) return;

  const supCats = gemma.feature_character.per_category;
  const oppCats = gemma.opposing_feature_character.per_category;

  // Compute per-category ratios
  const rows = [];
  for (const cat of Object.keys(supCats)) {
    const sg = supCats[cat].grammar_pct;
    const og = oppCats[cat]?.grammar_pct ?? 0;
    const total = supCats[cat].total;
    if (total === 0) continue;
    let ratio;
    if (sg === 0 && og === 0) ratio = 0;
    else if (sg === 0) ratio = Infinity;
    else ratio = og / sg;
    rows.push({ cat, sg, og, ratio });
  }
  // Sort: largest opposing > supporting first (the inversion), then where supporting beats opposing
  rows.sort((a, b) => {
    const aFlip = a.ratio !== 0 && a.ratio < 1;
    const bFlip = b.ratio !== 0 && b.ratio < 1;
    if (aFlip !== bFlip) return aFlip ? 1 : -1;
    return b.ratio - a.ratio;
  });

  const block = document.createElement("div");
  block.className = "enrichment-table";
  block.innerHTML = `
    <h3 style="font-family:var(--sans); font-size:13px; letter-spacing:0.04em; text-transform:uppercase; color:var(--faint); font-weight:600; margin:0 0 14px">
      Gemma 2 2B — where the grammar inversion lives, by task category
    </h3>
  `;
  const head = document.createElement("div");
  head.className = "enr-row";
  head.innerHTML = `
    <div class="cm-col-head">category</div>
    <div class="cm-col-head">supporting %</div>
    <div class="cm-col-head">opposing %</div>
    <div class="cm-col-head" style="text-align:center">×</div>
  `;
  block.appendChild(head);

  for (const r of rows) {
    let ratioStr;
    if (r.ratio === 0) ratioStr = "—";
    else if (!isFinite(r.ratio)) ratioStr = "∞";
    else ratioStr = `${r.ratio.toFixed(1)}×`;

    let cls = "";
    if (r.ratio !== 0 && r.ratio < 1) cls = "support-flip";
    else if (r.ratio >= 3) cls = "oppose-strong";

    const row = document.createElement("div");
    row.className = "enr-row";
    row.innerHTML = `
      <div>
        <strong>${r.cat}</strong>
        <div style="font-family:var(--sans); font-size:11px; color:var(--faint); margin-top:2px">${CATEGORY_BLURB[r.cat] ?? ""}</div>
      </div>
      <div class="enr-pct">${fmtPct(r.sg)}</div>
      <div class="enr-pct">${fmtPct(r.og)}</div>
      <div class="enr-ratio ${cls}">${ratioStr}</div>
    `;
    block.appendChild(row);
  }

  const note = document.createElement("p");
  note.style.cssText = "margin:14px 0 0; font-family:var(--sans); font-size:12px; color:var(--muted);";
  note.innerHTML = `
    <span style="display:inline-block; width:11px; height:11px; background:rgba(183,60,42,0.15); border-radius:2px; vertical-align:middle; margin-right:5px"></span>
    Strong inversion (opposing &gt;3× supporting) &nbsp; · &nbsp;
    <span style="display:inline-block; width:11px; height:11px; background:rgba(44,138,74,0.15); border-radius:2px; vertical-align:middle; margin-right:5px"></span>
    Inversion flipped (supporting &gt; opposing) — grammar features actually help on these tasks.
  `;
  block.appendChild(note);

  root.appendChild(block);
}
