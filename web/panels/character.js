// Character bars panel — grammar vs content composition of each model's
// load-bearing features. Shows supporting and (when available) opposing
// side-by-side, so the inversion is legible: supporting = mostly content,
// opposing = where the grammar features actually cluster.

export function renderCharacterBars(root, summary) {
  root.innerHTML = "";

  const models = Object.entries(summary.models);
  const hasOpposing = models.some(([, m]) =>
    m.opposing_feature_character && m.opposing_feature_character.total > 0
  );

  // Overall composition
  const overall = document.createElement("div");
  overall.className = "category-block";
  const t = document.createElement("p");
  t.className = "category-block-title";
  t.textContent = "Overall composition of load-bearing features";
  overall.appendChild(t);

  for (const [, m] of models) {
    overall.appendChild(makeModelBlock(m, hasOpposing));
  }
  root.appendChild(overall);

  // Per category
  for (const cat of summary.categories) {
    const block = document.createElement("div");
    block.className = "category-block";
    const head = document.createElement("p");
    head.className = "category-block-title";
    head.textContent = cat;
    block.appendChild(head);
    let anyShown = false;
    for (const [, m] of models) {
      const supCC = m.feature_character?.per_category?.[cat];
      const oppCC = m.opposing_feature_character?.per_category?.[cat];
      if (!supCC || supCC.total === 0) continue;
      block.appendChild(makeModelBlock(m, hasOpposing, supCC, oppCC));
      anyShown = true;
    }
    if (anyShown) root.appendChild(block);
  }

  const legend = document.createElement("div");
  legend.className = "legend";
  legend.innerHTML = `
    <span><span class="swatch" style="background:var(--grammar)"></span>grammar-flavored</span>
    <span><span class="swatch" style="background:var(--content)"></span>content-thematic</span>
    ${hasOpposing
      ? '<span style="color:#888;font-style:italic">Supporting = promotes target. Opposing = suppresses target.</span>'
      : '<span style="color:#888;font-style:italic">Opposing-side data pending second analysis pass.</span>'}
    <span style="color:#888;font-style:italic">Classification: strict keyword match on autointerp labels.</span>
  `;
  root.appendChild(legend);
}

function makeModelBlock(model, showOpposing, supOverride = null, oppOverride = null) {
  const supCC = supOverride || model.feature_character;
  const oppCC = oppOverride || model.opposing_feature_character;

  const row = document.createElement("div");
  row.className = "model-block";

  const lbl = document.createElement("div");
  lbl.className = "model-block-label";
  lbl.innerHTML = `
    <span class="swatch" style="background:${model.color}"></span>
    ${model.display_name}
  `;
  row.appendChild(lbl);

  if (showOpposing && oppCC && oppCC.total > 0) {
    const grid = document.createElement("div");
    grid.style.display = "grid";
    grid.style.gridTemplateColumns = "70px 1fr 8px 70px 1fr";
    grid.style.gap = "0";
    grid.style.alignItems = "center";
    grid.style.fontFamily = "var(--sans)";
    grid.style.fontSize = "12px";
    grid.style.color = "var(--muted)";

    grid.appendChild(spanLabel("supporting"));
    grid.appendChild(makeSplit(supCC));
    grid.appendChild(spacer());
    grid.appendChild(spanLabel("opposing"));
    grid.appendChild(makeSplit(oppCC));

    row.appendChild(grid);
  } else {
    row.appendChild(makeSplit(supCC));
  }
  return row;
}

function makeSplit(stats) {
  const wrap = document.createElement("div");
  wrap.style.display = "flex";
  wrap.style.flexDirection = "column";
  wrap.style.gap = "3px";

  const split = document.createElement("div");
  split.className = "split-bar";
  const gramPct = stats.grammar_pct || 0;
  const contPct = stats.content_pct || 0;
  const gram = document.createElement("div");
  gram.className = "grammar";
  gram.style.width = `${gramPct}%`;
  gram.textContent = gramPct >= 7 ? `${Math.round(gramPct)}%` : "";
  const cont = document.createElement("div");
  cont.className = "content";
  cont.style.width = `${contPct}%`;
  cont.textContent = contPct >= 7 ? `${Math.round(contPct)}%` : "";
  split.appendChild(gram);
  split.appendChild(cont);
  wrap.appendChild(split);

  const meta = document.createElement("div");
  meta.style.fontFamily = "var(--sans)";
  meta.style.fontSize = "11px";
  meta.style.color = "var(--faint)";
  meta.textContent = `${stats.total} feature·prompt slots`;
  wrap.appendChild(meta);
  return wrap;
}

function spanLabel(text) {
  const s = document.createElement("span");
  s.style.fontSize = "11px";
  s.style.textTransform = "uppercase";
  s.style.letterSpacing = "0.05em";
  s.style.color = "var(--faint)";
  s.style.paddingRight = "8px";
  s.textContent = text;
  return s;
}

function spacer() {
  return document.createElement("span");
}
