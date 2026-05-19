// Two compact panels:
// (a) character-inversion: Gemma's supporting vs opposing grammar share, side-by-side.
// (b) cross-model-inversion: same comparison across all loaded models.

const fmtPct = (v) => `${v.toFixed(1)}%`;

export function renderCharacterInversion(root, summary) {
  if (!root) return;
  root.innerHTML = "";

  // Single-model focus on Gemma 2 2B if present, otherwise first model.
  const gemma = summary.models.gemma ?? Object.values(summary.models)[0];
  if (!gemma) return;

  const sup = gemma.feature_character;
  const opp = gemma.opposing_feature_character;

  const block = document.createElement("div");
  block.className = "inversion-block";
  block.innerHTML = `<h3>Gemma 2 2B — share of grammar-flavored features (52 prompts)</h3>`;

  block.appendChild(makeRow("Supporting", "Promotes target", sup, "supporting"));
  block.appendChild(makeRow("Opposing", "Suppresses target", opp, "opposing"));

  const ratio = sup.grammar_pct > 0 ? opp.grammar_pct / sup.grammar_pct : Infinity;
  const ratioRow = document.createElement("div");
  ratioRow.className = "inv-row";
  ratioRow.style.borderTop = "1px solid var(--hairline)";
  ratioRow.style.marginTop = "6px";
  ratioRow.style.paddingTop = "16px";
  ratioRow.innerHTML = `
    <div class="inv-label" style="color:var(--faint); font-size:11px; letter-spacing:0.05em; text-transform:uppercase">enrichment</div>
    <div style="grid-column: 2 / span 3; font-family:var(--serif); font-size:15px; color:var(--ink); line-height:1.5">
      Grammar features are <strong style="color:var(--oppose)">${ratio.toFixed(1)}× more concentrated</strong>
      on the suppressing side than the promoting side.
    </div>
  `;
  block.appendChild(ratioRow);

  root.appendChild(block);
}

function makeRow(label, sublabel, stats, kind) {
  const row = document.createElement("div");
  row.className = "inv-row";
  const gramPct = stats.grammar_pct;
  const contPct = stats.content_pct;
  row.innerHTML = `
    <div class="inv-label">
      ${label}
      <div style="font-family:var(--sans); font-size:11px; color:var(--faint); font-weight:400; letter-spacing:0.04em; text-transform:uppercase">${sublabel}</div>
    </div>
    <div class="inv-bar">
      ${gramPct > 0 ? `<div class="inv-bar-fill grammar" style="width:${gramPct}%">${gramPct >= 6 ? fmtPct(gramPct) + ' grammar' : ''}</div>` : ''}
      <div class="inv-bar-rest" style="left:${gramPct}%; right:0; width:auto">
        ${contPct >= 30 ? fmtPct(contPct) + ' content' : ''}
      </div>
    </div>
    <div class="inv-arrow" style="visibility:hidden">→</div>
    <div style="grid-column: 4 / span 2; font-family:var(--sans); font-size:12px; color:var(--muted)">
      ${stats.total} feature slots in top-10 ${kind} sets across ${gramPct.toFixed(0)}% grammar
    </div>
  `;
  // 4-column layout: label / bar / count / semantic blurb
  row.innerHTML = `
    <div class="inv-label">
      ${label}
      <div style="font-family:var(--sans); font-size:10px; color:var(--faint); font-weight:400; letter-spacing:0.05em; text-transform:uppercase; margin-top:2px">${sublabel}</div>
    </div>
    <div class="inv-bar">
      <div class="inv-bar-fill grammar" style="width:${Math.max(2, gramPct)}%">${gramPct >= 5 ? fmtPct(gramPct) : ''}</div>
      <div class="inv-bar-rest" style="left:${gramPct}%; right:0;">${stats.content_pct >= 30 ? fmtPct(stats.content_pct) + ' content' : ''}</div>
    </div>
    <div class="inv-ratio">${stats.grammar}/${stats.total} grammar</div>
    <div style="font-family:var(--sans); font-size:12px; color:var(--muted)">
      ${kind === 'opposing' ? 'grammar features push <em style="color:var(--oppose)">against</em> the target' : 'grammar features push <em style="color:var(--support)">for</em> the target'}
    </div>
  `;
  return row;
}

export function renderCrossModelInversion(root, summary) {
  if (!root) return;
  root.innerHTML = "";

  const block = document.createElement("div");
  block.className = "cm-comparison";
  block.innerHTML = `<h3 style="font-family:var(--sans); font-size:13px; letter-spacing:0.04em; text-transform:uppercase; color:var(--faint); font-weight:600; margin:0 0 14px">Grammar share by model — supporting vs opposing</h3>`;

  // Header row
  const head = document.createElement("div");
  head.className = "cm-row";
  head.innerHTML = `
    <div class="cm-col-head">model</div>
    <div class="cm-col-head">% grammar — supporting</div>
    <div class="cm-col-head">% grammar — opposing</div>
    <div class="cm-col-head" style="text-align:center">enrichment</div>
  `;
  block.appendChild(head);

  // Order models smallest → largest by display, so the size sweep reads left-to-right.
  // Any model not listed here lands at the end.
  const order = [
    "pythia_70m",  // 70M
    "gpt2",        // 124M
    "qwen3_1_7b",  // 1.7B
    "gemma_1_2b",  // 2B (gen 1)
    "gemma",       // 2B (gen 2)
    "mistral_7b",  // 7B
    "gemma_9b",    // 9B
    "gemma_27b",   // 27B
  ];
  const knownEntries = order
    .filter((k) => k in summary.models)
    .map((k) => [k, summary.models[k]]);
  const unknownEntries = Object.entries(summary.models).filter(([k]) => !order.includes(k));
  const entries = [...knownEntries, ...unknownEntries];

  for (const [, m] of entries) {
    const sg = m.feature_character.grammar_pct;
    const og = m.opposing_feature_character.grammar_pct;
    const ratio = sg > 0 ? og / sg : (og > 0 ? Infinity : 0);
    const ratioStr = isFinite(ratio) ? `${ratio.toFixed(1)}×` : "∞";
    const ratioClass = ratio >= 2 ? "high" : "low";

    const row = document.createElement("div");
    row.className = "cm-row";

    if (m.has_labels === false) {
      // No autointerp labels for this model's SAE — we can't classify
      // grammar vs content. Show explicit "labels pending" instead of
      // misleading 0%.
      row.innerHTML = `
        <div class="cm-model" style="color:${m.color}">${m.display_name}</div>
        <div style="grid-column: 2 / span 3; color:var(--faint); font-size:12px; font-style:italic">
          load-bearing test ran (${m.n_prompts} prompts, hit rate ${Math.round(m.baseline_hit_rate*100)}%→${Math.round(m.ablated_hit_rate*100)}%, Δlog P = +${m.mean_log_p_drop.toFixed(2)} nats), but no autointerp labels cached for this SAE — grammar/content split not classified yet.
        </div>
      `;
    } else {
      row.innerHTML = `
        <div class="cm-model" style="color:${m.color}">${m.display_name}</div>
        <div class="cm-pct"><span class="mini-bar"><span style="width:${Math.min(100, sg * 4)}%"></span></span>${fmtPct(sg)}</div>
        <div class="cm-pct"><span class="mini-bar"><span style="width:${Math.min(100, og * 4)}%; background:${ratio >= 2 ? m.color : 'var(--grammar)'}"></span></span>${fmtPct(og)}</div>
        <div class="cm-mult ${ratioClass}">${ratioStr}</div>
      `;
    }
    block.appendChild(row);
  }

  const footer = document.createElement("p");
  footer.style.cssText = "margin:12px 0 0; font-family:var(--sans); font-size:12px; color:var(--muted);";
  footer.innerHTML = `Mini-bars scaled to 25% so small differences are visible. <strong>Only Gemma 2 2B</strong> exhibits the inversion (~3× enrichment of grammar features on the suppressing side). GPT-2 and Pythia stay flat — grammar features in their residual streams don't preferentially suppress.`;
  block.appendChild(footer);

  root.appendChild(block);
}
