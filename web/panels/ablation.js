// Ablation bars panel — per-category, per-model "before / after" visualization
// showing how much joint zero-ablation of the top-10 supporting features drops
// the hit rate, plus the mean Δlog-P annotation.

const fmtPct = (v) => `${Math.round(v * 100)}%`;
const fmtNats = (v) => `${v >= 0 ? "+" : ""}${v.toFixed(2)} nats`;

export function renderAblationBars(root, summary) {
  root.innerHTML = "";

  const models = Object.entries(summary.models);
  const categories = ["__all__", ...summary.categories];

  for (const cat of categories) {
    const title = cat === "__all__" ? "Overall" : cat;
    const block = makeBlock(title, models, (m) => {
      if (cat === "__all__") {
        return {
          baseline: m.baseline_hit_rate,
          ablated: m.ablated_hit_rate,
          drop: m.mean_log_p_drop,
          n: m.n_prompts,
        };
      }
      const c = m.per_category?.[cat];
      if (!c) return null;
      return {
        baseline: c.baseline_hit_rate,
        ablated: c.ablated_hit_rate,
        drop: c.mean_log_p_drop,
        n: c.n,
      };
    });
    if (block) root.appendChild(block);
  }
}

function makeBlock(title, models, pick) {
  const block = document.createElement("div");
  block.className = "category-block";

  const t = document.createElement("p");
  t.className = "category-block-title";
  t.textContent = title;
  block.appendChild(t);

  let rendered = 0;
  for (const [, m] of models) {
    const stats = pick(m);
    if (!stats) continue;
    rendered += 1;

    const row = document.createElement("div");
    row.className = "ablation-row";
    row.style.display = "grid";
    row.style.gridTemplateColumns = "132px 1fr 1fr 110px";
    row.style.gap = "12px";
    row.style.alignItems = "center";
    row.style.padding = "8px 0";
    row.style.borderTop = rendered > 1 ? "1px dashed var(--hairline)" : "none";
    row.style.fontFamily = "var(--sans)";
    row.style.fontSize = "13px";

    const label = document.createElement("div");
    label.style.textAlign = "right";
    label.innerHTML = `
      <div style="color:${m.color};font-weight:600">${m.display_name}</div>
      <div style="color:#888;font-size:11px">n=${stats.n}</div>
    `;
    row.appendChild(label);

    row.appendChild(makeMiniBar("baseline", stats.baseline, m.color, 0.55));
    row.appendChild(makeMiniBar("ablated",  stats.ablated,  m.color, 1.0));

    const annot = document.createElement("div");
    annot.style.fontSize = "12px";
    annot.style.color = "var(--muted)";
    annot.innerHTML = `Δlog P = <strong style="color:#444">${fmtNats(stats.drop)}</strong>`;
    row.appendChild(annot);

    block.appendChild(row);
  }
  return rendered ? block : null;
}

function makeMiniBar(label, value, color, opacity) {
  const wrap = document.createElement("div");
  wrap.style.display = "flex";
  wrap.style.flexDirection = "column";
  wrap.style.gap = "3px";

  const labelEl = document.createElement("div");
  labelEl.style.fontSize = "10.5px";
  labelEl.style.color = "var(--faint)";
  labelEl.style.textTransform = "uppercase";
  labelEl.style.letterSpacing = "0.05em";
  labelEl.textContent = label;
  wrap.appendChild(labelEl);

  const track = document.createElement("div");
  track.style.position = "relative";
  track.style.height = "16px";
  track.style.background = "var(--pill-bg)";
  track.style.borderRadius = "3px";
  track.style.overflow = "hidden";

  const bar = document.createElement("div");
  bar.style.position = "absolute";
  bar.style.left = "0";
  bar.style.top = "0";
  bar.style.bottom = "0";
  bar.style.width = `${Math.max(2, value * 100)}%`;
  bar.style.background = color;
  bar.style.opacity = String(opacity);
  bar.style.borderRadius = "3px";
  bar.style.display = "flex";
  bar.style.alignItems = "center";
  bar.style.padding = "0 7px";
  bar.style.color = "white";
  bar.style.fontWeight = "600";
  bar.style.fontSize = "10.5px";
  bar.textContent = value > 0.12 ? fmtPct(value) : "";
  track.appendChild(bar);

  if (value <= 0.12) {
    const tinyLabel = document.createElement("span");
    tinyLabel.style.position = "absolute";
    tinyLabel.style.left = `${Math.max(2, value * 100) + 4}%`;
    tinyLabel.style.top = "0";
    tinyLabel.style.bottom = "0";
    tinyLabel.style.display = "flex";
    tinyLabel.style.alignItems = "center";
    tinyLabel.style.color = "var(--muted)";
    tinyLabel.style.fontSize = "10.5px";
    tinyLabel.style.fontWeight = "600";
    tinyLabel.textContent = fmtPct(value);
    track.appendChild(tinyLabel);
  }

  wrap.appendChild(track);
  return wrap;
}
