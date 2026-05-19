// Headline stats — three big-number cards, one per model, showing the
// causal-collapse story at-a-glance.

const fmtPct = (v) => `${Math.round(v * 100)}%`;
const fmtNats = (v) => `${v >= 0 ? "+" : ""}${v.toFixed(2)} nats`;

export function renderHeadlineStats(root, summary) {
  if (!root) return;
  root.innerHTML = "";

  const models = Object.values(summary.models);

  // Auto-fill so 3 models pack 3-wide and 8 models wrap into 2 rows of 4.
  root.style.gridTemplateColumns = "repeat(auto-fit, minmax(170px, 1fr))";

  for (const m of models) {
    const card = document.createElement("div");
    card.className = "headline-stat";
    card.style.borderTop = `3px solid ${m.color}`;
    card.innerHTML = `
      <div class="model">model</div>
      <div class="model-name" style="color:${m.color}">${m.display_name}</div>
      <div class="big-number">
        ${fmtPct(m.baseline_hit_rate)}
        <span class="big-arrow">→</span>
        <strong style="color:${m.color}">${fmtPct(m.ablated_hit_rate)}</strong>
      </div>
      <div class="stat-foot">
        hit rate after joint-ablating the top-10 supporting features.<br/>
        target log-probability <span class="nats">${fmtNats(m.mean_log_p_drop)}</span> on average across ${m.n_prompts} prompts.
      </div>
    `;
    root.appendChild(card);
  }
}
