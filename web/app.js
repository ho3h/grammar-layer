// Single-file panel renderer — NO module imports, NO import map. Loads the
// inline data island first (always works), then optionally tries fetch for
// fresh data, then optionally loads Three.js from a CDN for the backbone graph.
// If any optional step fails, the rest of the page still renders.
//
// Plain <script> tag, no module semantics. Safe in any environment that runs JS.

(function () {
  "use strict";

  // ───────────────────────── status + error surface ──────────────────────
  var statusBar = document.getElementById("status-bar");
  function setStatus(status, text) {
    if (statusBar) {
      statusBar.innerHTML = '<span class="status-pill" data-status="' + status + '">' + text + "</span>";
    }
  }
  // Only show a status-bar JS error for failures inside this script.
  // Three.js module-specifier errors (when the import map isn't applied) are
  // expected in some sandboxed previews and shouldn't replace the status pill.
  window.addEventListener("error", function (e) {
    var msg = String(e.message || "");
    if (/module specifier|Failed to resolve module|three\.module\.js/i.test(msg)) return;
    setStatus("stale", "JS error: " + msg);
  });

  // ───────────────────────── helpers ─────────────────────────────────────
  function fmtPct(v) { return Math.round(v * 100) + "%"; }
  function fmtPctOne(v) { return v.toFixed(1) + "%"; }
  function fmtNats(v) { return (v >= 0 ? "+" : "") + v.toFixed(2) + " nats"; }
  function esc(s) {
    return String(s || "").replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  // ───────────────────────── data loading ────────────────────────────────
  var DATA_URL = "data/summary.json";
  function loadInline() {
    var island = document.getElementById("summary-island");
    if (!island || !island.textContent.trim()) return null;
    try { return JSON.parse(island.textContent); } catch (e) { return null; }
  }

  function tryFetch(cb) {
    if (typeof fetch !== "function") { cb(null); return; }
    try {
      fetch(DATA_URL, { cache: "no-cache" })
        .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
        .then(function (s) { cb(s, "fetch"); })
        .catch(function (e) { console.warn("fetch failed:", e && e.message); cb(null); });
    } catch (e) {
      cb(null);
    }
  }

  // ───────────────────────── headline cards ──────────────────────────────
  function renderHeadlineStats(root, summary) {
    if (!root) return;
    root.innerHTML = "";
    root.style.gridTemplateColumns = "repeat(auto-fit, minmax(170px, 1fr))";
    var keys = Object.keys(summary.models);
    for (var i = 0; i < keys.length; i++) {
      var m = summary.models[keys[i]];
      var card = document.createElement("div");
      card.className = "headline-stat";
      card.style.borderTop = "3px solid " + m.color;
      card.innerHTML =
        '<div class="model">model</div>' +
        '<div class="model-name" style="color:' + m.color + '">' + esc(m.display_name) + "</div>" +
        '<div class="big-number">' + fmtPct(m.baseline_hit_rate) +
          ' <span class="big-arrow">→</span> <strong style="color:' + m.color + '">' + fmtPct(m.ablated_hit_rate) + "</strong></div>" +
        '<div class="stat-foot">hit rate after joint-ablating the top-10 supporting features.<br/>' +
          'target log-probability <span class="nats">' + fmtNats(m.mean_log_p_drop) + "</span> on average across " + m.n_prompts + " prompts.</div>";
      root.appendChild(card);
    }
  }

  // ───────────────────────── character inversion (Gemma focus) ───────────
  function renderCharacterInversion(root, summary) {
    if (!root) return;
    root.innerHTML = "";
    var gemma = summary.models.gemma;
    if (!gemma) {
      var first = Object.values(summary.models)[0];
      gemma = first;
    }
    if (!gemma) return;
    var sup = gemma.feature_character;
    var opp = gemma.opposing_feature_character;

    var block = document.createElement("div");
    block.className = "inversion-block";
    block.innerHTML = "<h3>" + esc(gemma.display_name) + " — share of grammar-flavored features (" + gemma.n_prompts + " prompts)</h3>";

    block.appendChild(invRow("Supporting", "Promotes target", sup, "supporting"));
    block.appendChild(invRow("Opposing", "Suppresses target", opp, "opposing"));

    var ratio = sup.grammar_pct > 0 ? opp.grammar_pct / sup.grammar_pct : Infinity;
    var rRow = document.createElement("div");
    rRow.className = "inv-row";
    rRow.style.borderTop = "1px solid var(--hairline)";
    rRow.style.marginTop = "6px";
    rRow.style.paddingTop = "16px";
    rRow.innerHTML =
      '<div class="inv-label" style="color:var(--faint); font-size:11px; letter-spacing:0.05em; text-transform:uppercase">enrichment</div>' +
      '<div style="grid-column: 2 / span 3; font-family:var(--serif); font-size:15px; color:var(--ink); line-height:1.5">' +
        'Grammar features are <strong style="color:var(--oppose)">' + ratio.toFixed(1) + '× more concentrated</strong> on the suppressing side than the promoting side.' +
      '</div>';
    block.appendChild(rRow);
    root.appendChild(block);
  }

  function invRow(label, sub, stats, kind) {
    var row = document.createElement("div");
    row.className = "inv-row";
    var gramPct = stats.grammar_pct;
    row.innerHTML =
      '<div class="inv-label">' + label +
        '<div style="font-family:var(--sans); font-size:10px; color:var(--faint); font-weight:400; letter-spacing:0.05em; text-transform:uppercase; margin-top:2px">' + sub + '</div>' +
      '</div>' +
      '<div class="inv-bar">' +
        '<div class="inv-bar-fill grammar" style="width:' + Math.max(2, gramPct) + '%">' + (gramPct >= 5 ? fmtPctOne(gramPct) : "") + '</div>' +
        '<div class="inv-bar-rest" style="left:' + gramPct + '%; right:0;"></div>' +
      '</div>' +
      '<div class="inv-ratio">' + stats.grammar + '/' + stats.total + ' grammar</div>' +
      '<div style="font-family:var(--sans); font-size:12px; color:var(--muted)">' +
        (kind === "opposing" ? 'grammar features push <em style="color:var(--oppose)">against</em> the target' : 'grammar features push <em style="color:var(--support)">for</em> the target') +
      '</div>';
    return row;
  }

  // ───────────────────────── cross-model table ───────────────────────────
  function renderCrossModelInversion(root, summary) {
    if (!root) return;
    root.innerHTML = "";

    var block = document.createElement("div");
    block.className = "cm-comparison";
    block.innerHTML = '<h3 style="font-family:var(--sans); font-size:13px; letter-spacing:0.04em; text-transform:uppercase; color:var(--faint); font-weight:600; margin:0 0 14px">Grammar share by model — supporting vs opposing</h3>';

    var head = document.createElement("div");
    head.className = "cm-row";
    head.innerHTML =
      '<div class="cm-col-head">model</div>' +
      '<div class="cm-col-head">% grammar — supporting</div>' +
      '<div class="cm-col-head">% grammar — opposing</div>' +
      '<div class="cm-col-head" style="text-align:center">enrichment</div>';
    block.appendChild(head);

    var order = ["pythia_70m", "gpt2", "qwen3_1_7b", "gemma_1_2b", "gemma", "mistral_7b", "gemma_9b", "gemma_27b"];
    var done = {};
    var entries = [];
    for (var i = 0; i < order.length; i++) {
      if (order[i] in summary.models) {
        entries.push([order[i], summary.models[order[i]]]);
        done[order[i]] = true;
      }
    }
    var unkKeys = Object.keys(summary.models);
    for (var j = 0; j < unkKeys.length; j++) {
      if (!done[unkKeys[j]]) entries.push([unkKeys[j], summary.models[unkKeys[j]]]);
    }

    for (var k = 0; k < entries.length; k++) {
      var name = entries[k][0], m = entries[k][1];
      var sg = m.feature_character.grammar_pct;
      var og = m.opposing_feature_character.grammar_pct;
      var ratio = sg > 0 ? og / sg : (og > 0 ? Infinity : 0);
      var ratioStr = isFinite(ratio) ? ratio.toFixed(1) + "×" : "∞";
      var ratioClass = ratio >= 2 ? "high" : "low";
      var row = document.createElement("div");
      row.className = "cm-row";
      if (m.has_labels === false) {
        row.innerHTML =
          '<div class="cm-model" style="color:' + m.color + '">' + esc(m.display_name) + '</div>' +
          '<div style="grid-column: 2 / span 3; color:var(--faint); font-size:12px; font-style:italic">' +
            'load-bearing test ran (' + m.n_prompts + ' prompts, hit rate ' + fmtPct(m.baseline_hit_rate) + '→' + fmtPct(m.ablated_hit_rate) + ', Δlog P = ' + fmtNats(m.mean_log_p_drop) +
            '), but no autointerp labels cached for this SAE — grammar/content split not classified yet.' +
          '</div>';
      } else {
        row.innerHTML =
          '<div class="cm-model" style="color:' + m.color + '">' + esc(m.display_name) + '</div>' +
          '<div class="cm-pct"><span class="mini-bar"><span style="width:' + Math.min(100, sg * 4) + '%"></span></span>' + fmtPctOne(sg) + '</div>' +
          '<div class="cm-pct"><span class="mini-bar"><span style="width:' + Math.min(100, og * 4) + '%; background:' + (ratio >= 2 ? m.color : "var(--grammar)") + '"></span></span>' + fmtPctOne(og) + '</div>' +
          '<div class="cm-mult ' + ratioClass + '">' + ratioStr + '</div>';
      }
      block.appendChild(row);
    }

    var footer = document.createElement("p");
    footer.style.cssText = "margin:12px 0 0; font-family:var(--sans); font-size:12px; color:var(--muted);";
    footer.innerHTML = "Mini-bars scaled to 25% so small differences are visible. <strong>Only Gemma 2 2B</strong> exhibits the inversion (~3× enrichment of grammar features on the suppressing side). GPT-2 stays flat. The other models' grammar/content split is pending label-cache population.";
    block.appendChild(footer);

    root.appendChild(block);
  }

  // ───────────────────────── category enrichment table ──────────────────
  var CATEGORY_BLURB = {
    capital: "target after copula (specific noun)",
    weekday: "target after copula (specific noun)",
    code: "Python keyword after copula",
    syntactic: "noun completing transitive verb",
    "summarization-opener": "target word after summary-introducer",
    instruction: "single-word answer after directive",
    "factual-recall": "grammatical/conjunctive completion",
    pronoun: "referential element",
    reasoning: "logical conclusion",
    math: "numerical word",
    "multi-step-arithmetic": "numerical word",
    "named-entity": "category noun completing an entity",
  };

  function renderCategoryEnrichment(root, summary) {
    if (!root) return;
    root.innerHTML = "";
    var gemma = summary.models.gemma || Object.values(summary.models)[0];
    if (!gemma) return;
    var supCats = gemma.feature_character.per_category;
    var oppCats = gemma.opposing_feature_character.per_category;
    var rows = [];
    for (var cat in supCats) {
      var sg = supCats[cat].grammar_pct;
      var og = (oppCats[cat] && oppCats[cat].grammar_pct) || 0;
      var total = supCats[cat].total;
      if (total === 0) continue;
      var ratio;
      if (sg === 0 && og === 0) ratio = 0;
      else if (sg === 0) ratio = Infinity;
      else ratio = og / sg;
      rows.push({ cat: cat, sg: sg, og: og, ratio: ratio });
    }
    rows.sort(function (a, b) {
      var aFlip = a.ratio !== 0 && a.ratio < 1;
      var bFlip = b.ratio !== 0 && b.ratio < 1;
      if (aFlip !== bFlip) return aFlip ? 1 : -1;
      return b.ratio - a.ratio;
    });

    var block = document.createElement("div");
    block.className = "enrichment-table";
    block.innerHTML = '<h3 style="font-family:var(--sans); font-size:13px; letter-spacing:0.04em; text-transform:uppercase; color:var(--faint); font-weight:600; margin:0 0 14px">' + esc(gemma.display_name) + ' — where the grammar inversion lives, by task category</h3>';
    var head = document.createElement("div");
    head.className = "enr-row";
    head.innerHTML =
      '<div class="cm-col-head">category</div>' +
      '<div class="cm-col-head">supporting %</div>' +
      '<div class="cm-col-head">opposing %</div>' +
      '<div class="cm-col-head" style="text-align:center">×</div>';
    block.appendChild(head);

    for (var r = 0; r < rows.length; r++) {
      var row = rows[r];
      var ratioStr;
      if (row.ratio === 0) ratioStr = "—";
      else if (!isFinite(row.ratio)) ratioStr = "∞";
      else ratioStr = row.ratio.toFixed(1) + "×";
      var cls = "";
      if (row.ratio !== 0 && row.ratio < 1) cls = "support-flip";
      else if (row.ratio >= 3) cls = "oppose-strong";
      var rEl = document.createElement("div");
      rEl.className = "enr-row";
      rEl.innerHTML =
        '<div><strong>' + esc(row.cat) + '</strong>' +
          '<div style="font-family:var(--sans); font-size:11px; color:var(--faint); margin-top:2px">' + esc(CATEGORY_BLURB[row.cat] || "") + '</div>' +
        '</div>' +
        '<div class="enr-pct">' + fmtPctOne(row.sg) + '</div>' +
        '<div class="enr-pct">' + fmtPctOne(row.og) + '</div>' +
        '<div class="enr-ratio ' + cls + '">' + ratioStr + '</div>';
      block.appendChild(rEl);
    }

    var note = document.createElement("p");
    note.style.cssText = "margin:14px 0 0; font-family:var(--sans); font-size:12px; color:var(--muted);";
    note.innerHTML =
      '<span style="display:inline-block; width:11px; height:11px; background:rgba(183,60,42,0.15); border-radius:2px; vertical-align:middle; margin-right:5px"></span>' +
      'Strong inversion (opposing &gt;3× supporting) &nbsp; · &nbsp;' +
      '<span style="display:inline-block; width:11px; height:11px; background:rgba(44,138,74,0.15); border-radius:2px; vertical-align:middle; margin-right:5px"></span>' +
      'Inversion flipped (supporting &gt; opposing) — grammar features actually help on these tasks.';
    block.appendChild(note);
    root.appendChild(block);
  }

  // ───────────────────────── ablation bars (per-category) ───────────────
  function renderAblationBars(root, summary) {
    if (!root) return;
    root.innerHTML = "";
    var models = Object.entries(summary.models);
    var cats = ["__all__"].concat(summary.categories);
    for (var c = 0; c < cats.length; c++) {
      var cat = cats[c];
      var title = cat === "__all__" ? "Overall" : cat;
      var block = abBlock(title, models, cat);
      if (block) root.appendChild(block);
    }
  }
  function abBlock(title, models, cat) {
    var block = document.createElement("div");
    block.className = "category-block";
    var t = document.createElement("p");
    t.className = "category-block-title";
    t.textContent = title;
    block.appendChild(t);
    var rendered = 0;
    for (var i = 0; i < models.length; i++) {
      var m = models[i][1];
      var stats;
      if (cat === "__all__") {
        stats = { baseline: m.baseline_hit_rate, ablated: m.ablated_hit_rate, drop: m.mean_log_p_drop, n: m.n_prompts };
      } else {
        var c = m.per_category && m.per_category[cat];
        if (!c) continue;
        stats = { baseline: c.baseline_hit_rate, ablated: c.ablated_hit_rate, drop: c.mean_log_p_drop, n: c.n };
      }
      rendered += 1;
      var row = document.createElement("div");
      row.className = "ablation-row";
      row.style.cssText = "display:grid; grid-template-columns:132px 1fr 1fr 110px; gap:12px; align-items:center; padding:8px 0; border-top:" + (rendered > 1 ? "1px dashed var(--hairline)" : "none") + "; font-family:var(--sans); font-size:13px;";
      row.innerHTML =
        '<div style="text-align:right">' +
          '<div style="color:' + m.color + '; font-weight:600">' + esc(m.display_name) + '</div>' +
          '<div style="color:#888;font-size:11px">n=' + stats.n + '</div>' +
        '</div>' +
        miniBar("baseline", stats.baseline, m.color, 0.55) +
        miniBar("ablated", stats.ablated, m.color, 1.0) +
        '<div style="font-size:12px; color:var(--muted)">Δlog P = <strong style="color:#444">' + fmtNats(stats.drop) + '</strong></div>';
      block.appendChild(row);
    }
    return rendered ? block : null;
  }
  function miniBar(label, value, color, opacity) {
    var pct = Math.max(2, value * 100);
    var labelHtml = value > 0.12 ? fmtPct(value) : "";
    return '<div style="display:flex; flex-direction:column; gap:3px">' +
      '<div style="font-size:10.5px; color:var(--faint); text-transform:uppercase; letter-spacing:0.05em">' + label + '</div>' +
      '<div style="position:relative; height:16px; background:var(--pill-bg); border-radius:3px; overflow:hidden">' +
        '<div style="position:absolute; left:0; top:0; bottom:0; width:' + pct + '%; background:' + color + '; opacity:' + opacity + '; border-radius:3px; display:flex; align-items:center; padding:0 7px; color:white; font-weight:600; font-size:10.5px">' + labelHtml + '</div>' +
      '</div>' +
    '</div>';
  }

  // ───────────────────────── Three.js backbone (optional, soft-fail) ────
  function renderBackbone(root, summary) {
    if (!root) return;
    // We don't ship Three.js bundled. Try loading it dynamically; on
    // failure, leave a visible note instead of taking down the page.
    var script = document.createElement("script");
    script.type = "module";
    script.textContent =
      'import * as THREE from "https://unpkg.com/three@0.165.0/build/three.module.js";\n' +
      'import { OrbitControls } from "https://unpkg.com/three@0.165.0/examples/jsm/controls/OrbitControls.js";\n' +
      'window.__three = THREE; window.__OrbitControls = OrbitControls;\n' +
      'window.dispatchEvent(new CustomEvent("three-ready"));';
    script.onerror = function () { backboneFallback(root, "Three.js script failed to load"); };
    document.head.appendChild(script);

    var armed = false;
    setTimeout(function () { if (!armed) backboneFallback(root, "Three.js never reported ready (sandboxed preview likely blocks unpkg.com)"); }, 5000);
    window.addEventListener("three-ready", function () {
      armed = true;
      try {
        drawBackbone(root, summary, window.__three, window.__OrbitControls);
      } catch (e) { backboneFallback(root, e.message); }
    });
  }
  function backboneFallback(root, why) {
    root.innerHTML =
      '<div style="padding:32px; color:#666; font-family:var(--sans); font-size:13px">' +
        '<strong style="color:#9c2a2a">3D backbone graph unavailable in this preview.</strong><br/>' +
        'Three.js wasn\'t reachable here. To see the full visualisation, serve the page locally:<br/>' +
        '<code style="background:var(--pill-bg); padding:2px 6px; border-radius:3px">cd web &amp;&amp; python3 -m http.server 8770</code><br/>' +
        '<span style="color:#999">' + esc(why) + '</span>' +
      '</div>';
  }

  var CATEGORY_COLORS = {
    capital: "#d35400", weekday: "#2980b9", math: "#27ae60",
    "named-entity": "#8e44ad", syntactic: "#16a085", reasoning: "#c0392b",
    instruction: "#7f8c8d", "factual-recall": "#e67e22", code: "#34495e",
    "multi-step-arithmetic": "#1abc9c", pronoun: "#9b59b6",
    "summarization-opener": "#f39c12",
  };

  function drawBackbone(root, summary, THREE, OrbitControls) {
    root.innerHTML = "";
    var modelEntries = Object.entries(summary.models);
    if (!modelEntries.length) return;

    var toolbar = document.createElement("div");
    toolbar.style.cssText = "position:absolute; top:12px; left:12px; z-index:5; display:flex; gap:6px; font-family:var(--sans); font-size:12px;";
    var select = document.createElement("select");
    select.style.cssText = "padding:5px 10px; border:1px solid var(--hairline); border-radius:4px; background:white; font:inherit;";
    for (var i = 0; i < modelEntries.length; i++) {
      var opt = document.createElement("option");
      opt.value = modelEntries[i][0];
      opt.textContent = modelEntries[i][1].display_name;
      select.appendChild(opt);
    }
    var sideToggle = document.createElement("select");
    sideToggle.style.cssText = select.style.cssText;
    [["supporting","supporting features"],["opposing","opposing features"]].forEach(function (p) {
      var o = document.createElement("option"); o.value = p[0]; o.textContent = p[1]; sideToggle.appendChild(o);
    });
    toolbar.appendChild(select); toolbar.appendChild(sideToggle);
    root.appendChild(toolbar);

    var host = document.createElement("div");
    host.style.cssText = "position:absolute; inset:0;";
    root.appendChild(host);

    var cleanup = null;
    function render() {
      if (cleanup) cleanup();
      cleanup = drawOne(host, summary.models[select.value], sideToggle.value, THREE, OrbitControls);
    }
    select.addEventListener("change", render);
    sideToggle.addEventListener("change", render);
    render();
  }
  function drawOne(root, modelData, side, THREE, OrbitControls) {
    while (root.firstChild) root.removeChild(root.firstChild);
    var w = root.clientWidth || root.parentElement.clientWidth;
    var h = root.clientHeight || root.parentElement.clientHeight;
    var scene = new THREE.Scene();
    scene.background = new THREE.Color(0xfafaf7);
    var cam = new THREE.PerspectiveCamera(45, w / h, 0.1, 2000);
    cam.position.set(0, 0, 90);
    var renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
    renderer.setSize(w, h);
    root.appendChild(renderer.domElement);
    var controls = new OrbitControls(cam, renderer.domElement);
    controls.enableDamping = true; controls.dampingFactor = 0.08;
    scene.add(new THREE.AmbientLight(0xffffff, 0.85));
    var dl = new THREE.DirectionalLight(0xffffff, 0.45); dl.position.set(50,80,100); scene.add(dl);

    var edges = side === "opposing" ? (modelData.opposing_edges || []) : (modelData.backbone_edges || []);
    var g = buildGraph(edges);
    if (!g.nodes.length) {
      root.innerHTML = '<div style="padding:24px;color:#888;font-family:var(--sans);font-size:13px">No ' + side + ' edges for ' + modelData.display_name + '.</div>';
      return function(){};
    }
    layout(g.nodes, g.edges);

    var sphere = new THREE.SphereGeometry(1, 14, 10);
    var meshes = [];
    for (var i = 0; i < g.nodes.length; i++) {
      var n = g.nodes[i];
      var isP = n.kind === "prompt";
      var color = isP ? (CATEGORY_COLORS[n.category] || "#95a5a6") : (n.isGrammar ? "#4a6fa5" : "#8a7752");
      var r = isP ? 1.6 : (n.isGrammar ? 1.0 : 0.7);
      var mat = new THREE.MeshLambertMaterial({ color: color });
      var mesh = new THREE.Mesh(sphere, mat);
      mesh.scale.set(r, r, r);
      mesh.position.set(n.x, n.y, n.z);
      mesh.userData.node = n;
      scene.add(mesh); meshes.push(mesh);
    }
    var posArr = new Float32Array(g.edges.length * 6);
    var colArr = new Float32Array(g.edges.length * 6);
    for (var j = 0; j < g.edges.length; j++) {
      var a = g.nodes[g.edges[j].a], b = g.nodes[g.edges[j].b], off = j * 6;
      posArr[off]=a.x; posArr[off+1]=a.y; posArr[off+2]=a.z;
      posArr[off+3]=b.x; posArr[off+4]=b.y; posArr[off+5]=b.z;
      var c = new THREE.Color((b.kind === "feature" && b.isGrammar) ? "#4a6fa5" : "#8a7752");
      for (var p = 0; p < 2; p++) { colArr[off+p*3]=c.r; colArr[off+p*3+1]=c.g; colArr[off+p*3+2]=c.b; }
    }
    var lg = new THREE.BufferGeometry();
    lg.setAttribute("position", new THREE.BufferAttribute(posArr, 3));
    lg.setAttribute("color", new THREE.BufferAttribute(colArr, 3));
    scene.add(new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.28 })));

    var stopped = false;
    function tick() {
      if (stopped) return;
      requestAnimationFrame(tick);
      controls.update();
      renderer.render(scene, cam);
    }
    tick();
    return function () { stopped = true; renderer.dispose(); };
  }
  function buildGraph(edges) {
    var pIdx = {}, fIdx = {}, nodes = [], out = [];
    for (var i = 0; i < edges.length; i++) {
      var e = edges[i];
      var aKey = "p:" + e.prompt_id;
      if (!(aKey in pIdx)) {
        pIdx[aKey] = nodes.length;
        nodes.push({ kind: "prompt", id: e.prompt_id, category: e.prompt_category,
          x: (Math.random()-0.5)*60, y: (Math.random()-0.5)*60, z: (Math.random()-0.5)*60,
          vx:0, vy:0, vz:0 });
      }
      var bKey = "f:" + e.feature_index;
      if (!(bKey in fIdx)) {
        fIdx[bKey] = nodes.length;
        nodes.push({ kind: "feature", feature_index: e.feature_index, isGrammar: !!e.feature_is_grammar,
          x: (Math.random()-0.5)*60, y: (Math.random()-0.5)*60, z: (Math.random()-0.5)*60,
          vx:0, vy:0, vz:0 });
      }
      out.push({ a: pIdx[aKey], b: fIdx[bKey] });
    }
    return { nodes: nodes, edges: out };
  }
  function layout(nodes, edges) {
    for (var it = 0; it < 200; it++) {
      for (var i = 0; i < nodes.length; i++) {
        var a = nodes[i];
        for (var j = i+1; j < nodes.length; j++) {
          var b = nodes[j];
          var dx=a.x-b.x, dy=a.y-b.y, dz=a.z-b.z, d2=dx*dx+dy*dy+dz*dz+0.01, d=Math.sqrt(d2), f=32/d2;
          var fx=dx/d*f, fy=dy/d*f, fz=dz/d*f;
          a.vx+=fx; a.vy+=fy; a.vz+=fz;
          b.vx-=fx; b.vy-=fy; b.vz-=fz;
        }
      }
      for (var k = 0; k < edges.length; k++) {
        var aa = nodes[edges[k].a], bb = nodes[edges[k].b];
        var dx=bb.x-aa.x, dy=bb.y-aa.y, dz=bb.z-aa.z, d=Math.sqrt(dx*dx+dy*dy+dz*dz)+0.01;
        var f=(d-6)*0.06, fx=dx/d*f, fy=dy/d*f, fz=dz/d*f;
        aa.vx+=fx; aa.vy+=fy; aa.vz+=fz;
        bb.vx-=fx; bb.vy-=fy; bb.vz-=fz;
      }
      for (var m = 0; m < nodes.length; m++) {
        var n = nodes[m];
        n.vx -= n.x*0.012; n.vy -= n.y*0.012; n.vz -= n.z*0.012;
        n.vx *= 0.86; n.vy *= 0.86; n.vz *= 0.86;
        n.x += n.vx; n.y += n.vy; n.z += n.vz;
      }
    }
  }

  // ───────────────────────── boot ────────────────────────────────────────
  function boot(summary, source) {
    if (!summary) {
      setStatus("stale", "No data available — run scripts/export_web_data.py + scripts/embed_web_data.py.");
      return;
    }
    var nP = Object.values(summary.models)[0] ? Object.values(summary.models)[0].n_prompts : 0;
    var ts = summary.generated_at ? new Date(summary.generated_at).toLocaleString() : "unknown";
    setStatus("ready", Object.keys(summary.models).length + " models · " + nP + " prompts · generated " + ts + (source === "inline" ? " · inline" : ""));
    var ds = document.getElementById("data-source");
    if (ds) ds.textContent = source === "inline" ? "inline data island" : DATA_URL;

    safe(function () { renderHeadlineStats(document.getElementById("headline-stats"), summary); }, "headline");
    safe(function () { renderCharacterInversion(document.getElementById("character-inversion"), summary); }, "character");
    safe(function () { renderCrossModelInversion(document.getElementById("cross-model-inversion"), summary); }, "crossmodel");
    safe(function () { renderCategoryEnrichment(document.getElementById("category-enrichment"), summary); }, "enrichment");
    safe(function () { renderAblationBars(document.getElementById("ablation-bars"), summary); }, "ablation");
    safe(function () { renderBackbone(document.getElementById("backbone-graph"), summary); }, "backbone");
  }
  function safe(fn, name) {
    try { fn(); } catch (e) { console.error(name + " failed:", e); }
  }

  // Try fetch first; fall back to inline.
  var inline = loadInline();
  tryFetch(function (data, source) {
    boot(data || inline, data ? "fetch" : "inline");
  });
})();
