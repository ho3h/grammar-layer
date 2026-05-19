// Three.js backbone graph — prompts and features as nodes, edges = "feature is
// in this prompt's top-10 supporting set". Force-directed layout in 3D, hover
// to inspect.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const CATEGORY_COLORS = {
  capital: "#d35400",
  weekday: "#2980b9",
  math: "#27ae60",
  "named-entity": "#8e44ad",
  syntactic: "#16a085",
  reasoning: "#c0392b",
  instruction: "#7f8c8d",
  "factual-recall": "#e67e22",
  code: "#34495e",
  "multi-step-arithmetic": "#1abc9c",
  pronoun: "#9b59b6",
  "summarization-opener": "#f39c12",
};
const DEFAULT_CATEGORY_COLOR = "#95a5a6";

const FEATURE_GRAMMAR_COLOR = "#5b9c3b";
const FEATURE_CONTENT_COLOR = "#8a7752";

const PROMPT_RADIUS = 1.6;
const FEATURE_RADIUS = 0.7;
const FEATURE_GRAMMAR_RADIUS = 1.0;

export function renderBackboneGraph(root, summary) {
  root.innerHTML = "";

  const modelEntries = Object.entries(summary.models);
  if (modelEntries.length === 0) return;

  // Selector toolbar
  const toolbar = document.createElement("div");
  toolbar.style.position = "absolute";
  toolbar.style.top = "12px";
  toolbar.style.left = "12px";
  toolbar.style.zIndex = "5";
  toolbar.style.display = "flex";
  toolbar.style.gap = "6px";
  toolbar.style.fontFamily = "var(--sans)";
  toolbar.style.fontSize = "12px";

  const select = document.createElement("select");
  select.style.padding = "5px 10px";
  select.style.border = "1px solid var(--hairline)";
  select.style.borderRadius = "4px";
  select.style.background = "white";
  select.style.font = "inherit";
  for (const [name, m] of modelEntries) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = m.display_name;
    select.appendChild(opt);
  }
  toolbar.appendChild(select);

  const sideToggle = document.createElement("select");
  sideToggle.style.padding = "5px 10px";
  sideToggle.style.border = "1px solid var(--hairline)";
  sideToggle.style.borderRadius = "4px";
  sideToggle.style.background = "white";
  sideToggle.style.font = "inherit";
  for (const [val, txt] of [["supporting", "supporting features"], ["opposing", "opposing features"]]) {
    const opt = document.createElement("option");
    opt.value = val;
    opt.textContent = txt;
    sideToggle.appendChild(opt);
  }
  toolbar.appendChild(sideToggle);

  root.appendChild(toolbar);

  const canvasHost = document.createElement("div");
  canvasHost.style.position = "absolute";
  canvasHost.style.inset = "0";
  root.appendChild(canvasHost);

  let cleanup = null;
  const render = () => {
    if (cleanup) cleanup();
    const modelName = select.value;
    const side = sideToggle.value;
    cleanup = renderOneGraph(canvasHost, summary.models[modelName], modelName, side);
  };
  select.addEventListener("change", render);
  sideToggle.addEventListener("change", render);
  render();
}

function renderOneGraph(root, modelData, modelName, side) {
  while (root.firstChild) root.removeChild(root.firstChild);

  const width = root.clientWidth || root.parentElement.clientWidth;
  const height = root.clientHeight || root.parentElement.clientHeight;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xfafaf7);

  const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 2000);
  camera.position.set(0, 0, 90);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
  renderer.setSize(width, height);
  root.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  scene.add(new THREE.AmbientLight(0xffffff, 0.85));
  const dir = new THREE.DirectionalLight(0xffffff, 0.45);
  dir.position.set(50, 80, 100);
  scene.add(dir);

  const edgeRows = side === "opposing"
    ? (modelData.opposing_edges || [])
    : (modelData.backbone_edges || []);
  const { nodes, edges } = buildGraph({ ...modelData, backbone_edges: edgeRows });
  if (nodes.length === 0) {
    const msg = document.createElement("div");
    msg.style.padding = "24px";
    msg.style.color = "#888";
    msg.style.fontFamily = "var(--sans)";
    msg.textContent = `No ${side} edges yet for ${modelData.display_name}. Re-run the analysis with the updated script (saves both signs in one pass) and re-export.`;
    root.appendChild(msg);
    return () => {};
  }

  layoutForceDirected(nodes, edges, { iterations: 220, repulsion: 32, edgeRest: 6, gravity: 0.012 });

  const nodeMeshes = [];
  const sphereLow = new THREE.SphereGeometry(1, 14, 10);
  for (const n of nodes) {
    const isPrompt = n.kind === "prompt";
    const color = isPrompt
      ? CATEGORY_COLORS[n.category] || DEFAULT_CATEGORY_COLOR
      : (n.isGrammar ? FEATURE_GRAMMAR_COLOR : FEATURE_CONTENT_COLOR);
    const r = isPrompt ? PROMPT_RADIUS : (n.isGrammar ? FEATURE_GRAMMAR_RADIUS : FEATURE_RADIUS);
    const mat = new THREE.MeshLambertMaterial({ color });
    const mesh = new THREE.Mesh(sphereLow, mat);
    mesh.scale.set(r, r, r);
    mesh.position.set(n.x, n.y, n.z);
    mesh.userData.node = n;
    scene.add(mesh);
    nodeMeshes.push(mesh);
    n.mesh = mesh;
  }

  // Edges as a single line segments primitive
  const positions = new Float32Array(edges.length * 6);
  const colors = new Float32Array(edges.length * 6);
  for (let i = 0; i < edges.length; i++) {
    const a = nodes[edges[i].a];
    const b = nodes[edges[i].b];
    const off = i * 6;
    positions[off + 0] = a.x; positions[off + 1] = a.y; positions[off + 2] = a.z;
    positions[off + 3] = b.x; positions[off + 4] = b.y; positions[off + 5] = b.z;
    const eColor = new THREE.Color((b.kind === "feature" && b.isGrammar) || (a.kind === "feature" && a.isGrammar)
      ? FEATURE_GRAMMAR_COLOR
      : FEATURE_CONTENT_COLOR);
    for (let j = 0; j < 2; j++) {
      colors[off + j * 3 + 0] = eColor.r;
      colors[off + j * 3 + 1] = eColor.g;
      colors[off + j * 3 + 2] = eColor.b;
    }
  }
  const lineGeom = new THREE.BufferGeometry();
  lineGeom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  lineGeom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  const lineMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.28 });
  const lines = new THREE.LineSegments(lineGeom, lineMat);
  scene.add(lines);

  // Hover tooltip
  let tip = document.getElementById("hover-tip");
  if (!tip) {
    tip = document.createElement("div");
    tip.id = "hover-tip";
    document.body.appendChild(tip);
  }
  const raycaster = new THREE.Raycaster();
  raycaster.params.Mesh.threshold = 0.01;
  const mouse = new THREE.Vector2();
  let hovered = null;

  renderer.domElement.addEventListener("mousemove", (ev) => {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObjects(nodeMeshes);
    if (hits.length) {
      const h = hits[0].object.userData.node;
      if (h !== hovered) hovered = h;
      tip.style.display = "block";
      tip.style.left = `${ev.clientX + 14}px`;
      tip.style.top = `${ev.clientY + 14}px`;
      if (h.kind === "prompt") {
        tip.innerHTML = `
          <div class="tip-title">${escapeHtml(h.prompt)}</div>
          <div class="tip-meta">${h.category} · target: ${escapeHtml(h.target)} · ${h.degree} load-bearing features</div>
        `;
      } else {
        const dropTxt = h.totalDrop ? ` · Σ Δlog P = +${h.totalDrop.toFixed(2)}` : "";
        tip.innerHTML = `
          <div class="tip-title">feat ${h.feature_index} — ${escapeHtml(h.label || "(no label)")}</div>
          <div class="tip-meta">${h.isGrammar ? "grammar-flavored" : "content-thematic"} · in top-10 of ${h.degree} prompts${dropTxt}</div>
        `;
      }
    } else {
      hovered = null;
      tip.style.display = "none";
    }
  });

  renderer.domElement.addEventListener("mouseleave", () => {
    hovered = null;
    tip.style.display = "none";
  });

  // Resize
  const onResize = () => {
    const w = root.clientWidth;
    const h = root.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  };
  window.addEventListener("resize", onResize);

  // Render loop with auto-rotate when idle
  let lastInteraction = performance.now();
  let stopped = false;
  controls.addEventListener("start", () => { lastInteraction = performance.now(); });
  const tick = () => {
    if (stopped) return;
    requestAnimationFrame(tick);
    if (!hovered && performance.now() - lastInteraction > 2500) {
      scene.rotation.y += 0.0017;
    }
    controls.update();
    renderer.render(scene, camera);
  };
  tick();

  // Legend
  const legendEl = document.getElementById("backbone-legend");
  if (legendEl) {
    const cats = Object.keys(CATEGORY_COLORS).filter((c) =>
      nodes.some((n) => n.kind === "prompt" && n.category === c)
    );
    const swatches = cats.map((c) =>
      `<span><span class="swatch" style="background:${CATEGORY_COLORS[c]}"></span>${c}</span>`
    ).join("");
    legendEl.innerHTML = `
      ${swatches}
      <span style="margin-left:18px"><span class="swatch" style="background:${FEATURE_GRAMMAR_COLOR}"></span>feature (grammar)</span>
      <span><span class="swatch" style="background:${FEATURE_CONTENT_COLOR}"></span>feature (content)</span>
      <span style="color:#888;font-style:italic">Drag to rotate · hover for label</span>
    `;
  }

  // Cleanup function — called when caller wants to dispose this graph (e.g. on
  // model/side switch). Stops the render loop, releases GPU resources.
  return () => {
    stopped = true;
    window.removeEventListener("resize", onResize);
    renderer.dispose();
    if (renderer.domElement.parentElement) {
      renderer.domElement.parentElement.removeChild(renderer.domElement);
    }
  };
}

function buildGraph(modelData) {
  const edges = modelData.backbone_edges || [];
  const promptIndex = new Map();
  const featureIndex = new Map();
  const nodes = [];

  const ensurePrompt = (e) => {
    if (!promptIndex.has(e.prompt_id)) {
      promptIndex.set(e.prompt_id, nodes.length);
      nodes.push({
        kind: "prompt",
        id: e.prompt_id,
        prompt: e.prompt || e.prompt_id,
        category: e.prompt_category,
        target: e.target || "",
        x: (Math.random() - 0.5) * 60,
        y: (Math.random() - 0.5) * 60,
        z: (Math.random() - 0.5) * 60,
        vx: 0, vy: 0, vz: 0,
        degree: 0,
      });
    }
    return promptIndex.get(e.prompt_id);
  };
  const ensureFeature = (e) => {
    if (!featureIndex.has(e.feature_index)) {
      featureIndex.set(e.feature_index, nodes.length);
      nodes.push({
        kind: "feature",
        feature_index: e.feature_index,
        label: e.feature_label || "",
        isGrammar: !!e.feature_is_grammar,
        x: (Math.random() - 0.5) * 60,
        y: (Math.random() - 0.5) * 60,
        z: (Math.random() - 0.5) * 60,
        vx: 0, vy: 0, vz: 0,
        degree: 0,
        totalDrop: 0,
      });
    }
    return featureIndex.get(e.feature_index);
  };

  const e_out = [];
  for (const e of edges) {
    const a = ensurePrompt(e);
    const b = ensureFeature(e);
    nodes[a].degree++;
    nodes[b].degree++;
    nodes[b].totalDrop += Math.max(0, e.single_log_p_drop || 0);
    e_out.push({ a, b });
  }
  return { nodes, edges: e_out };
}

function layoutForceDirected(nodes, edges, opts) {
  const { iterations = 150, repulsion = 30, edgeRest = 5, gravity = 0.01, damping = 0.86 } = opts || {};
  for (let it = 0; it < iterations; it++) {
    // Pairwise repulsion (O(N²); fine for N≈200)
    for (let i = 0; i < nodes.length; i++) {
      const a = nodes[i];
      for (let j = i + 1; j < nodes.length; j++) {
        const b = nodes[j];
        const dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
        const d2 = dx * dx + dy * dy + dz * dz + 0.01;
        const f = repulsion / d2;
        const d = Math.sqrt(d2);
        const fx = (dx / d) * f, fy = (dy / d) * f, fz = (dz / d) * f;
        a.vx += fx; a.vy += fy; a.vz += fz;
        b.vx -= fx; b.vy -= fy; b.vz -= fz;
      }
    }
    // Edge springs
    for (const e of edges) {
      const a = nodes[e.a], b = nodes[e.b];
      const dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz) + 0.01;
      const f = (d - edgeRest) * 0.06;
      const fx = (dx / d) * f, fy = (dy / d) * f, fz = (dz / d) * f;
      a.vx += fx; a.vy += fy; a.vz += fz;
      b.vx -= fx; b.vy -= fy; b.vz -= fz;
    }
    // Gravity toward origin
    for (const n of nodes) {
      n.vx -= n.x * gravity;
      n.vy -= n.y * gravity;
      n.vz -= n.z * gravity;
      n.vx *= damping; n.vy *= damping; n.vz *= damping;
      n.x += n.vx; n.y += n.vy; n.z += n.vz;
    }
  }
}

function escapeHtml(s) {
  return String(s || "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}
