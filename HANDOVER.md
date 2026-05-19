# Neograph — handover for the next agent

*Last updated 2026-05-12 evening, by the agent who built the substrate and the
12-circuit cross-model finding.*

> **You're inheriting a finding worth fighting for and a clear plan to make it
> indelible.** The current bilateral observation ("Gemma routes through a grammar
> layer, GPT-2 doesn't") is publishable as a LessWrong post but vulnerable on
> three axes: sample size on prompts, sample size on models, and causation vs
> correlation. Theo (the project owner) has laid out the strengthening plan. Your
> job is to execute it in order, lock down causation first on one model, then
> scale. Do not start from scratch — read this whole file, then `reports/findings.md`
> + `reports/writeup_v2_draft.md`, then begin.

---

## 1. What the finding currently says

> Across 12 next-token-prediction prompts spanning capitals, weekdays, arithmetic,
> named-entity completion, and syntactic continuation, Gemma 2 2B recruits a small
> set of **structural** SAE features ("beginning of text", "control and authority",
> "forms of to-be", "statements of existence", "legal/technical terminology") in
> 5–12 of 12 circuits each. **Feat 6631 fires in 12/12.** GPT-2 small on the same
> 12 prompts recruits **content-thematic** features instead (politics, statistics,
> capital cities, URLs, preferences). The GPT-2 SAE *has* features for Gemma's
> predicates at label cosine 0.87–0.93 — but they participate in 0/12 GPT-2
> circuits. **Vocabulary universality without circuit universality.**

This holds against the layer-asymmetry objection: GPT-2 L4 / L8 / L10 / L11 all show
the same content-thematic pattern, no predicate backbone emerges at any sampled depth.

The brain-scan analogy is real and useful: Gemma resembles the **multi-demand
network** (frontoparietal regions firing across diverse cognitive tasks; Duncan
2010+); GPT-2 resembles **task-positive networks** (regions activating by content
type). It's a productive metaphor for non-technical readers and accurate at the
structural level. See writeup §"A brain-scan analogy that holds up".

The name in this project: **the grammar layer**. Older artifacts may still say
"predicate backbone" — they're the same thing.

---

## 2. The three attacks any reviewer will mount

| Attack | Current state | Fix |
|---|---|---|
| "12 prompts is anecdote, not result" | 12 prompts, 4 categories | Scale to ≥50 prompts across ≥8 categories. Priority #2 below. |
| "2 models is a coincidence, not a pattern" | Gemma 2 2B + GPT-2 small | Pythia scale curve (70M → 12B). Priority #3. |
| "Attribution = correlation, not causation" | Zero-ablation patching gives per-feature attribution but doesn't show backbone features are *load-bearing* | Ablate top-3 backbone features jointly during generation, measure hit-rate drop. Priority #1. |

These three together are the bundle. Theo's words: "If those three land cleanly, you
have an arXiv preprint, not a LessWrong post."

---

## 3. Priority queue (execute in this exact order)

### P1 — Ablation experiment (1 day) — DO THIS FIRST

The single highest-leverage experiment available. Upgrades the headline from
correlational ("feat 6631 participates in 12/12 circuits") to causal ("feat 6631 is
load-bearing across 12/12 circuits — ablate it, completion breaks uniformly").

**Why this is first**: every scaling experiment downstream (50 prompts, Pythia
curve) is expensive if the underlying claim is still correlational. Pin down
causation on one model first.

**Design**:
- Pick the top 5 backbone features from `reports/predicate_alignment.json`:
  - feat 6631 ("beginning of text", in 12/12 circuits)
  - feat 9768 ("control and authority", 11/12)
  - feat 15596 ("forms of to-be", 7/12, *opposing*)
  - feat 13414 ("statements of existence", 7/12)
  - feat 12927 ("statements of existence", 7/12)
- Three conditions per prompt:
  1. **Baseline**: no ablation, measure target hit rate + log P(target).
  2. **Single ablate**: ablate one backbone feature at a time, 5 features × 12
     prompts = 60 runs.
  3. **Joint ablate**: ablate top-3 jointly (6631 + 9768 + 13414).
- Metric: target hit rate (does next token equal expected target), log P(target),
  log P(any reasonable completion) — the last one detects "we broke completion
  generally" vs "we broke the *correct* completion".
- **Critical**: the ablation must happen at the SAE-feature post-encode hook, at
  the *last position only* (the steering position). Ablating earlier positions
  measures a different thing.
- Per-category breakdown: capitals / weekdays / math / named-entities / syntactic.
  If hit-rate drops uniformly across categories under joint ablation → load-bearing.
  If hit-rate drops only for some category → backbone is more limited than claimed.

**Files to create**:
- `scripts/causal_ablation.py` — new script. Adapt the hook pattern from
  `scripts/causal_attribution_v2.py` lines 73–95 (zero-ablation hook). The trick is
  passing a *list* of feature indices to ablate jointly, not just one.
- `reports/ablation_causality.json` — per (condition, prompt) results.
- `reports/ablation_causality.md` — readable summary table.
- `reports/viz_ablation.png` — bar chart, 3 conditions × 5 categories.

**Exit criterion**: Joint-ablation of top-3 backbone features drops target hit rate
by ≥40% across all 5 categories, with the drop visible in log P(target) of ≥1.0 nat
on average. If yes → the headline is causal. If no → re-examine which features
constitute "the" backbone.

**Reuse**: the model+SAE loading and the zero-ablation hook signature from
`scripts/causal_attribution_v2.py`. Don't rewrite those.

**Hook reminder** (sae-lens 4.x quirk; lost an hour to this earlier):
TransformerLens passes `hook` as a kwarg, so use `def ablate(act, **kwargs)`, not
`def ablate(act, hook)`. See `scripts/causal_attribution_v2.py:80`.

---

### P2 — Scale prompts from 12 to 50 (2 days)

Removes the "12 cherry-picked prompts" objection. Pure elbow grease; no methodology
novelty. The 50-prompt set lives at `data/prompts_50.json` (already stubbed by the
previous agent — see below).

**Domain coverage** (Theo's note):
- ≥4 each of: capitals, weekdays (with deeper temporal arithmetic), basic math,
  named-entity completion, syntactic continuation.
- Add ≥4 each of: reasoning chains ("If all A are B…"), instruction-following
  ("Please respond with 'yes' or 'no':…"), factual recall, simple code completion,
  multi-step arithmetic, pronoun resolution, summarization openings.

**The dataset is already drafted** at `data/prompts_50.json` (50 prompts, 11 domains).
Read it; trim or expand domain breadth per your judgement; add or remove items so
that each domain has ≥4 prompts and each prompt has a single-token expected target.

**Files to update**:
- `scripts/causal_attribution_v2.py` already reads from `data/causal_prompts.json`.
  Add a `--prompts-file` flag or just point it at `data/prompts_50.json`.
- Re-run for both Gemma and GPT-2: ~10 min Gemma, ~5 min GPT-2 on M5 Max MPS.
- Re-run `scripts/predicate_alignment.py` afterwards — the predicate backbone
  should *grow* with more data (more circuits → more recruitment evidence).
- Append the new results to `reports/findings.md` and the writeup.

**Exit criterion**: with N=50, feat 6631 should still be in ≥45/50 circuits. If it
drops below 40/50 the original 12 was anomalously concentrated and the finding
weakens; investigate which domains it doesn't fire on.

---

### P3 — Pythia scale curve (3–5 days)

The single experiment that promotes this from bilateral to scale-dependent. EleutherAI
ships SAEs across the Pythia family at multiple scales. Run the predicate-backbone
test at every available size.

**Two outcomes, both publishable**:
1. **Backbone emerges at some scale.** "Grammar-layer routing emerges at parameter
   count X in the Pythia family." That's an *emergence* claim — top-of-arXiv.
2. **Backbone never emerges in Pythia.** Predicate routing isn't a function of scale
   but of model family or training distribution. Pythia and GPT-2 share lineage;
   Gemma 2 doesn't. That tracks Google's training recipe specifically. Still real.

**Pythia + SAE releases to test** (verify availability in SAELens before promising
any specific scale):
- Pythia 70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B — `EleutherAI/sae-pythia-{size}-{config}`
- Layer pick: roughly middle-late layer of each model. For Pythia-410M (24 layers)
  → L16. For Pythia-1.4B (24 layers) → L16. For Pythia-2.8B (32 layers) → L20.
- Use the same 50-prompt set as P2. Reuse `causal_attribution_v2.py` with a new
  `ModelSpec` per Pythia size.

**Engineering notes**:
- Pythia is ungated; no HF auth required.
- Pythia 6.9B and 12B may not fit in 128GB with the SAE loaded. Confirm before
  starting; fall back to bf16 or skip the largest sizes.
- The disk footprint of 8 model weights is ~80GB. Verify `df -h` before kickoff.
- Use sae-lens's `get_pretrained_saes_directory()` from `sae_lens.loading.pretrained_saes_directory`
  to enumerate available Pythia SAEs (the snippet is in `scripts/cross_model_gpt2.py`
  comments).

**Files to create**:
- `scripts/pythia_scale_sweep.py` — loops over Pythia sizes, runs 50-prompt
  causal_attribution, writes per-model multi-circuit features + labels.
- `reports/pythia_scale_curve.json` — per-size: top backbone features + label-cosine
  similarity to Gemma's grammar layer.
- `reports/viz_pythia_emergence.png` — x-axis: Pythia parameter count (log scale);
  y-axis: max-circuit-count for any feature labelled as predicate-style. If there's a
  knee, that's the emergence point.

**Exit criterion**: clear, monotonic answer either way. Either backbone emerges and
you can name the size, or it never emerges across 8 scales spanning 100× parameter
count.

---

### P4 — Width-65k Gemma Scope (1 day)

Tests whether the grammar layer is robust to SAE width. PRD §11 anticipated this.

- Replace `gemma-scope-2b-pt-res-canonical` width-16k with the width-65k canonical
  release at the same layer (L20).
- Re-run the 12-prompt (or 50-prompt) causal_attribution.
- Compare: does feat-equivalent-of-6631 exist in width-65k? Does it still fire in
  ~12/12 circuits? Or does the predicate concept shatter into multiple finer features?

If backbone persists at width-65k → the SAE-width objection dies, and the finding is
width-stable. If it disperses → arguably more interesting; backbone is an
SAE-width-dependent emergent property (Bhalla et al.'s "Do SAEs Capture Concept
Manifolds" would predict this).

---

### P5 — Attribution-method replication (2 days)

Zero-ablation is one attribution method. Replicate the top-5 backbone features under
two others on a subset of prompts (e.g. capitals + weekdays = 7 prompts):
- **Mean-ablation**: replace the feature's activation with its corpus mean instead
  of zero.
- **Attribution patching** (Stolfo et al. 2023) or **EAP-IG**: integrated-gradient
  attribution along the activation axis. Cheaper than full ablation.

If backbone appears under all three methods → method-robust. If only zero-ablation
→ you've identified a method artifact, which is also worth knowing.

---

### P6 — Behavioral signature (2 days)

Connect the internal finding to external behavior. Generate 500-token continuations
from both models on matched prompts; measure:
- Modal-verb density ("would", "could", "should", "must", "may")
- Hedge-word frequency ("perhaps", "possibly", "generally", "typically")
- Declarative-vs-descriptive ratios (count of "X is Y" patterns vs adjective-heavy
  description)
- Sentence-opener distribution (does Gemma start sentences with "This", "There", "It"
  more than GPT-2?)

If Gemma's generations are quantifiably more grammatical / declarative / hedge-laden,
the internal grammar-layer claim has external behavioral support. *This* is the
experiment that lets the finding cross over to a broader audience.

---

## 4. What NOT to do

- **Don't skip the ablation.** Theo's note is explicit: "Lock down causation on one
  model first." Every downstream experiment is expensive if the underlying claim is
  correlational.
- **Don't add a third individual model (Mistral, Qwen, Llama 3).** The Pythia scale
  curve replaces this — eight scales of one model family is a better experiment than
  three random models.
- **Don't add interactive UI features to the Three.js viewer.** It works.
- **Don't refit manifolds.** The manifold story is honestly downgraded to
  "cluster + waypoint scaffolding" in the writeup; don't re-claim differential-geometry
  manifold reconstruction. The follow-up loop-closer (fit a principal curve through
  the *grammar-layer activations* across 12 prompts and test trajectory steering for
  *style*) is the right manifold experiment — but it's still queued behind P1–P6.
- **Don't move the Cypher Q-CAUSE-5 finding to the headline.** It's the substrate
  argument, not the science argument. The science is the grammar layer + cross-model
  routing distinction.
- **Don't introduce new model architectures** (Mamba, RWKV, state-space) until P1–P3
  land. They're paper-grade follow-ups, not strengthening.

---

## 5. State snapshot — what exists, where

### Code

```
neograph/
  HANDOVER.md                          # this file
  README.md                            # project overview + build status
  pyproject.toml                       # uv-managed deps
  .env                                 # NEO4J creds + HF_TOKEN (gitignored)
  .neograph-db/                        # project-local Neo4j 2026.03.1 + GDS 2026.04 + APOC
  apps/grammar_layer/                  # Three.js interactive viewer (P1–P5 done)
    index.html
    data.json                          # 2.5 MB UMAP coords + circuit data
    README.md
  bloom/neograph-perspective.json
  cypher/{00_constraints,01_indexes}.cypher
  data/
    causal_prompts.json                # the original 12 prompts
    prompts_50.json                    # NEW: 50-prompt expansion for P2 (drafted, needs review)
    synthetic/{rhymes,weekdays}.json
    staging/                           # parquet caches (10k+ prompts, 16k×topk, etc.)
  reports/
    findings.md                        # the canonical accumulated-findings doc
    writeup_v2_draft.md                # LessWrong-ready draft, "The Grammar Layer" headline
    viz_grammar_layer.{png,gif}        # hero viz + Twitter GIF
    viz_grammar_layer_{gemma,gpt2}.png # single-model panels
    viz_vocab_links.png                # cross-model label-cosine arcs
    causal_circuits_{gemma,gpt2}.json  # 12 circuits per model, INCLUDES edges populated in DB
    predicate_alignment.json           # top-30 multi-circuit features per model + cross-model NN
    gpt2_layer_sweep.{json,summary.md} # L4/L8/L10/L11 layer-asymmetry robustness check
    cross_model_universality.json      # earlier Hungarian alignment (retracted in writeup)
    label_clustering.json              # weekday/money/code/prefix concentration per Leiden community
    leiden_gamma_sweep.json            # γ sweep showing resolution doesn't rescue Goodfire indices
    umap_{gemma,gpt2}_3d.npy           # cached 3D projections (use these, don't recompute)
    circuits_per_feature_{gemma,gpt2}.npy
  scripts/
    00_bootstrap_neo4j.sh              # idempotent Neo4j setup
    migrate.py                         # apply schema (constraints + 4 vector indexes)
    01_load_model_and_sae.py           # P1 smoke
    02_seed_corpus.py                  # P2 prompts.parquet (Pile + synth)
    03_capture_activations.py          # P2 model+SAE activations
    04_ingest_features.py              # P2 features + labels (uses sae-lens 4.x API)
    05_build_relations.py              # P3 PMI/decoder/label cosine
    06_communities_and_manifolds.py    # P3 Leiden + P4 manifold fits
    07_eval_steering.py                # P6 trajectory vs linear (single-step, both 100% hit)
    causal_attribution.py              # N=2 version (predecessor of v2; can delete)
    causal_attribution_v2.py           # ★ KEY ★ adapt this for P1 ablation experiment
    cross_model_gpt2.py                # GPT-2 ingest pipeline
    cross_model_finish.py              # Leiden+matching tail after relations are written
    cross_model_universality.py        # Hungarian assignment (the retracted approach)
    gpt2_layer_sweep.py                # ★ KEY ★ adapt this for Pythia scale curve in P3
    predicate_alignment.py             # ★ KEY ★ vocabulary-vs-circuit cross-model test
    eval_label_clustering.py           # per-concept Leiden concentration
    sweep_leiden.py                    # γ sweep
    viz_grammar_layer.py               # the hero image renderer
    viz_cross_model.py                 # cross-model concept bar chart
    viz_for_theo.py                    # community sizes + concept concentration
    prefetch_labels.py                 # idempotent Neuronpedia cache warmer
    verify.py                          # end-to-end exit-criterion verifier
    export_viz_data.py                 # JSON blob for Three.js
    run_all.sh                         # full P1→P6 orchestrator
  src/neograph/
    config.py                          # SAE/MODEL/PATHS constants
    cypher.py                          # NeographClient + Q1-Q6 helpers
    labels.py                          # Neuronpedia + Anthropic fallback + MiniLM
    relations.py                       # PMI/Jaccard/cosine math
    activations.py                     # capture pipeline
    steering.py                        # linear + manifold hooks
    evals.py                           # NMI vs Goodfire + steering summary
    util.py                            # logging, chunking, hashing
    manifold/{fit,write,concept,viz}.py # PCA + Kégl polyline + cyclic spline
  tests/
    test_manifold_fit.py               # 4 tests, all pass
    test_relations.py                  # 2 tests, all pass
    test_neo4j_smoke.py                # 1 end-to-end ingest smoke test, passes
```

### Neo4j

- `bolt://localhost:7693` in `.neograph-db/` (NOT the orbweaver instance on 7687)
- User `neo4j`, password `neograph_local_dev`
- 16,384 Gemma features + 24,576 GPT-2 features in same store, partitioned by `sae_id`
- 24 `Circuit` nodes (12 Gemma + 12 GPT-2), each with 50 `:INCLUDES` edges
- 18 Gemma Leiden communities, 14 GPT-2 Leiden communities
- 18 `Manifold` nodes with `:HAS_WAYPOINT` chains (community-cluster scaffolding,
  not true manifold reconstruction)
- 4 vector indexes online: `feat_decoder`, `feat_encoder`, `label_emb`,
  `waypoint_centroid`
- GDS 2026.04.0, APOC 2026.03.1

---

## 6. Running commands

### One-time
```bash
# Restart Neo4j if it died
nohup ./.neograph-db/bin/neo4j console > ./.neograph-db/logs/console.log 2>&1 &

# Verify schema + plugins
cypher-shell -a bolt://localhost:7693 -u neo4j -p neograph_local_dev \
  "RETURN gds.version() AS gds, apoc.version() AS apoc"
```

### Daily
```bash
# Run any script
uv run python scripts/<name>.py

# Verify end-to-end state
uv run python scripts/verify.py

# Run tests
.venv/bin/python -m pytest -W ignore tests/

# Open the interactive viewer
cd apps/grammar_layer && python3 -m http.server 8765
# → open http://localhost:8765
```

### Reproducing earlier results
```bash
uv run python scripts/causal_attribution_v2.py --model gemma  # 12 prompts, ~2 min
uv run python scripts/causal_attribution_v2.py --model gpt2   # ~1 min
uv run python scripts/predicate_alignment.py                  # ~10 sec
uv run python scripts/gpt2_layer_sweep.py --layers 4 8 10 11  # ~5 min
uv run python scripts/viz_grammar_layer.py                    # ~20 sec
uv run python scripts/export_viz_data.py                      # ~10 sec
```

---

## 7. Known landmines (learn from earlier mistakes)

1. **HF auth for Gemma**. Gemma 2 2B is gated. `HF_TOKEN` must be set in `.env`.
   The current token belongs to user `hopski`. Don't touch it.

2. **uv editable install + `.pth` quirk**. `uv pip install -e .` writes a `.pth`
   file that *site.py doesn't process* on this machine. Workaround:
   `.venv/lib/python3.12/site-packages/sitecustomize.py` adds `src/` to `sys.path`
   explicitly. If you blow away the venv, re-create this file or imports break.

3. **sae-lens 4.x API**. `SAE.from_pretrained` returns *just the SAE*, not a tuple.
   `HookedSAETransformer` (not vanilla `HookedTransformer`) has `run_with_cache_with_saes`
   and `run_with_hooks_with_saes`. The hook signature is `def hook(act, **kwargs)`
   — TransformerLens passes `hook` as a kwarg.

4. **GDS Cypher syntax**. `gds.graph.project.cypher` is deprecated in GDS 2026.04
   and rejects `undirectedRelationshipTypes`. Use the aggregating `gds.graph.project`
   function form (see `scripts/sweep_leiden.py` for the working pattern).

5. **Neo4j `CALL` blocks without scope clause** generate deprecation warnings in
   2026.03. Use `CALL () { … }` form. Old `CALL { … }` still works but warns.

6. **Three Neo4j instances on this box**. Don't confuse them:
   - `7687` = `neo4j-local` aka orbweaver (USER DATA, DO NOT TOUCH, no GDS)
   - `7688` = Homebrew Neo4j (idle, Community, no GDS, no special data)
   - `7689` = Neo4j Desktop 2 DBMS (Enterprise, has APOC but no GDS)
   - `7693` = **Neograph** — yours.

7. **MPS forward-pass parity**. Verified at P1 smoke (max |Δ|=9.77e-04 vs CPU);
   no need to re-verify unless Gemma 2 hybrid attention changes.

8. **Disk**. M5 Max has ~85 GiB free as of 2026-05-12. Pythia 7B + SAE = ~14GB per
   model. Run `df -h` before kicking off P3.

---

## 8. The writeup state

- `reports/writeup_v2_draft.md` — current LessWrong-ready draft. Opens with cognitive-
  style hook and brain-scan analogy, hero image as second element, predicate-backbone
  table in §1, structural cross-model test in §2, layer sweep in §3a, honest manifold
  framing in §4, "so what" coda in §8.

- After P1 (ablation), add §1.5 "Load-bearing": ablation table showing hit-rate drop
  per category. Replace correlational sentences in §1 with causal ones.

- After P2 (50 prompts), revise N=12 → N=50 throughout. Confirm feat 6631 ≥45/50.

- After P3 (Pythia curve), the writeup pivots from "bilateral cross-model" to
  "scale-curve emergence" (if positive) or "family-specific" (if Pythia stays
  GPT-2-like). Either reframe is bigger than the current bilateral framing.

- The Three.js viewer (`apps/grammar_layer/`) is ready to embed. Iframe target:
  whatever static host Theo prefers (Cloudflare Pages, Vercel, GitHub Pages).

---

## 9. Naming/framing checklist before shipping

- "the grammar layer" everywhere, with one parenthetical mention of the technical
  synonym "predicate backbone" the first time. ✅ already done in writeup.
- The negative-space sentence: "GPT-2 has the same words in its dictionary. It just
  doesn't use those words to answer questions." ✅ in writeup §1 and §8.
- "We don't read this as GPT-2 failing to use predicate features. Both models complete
  the prompts successfully." → routing strategy, not capability claim. ✅ in §1.
- The brain-scan analogy with multi-demand network. ✅ added 2026-05-12 evening.
- The Cypher Q-CAUSE-5 query is in §3 not the headline. ✅ correct ordering.
- The §"what this writeup is not" section. ✅ §6.

---

## 10. Goodfire / community outreach (post-ship)

The original framing came from Goodfire's Geiger/Lubana/Fel/McGrath/Lewis/Merullo/Byun.
After the science lands, the right move is a "here's where your framing took us" note
to them with the writeup attached. Don't cold-pitch; the work itself is the overture.

The Neo4j-substrate-advocacy post is *separate*. Write it after the science post lands
and is well-received. Different audiences, no cross-contamination.

---

## 11. The follow-up that closes the loop

Theo's note (2026-05-12 evening) suggested this for after the current strengthening
bundle: fit a principal curve through the *grammar-layer features' activations across
the 12 (or 50) prompts*, treating them as a low-dimensional cloud. That cloud is the
manifold of "the structural skeleton of completion." Then test trajectory steering
along it: does moving along the curve modulate *style* (assertive ↔ hedged,
declarative ↔ descriptive) rather than *content*?

If yes, the manifold framing and the routing framing unify into one story: "the
geometric manifold the model traverses to compose completions is the grammar layer,
and we can steer style by moving along it." That's the experiment that turns this
from a routing finding into a Goodfire-grade unified result.

Don't do it before P1-P3 land. Put it at the top of the follow-up list.

---

## 12. If you only do one thing

**Run the ablation experiment (P1) and report whether joint-ablation of top-3
backbone features drops target hit rate uniformly across all 5 categories.** That
single number — call it the "load-bearing index" — is what turns the current finding
from "correlational pattern" to "causal claim that survives a reviewer".

Everything else can wait.

Good luck.
