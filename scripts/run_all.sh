#!/usr/bin/env bash
# Run the full Neograph pipeline P1→P6 in order. Stops on first failure.
# Each step prints a [exit-criterion: PASS|FAIL] marker.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

UV="${UV:-uv}"

echo "=== P1: bootstrap + schema + smoke ==="
bash scripts/00_bootstrap_neo4j.sh
"$UV" run python scripts/migrate.py
"$UV" run python scripts/01_load_model_and_sae.py

echo "=== P2: corpus + activations + features ==="
"$UV" run python scripts/02_seed_corpus.py
"$UV" run python scripts/03_capture_activations.py
"$UV" run python scripts/04_ingest_features.py

echo "=== P3: relations + Leiden ==="
"$UV" run python scripts/05_build_relations.py

echo "=== P3+P4: communities + manifolds ==="
"$UV" run python scripts/06_communities_and_manifolds.py

echo "=== P6: steering eval ==="
"$UV" run python scripts/07_eval_steering.py

echo "=== Done. Summary ==="
grep -h "exit-criterion" .neograph-db/logs/console.log 2>/dev/null || true
echo "Reports: reports/p6_steering.json + reports/p6_steering.png"
