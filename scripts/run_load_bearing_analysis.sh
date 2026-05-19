#!/usr/bin/env bash
# End-to-end orchestrator for the load-bearing-feature analysis.
#
# Runs the supporting+opposing ablation on every model in MODELS (default:
# Gemma 2 2B, GPT-2 small) using prompts_50.json, then re-exports web/data/summary.json.
#
# Add gemma_9b to MODELS once the 9B run is comfortable (≈25-min compute on M5 Max).
#
# Usage:
#   bash scripts/run_load_bearing_analysis.sh                  # default models
#   MODELS="gemma gpt2 gemma_9b" bash scripts/run_load_bearing_analysis.sh
#   PROMPTS=data/causal_prompts.json bash scripts/run_load_bearing_analysis.sh  # short 12-prompt set
#
set -euo pipefail

cd "$(dirname "$0")/.."

MODELS="${MODELS:-gemma gpt2}"
PROMPTS="${PROMPTS:-data/prompts_50.json}"
SIGN="${SIGN:-positive}"
TOP_K="${TOP_K:-10}"

mkdir -p logs reports web/data

# Suffix from prompts file: data/prompts_50.json -> 50; data/causal_prompts.json -> 12
N_PROMPTS=$(python3 -c "import json; print(len(json.load(open('$PROMPTS'))))")

for MODEL in $MODELS; do
  OUT="reports/load_bearing_pos${TOP_K}_${MODEL}_${N_PROMPTS}.json"
  LOG="logs/${MODEL}_${N_PROMPTS}_$(date +%H%M%S).log"
  echo "=== ${MODEL}  →  ${OUT}  (log: ${LOG}) ==="
  uv run python scripts/load_bearing_topk.py \
    --model "${MODEL}" \
    --prompts-file "${PROMPTS}" \
    --top-k "${TOP_K}" \
    --sign "${SIGN}" \
    --output "${OUT}" > "${LOG}" 2>&1
  tail -5 "${LOG}"
done

echo
echo "=== Exporting web data ==="
uv run python scripts/export_web_data.py

echo
echo "Done. Serve the explainer with:"
echo "  cd web && python3 -m http.server 8770"
