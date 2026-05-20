#!/bin/bash
# Run the remaining experiments sequentially to avoid MPS contention.
#
# Each step is idempotent — checks for the output file before running.
# Logs to /tmp/batch_<step>.log.

set -u

ROOT="/Users/tedsandtads/Documents/GitHub/graphgeometry/.claude/worktrees/distracted-khorana-a653e7"
cd "$ROOT"

# Wait until Gemma 9B L31 + cross-task Gemma have both landed (poll every 60s).
echo "[batch] waiting for in-flight runs to finish ..."
while ps -p 15745 > /dev/null 2>&1 || ps -p 15774 > /dev/null 2>&1; do
  sleep 60
  echo "[batch]   still waiting at $(date +%H:%M:%S)"
done
echo "[batch] both in-flight runs exited"

# Cross-task across the remaining models
for MODEL in gpt2 pythia_70m gemma_1_2b; do
  OUT="reports/cross_task_${MODEL}.json"
  if [ -f "$OUT" ]; then
    echo "[batch] $OUT already exists, skipping"
    continue
  fi
  LOG="/tmp/cross_task_${MODEL}.log"
  echo "[batch] cross-task: $MODEL → $OUT (log $LOG)"
  uv run python scripts/load_bearing_topk.py \
    --model "$MODEL" --prompts-file data/cross_task_prompts.json \
    --top-k 10 --sign positive \
    --labels-cache "data/labels_cache_${MODEL}.json" \
    --output "$OUT" > "$LOG" 2>&1 \
    || { echo "[batch] FAIL on $MODEL — see $LOG"; }
done

# Behavioral n=300 — n-seeds 20 × 15 prompts = 300 generations per model
for MODEL in gemma gpt2 pythia_70m gemma_1_2b; do
  OUT="reports/generations_${MODEL}_n300.json"
  if [ -f "$OUT" ]; then
    echo "[batch] $OUT already exists, skipping"
    continue
  fi
  LOG="/tmp/gen_n300_${MODEL}.log"
  echo "[batch] generations n=300: $MODEL → $OUT (log $LOG)"
  uv run python scripts/generate_continuations.py \
    --model "$MODEL" --n-seeds 20 --max-new-tokens 300 \
    --output "$OUT" > "$LOG" 2>&1 \
    || { echo "[batch] FAIL on $MODEL gens — see $LOG"; }
done

# Behavioral analysis on n=300
if [ ! -f reports/behavior_metrics_n300.json ]; then
  echo "[batch] analyse behavior n=300"
  uv run python scripts/analyze_behavior.py \
    --inputs \
      reports/generations_gemma_n300.json \
      reports/generations_gemma_1_2b_n300.json \
      reports/generations_pythia_70m_n300.json \
      reports/generations_gpt2_n300.json \
    --output reports/behavior_metrics_n300.json \
    > /tmp/analyze_behavior_n300.log 2>&1 \
    || { echo "[batch] FAIL on behavior analysis — see /tmp/analyze_behavior_n300.log"; }
fi

echo "[batch] done at $(date +%H:%M:%S)"
