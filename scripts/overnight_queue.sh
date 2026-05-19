#!/usr/bin/env bash
# Overnight model queue — runs each pending model serially, re-exporting the
# web page's data file after each success so the page is always up-to-date.
# A failure on one model just logs and continues to the next.
#
# Models are listed in safest-first order: smaller / known-good first, so a
# tokenizer or memory blowup on the big ones doesn't block the smaller ones.

set -u

cd "$(dirname "$0")/.."

# Prevent macOS from idle-sleeping during the run. -i = inhibit idle sleep, -m
# = inhibit disk sleep. We caffeinate the SHELL itself (and all subprocesses)
# by re-exec'ing under caffeinate if we're not already inside it.
if [ -z "${CAFFEINATED:-}" ]; then
  exec env CAFFEINATED=1 caffeinate -im -- bash "$0" "$@"
fi

LOGDIR="logs"
mkdir -p "$LOGDIR" web/data reports

PROMPTS="data/prompts_50.json"
TOP_K=10
SIGN="positive"

# Queue. Each line: model_key. The script skips any with an existing result
# file, so re-running after a partial completion is idempotent. Smaller / less
# risky models first, so a failure on the big ones doesn't block the rest.
QUEUE=(
  "qwen3_1_7b"
  "gemma_1_2b"
  "mistral_7b"
  "gemma_9b"
  "gemma_27b"
)

for MODEL in "${QUEUE[@]}"; do
  OUT="reports/load_bearing_pos${TOP_K}_${MODEL}_50.json"
  LOG="$LOGDIR/${MODEL}_50_$(date +%Y%m%d_%H%M%S).log"
  if [ -f "$OUT" ]; then
    echo "[$(date +%H:%M:%S)] $MODEL already has results, skipping."
    continue
  fi
  echo "[$(date +%H:%M:%S)] === Starting $MODEL (log: $LOG) ==="
  if uv run python scripts/load_bearing_topk.py \
      --model "$MODEL" \
      --prompts-file "$PROMPTS" \
      --top-k "$TOP_K" \
      --sign "$SIGN" \
      --output "$OUT" > "$LOG" 2>&1; then
    echo "[$(date +%H:%M:%S)] $MODEL FINISHED. Re-exporting..."
    uv run python scripts/export_web_data.py 2>&1 | tail -8
    uv run python scripts/embed_web_data.py 2>&1 | tail -2
  else
    echo "[$(date +%H:%M:%S)] $MODEL FAILED (see $LOG). Continuing."
    tail -20 "$LOG"
  fi
  echo
done

echo "[$(date +%H:%M:%S)] === overnight queue done ==="
