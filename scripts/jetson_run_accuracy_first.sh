#!/usr/bin/env bash
set -euo pipefail

# Accuracy-first benchmark runner for Jetson.
#
# It temporarily switches to max-performance mode, runs quality-focused
# FP16 benchmarks, optionally runs AWQ comparison, and restores the original
# power mode on exit.
#
# Usage on Jetson:
#   cd ~/VQA_Project-COMP4901D-
#   bash scripts/jetson_run_accuracy_first.sh
#
# Optional env vars:
#   JETSON_MAX_PERF_MODE=1
#   COMP_LIST="576 192 81"
#   MAX_NEW_TOKENS_EVAL=24
#   PROMPT_STYLE=label_plus_action
#   RUN_AWQ_COMPARE=1
#   LLM_AWQ_DIR=quantized/qwen2p5_0p5b_awq_int4

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="${PYTHONPATH:-}:$ROOT_DIR"

TARGET_MODE="${JETSON_MAX_PERF_MODE:-1}"
COMP_LIST="${COMP_LIST:-576 192 81}"
MAX_NEW_TOKENS_EVAL="${MAX_NEW_TOKENS_EVAL:-24}"
RUN_AWQ_COMPARE="${RUN_AWQ_COMPARE:-0}"
LLM_AWQ_DIR="${LLM_AWQ_DIR:-quantized/qwen2p5_0p5b_awq_int4}"
PROMPT_STYLE="${PROMPT_STYLE:-label_plus_action}"
POWER_CONTROL_ENABLED=1

query_mode_num() {
  sudo nvpmodel -q 2>/dev/null | awk '/Current mode:/{print $3; exit}'
}

restore_mode() {
  if [[ -n "${ORIG_MODE_NUM:-}" ]]; then
    echo "[restore] Restoring nvpmodel mode to ${ORIG_MODE_NUM} ..."
    if sudo nvpmodel -m "$ORIG_MODE_NUM"; then
      echo "[restore] Mode restored."
      sudo nvpmodel -q || true
    else
      echo "[restore] WARNING: failed to restore mode ${ORIG_MODE_NUM}." >&2
    fi
  fi
}

if ! command -v nvpmodel >/dev/null 2>&1; then
  echo "nvpmodel command not found. Run this only on Jetson." >&2
  exit 1
fi

# Use non-interactive sudo when available; otherwise continue benchmarks
# without mode switching so CI/remote automation can still run.
if ! sudo -n true >/dev/null 2>&1; then
  POWER_CONTROL_ENABLED=0
  echo "[power] WARNING: non-interactive sudo is unavailable; mode switch/restore will be skipped." >&2
fi

ORIG_MODE_NUM=""
if [[ "$POWER_CONTROL_ENABLED" == "1" ]]; then
  ORIG_MODE_NUM="$(query_mode_num || true)"
  if [[ -z "$ORIG_MODE_NUM" ]]; then
    echo "Could not detect current nvpmodel mode." >&2
    exit 1
  fi
fi

trap restore_mode EXIT

if [[ "$POWER_CONTROL_ENABLED" == "1" ]]; then
  echo "[power] Original nvpmodel mode: ${ORIG_MODE_NUM}"
  echo "[power] Switching to benchmark mode: ${TARGET_MODE}"
  sudo nvpmodel -m "$TARGET_MODE"
  sudo nvpmodel -q
else
  echo "[power] Running benchmark without nvpmodel change."
fi

if [[ "$POWER_CONTROL_ENABLED" == "1" ]] && command -v jetson_clocks >/dev/null 2>&1; then
  echo "[power] Applying jetson_clocks for stable max throughput during tests"
  sudo jetson_clocks
fi

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-jetson.txt

test -f data/eval/labels.json

IFS=' ' read -r -a COMP_ARR <<< "$COMP_LIST"

mkdir -p reports
python3 scripts/benchmark_compression.py \
  --labels data/eval/labels.json \
  --siglip google/siglip-base-patch16-384 \
  --llm Qwen/Qwen2.5-0.5B-Instruct \
  --llm_mode fp16 \
  --compression "${COMP_ARR[@]}" \
  --max_new_tokens 32 \
  --max_new_tokens_eval "$MAX_NEW_TOKENS_EVAL" \
  --prompt_style "$PROMPT_STYLE" \
  --out reports/fp16_accuracy_first_benchmark.json

echo "[done] FP16 accuracy-first benchmark: reports/fp16_accuracy_first_benchmark.json"

if [[ "$RUN_AWQ_COMPARE" == "1" ]]; then
  if [[ -d "$LLM_AWQ_DIR" ]]; then
    python3 scripts/benchmark_compression.py \
      --labels data/eval/labels.json \
      --siglip google/siglip-base-patch16-384 \
      --llm "$LLM_AWQ_DIR" \
      --llm_mode awq \
      --compression "${COMP_ARR[@]}" \
      --max_new_tokens 32 \
      --max_new_tokens_eval "$MAX_NEW_TOKENS_EVAL" \
      --prompt_style "$PROMPT_STYLE" \
      --out reports/awq_accuracy_first_benchmark.json
    echo "[done] AWQ accuracy-first benchmark: reports/awq_accuracy_first_benchmark.json"
  else
    echo "[skip] RUN_AWQ_COMPARE=1 but AWQ dir is missing: $LLM_AWQ_DIR"
  fi
fi

restore_mode
trap - EXIT

if [[ "$POWER_CONTROL_ENABLED" == "1" ]]; then
  echo "[final] Completed accuracy-first run and restored original power mode."
else
  echo "[final] Completed accuracy-first run (power mode unchanged due sudo limits)."
fi
