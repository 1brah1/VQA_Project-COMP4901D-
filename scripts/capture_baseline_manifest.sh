#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f "$ROOT_DIR/.venv/bin/activate" ]; then
  source "$ROOT_DIR/.venv/bin/activate"
fi

export PYTHONPATH="${PYTHONPATH:-}:$ROOT_DIR"

RESULTS_FILE="${1:-}"
if [ -z "$RESULTS_FILE" ]; then
  if [ -f "reports/vqa_with_tts_results.json" ]; then
    RESULTS_FILE="reports/vqa_with_tts_results.json"
  elif [ -f "reports/fp16_benchmark.json" ]; then
    RESULTS_FILE="reports/fp16_benchmark.json"
  else
    RESULTS_FILE="reports/vqa_results.json"
  fi
fi

OUT_FILE="${2:-reports/baseline_run_manifest.json}"

echo "[baseline-manifest] results=$RESULTS_FILE"
echo "[baseline-manifest] out=$OUT_FILE"
python3 scripts/capture_run_manifest.py --results "$RESULTS_FILE" --out "$OUT_FILE"
