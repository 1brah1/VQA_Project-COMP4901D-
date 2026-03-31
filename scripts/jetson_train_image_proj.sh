#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f "$ROOT_DIR/.venv/bin/activate" ]; then
  source "$ROOT_DIR/.venv/bin/activate"
fi

export PYTHONPATH="${PYTHONPATH:-}:$ROOT_DIR"
export TOKENIZERS_PARALLELISM=false

CKPT_DIR="${CKPT_DIR:-checkpoints/image_proj}"
MAX_RETRIES="${MAX_RETRIES:-20}"
RETRY_SLEEP_SECS="${RETRY_SLEEP_SECS:-15}"

EPOCHS="${EPOCHS:-30}"
MAX_STEPS="${MAX_STEPS:-400}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
SAVE_EVERY="${SAVE_EVERY:-20}"
EMPTY_CACHE_STEPS="${EMPTY_CACHE_STEPS:-20}"
LR="${LR:-2e-4}"
COMPRESSION="${COMPRESSION:-192}"

attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "[jetson-train] attempt=$attempt/$MAX_RETRIES"

  set +e
  python3 scripts/train_image_proj_jetson.py \
    --labels data/eval/labels.json \
    --compression "$COMPRESSION" \
    --epochs "$EPOCHS" \
    --max-steps "$MAX_STEPS" \
    --grad-accum "$GRAD_ACCUM" \
    --save-every "$SAVE_EVERY" \
    --empty-cache-steps "$EMPTY_CACHE_STEPS" \
    --lr "$LR" \
    --checkpoint-dir "$CKPT_DIR" \
    --resume latest
  code=$?
  set -e

  if [ "$code" -eq 0 ]; then
    echo "[jetson-train] training completed successfully"
    exit 0
  fi

  echo "[jetson-train] training exited with code=$code"
  if [ "$attempt" -ge "$MAX_RETRIES" ]; then
    echo "[jetson-train] reached MAX_RETRIES=$MAX_RETRIES; giving up"
    exit "$code"
  fi

  echo "[jetson-train] resuming from latest checkpoint after ${RETRY_SLEEP_SECS}s"
  sleep "$RETRY_SLEEP_SECS"
done
