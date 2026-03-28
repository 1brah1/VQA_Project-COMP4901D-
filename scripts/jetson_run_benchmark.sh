#!/usr/bin/env bash
set -euo pipefail

# Jetson Orin NX run helper (user will SSH in and run this).
#
# Expected usage on Jetson:
#   cd /path/to/VQA_Project-COMP4901D-
#   bash scripts/jetson_run_benchmark.sh
#
# These files must use LF line endings only (Unix). If you see
# "set: pipefail: invalid option", run: sed -i 's/\r$//' scripts/jetson_*.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# 1) Install Python deps
# - Assumes Jetson already has a working torch/torchvision install.
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-jetson.txt

# 2) Ensure dataset labels exist
test -f data/eval/labels.json

# 3) Run FP16 baseline benchmark (compression sweep)
mkdir -p reports
python3 scripts/benchmark_compression.py \
  --labels data/eval/labels.json \
  --siglip google/siglip-base-patch16-384 \
  --llm Qwen/Qwen2.5-0.5B-Instruct \
  --llm_mode fp16 \
  --max_new_tokens 32 \
  --max_new_tokens_eval 12 \
  --out reports/fp16_benchmark.json

echo "FP16 benchmark done: reports/fp16_benchmark.json"

# 4) (Optional) AWQ INT4 benchmark
LLM_AWQ_DIR="${LLM_AWQ_DIR:-quantized/qwen2p5_0p5b_awq_int4}"

if [ "${QUANTIZE_AWQ_FIRST:-0}" = "1" ]; then
  echo "QUANTIZE_AWQ_FIRST=1: running scripts/jetson_quantize_llm_awq.sh ..."
  bash scripts/jetson_quantize_llm_awq.sh
fi

if [ -d "$LLM_AWQ_DIR" ]; then
  python3 -m pip install -r requirements-quant.txt
  python3 scripts/benchmark_compression.py \
    --labels data/eval/labels.json \
    --siglip google/siglip-base-patch16-384 \
    --llm "$LLM_AWQ_DIR" \
    --llm_mode awq \
    --max_new_tokens 32 \
    --max_new_tokens_eval 12 \
    --out reports/awq_int4_benchmark.json
  echo "AWQ benchmark done: reports/awq_int4_benchmark.json"
else
  echo "Skipping AWQ benchmark (missing dir): $LLM_AWQ_DIR"
  echo "  Option A — On Jetson:  QUANTIZE_AWQ_FIRST=1 bash scripts/jetson_run_benchmark.sh"
  echo "  Option B — On Jetson:  bash scripts/jetson_quantize_llm_awq.sh"
  echo "  Option C — On PC:     python scripts/quantize_llm_awq.py --out quantized/qwen2p5_0p5b_awq_int4"
  echo "                        then scp -r quantized/qwen2p5_0p5b_awq_int4 user@jetson:~/VQA_Project-COMP4901D-/quantized/"
  echo "If AWQ script fails with 'pipefail': sed -i 's/\r$//' scripts/jetson_*.sh"
  echo "FP16 results above are still valid without AWQ."
fi
