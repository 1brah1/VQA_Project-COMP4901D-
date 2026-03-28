#!/usr/bin/env bash
set -euo pipefail

# Optional: run AWQ INT4 quantization on Jetson (slow; requires autoawq pip install).
# Must use LF line endings only. Fix CRLF: sed -i 's/\r$//' scripts/jetson_quantize_llm_awq.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 -m pip install -r requirements-quant.txt

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
OUT_DIR="${OUT_DIR:-quantized/qwen2p5_0p5b_awq_int4}"

python3 scripts/quantize_llm_awq.py --model "$MODEL_NAME" --out "$OUT_DIR"
echo "Quantization saved to: $OUT_DIR"
