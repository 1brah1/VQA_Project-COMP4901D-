## Jetson (Orin NX 16GB) benchmark recipe

### 1) FP16 baseline (recommended first)

Install dependencies (Jetson PyTorch is usually installed via NVIDIA wheels; then):

```bash
pip install -r requirements.txt
```

Run the compression sweep:

```bash
python scripts/benchmark_compression.py --out reports/fp16_benchmark.json
```

### 2) INT4 (LLM via AWQ)

```bash
pip install -r requirements-quant.txt
python scripts/quantize_llm_awq.py --model Qwen/Qwen2.5-0.5B-Instruct --out quantized/qwen2p5_0p5b_awq_int4
```

Then adjust your benchmark to point `--llm` to the quantized folder, and load with `src/vlm/llm_loader.py::load_llm_awq` (wire-up depends on your run script).

