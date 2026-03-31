# VQA Project (COMP4901D): Jetson Real-Time VQA + TTS

This repository contains an edge-oriented visual navigation assistant pipeline for Jetson Orin NX.

Current runtime architecture:
- Vision encoder: SigLIP (`google/siglip-base-patch16-384`)
- LLM: Qwen2.5-0.5B-Instruct
- Compression: deterministic spatial token compression (`576 -> 192/81/36/9`)
- TTS: fallback chain with `piper,silero,pyttsx3` (Piper-first)

## March 31, 2026 status update

Completed in this cycle:
- Migrated active TTS path to fallback backends (Piper-first) for Jetson compatibility.
- Cleaned Jetson disk from low-space emergency state and restored preflight pass.
- Validated integrated pipeline end-to-end with strict WAV artifact checks.
- Ran full compression sweeps under FP16 and AWQ mode requests.
- Tuned integrated run settings for better accuracy/latency balance.

## Key reproducible commands (Jetson)

Activate env and set import path:

```bash
cd ~/VQA_Project-COMP4901D-
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.
```

Preflight validation:

```bash
python scripts/jetson_preflight_check.py
python scripts/validate_integration.py
```

Full 1.5B fine-tune on Jetson (full weights + image projection):

```bash
python scripts/train_image_proj_jetson.py \
  --labels data/train/labels.json \
  --llm Qwen/Qwen2.5-1.5B-Instruct \
  --compression 192 \
  --train-mode full \
  --output-dir models/qwen2p5_1p5b_full \
  --epochs 3 \
  --grad-accum 8 \
  --amp \
  --grad-checkpointing
```

Notes:
- If FP16 is unstable on Jetson for 1.5B, re-run with `--allow-fp32-fallback` (higher memory).
- Outputs: full model weights in `models/qwen2p5_1p5b_full/` plus `image_proj.pt`.

Use a trained image projection during inference/benchmarks:

```bash
python scripts/run_integrated.py \
  --labels data/eval/labels.json \
  --llm models/qwen2p5_1p5b_full \
  --image-proj models/qwen2p5_1p5b_full \
  --compression 192 \
  --profile label_only_eval

python scripts/benchmark_compression.py \
  --labels data/eval/labels.json \
  --llm models/qwen2p5_1p5b_full \
  --image-proj models/qwen2p5_1p5b_full \
  --llm_mode fp16 \
  --compression 576 192 81 36 9 \
  --max_new_tokens_eval 12 \
  --out reports/sweeps/fp16_qwen1p5b_full_c576_192_81_36_9.json
```

LoRA adapter integrated run (uses image_proj.pt from the adapter directory if present):

```bash
python scripts/run_integrated_lora.py \
  --labels data/eval/labels.json \
  --llm Qwen/Qwen2.5-1.5B-Instruct \
  --lora-adapter models/lora_accessibility_vqa \
  --compression 192 \
  --profile label_only_eval \
  --max-new-tokens 4
```

Best current integrated run (full labels + TTS + per-sample WAVs):

```bash
python scripts/run_integrated.py \
  --labels data/eval/labels.json \
  --compression 192 \
  --profile label_only_eval \
  --max-new-tokens 4 \
  --enable-tts \
  --tts-fallback piper,silero,pyttsx3 \
  --strict-demo \
  --warmup-images 0 \
  --keep-per-sample-wavs \
  --output-dir reports/jetson_voice_tuned
```

FP16 compression sweep used for report figures:

```bash
python scripts/benchmark_compression.py \
  --labels data/eval/labels.json \
  --llm_mode fp16 \
  --compression 576 192 81 36 9 \
  --max_new_tokens_eval 4 \
  --prompt_style label_only \
  --out reports/sweeps/fp16_c576_192_81_36_9_t4_labelonly.json
```

AWQ-mode sweep command used for comparison:

```bash
python scripts/benchmark_compression.py \
  --labels data/eval/labels.json \
  --llm quantized/qwen2p5_0p5b_awq_int4 \
  --llm_mode awq \
  --compression 576 192 81 36 9 \
  --max_new_tokens_eval 4 \
  --prompt_style label_only \
  --out reports/sweeps/awq_c576_192_81_36_9_t4_labelonly.json
```

Realtime demo (laptop capture + Jetson inference):

1) On Jetson (server):

```bash
python scripts/realtime_vqa_server.py \
  --llm models/qwen2p5_1p5b_full \
  --image-proj models/qwen2p5_1p5b_full \
  --task obstacles \
  --compression 192
```

2) On laptop (client + local speaker):

```bash
# Open SSH tunnel (uses ~/.ssh/config alias comp4901d-jetson)
ssh -L 5005:localhost:5005 comp4901d-jetson

# In another terminal (on the laptop)
python scripts/realtime_vqa_client.py --host 127.0.0.1 --port 5005 --interval-ms 800
```

Laptop requirements for the client:
- `opencv-python` for webcam capture
- Optional `pyttsx3` for TTS playback

## Latest measured results

### Integrated tuned run (`compression=192`, `max_new_tokens=4`, label-only profile, TTS on)

From `reports/jetson_voice_tuned/report_20260331_021401.json`:
- Success: `28/28`
- WAV artifacts: `28/28`
- Active fallback backend: `piper`
- Accuracy vs labels: `39.29%`
- Unknown prediction rate: `39.29%`
- E2E latency: `mean 627 ms`, `p50 559 ms`, `p95 621 ms`

### Compression sweep summary (`accuracy_gt_known`, `unknown_rate`, mean total sec/sample)

FP16 (`reports/sweeps/fp16_c576_192_81_36_9_t4_labelonly.json`):

| Compression | Accuracy (GT known) | Unknown rate | Mean total time (s) |
|---|---:|---:|---:|
| 576 | 0.2500 | 0.4643 | 0.7471 |
| 192 | 0.3929 | 0.3929 | 0.4428 |
| 81  | 0.5000 | 0.3214 | 0.3654 |
| 36  | 0.5000 | 0.3571 | 0.3464 |
| 9   | 0.4286 | 0.3571 | 0.3727 |

AWQ-mode request (`reports/sweeps/awq_c576_192_81_36_9_t4_labelonly.json`):

| Compression | Accuracy (GT known) | Unknown rate | Mean total time (s) |
|---|---:|---:|---:|
| 576 | 0.2500 | 0.6071 | 0.7475 |
| 192 | 0.3214 | 0.4286 | 0.4098 |
| 81  | 0.4643 | 0.3571 | 0.3668 |
| 36  | 0.3929 | 0.3214 | 0.3726 |
| 9   | 0.3214 | 0.4286 | 0.4017 |

Note on AWQ on Jetson:
- Current aarch64 loader path reports AWQ GEMM incompatibility and falls back to FP16 runtime for generation.
- Keep this caveat explicit in report discussion when interpreting AWQ-mode numbers.

## Artifacts generated in this cycle

- Tuned integrated run: `reports/jetson_voice_tuned/`
- Compression sweeps: `reports/sweeps/`
- Prior run artifacts intentionally pruned to reduce local disk usage.

## Academic write-up

See `report.md` for the academic-facing narrative, methodology, experiments, and discussion aligned with course deliverables.

## References

See `REFERENCES.md` for the primary papers and model cards used in this project.
