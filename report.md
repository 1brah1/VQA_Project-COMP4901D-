# Academic Report: Real-Time VQA Navigation Assistant on Jetson

Prepared for COMP4901D (Group 9)

## Abstract

This project develops and evaluates a real-time visual navigation assistant for edge deployment on NVIDIA Jetson Orin NX. The system combines a compact vision-language model pipeline with deterministic token compression and local text-to-speech feedback. We evaluate latency and prediction quality under multiple compression levels and runtime modes, then tune decoding to improve real-world responsiveness.

## 1. Problem Statement

Assistive navigation requires low end-to-end delay from scene capture to user feedback. On constrained edge hardware, this requires optimizing both model throughput and output reliability. The practical target is a robust pipeline that produces understandable guidance with minimal lag while preserving enough scene detail for safety-relevant tasks.

## 2. System Overview

Pipeline stages:
1. Image encoding with SigLIP (`google/siglip-base-patch16-384`).
2. Spatial token compression from 576 tokens to a target set (`576, 192, 81, 36, 9`).
3. Prefix VLM inference with Qwen2.5-0.5B.
4. Task-specific classification extraction from generated text.
5. Speech rendering through fallback TTS backends (`piper,silero,pyttsx3`), with Piper as preferred runtime.

## 3. Methods and Experimental Design

### 3.1 Compression Study

We sweep all supported targets for a 24x24 token grid:
- 576 (no compression)
- 192
- 81
- 36
- 9

For each target, we record:
- `accuracy_gt_known`
- `unknown_rate`
- stage timing summaries (`total.mean`, `total.p50`)

### 3.2 Runtime Modes

Two benchmark modes were executed:
- FP16 mode (`--llm_mode fp16`)
- AWQ-mode request (`--llm_mode awq` with quantized directory)

Important implementation caveat:
- On Jetson aarch64, current loader detects AWQ GEMM incompatibility and falls back to FP16 generation. Therefore AWQ-mode results in this cycle should be interpreted as requested-mode behavior with fallback, not pure INT4 kernel execution.

### 3.3 Integrated End-to-End Tuning

We compared integrated settings and selected:
- `--compression 192`
- `--profile label_only_eval`
- `--max-new-tokens 4`
- TTS enabled with `piper,silero,pyttsx3`

This setting gave the best balance between latency and observed prediction quality in integrated full-run validation.

## 4. Results

### 4.1 Compression Sweep Results (FP16)

Source: `reports/sweeps/fp16_c576_192_81_36_9_t4_labelonly.json`

| Compression | Accuracy (GT known) | Unknown rate | Mean total time (s) |
|---|---:|---:|---:|
| 576 | 0.2500 | 0.4643 | 0.7471 |
| 192 | 0.3929 | 0.3929 | 0.4428 |
| 81  | 0.5000 | 0.3214 | 0.3654 |
| 36  | 0.5000 | 0.3571 | 0.3464 |
| 9   | 0.4286 | 0.3571 | 0.3727 |

Observation:
- Moderate-to-strong compression improves throughput substantially.
- Best benchmark accuracy among tested settings occurs at 81/36 in this sweep configuration.

### 4.2 Compression Sweep Results (AWQ-mode request)

Source: `reports/sweeps/awq_c576_192_81_36_9_t4_labelonly.json`

| Compression | Accuracy (GT known) | Unknown rate | Mean total time (s) |
|---|---:|---:|---:|
| 576 | 0.2500 | 0.6071 | 0.7475 |
| 192 | 0.3214 | 0.4286 | 0.4098 |
| 81  | 0.4643 | 0.3571 | 0.3668 |
| 36  | 0.3929 | 0.3214 | 0.3726 |
| 9   | 0.3214 | 0.4286 | 0.4017 |

Interpretation note:
- Since runtime falls back to FP16 on this Jetson path, these values reflect mode-requested behavior under fallback constraints.

### 4.3 Integrated Tuned Pipeline + TTS

Source: `reports/jetson_voice_tuned/report_20260331_021401.json`

- Samples processed: `28/28`
- Success count: `28/28`
- WAV artifacts: `28/28`
- Active backend used: `piper`
- Accuracy against labels: `39.29%`
- Unknown rate: `39.29%`
- E2E latency: `mean 627 ms`, `p50 559 ms`, `p95 621 ms`

Compared to earlier integrated configuration (`max_new_tokens=12`, sentence demo profile), tuning reduced latency and reduced unknown output frequency while maintaining robust audio artifact generation.

## 5. Discussion

1. Compression is an effective lever for latency reduction on Jetson.
2. Very low decoding budgets (`max_new_tokens=4`) can reduce instruction-echo failures in this setup.
3. TTS reliability improved substantially after migration to fallback backends and Piper-first runtime.
4. AWQ deployment on Jetson remains constrained by kernel/tooling compatibility; documented fallback behavior is required for transparent reporting.

## 6. Threats to Validity

1. Dataset size is limited (28 samples), so absolute percentages may vary with larger or more diverse data.
2. Label extraction from generated short text can be sensitive to prompt phrasing.
3. AWQ-mode findings are impacted by fallback to FP16 on current aarch64 path.

## 7. Conclusion

The project now has a reproducible, edge-valid pipeline with full audio artifact generation and systematic compression benchmarking. Current best operational setting for full integrated use is `compression=192` with `max_new_tokens=4` and Piper fallback TTS. Compression sweeps provide report-ready evidence for speed-quality trade-offs and support further ablation in final submission.

## 8. Reproducibility Checklist

1. Run preflight:
   - `python scripts/jetson_preflight_check.py`
2. Validate integration:
   - `python scripts/validate_integration.py`
3. Run tuned integrated test:
   - `python scripts/run_integrated.py --labels data/eval/labels.json --compression 192 --profile label_only_eval --max-new-tokens 4 --enable-tts --tts-fallback piper,silero,pyttsx3 --strict-demo --warmup-images 0 --keep-per-sample-wavs --output-dir reports/jetson_voice_tuned`
4. Run sweeps:
   - FP16 and AWQ-mode commands listed in `README.md`.
