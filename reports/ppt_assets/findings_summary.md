# Findings Summary (Compression Benchmarks)

## Design optimization objective
- Reduce SigLIP patch tokens (576 -> 192/81/36/9) to lower end-to-end latency while keeping classification accuracy usable.
- Keep the pipeline edge-friendly: deterministic compression + short fixed-length outputs for reliable parsing.

## System constraints (Jetson)
- Memory and latency limits on Jetson Orin NX require smaller models and shorter generations.
- AWQ on aarch64 can fall back to FP16, so AWQ-request results are not pure INT4 kernels.
- VibeVoice is not compatible with the Jetson Python/transformers stack; Silero is the usable fallback.

## Solutions utilized
- Deterministic token compression before VLM inference to trade resolution for speed.
- Short fixed-length outputs to reduce unknowns and parsing errors.
- Optional LoRA adapters and self-speculative decoding (PPSD-inspired) for future speedups.

## Key benchmark findings
| Series | Best accuracy (GT known) | Compression | Best total mean (ms) | Compression |
|---|---:|---:|---:|---:|
| 0.5B AWQ-request | 0.464 | 81 | 366.8 | 81 |
| 0.5B FP16 | 0.500 | 36 | 346.4 | 36 |
| 0.5B FP16 + LoRA | 0.750 | 36 | 479.0 | 9 |
| 1.5B FP16 | 0.000 | 9 | 679.6 | 9 |
| 1.5B FP16 + LoRA | 0.821 | 192 | 1954.9 | 9 |

## Notes
- Note: 1.5B FP16 produced 0 accuracy across compressions (likely model or prompt instability on device).

## Charts generated
- accuracy_vs_compression.png
- unknown_rate_vs_compression.png
- latency_vs_compression.png
- memory_vs_compression.png