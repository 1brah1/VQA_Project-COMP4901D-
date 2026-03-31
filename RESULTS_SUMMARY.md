# VQA Results Summary (Concise)

**Last updated:** March 31, 2026

## Baseline (reports/vqa_results.json)

- Total samples: 28 images
- Overall accuracy: 46.4% (13/28 correct)
- By task:
  - Stairs: 66.7% (6/9)
  - Obstacles: 40.0% (4/10)
  - Crosswalk signal: 33.3% (3/9)
- E2E latency: mean 1008 ms, median 616 ms
- Bottleneck: VLM inference (~85% of total time)

## Latest integrated run (Jetson, TTS enabled)

- Source: reports/jetson_voice_tuned/report_20260331_021401.json
- Accuracy vs labels: 39.29%
- Unknown prediction rate: 39.29%
- E2E latency: mean 627 ms, p50 559 ms, p95 621 ms

## Fine-tuning status

- Full 1.5B fine-tune path: scripts/train_image_proj_jetson.py --train-mode full
- LoRA integrated eval: scripts/run_integrated_lora.py (uses image_proj.pt if present)
- Always record the adapter path and image_proj path for tuned runs

## Where to find full detail

- reports/FINAL_REPORT.md — complete narrative + recommendations
- reports/vqa_analysis.md — detailed tables and error modes
- reports/vqa_dashboard.html — interactive charts
- reports/vqa_results.json — raw per-sample metrics

## Reproduce the baseline summary

```bash
python analyze_results.py
python create_dashboard.py
```

## Benchmark artifacts

- Compression sweeps and Jetson runs live under reports/sweeps/ and reports/jetson_voice_tuned/
- AWQ and FP16 benchmark JSON files are kept in reports/ for side-by-side comparison

## Notes for new runs

- Use label-only prompts for strict classification benchmarks.
- Keep max_new_tokens low (8-16) to reduce over-generation.
- Always record the image projection path used for fine-tuned runs.

