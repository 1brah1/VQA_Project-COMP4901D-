# 12-Minute Presentation Outline (with Live Demo)

Target length: ~12 minutes total, including a 2.5-minute live demo.

## Slide 1 (0:30) Title
- Project name, team, course
- One-line goal: low-latency VQA for assistive navigation on Jetson

## Slide 2 (1:00) Problem + Motivation
- Why edge VQA matters for visually impaired navigation
- Constraints: latency, power, memory

## Slide 3 (1:00) System Overview
- Pipeline: SigLIP -> token compression -> Prefix VLM -> classification -> TTS
- Show the flow diagram

## Slide 4 (1:00) Optimization Objective
- Compression targets: 576 -> 192/81/36/9
- Trade accuracy vs latency

## Slide 5 (1:30) Accuracy vs Compression (Benchmark)
- Show: reports/ppt_assets/accuracy_vs_compression.png
- Callouts: 0.5B baseline vs 0.5B LoRA vs 1.5B LoRA
- Note: small eval set (28 images)

## Slide 6 (1:00) Latency and Memory Tradeoffs
- Show: reports/ppt_assets/latency_vs_compression.png
- Show: reports/ppt_assets/memory_vs_compression.png
- Highlight Jetson constraints and FP32 fallback for 1.5B

## Slide 7 (1:00) Fine-Tuning Impact (LoRA)
- Best accuracy: 0.5B LoRA 0.75 @ compression 36
- Best accuracy: 1.5B LoRA 0.821 @ compression 192 (FP32 fallback)

## Slide 8 (1:00) Deployment Constraints and Fixes
- AWQ request falls back to FP16 on aarch64
- VibeVoice not usable on Jetson; Silero fallback
- Hidden-size checks and prompt tightening

## Slide 9 (0:30) Demo Setup
- What the demo shows (live classification + spoken output)
- Briefly describe input and output

## Slide 10 (2:30) Live Demo
- Run the demo (camera or sample image)
- Show label output and timing

## Slide 11 (1:00) Key Takeaways
- Compression is the main speed lever
- LoRA improves accuracy on the small dataset
- 0.5B is the practical Jetson target

## Slide 12 (1:00) Next Steps (Post-Prototype)
- More data + stronger eval
- Compression refinements (OmniVLM-inspired)
- Event map integration (Morales 2025)
- Further speculative decoding tuning (PPSD-inspired)
