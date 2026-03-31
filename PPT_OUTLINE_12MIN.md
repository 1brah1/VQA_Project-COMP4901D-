# 12-Minute Presentation Slide Content (with Live Demo)

Target length: ~12 minutes total, including a 2.5-minute live demo.

Use this as direct copy/paste guidance for each slide (text + images to paste).

## Slide 1 (0:30) Title
Text to add:
- Real-Time VQA Navigation Assistant on Jetson
- COMP4901D Group 9
- Team members + date
- One-line goal: low-latency VQA for assistive navigation on Jetson
Images to paste:
- None (clean title slide)

## Slide 2 (1:00) Problem + Motivation
Text to add:
- Edge VQA matters for visually impaired navigation
- Key constraints: latency, power, memory
- Tasks: crosswalk signal, stairs, obstacles
Images to paste (3 thumbnails in a row):
- data/eval/images/crosswalk/Crosswalk_2.png
- data/eval/images/stairs/Stairs_3.png
- data/eval/images/obstacles/Obstacle_1.png

## Slide 3 (1:00) System Overview
Text to add (pipeline bullets):
- SigLIP encoder (384)
- Token compression 576 -> 192/81/36/9
- Prefix VLM (Qwen2.5-0.5B)
- Label extraction
- TTS (piper, silero, pyttsx3)
Images to paste:
- Build a simple flow diagram directly in PPT with five boxes and arrows

## Slide 4 (1:00) Optimization Objective
Text to add:
- Sweep compression targets: 576, 192, 81, 36, 9
- Measure accuracy_gt_known, unknown rate, latency
- Eval set size: 28 images
Images to paste:
- None (use a small table or bullet list)

## Slide 5 (1:30) Accuracy vs Compression (Benchmark)
Text to add:
- Compare 0.5B baseline vs 0.5B LoRA vs 1.5B LoRA
- Note: small eval set (28 images)
Images to paste:
- reports/ppt_assets/accuracy_vs_compression.png

## Slide 6 (1:00) Latency and Memory Tradeoffs
Text to add:
- Compression is the main speed lever
- 1.5B on Jetson needs FP32 fallback
Images to paste:
- reports/ppt_assets/latency_vs_compression.png
- reports/ppt_assets/memory_vs_compression.png

## Slide 7 (1:00) Fine-Tuning Impact (LoRA)
Text to add (as a 3-row table):
- 0.5B FP16 baseline: best accuracy_gt_known 0.50 @ compression 36
- 0.5B LoRA: best accuracy_gt_known 0.75 @ compression 36
- 1.5B LoRA: best accuracy_gt_known 0.821 @ compression 192 (FP32 fallback)
Images to paste:
- None (table only)

## Slide 8 (1:00) Deployment Constraints and Fixes
Text to add:
- AWQ request falls back to FP16 on aarch64
- VibeVoice not usable on Jetson; Silero fallback
- Hidden-size checks + prompt tightening to reduce errors
Images to paste:
- None

## Slide 9 (0:30) Demo Setup
Text to add:
- Live classification + spoken output
- Input: single image (crosswalk/obstacle/stairs)
- Output: label + response + audio
Images to paste:
- data/eval/images/obstacles/Obstacle_1.png

## Slide 10 (2:30) Live Demo
Text to add:
- Run the demo (camera or sample image)
- Show label output and timing
- Play the audio output
Media to embed:
- reports/jetson_silero_lora_0p5b_c192_v3/combined_tts.wav
Images to paste:
- None (live demo slide)

## Slide 11 (1:00) Key Takeaways
Text to add:
- Compression is the main speed lever
- LoRA improves accuracy on a small dataset
- 0.5B is the practical Jetson target
Images to paste:
- None

## Slide 12 (1:00) Next Steps (Post-Prototype)
Text to add:
- More data + stronger evaluation
- Compression refinements (OmniVLM-inspired)
- Event map integration (Morales 2025)
- Speculative decoding tuning (PPSD-inspired)
Images to paste:
- None

## Slide 13 (0:45) References
Text to add (bulleted list):
- SigLIP: Zhai et al., "Sigmoid Loss for Language Image Pre-training" (2023). arXiv:2303.15343
- Qwen2.5: Qwen technical report (2023). arXiv:2309.16609; model card: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- LoRA: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021). arXiv:2106.09685
- AWQ: Lin et al., "AWQ: Activation-aware Weight Quantization" (2023). arXiv:2306.00978
- PPSD inspiration: Leviathan et al., "Fast Inference via Speculative Decoding" (2023). arXiv:2302.01318; self-speculative overview arXiv:2305.10427
- TTS: Piper (https://github.com/rhasspy/piper), Silero (https://github.com/snakers4/silero-models), VibeVoice (https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B)
- OmniVLM (compression inspiration): add full citation here
- Morales 2025 event map: add full citation here
Images to paste:
- None
