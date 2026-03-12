Here is the complete project proposal draft in Markdown format.

---

# Project Proposal: A Real-Time, Low-Latency Visual Navigation Assistant for the Visually Impaired Using Optimized Streaming VLMs

**Prepared by:** Ibrahim and Hashim
**Date:** March 12, 2026
**Course:** COMP4901D
**Project Group:** 9 - Real-Time VQA-based Navigation Assistance for Visually Impaired Users

---

## 1. Introduction and Motivation

Individuals with visual impairments face significant challenges navigating dynamic urban environments. Real-time assistive technology that can interpret a live scene and provide immediate, intuitive audio feedback has the potential to dramatically improve safety and independence. Recent advancements in Vision-Language Models (VLMs) and Visual Question Answering (VQA) offer a path toward such systems by enabling a nuanced, language-based understanding of complex scenes (Morales et al., ICRA 2025). However, deploying these models on resource-constrained edge hardware, like the NVIDIA Jetson platform, presents a critical challenge: minimizing end-to-end latency. Delays of even a few hundred milliseconds between capturing an image and delivering audio guidance can be dangerous for a navigating user.

This project aims to design, implement, and rigorously evaluate a real-time navigation assistance system that addresses this latency bottleneck, with the explicit goal of meeting the **Top-tier criteria for Group 9**. The core innovation will be the integration and optimization of three key components: a highly efficient, edge-deployed VLM, a streaming text-to-speech (TTS) engine, and a novel pipelined inference architecture.

---

## 2. Background and Related Work

This proposal builds upon three state-of-the-art research efforts and one open-source model:

| Source | Key Contribution | How We'll Use It |
|--------|------------------|------------------|
| **Morales et al. (ICRA 2025)** | VQA-driven Event Maps for navigation | Application framework, navigation tasks, evaluation methodology |
| **OmniVLM (arXiv:2412.11475)** | 9x token compression (729 → 81 tokens) | Optimization 1: reduce visual tokens |
| **PPSD (arXiv:2509.19368)** | 2.01–3.81x speedup via verify-while-draft pipeline | Optimization 2: accelerate text generation |
| **Microsoft VibeVoice (GitHub)** | Streaming TTS, ~200ms first-audio latency | Real-time audio output |

**Morales et al. (ICRA 2025)** provides a compelling real-world framework. It demonstrates how a VQA model on a smart glasses platform can answer targeted questions about urban scenes (e.g., crosswalk signals, obstacles) to create "Event Maps" for user guidance. This validates the use of VQA for this specific application and provides a clear set of navigation tasks to target.

**OmniVLM** presents a perfect candidate for edge deployment. Its key contribution is a **9x token compression mechanism** (reducing visual tokens from 729 to 81), which dramatically speeds up inference (achieving a 9.1x faster time-to-first-token on a laptop) while maintaining strong performance on benchmarks like ScienceQA. This architecture is ideal for minimizing the visual encoding bottleneck.

**PPSD** introduces a powerful technique for inference acceleration. PPSD uses a **verify-while-draft pipeline parallelism** to overlap the draft and verification stages of speculative decoding, achieving **2.01x to 3.81x speedups** in LLM inference. This methodology is directly applicable to accelerating the language generation component of our VLM.

**Microsoft VibeVoice-Realtime** (0.5B parameters) is explicitly designed for streaming audio, supporting **streaming text input** and achieving a **first-audible latency of ~200-300ms**. This allows the system to begin speaking as soon as the first words of a description are generated, perfectly complementing a pipelined VLM.

This project uniquely combines these advancements: using OmniVLM's token compression for efficient scene encoding, PPSD's pipelining to accelerate text generation, and VibeVoice's streaming capability for rapid audio feedback, all orchestrated to run concurrently on edge hardware.

---

## 3. Project Objectives and Alignment with Top-Tier Criteria

The primary objective is to build a functional prototype that meets all the requirements for a **Top-tier** evaluation as defined in the Group 9 rubric. This means the project will:

1. **Build a Fully Functioning Pipeline on Edge Hardware:** Integrate a lightweight VLM (based on OmniVLM principles) with a streaming TTS engine (VibeVoice-Realtime) on an NVIDIA Jetson Orin NX.

2. **Implement and Measure at Least Two Latency Optimizations:**
   - **Optimization 1 (Token Compression):** Implement the 9x token compression technique from OmniVLM to reduce the VLM's visual processing load. We will test multiple compression ratios (729, 243, 81, 27, 9 tokens) to find the optimal balance.
   - **Optimization 2 (Pipeline Parallelism):** Implement the verify-while-draft pipeline parallelism inspired by PPSD to overlap the VLM's text generation stages.

3. **Provide a Full Latency Breakdown:** Profile and report the end-to-end delay by breaking it down into its component parts: frame capture, preprocessing, VLM inference (split into vision encoding and language generation), TTS synthesis (time to first audio), and audio playback buffering.

4. **Analyze the Trade-off Between Speed and Description Quality:** Systematically evaluate how different optimization settings (compression ratios, pipeline depths) affect both latency and the perceived usefulness/accuracy of the generated descriptions.

5. **Conduct Simulated Navigation Scenarios:** Perform user testing in controlled, simulated environments following the ICRA 2025 paper's four task categories (crosswalks, stairs, obstacles, building fronts) to quantitatively measure how the system's response time impacts task success rates.

---

## 4. Proposed Methodology and System Architecture

The proposed system will be developed and tested on the NVIDIA Jetson Orin NX (16GB) with a standard USB webcam. The software stack will be based on Python and PyTorch.

The architecture is a three-stage pipeline designed for concurrency:

### Stage 1: Vision Encoding & Compression (Optimization 1)

- A frame is captured from the webcam
- A SigLIP vision encoder processes the frame at 384×384 resolution, producing 729 visual tokens (27×27 grid)
- A **token compression module** (implementing OmniVLM's reshape-based compression) reduces the visual token count
- We will implement **five compression ratios** for comparative analysis: 729 (baseline), 243, 81, 27, and 9 tokens
- This stage outputs a compressed visual embedding

### Stage 2: Pipelined VLM Decoding (Optimization 2)

- The compressed visual tokens and the user's implicit query (e.g., "Describe obstacles") are fed into a small language model backbone (Qwen2.5-0.5B)
- A **verify-while-draft pipeline** (inspired by PPSD) is implemented:
  - The first E layers act as a "draft" to propose subsequent tokens
  - The remaining layers verify the current token in parallel
  - Draft and verification stages overlap, hiding latency
  - Rejected tokens are identified immediately with zero wasted computation
- We will optionally compare performance across **two model architectures** (Qwen2.5-0.5B and Phi-2/TinyLLaMA) to test generalization

### Stage 3: Streaming Audio Synthesis (VibeVoice-Realtime)

- As soon as the first few tokens of the text description are generated and verified, they are streamed to VibeVoice-Realtime
- VibeVoice's interleaved, windowed design begins synthesizing audio immediately while the VLM continues generating
- First audible speech is achieved in **~200-300ms**
- Audio playback overlaps with continued text generation

**Latency Profiling:** Each stage will be instrumented to record timestamps, allowing for detailed breakdown of p50, p95, and p99 latencies. The target is **95th percentile end-to-end latency under 500ms** from frame capture to start of audio output.

---

## 5. Evaluation Plan

Evaluation is twofold, directly mapping to Top-tier requirements with an expanded investigation into token compression ratios.

### Quantitative Evaluation

**Full Latency Breakdown:**

- p50, p95, p99 latency for each stage: capture, vision encoding, token compression, VLM drafting, VLM verification, TTS first-chunk, playback buffering

**Token Compression Ratio Analysis:**

We will test five compression ratios, inspired by OmniVLM's methodology:

| Tokens | Ratio | Category |
|--------|-------|----------|
| 729 | 1x (baseline) | No compression |
| 243 | 3x | Light compression |
| 81 | 9x | Moderate (OmniVLM sweet spot) |
| 27 | 27x | Aggressive |
| 9 | 81x | Extreme |

For each configuration, we will measure:
- End-to-end latency
- Vision encoding time
- VLM inference time
- VQA accuracy on navigation-relevant tasks (crosswalk detection, obstacle recognition, stair detection)
- Memory usage and throughput

**Optimization Ablation:**

We will compare four configurations:
1. Baseline (naive sequential pipeline)
2. Compression-only
3. Pipeline-only
4. Full system (both optimizations)

This isolates each optimization's individual contribution.

### Qualitative Evaluation (Simulated Navigation)

Following the ICRA 2025 paper's methodology, we will test **four navigation scenarios**:

| Scenario | Task | Based On |
|----------|------|----------|
| Crosswalk | Distinguish red vs. green walk signals | ICRA Table I & II |
| Stairs | Detect steps or staircases | ICRA Table I & II |
| Obstacles | Detect generic blockages on sidewalk | ICRA Table I & II (hardest task) |
| Building Fronts | Identify restaurants/cafes | ICRA Table I & II |

For each scenario, across multiple compression configurations, we will measure:
- Task success rate
- Time to navigate
- Number of collisions/stumbles
- Participant feedback on speed vs. description usefulness

This will identify the optimal compression ratio where maximum speedup is achieved without sacrificing safety-critical information.

---

## 6. Expected Outcomes and Contributions

By project completion, we will deliver a submission meeting all Top-tier criteria for Group 9:

1. ✅ Fully functioning end-to-end prototype on Jetson Orin NX edge hardware
2. ✅ Two latency optimizations implemented and measured (token compression + pipeline parallelism)
3. ✅ Detailed stage-wise latency breakdown (capture, encoding, compression, draft/verify, TTS, playback)
4. ✅ Comprehensive compression ratio analysis (729→243→81→27→9 tokens) with accuracy metrics for navigation tasks
5. ✅ Speed-quality trade-off analysis based on simulated user testing and task success rates
6. ✅ Documented codebase and final report with implementation details and recommendations

---

## 7. Project Timeline (March 12 – May 8, 2026)

| Section | Date Range | Key Tasks |
|:---:|:---:|:---|
| **1. Foundation & Mid-Term Progress** | Mar 12 – Apr 1 | • Set up Jetson environment (PyTorch, drivers)<br>• Test VibeVoice TTS locally<br>• Select base VLM (Qwen2.5-0.5B + SigLIP)<br>• Build naive pipeline: capture → VLM → TTS<br>• Measure baseline latency<br>• Prepare mid-term presentation |
| **2. Optimization 1: Token Compression** | Apr 2 – Apr 15 | • Implement 9x token compression (reshape-based)<br>• Implement all five compression ratios<br>• Integrate into vision encoder pipeline<br>• Measure latency improvement vs. baseline |
| **3. Optimization 2: Pipeline Parallelism** | Apr 16 – Apr 29 | • Design verify-while-draft architecture<br>• Implement early-exit draft mechanism<br>• Create pipeline stages for draft & verification<br>• Integrate streaming TTS<br>• Measure combined latency improvement |
| **4. Comprehensive Evaluation & Final Delivery** | Apr 30 – May 8 | • Implement stage-wise latency instrumentation<br>• Run ablation study (all configurations)<br>• Conduct user testing (4 navigation scenarios)<br>• Analyze speed-quality trade-off<br>• Write final report & prepare presentation |

**Key Milestones:**
- **Mid-Term Proposal:** March 27 – April 1
- **Final Proposal:** May 6–8

---

## 8. Hardware and Software Requirements

**Hardware:**

- NVIDIA Jetson Orin NX (16GB)
- Standard USB webcam (or Intel RealSense D435)

**Software:**

- Ubuntu 20.04+ / JetPack SDK
- Python 3.8+ with PyTorch
- VibeVoice-Realtime TTS engine
- VLM components: Qwen2.5-0.5B, SigLIP vision encoder
- Profiling tools (custom instrumentation, NVIDIA Nsight)
- OpenCV for camera capture

---

## 9. References

1. Morales, J., Gebregziabher, B., Cabaneros, A., & Sanchez-Riera, J. (2025). VQA-driven Event Maps for Assistive Navigation for People with Low Vision in Urban Environments. *2025 IEEE International Conference on Robotics and Automation (ICRA)*. https://ieeexplore.ieee.org/document/11128754

2. Chen, W., Li, Z., & Xin, S. (2024). OmniVLM: A Token-Compressed, Sub-Billion-Parameter Vision-Language Model for Efficient On-Device Inference. *arXiv preprint arXiv:2412.11475*. https://arxiv.org/abs/2412.11475

3. Li, R., Li, Z., Shi, Y., Shao, J., Zhang, C., & Li, X. (2025). PPSD: Pipeline Parallelism is All You Need for Optimized Early-Exit Based Self-Speculative Decoding. *arXiv preprint arXiv:2509.19368*. https://arxiv.org/abs/2509.19368

4. Microsoft. (2025). VibeVoice-Realtime: Lightweight real-time text-to-speech with streaming support. *GitHub repository*. https://github.com/microsoft/VibeVoice

---
