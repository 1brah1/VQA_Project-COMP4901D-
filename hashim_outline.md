# COMP4901D: Visual Navigation VQA — Hashim's track (Inference & Audio)

This document covers **what Hashim built**: the accelerated inference pipeline (speculative decoding) and the real-time TTS integration. For the vision baseline and evaluation framework see [`ibrahim_outline.md`](ibrahim_outline.md). For Jetson setup see [`AGENTS.md`](AGENTS.md).

---

## 1. What this track owns

| Responsibility | Files |
|----------------|-------|
| Accelerated VLM inference (PPSD speculative decoding) | `src/vlm/pipelined_vlm.py` |
| Real-time TTS streaming (VibeVoice bridge) | `src/tts/streaming_bridge.py`, `src/tts/__init__.py` |
| End-to-end pipeline runner with latency table | `scripts/run_pipelined.py` |
| Standalone TTS demo / smoke test | `tts_demo.py` |

---

## 2. Accelerated inference — PPSD self-speculative decoding

### What it is

`SelfSpeculativeVLM` (`src/vlm/pipelined_vlm.py`) wraps `SimplePrefixVLM` with **self-speculative decoding** inspired by PPSD (Parallel Prefix Speculative Decoding). No second model is needed — it splits Qwen2.5-0.5B's own 24 layers into:

- **Draft head** — layers 0–11 (half depth), runs K times per round to propose K speculative tokens cheaply.
- **Verify pass** — all 24 layers, verifies K+1 tokens in a **single batched forward pass**.

```
Round k:
  ┌─────────────────────────────────────────────────────────┐
  │  pending_tok ──► [Draft ×K] ──► draft_ids[0..K-1]      │
  │              ──► [Verify once on K+1 toks]              │
  │                    ↓                                     │
  │  Accept longest matching prefix + bonus token            │
  └─────────────────────────────────────────────────────────┘
```

### Why it's faster

| | Layer-ops per confirmed token (K=4, E=12) |
|--|--|
| Baseline sequential | 5 × 24 = **120** layer-ops / round |
| Speculative (ours) | 4 × 12 + 24 = **72** layer-ops / round |
| Theoretical speedup | **1.67×** |
| Empirical (short nav text) | **~1.3–1.5×** |

### Key interface

```python
from src.vlm.pipelined_vlm import SelfSpeculativeVLM, SpecStats
from src.vlm.model import SimplePrefixVLM

vlm = SimplePrefixVLM(...)
spec = SelfSpeculativeVLM(vlm, split_layer=12, K=4)

# Streaming (yields tokens as they are confirmed)
gen = spec.generate_streaming(image_tokens, system_prompt, user_prompt)
try:
    while True:
        text_chunk, accepted = next(gen)
        print(text_chunk, end="", flush=True)
except StopIteration as e:
    stats: SpecStats = e.value   # acceptance rate, speedup, timings

# Non-streaming
text = spec.generate(image_tokens, system_prompt, user_prompt)

# Head-to-head benchmark vs baseline
results = spec.benchmark_vs_baseline(image_tokens, system_prompt, user_prompt)
print(results["speedup"])
```

### SpecStats fields

| Field | Meaning |
|-------|---------|
| `acceptance_rate` | Fraction of draft tokens accepted by verifier |
| `tokens_per_verify_pass` | Average confirmed tokens per round |
| `speedup` | Estimated wall-clock speedup vs sequential |
| `prefill_ms / draft_ms / verify_ms` | Stage-level timing |

---

## 3. Real-time TTS — VibeVoice streaming bridge

### Model

**VibeVoice-Realtime-0.5B** (`microsoft/VibeVoice-Realtime-0.5B`): a flow-matching TTS model that generates speech in streaming chunks (~0.5s/chunk) at **24 kHz**.

### Two classes in `src/tts/streaming_bridge.py`

#### `VibeVoiceTTSService`

Standalone wrapper — load once, call `stream(text)` any number of times.

```python
from src.tts.streaming_bridge import VibeVoiceTTSService

svc = VibeVoiceTTSService(
    model_path="microsoft/VibeVoice-Realtime-0.5B",
    voices_dir="/path/to/VibeVoice/demo/voices/streaming_model",
    device="cuda",
    inference_steps=5,
)
svc.load()

for chunk in svc.stream("Obstacle ahead, move right."):
    # chunk: np.ndarray float32, shape (N,), samplerate=24000
    sounddevice.play(chunk, samplerate=24000, blocking=True)
```

#### `WordBufferedTTSBridge`

Connects directly to a `SelfSpeculativeVLM` streaming generator. Buffers VLM tokens until `word_threshold` complete words arrive, then fires TTS in a **background thread** so VLM decoding continues in parallel.

```python
from src.tts.streaming_bridge import WordBufferedTTSBridge

bridge = WordBufferedTTSBridge(tts_service=svc, word_threshold=3)
bridge.start()

gen = spec.generate_streaming(image_tokens, system_prompt, user_prompt)
try:
    while True:
        chunk, _ = next(gen)
        bridge.feed(chunk)   # pushes tokens into the bridge
except StopIteration:
    pass

bridge.flush()   # speak any leftover words
bridge.stop()

events = bridge.events  # BridgeEvents with TTFT, TTFA, E2E timing
```

### Timing events (`BridgeEvents`)

| Event | Meaning |
|-------|---------|
| `vlm_start` | First token from VLM |
| `tts_triggered` | Bridge triggered TTS (after word_threshold words) |
| `first_audio` | First audio chunk available |
| `e2e_ms` | Image loaded → first audio heard |

**Target**: `e2e_ms < 300 ms` for navigation instructions.

---

## 4. End-to-end pipeline runner (`scripts/run_pipelined.py`)

Runs the full stack on every image in `data/eval/labels.json` and prints a Markdown latency table:

```
Stage                            |    p50 (ms) |    p95 (ms) |   mean (ms)
Capture (load image)             |         0.3 |         0.6 |         0.3
Compression (SigLIP + pool)      |        42.1 |        46.8 |        43.0
VLM TTFT (first token)           |       118.3 |       134.5 |       119.7
VLM Total (all tokens)           |       312.1 |       338.4 |       315.2
TTS TTFA (first audio)           |       210.4 |       240.1 |       215.3
E2E → First Audio                |       440.7 |       480.2 |       445.5
E2E Total                        |      1240.2 |      1380.1 |      1260.4
```

**Usage:**

```bash
# VLM-only (no TTS):
python scripts/run_pipelined.py --labels data/eval/labels.json --no_tts

# Full pipeline with TTS:
python scripts/run_pipelined.py \
    --labels data/eval/labels.json \
    --tts microsoft/VibeVoice-Realtime-0.5B \
    --voices_dir /path/to/VibeVoice/demo/voices/streaming_model

# Benchmark without playing audio:
python scripts/run_pipelined.py --labels data/eval/labels.json \
    --tts microsoft/VibeVoice-Realtime-0.5B \
    --voices_dir /path/to/VibeVoice/demo/voices/streaming_model \
    --no_play
```

---

## 5. Integration with Ibrahim's track

Hashim's `SelfSpeculativeVLM` is a **drop-in replacement** for `SimplePrefixVLM`:

- **Same inputs**: `image_tokens` (from SigLIP + compression), `system_prompt`, `user_prompt`
- **Same output format**: plain text string (normalized verdict: `red`, `green`, `yes`, `no`, `unknown`)
- **Additional**: streaming generator + `SpecStats` for latency profiling

If Ibrahim changes tensor shapes or SigLIP output dimensions, update `_build_prefix_embeds` in `pipelined_vlm.py` accordingly.

**Suggested split of concerns:**

| Ibrahim | Hashim |
|---------|--------|
| Vision encoding, token compression | VLM inference speed, speculative decoding tuning |
| Eval accuracy, prompt engineering | TTS integration, audio latency |
| Jetson benchmarking | End-to-end pipeline script, timing reports |

---

## 6. VibeVoice on Windows (dev machine)

Requirements (Python 3.11, CUDA 12.7):
```
torch==2.7.0+cu124
transformers==4.57.x   (pinned via vibevoice optional dep)
diffusers, sounddevice, scipy
```

Running the standalone demo:
```bash
cd VQA_Project-COMP4901D-
py -3.11 tts_demo.py
# Output: tts_output.wav (open manually if sounddevice fails)
```

Key fix applied to VibeVoice for transformers ≥ 4.57 compatibility:
- `MockCacheLayer.get_mask_sizes(cache_position)` returns `past_kv + query_length` to match the new `DynamicLayer` API introduced in transformers 4.57.
- File: `vibevoice/modular/modeling_vibevoice_streaming_inference.py` (lines ~38–85).

---

## 7. VibeVoice on Jetson (planned)

**Status**: setup in progress at `~/vibevoice_test/`

Jetson constraints:
- Python 3.8.10 — can't install transformers > 4.46.3
- transformers 4.46.3 is shared from VQA venv via `.pth` (no duplicate ~2 GB torch install)
- `FlashAttentionKwargs` compat shim needed at import time (4.46.3 doesn't have it)

Test script: `~/vibevoice_test/tts_test.py` (transferred from Windows)

Expected inference device: **CUDA** (Orin GPU, float16)

---

## 8. Latency budget

Target: **first audio ≤ 300 ms** from image capture.

| Stage | Budget |
|-------|--------|
| SigLIP + compression | ~50 ms |
| VLM TTFT (first token) | ~120 ms |
| TTS TTFA (first audio chunk) | ~130 ms after trigger |
| **E2E → first audio** | **≤ 300 ms** |

Speculative decoding primarily reduces **VLM Total** (all tokens), which matters most when the TTS `word_threshold` is ≥ 3 words — the bridge has to wait for the first 3 words before firing TTS, so faster token output directly compresses E2E latency.
