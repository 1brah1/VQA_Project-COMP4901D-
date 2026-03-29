# COMP4901D — Midterm Proposal: Agent Onboarding & Task List

> **Branch**: `tts-implementation`
> **Repo**: `https://github.com/1brah1/VQA_Project-COMP4901D-`
> **Last verified**: 2026-03-29
> **Purpose**: This document gives a coding agent (or new contributor) full context on what has been built and exactly what still needs to be done before the midterm proposal deadline.

---

## 0. Project Overview

Assistive navigation system for visually impaired users running on a **Jetson Orin NX**. Given a camera image, the system must:

1. Encode the image (SigLIP vision encoder)
2. Compress vision tokens (adaptive average-pool, 729 → 81 tokens)
3. Run a VLM (Qwen2.5-0.5B) to produce a short navigation verdict
4. Convert the verdict to speech (VibeVoice TTS) and play it to the user

**Latency target**: ≤ 300 ms from image capture to **first spoken word**.

### Two parallel tracks

| Track | Owner | Focus |
|-------|-------|-------|
| Vision / Accuracy | Ibrahim | SigLIP encoding, token compression, eval accuracy, prompt tuning |
| Inference Speed / Audio | Hashim | PPSD speculative decoding, VibeVoice TTS, E2E latency |

---

## 1. Repo & Environment Setup

### 1.1 Directory layout

```
VQA_Project-COMP4901D-/
  src/
    vision/                   # SigLIP encoder + token compression
      siglip_encoder.py       # SiglipPatchEncoder class
      token_compression.py    # compress_27x27_tokens()
    vlm/
      model.py                # SimplePrefixVLM — baseline VLM wrapper
      llm_loader.py           # loads Qwen2.5-0.5B-Instruct
      pipelined_vlm.py        # SelfSpeculativeVLM (PPSD, Hashim's track)
    tts/
      streaming_bridge.py     # VibeVoiceTTSService + WordBufferedTTSBridge
      __init__.py
    prompts/
      system_prompt.txt       # shared system prompt for all tasks
  scripts/
    run_once.py               # single-image baseline inference
    run_pipelined.py          # full E2E pipeline with latency table
    benchmark_compression.py  # token compression ablation
    jetson_run_benchmark.sh   # shell runner for Jetson benchmarks
    jetson_quantize_llm_awq.sh
    quantize_llm_awq.py       # AWQ INT4 quantization script
  data/eval/
    labels.json               # 16-image eval set (crosswalk/stairs/obstacles)
    images/crosswalk/         # 5 PNG images
    images/stairs/            # 5 PNG images
    images/obstacles/         # 6 PNG images
  AGENTS.md                   # Jetson environment guide (read first)
  hashim_outline.md           # Hashim track detail doc
  ibrahim_outline.md          # Ibrahim track detail doc
  requirements.txt            # Windows/Linux dev deps
  requirements-jetson.txt     # Jetson deps (intentionally omits torch)
```

### 1.2 Dev machine (Windows, Python 3.11)

```bash
# VibeVoice repo is a sibling directory: C:/Users/hash_/VibeVoice/
# Run TTS demo:
cd VQA_Project-COMP4901D-
py -3.11 tts_demo.py
# Output: tts_output.wav
```

### 1.3 Jetson Orin NX

- **SSH**: `comp4901d@10.89.149.159` (password: `comp4901d`)
- **Python**: 3.8.10 — no syntax ≥ 3.10 (use `typing.Union`, `Optional`)
- **PyTorch**: `2.1.0a0+gitXXX` (Jetson-specific build, **never** `pip install torch`)
- **Transformers**: 4.46.3 (max for Python 3.8)
- **Disk**: ~98% full — be careful with large downloads; use `df -h` before installing anything

#### Jetson venv structure

```
/home/comp4901d/
  VQA_Project-COMP4901D-/
    .venv/                    # main VQA project venv
  VibeVoice/                  # TTS source (transferred via SFTP)
  vibevoice_test/             # isolated TTS test environment
    .venv/
      lib/python3.8/site-packages/
        shared_vqa.pth        # points to VQA venv's site-packages (shares torch)
        vibevoice.pth         # points to /home/comp4901d/VibeVoice/
        diffusers/            # minimal hand-written stub (see §3.4)
        sounddevice/          # installed
        scipy/                # installed
    model/                    # VibeVoice-Realtime-0.5B weights (1.9 GB)
    voices/en-Davis_man.pt    # voice preset (2.4 MB)
    tts_test.py               # standalone TTS smoke test
    run_pipeline.py           # integrated VQA + TTS pipeline (verified working)
```

**Activate vibevoice_test venv:**
```bash
source /home/comp4901d/vibevoice_test/.venv/bin/activate
```

**Activate VQA project venv:**
```bash
source /home/comp4901d/VQA_Project-COMP4901D-/.venv/bin/activate
```

---

## 2. What Has Been Built (Completed Work)

### 2.1 Vision Pipeline (Ibrahim's track — baseline complete)

| Component | File | Status |
|-----------|------|--------|
| SigLIP image encoder | `src/vision/siglip_encoder.py` | ✅ Working |
| Token compression (729→81) | `src/vision/token_compression.py` | ✅ Working |
| Prefix VLM (Qwen2.5-0.5B) | `src/vlm/model.py` | ✅ Working |
| Eval benchmark runner | `scripts/run_once.py` | ✅ Working |
| AWQ INT4 quantization scripts | `scripts/quantize_llm_awq.py` | ✅ Scripted |

**VQA interface:**
```python
from src.vision.siglip_encoder import SiglipPatchEncoder
from src.vision.token_compression import compress_27x27_tokens
from src.vlm.model import SimplePrefixVLM

encoder = SiglipPatchEncoder.from_pretrained("google/siglip-so400m-patch14-384")
image_tokens = encoder.encode(image)                       # (1, 729, 1152)
compressed   = compress_27x27_tokens(image_tokens, target_tokens=81)  # (1, 81, 1152)

vlm = SimplePrefixVLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", ...)
text = vlm.generate(compressed, system_prompt, user_prompt)
# e.g. "No obstacle ahead."
```

### 2.2 PPSD Speculative Decoding (`src/vlm/pipelined_vlm.py`)

`SelfSpeculativeVLM` wraps `SimplePrefixVLM` with self-speculative decoding — no second model needed. Splits Qwen2.5-0.5B's 24 decoder layers into:

- **Draft head** (layers 0–11): runs K=4 times cheaply to propose 4 tokens
- **Verify pass** (all 24 layers): verifies K+1 tokens in one batched forward

```
Theoretical speedup:  4×12 + 24 = 72 layer-ops vs 5×24 = 120  →  ~1.67×
Empirical (nav text): ~1.3–1.5×
```

**Status**: Code complete, **not yet benchmarked on Jetson**. Benchmark is task 4.1 below.

```python
from src.vlm.pipelined_vlm import SelfSpeculativeVLM

spec = SelfSpeculativeVLM(vlm, split_layer=12, K=4)

# Streaming (yields tokens as confirmed)
gen = spec.generate_streaming(image_tokens, system_prompt, user_prompt)

# Non-streaming
text = spec.generate(image_tokens, system_prompt, user_prompt)

# Benchmark vs baseline
results = spec.benchmark_vs_baseline(image_tokens, system_prompt, user_prompt)
print(results["speedup"])  # float
```

### 2.3 VibeVoice TTS — Jetson Setup (vibevoice_test)

**Fully working on Jetson** as of 2026-03-29. Generates speech via CUDA GPU in float32.

#### Five required compat patches (apply before any vibevoice import):

```python
# Patch 1 — FlashAttentionKwargs missing in transformers 4.46.3
import transformers.modeling_flash_attention_utils as _m
if not hasattr(_m, 'FlashAttentionKwargs'):
    class FlashAttentionKwargs(dict): pass
    _m.FlashAttentionKwargs = FlashAttentionKwargs

# Patch 2 — BaseStreamer not re-exported at generation level
import transformers.generation as _gen
if not hasattr(_gen, 'BaseStreamer'):
    from transformers.generation.streamers import BaseStreamer as _BS
    _gen.BaseStreamer = _BS

# Patch 3 — _prepare_generation_config gets extra bool positional arg
from transformers import GenerationMixin as _GM
_orig = _GM._prepare_generation_config
def _pgc_compat(self, generation_config=None, *args, **kwargs):
    return _orig(self, generation_config, **kwargs)
_GM._prepare_generation_config = _pgc_compat

# After model.load():
model.eval()
model = model.to(torch.float32)   # float16 crashes Jetson CUDA eager attention

# Patch 4 — speech_start_id leaks into language_model.forward kwargs
# Patch 5 — verbose leaks into tts_language_model.forward kwargs
import inspect, types
def _filtered_fwd(orig):
    accepted = set(inspect.signature(orig).parameters.keys())
    def _f(*a, **kw): return orig(*a, **{k: v for k,v in kw.items() if k in accepted})
    return _f
model.model.language_model.forward = _filtered_fwd(model.model.language_model.forward)
model.model.tts_language_model.forward = _filtered_fwd(model.model.tts_language_model.forward)
```

#### Confirmed output (run on Jetson 2026-03-29):

```
Verdict:     no
Spoken:      "Path looks clear. Proceed with care."
Audio:       2.53s WAV file
VLM latency: 781 ms
TTS TTFA:    478 ms   ← time to first audio chunk
TTS total:   7.7 s    ← full audio generation
E2E (cold):  37 s     ← includes all model loads
```

### 2.4 TTS Streaming Bridge (`src/tts/streaming_bridge.py`)

Two classes designed for the warm-start production pipeline:

**`VibeVoiceTTSService`** — load once, stream many:
```python
svc = VibeVoiceTTSService(model_path=..., voices_dir=..., device="cuda", inference_steps=5)
svc.load()
for chunk in svc.stream("Obstacle ahead, move right."):
    sounddevice.play(chunk, samplerate=24000, blocking=True)
```

**`WordBufferedTTSBridge`** — connects to speculative VLM streaming output:
```python
bridge = WordBufferedTTSBridge(tts_service=svc, word_threshold=3)
bridge.start()
gen = spec.generate_streaming(image_tokens, system_prompt, user_prompt)
for chunk, _ in gen:
    bridge.feed(chunk)
bridge.flush()
events = bridge.events   # BridgeEvents with TTFT, TTFA, e2e_ms
```

### 2.5 Minimal `diffusers` Stub (Jetson only)

Real diffusers is broken on Jetson (v0.36 uses `torch.xpu`; v0.21 uses removed `cached_download`). A hand-written 6-file stub lives at:

```
/home/comp4901d/vibevoice_test/.venv/lib/python3.8/site-packages/diffusers/
  __init__.py
  configuration_utils.py      # ConfigMixin, register_to_config, FrozenDict (with __getattr__)
  utils/__init__.py            # deprecate (no-op), randn_tensor
  utils/torch_utils.py        # randn_tensor implementation
  schedulers/__init__.py
  schedulers/scheduling_utils.py  # KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput
```

`FrozenDict.__getattr__` is required because `dpm_solver.py` accesses config as `self.config.lambda_min_clipped`.

---

## 3. Midterm Tasks — Priority Ordered

### 🔴 P1 — Must Have (blocks demo)

#### Task 1.1 — Warm-start persistent pipeline

**Problem**: Cold-start E2E is 37 s (loads SigLIP + Qwen + VibeVoice fresh). In a real demo, models must stay loaded.
**Goal**: Single Python process loads all models once, then loops accepting images.
**File to create/modify**: `scripts/run_pipelined.py` or new `scripts/demo_loop.py`

```python
# Pseudocode for target behaviour
models = load_all_models()          # done once, ~30s
while True:
    image_path = input("Image path: ")
    result = pipeline(models, image_path)
    print(f"Verdict: {result.verdict}  E2E: {result.e2e_ms} ms")
```

**Acceptance criteria**: Second and subsequent inferences run at hot-path speed (~1.3 s E2E).

---

#### Task 1.2 — Real-time audio playback (not just WAV save)

**Problem**: `run_pipeline.py` currently saves to `.wav` only.
**Goal**: Audio plays through Jetson's audio output as TTS generates chunks (streaming playback).

```python
import sounddevice as sd
for chunk in tts_service.stream(phrase):
    sd.play(chunk, samplerate=24000, blocking=True)
```

**File**: `vibevoice_test/run_pipeline.py` and `scripts/run_pipelined.py`
**Note**: `sounddevice` is already installed in `vibevoice_test/.venv`. On Jetson, verify audio device with `python -c "import sounddevice; print(sounddevice.query_devices())"`.

---

#### Task 1.3 — Move vibevoice_test code into main VQA repo

**Problem**: Compat patches + Jetson pipeline live in `~/vibevoice_test/` which is not version-controlled.
**Goal**: Bring all production Jetson code into `VQA_Project-COMP4901D-`.

Steps:
1. Copy `/home/comp4901d/vibevoice_test/run_pipeline.py` → `scripts/run_jetson_pipeline.py` in repo
2. Copy `/home/comp4901d/vibevoice_test/tts_test.py` → `scripts/tts_smoke_test.py`
3. Copy the `diffusers/` stub → `src/tts/diffusers_stub/` with a `README` explaining why it exists
4. Ensure all compat patches live in `src/tts/__init__.py` under `def apply_jetson_patches()`
5. Update `AGENTS.md` with the new paths

---

### 🟡 P2 — Performance (to hit ≤300 ms)

#### Task 2.1 — Benchmark PPSD on Jetson

**Goal**: Run `SelfSpeculativeVLM.benchmark_vs_baseline()` on all 16 eval images, record:
- Baseline VLM latency (ms)
- Speculative VLM latency (ms)
- Acceptance rate (fraction of draft tokens kept)
- Actual speedup

**Command** (once Task 1.3 is done):
```bash
source ~/VQA_Project-COMP4901D-/.venv/bin/activate
cd ~/VQA_Project-COMP4901D-
python scripts/run_pipelined.py --labels data/eval/labels.json --no_tts --benchmark_spec
```

**Target**: ≥1.3× speedup at ≥50% acceptance rate for short navigation sentences.

---

#### Task 2.2 — Tune TTS inference steps

**Current**: 5 diffusion steps (TTS total = 7.7 s, TTFA = 478 ms)
**Goal**: Find minimum steps that keep audio intelligible. Try 3 and 4 steps.

```python
# In tts_test.py, change:
svc = VibeVoiceTTSService(..., inference_steps=3)  # try 3, 4
# Measure TTFA and listen to output quality
```

**Acceptance**: TTFA < 350 ms, speech clearly intelligible.

---

#### Task 2.3 — Memory overlap: VLM + TTS in same process

**Current**: `run_pipeline.py` frees VQA models (`del encoder, vlm; torch.cuda.empty_cache()`) before loading TTS — sequential, not overlapped.
**Goal**: Profile whether all models fit simultaneously:
- SigLIP encoder: ~400 MB float32
- Qwen2.5-0.5B: ~1.1 GB float32
- VibeVoice-0.5B: ~1.9 GB float32
- **Total**: ~3.4 GB — Jetson has 15 GB unified memory → should fit

If they fit, remove the `del` / `empty_cache()` step so the warm pipeline doesn't need to reload TTS between inferences.

---

#### Task 2.4 — Measure true hot-path E2E latency

Once Tasks 1.1 + 2.3 are done, run 10 consecutive inferences and record:

| Metric | Target |
|--------|--------|
| VLM TTFT | ≤ 120 ms |
| TTS TTFA | ≤ 180 ms |
| **E2E → first audio** | **≤ 300 ms** |
| E2E total (full phrase) | ≤ 2000 ms |

Write results to `reports/jetson_latency.md`.

---

### 🟢 P3 — Eval & Polish (for proposal quality)

#### Task 3.1 — Full accuracy benchmark with latency

Run the 16-image eval suite and produce a combined accuracy + latency report:

```bash
python scripts/run_pipelined.py \
    --labels data/eval/labels.json \
    --tts microsoft/VibeVoice-Realtime-0.5B \
    --voices_dir /home/comp4901d/vibevoice_test/voices \
    --no_play --output_report reports/midterm_benchmark.md
```

Report format:
```
Task            | Accuracy | VLM p50 (ms) | TTS TTFA p50 (ms) | E2E p50 (ms)
crosswalk_signal|   4/5    |     790      |        490        |     1280
stairs          |   5/5    |     810      |        460        |     1270
obstacles       |   5/6    |     800      |        500        |     1300
```

---

#### Task 3.2 — AGENTS.md update for TTS track

Add a new section to `AGENTS.md` covering:
- `vibevoice_test/` venv structure and activation
- The 5 compat patches and why each exists
- `diffusers` stub explanation
- How to run the integrated pipeline (`run_pipeline.py`)
- Known gotchas (float32 required, no `generate_streaming`, no `attn_implementation='sdpa'`)

---

#### Task 3.3 — Demo script for proposal presentation

Create `scripts/demo.py` — single command interactive demo:

```bash
python scripts/demo.py \
    --mode interactive \
    --tts /home/comp4901d/vibevoice_test/model \
    --voices /home/comp4901d/vibevoice_test/voices
```

Behaviour:
- Loads all models once (shows loading time)
- Prompts `"Image path (or 'q' to quit): "`
- For each image: prints verdict, plays audio, prints latency table
- Shows running average latency across all queries

---

#### Task 3.4 — Ibrahim / Hashim merge sync

When Ibrahim pushes updates to `main`, merge into `tts-implementation`:
```bash
git fetch origin
git merge origin/main --no-ff -m "merge: sync vision updates from main"
```

Potential conflict points:
- `src/vlm/model.py` — if Ibrahim changes `SimplePrefixVLM.generate()` signature
- `src/vision/token_compression.py` — if compression API changes, update `SelfSpeculativeVLM._build_prefix_embeds()`
- `data/eval/labels.json` — if more eval images added

---

## 4. Latency Budget Reference

```
Image capture
    ↓  ~0 ms
SigLIP encode (729 tokens)
    ↓  ~50 ms
Token compression (81 tokens)
    ↓  ~5 ms
VLM first token (PPSD speculative)
    ↓  ~120 ms         ← WordBufferedTTSBridge triggers TTS here (after 3 words)
TTS first audio chunk
    ↓  ~130 ms
─────────────────────────────
E2E → FIRST SPOKEN WORD:  ~300 ms  ← TARGET
```

---

## 5. File Transfer (Windows ↔ Jetson)

**Why paramiko, not scp**: Windows doesn't ship `scp`; paramiko is already installed.

```python
import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('10.89.149.159', username='comp4901d', password='comp4901d')
sftp = ssh.open_sftp()

sftp.put('local_file.py', '/home/comp4901d/remote_file.py')   # upload
sftp.get('/home/comp4901d/output.wav', 'local_output.wav')    # download

sftp.close()
ssh.close()
```

---

## 6. Known Gotchas (don't repeat these mistakes)

| Issue | Root cause | Fix |
|-------|-----------|-----|
| `git clone` bus error (exit 135) on Jetson | OOM during clone | Transfer files via SFTP instead |
| `torch.xpu` AttributeError with diffusers 0.36 | Jetson torch 2.1.0 has no XPU | Use hand-written diffusers stub |
| `cached_download` ImportError with diffusers 0.21 | Removed from huggingface_hub 0.36 | Same — use stub |
| `FlashAttentionKwargs` ImportError | Added in transformers 4.50+ | Inject stub class (Patch 1) |
| `generate_streaming` AttributeError | Method doesn't exist, only `generate` | Use `model.generate()` with `AudioStreamer` |
| `attn_implementation='sdpa'` ImportError | Requires torch ≥ 2.1.1; Jetson has 2.1.0a | Remove this argument entirely |
| `expected scalar type Float but found Half` | float16 + Jetson CUDA eager attention | Load in `torch_dtype=torch.float32` + `.to(torch.float32)` after `eval()` |
| `mat1/mat2 dtype mismatch float/bfloat16` | Some checkpoint weights survive as bfloat16 | Explicit `model.to(torch.float32)` after `model.eval()` |
| `speech_start_id` unexpected kwarg | Speech IDs from model_kwargs leak into `language_model.forward` | Patch 4: filter kwargs to accepted params |
| `FrozenDict` has no attribute `lambda_min_clipped` | `dpm_solver.py` uses attribute-style access | Add `__getattr__` to FrozenDict stub |
| `pip install -e .` fails (pip 20.0.2) | Old pip doesn't support pyproject-only editable install | Use `.pth` file to add path instead |

---

## 7. Quick Task Checklist

```
Midterm Proposal Tasks
======================

P1 — Blocking
[ ] 1.1  Warm-start persistent pipeline (load once, loop)
[ ] 1.2  Real-time audio playback via sounddevice on Jetson
[ ] 1.3  Move vibevoice_test code into main repo + version control

P2 — Performance
[ ] 2.1  Benchmark PPSD speculative decoding on Jetson (16 images)
[ ] 2.2  Tune TTS inference steps (test 3 and 4 steps for faster TTFA)
[ ] 2.3  Profile all-models-in-memory (remove del/empty_cache if they fit)
[ ] 2.4  Measure true hot-path E2E latency, write reports/jetson_latency.md

P3 — Polish
[ ] 3.1  Full 16-image accuracy + latency benchmark report
[ ] 3.2  Update AGENTS.md with TTS track documentation
[ ] 3.3  Demo script (interactive, warm, prints latency table)
[ ] 3.4  Merge sync with Ibrahim's main branch updates
```
