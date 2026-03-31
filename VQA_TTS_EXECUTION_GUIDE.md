# VQA + TTS Pipeline Execution Guide

## Quick Start

### Step 1: Connect to Jetson (SSH)
```bash
ssh comp4901d@10.89.149.159
```

### Step 2: Navigate to Project
```bash
cd ~/VQA_Project-COMP4901D-
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Step 3: Run Canonical Integrated Pipeline

Recommended for demos (persistent model loading + unified schema):

```bash
python scripts/run_integrated.py \
  --labels data/eval/labels.json \
  --compression 192 \
  --profile sentence_demo_fast \
  --tts microsoft/VibeVoice-Realtime-0.5B \
  --voices-dir ~/vibevoice_test/voices \
  --strict-demo \
  --warmup-images 1 \
  --output-dir reports/integrated_demo
```

### Step 4: Legacy VQA Pipeline WITH TTS

**Option A: Basic TTS (with pyttsx3 fallback)**
```bash
python scripts/run_vqa_with_tts.py \
  --llm_mode fp16 \
  --enable_tts \
  --max_new_tokens 24 \
  --output reports/vqa_with_tts_results.json \
  --audio_dir reports/vqa_audio
```

By default, this now keeps one merged WAV file at `reports/vqa_audio/combined_tts.wav`.
To keep per-sample WAVs instead, add `--keep_per_sample_wavs`.

**Option B: Use VibeVoice TTS (if installed)**
```bash
python scripts/run_vqa_with_tts.py \
  --llm_mode fp16 \
  --enable_tts \
  --use_vibevoice \
  --max_new_tokens 24 \
  --output reports/vqa_with_tts_results.json \
  --audio_dir reports/vqa_audio
```

**Option C: Use AWQ Quantized Model + TTS**
```bash
python scripts/run_vqa_with_tts.py \
  --llm_mode awq \
  --llm quantized/qwen2p5_0p5b_awq_int4 \
  --enable_tts \
  --max_new_tokens 24 \
  --output reports/vqa_with_tts_awq_results.json \
  --audio_dir reports/vqa_audio_awq
```

### Step 5: Monitor Progress
The script will:
- Process 28 images (should take 5-15 minutes depending on TTS backend)
- Generate text responses
- Generate WAV files (24kHz, mono, 16-bit)
- Save JSON results with metrics

### Step 6: Collect Results
After the run completes:

```bash
# Check results JSON
ls -lh reports/vqa_with_tts_results.json

# Check audio files
ls -lh reports/vqa_audio/*.wav | head -5
echo "Total WAV files: $(ls reports/vqa_audio/*.wav 2>/dev/null | wc -l)"

# Quick stats
python -c "
import json
with open('reports/vqa_with_tts_results.json') as f:
    data = json.load(f)
    stats = data['statistics']
    print(f\"Accuracy: {stats['overall_accuracy']*100:.1f}% ({stats['correct_predictions']}/{stats['total_samples']})\")"
print(f\"E2E Latency: {stats['avg_e2e_latency_ms']:.1f}ms avg\")
print(f\"TTS Latency: {stats['avg_tts_latency_ms']:.1f}ms avg\")
```

## Step 7: Transfer WAV Files to Local Machine

## Step 8: One-Command Pre-Demo Validation

```bash
bash scripts/pre_demo_validate.sh
```

This runs:
- preflight checks
- text-only smoke run
- optional strict TTS smoke run (if voices are present)
- run manifest capture
- schema sanity check

## Step 9: Baseline Manifest Command

```bash
bash scripts/capture_baseline_manifest.sh
```

Optional explicit inputs:

```bash
bash scripts/capture_baseline_manifest.sh reports/vqa_with_tts_results.json reports/baseline_run_manifest.json
```

### On LOCAL machine (Windows):
```powershell
# Create local audio directory
mkdir reports\vqa_audio -Force

# Copy WAV files from Jetson using SCP
scp -r comp4901d@10.89.149.159:~/VQA_Project-COMP4901D-/reports/vqa_audio/*.wav reports\vqa_audio\

# Copy JSON results too
scp comp4901d@10.89.149.159:~/VQA_Project-COMP4901D-/reports/vqa_with_tts_results.json reports\
```

### On LOCAL machine (macOS/Linux):
```bash
mkdir -p reports/vqa_audio
scp -r comp4901d@10.89.149.159:~/VQA_Project-COMP4901D-/reports/vqa_audio/*.wav reports/vqa_audio/
scp comp4901d@10.89.149.159:~/VQA_Project-COMP4901D-/reports/vqa_with_tts_results.json reports/
```

## Output Structure

```
reports/vqa_with_tts_results.json     ← Main results file with all metrics
reports/vqa_audio/                    ← Directory with WAV files
  └─ Crosswalk_1.wav
  └─ Crosswalk_2.wav
  └─ Stairs_1.wav
  └─ Obstacle_1.wav
  ├─ ... (28 files total)
```

## Results JSON Structure

```json
{
  "timestamp": "2026-03-30 12:30:45",
  "device": "cuda",
  "run_config": {
    "llm_mode": "fp16",
    "compression": 192,
    "enable_tts": true,
    "use_vibevoice": false
  },
  "statistics": {
    "total_samples": 28,
    "overall_accuracy": 0.464,
    "correct_predictions": 13,
    "avg_vlm_latency_ms": 852.7,
    "avg_e2e_latency_ms": 1008.1,
    "avg_tts_latency_ms": 450.5
  },
  "results": [
    {
      "id": "Crosswalk_1",
      "image": "Crosswalk_1.png",
      "task": "crosswalk_signal",
      "response": "red",
      "pred": "red",
      "gt": "red",
      "correct": true,
      "vlm_ttft_ms": 300.5,
      "e2e_total_ms": 1050.2,
      "tts_ttfa_ms": 450.5,
      "tts_metrics": {
        "success": true,
        "wav_path": "reports/vqa_audio/Crosswalk_1.wav",
        "generation_ms": 450.5
      }
    },
    ...
  ]
}
```

## Troubleshooting

### TTS Not Available
If pyttsx3/VibeVoice fails:
- Script falls back to generating JSON without audio
- Check logs for specific errors
- Try installing: `pip install pyttsx3`

### Out of Memory
If you run into OOM:
- Use `awq` mode (more memory efficient)
- Reduce batch processing if parallelized
- Monitor with `nvidia-smi` on Jetson

### Slow Processing
- On Jetson, ~2-3 samples/minute with TTS is normal
- ~5-10 samples/minute without TTS
- Total runtime: 5-15 minutes for 28 samples

### Audio Quality Issues
- If using pyttsx3: speech may sound robotic but is useful for testing
- If using VibeVoice: requires model download on first run (~500MB)
- Adjust inference steps in code for quality vs. speed tradeoff

## Next: Analyze Results

After collecting files locally:

```python
# Quick analysis
import json
with open('reports/vqa_with_tts_results.json') as f:
    data = json.load(f)

# Count successful audio files
audio_success = [r for r in data['results'] if r.get('tts_metrics', {}).get('success')]
print(f"Audio files generated: {len(audio_success)}/{len(data['results'])}")

# Accuracy by task
for task in ['crosswalk_signal', 'stairs', 'obstacles']:
    task_results = [r for r in data['results'] if r['task'] == task]
    correct = sum(1 for r in task_results if r['correct'])
    print(f"{task}: {correct}/{len(task_results)} ({correct/len(task_results)*100:.1f}%)")
```

Or use the analysis script:
```bash
python analyze_results.py  # Works with both vqa_results.json and vqa_with_tts_results.json
```

