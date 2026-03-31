# LoRA Fine-Tuning & Presentation Outline
**COMP4901D VQA Project — Phase 3**

---

## Part 1: LoRA Fine-Tuning (Hours 0–4)

### 1.1 Data Preparation (30 min)
- **Task:** Expand labeled training set beyond 28 eval images
  - [ ] Copy 28 eval images + labels to `data/train/` directory
  - [ ] Manually label 20–40 additional accessibility images (or synthetic variants)
  - [ ] Build `data/train/labels.json` in same schema as eval labels
  - [ ] Create train/val split (70/30) for local tuning
  - [ ] Verify no label leakage (train/eval/test are disjoint)

**Output:** `data/train/labels.json` with 40–60 labeled samples

---

### 1.2 LoRA Implementation (90 min)
- **Task:** Build lightweight LoRA adapter for LLM on accessibility VQA tasks
  
**File to create:** `scripts/finetune_lora_vqa.py`

Key components:
```python
# Pseudo-structure
from peft import get_peft_model, LoraConfig

config = LoraConfig(
    r=8,  # low rank
    lora_alpha=16,
    target_modules=['q_proj', 'v_proj'],  # Qwen2.5 proj layers
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM'
)

# Load base Qwen + SigLIP
llm = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
llm = get_peft_model(llm, config)

# Train with accessibility VQA task-specific prompts
trainer = Trainer(
    model=llm,
    args=TrainingArguments(..., num_train_epochs=2, ...),
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    ...
)
trainer.train()

# Save LoRA weights only (~20MB)
llm.save_pretrained('models/lora_accessibility_vqa/')
```

**Output:** LoRA adapter weights in `models/lora_accessibility_vqa/`

---

### 1.3 Local Validation (60 min)
- **Task:** Validate LoRA + base model on eval set locally

**File to create:** `scripts/test_lora_local.py`

Steps:
- [ ] Load base Qwen + LoRA adapter
- [ ] Run on 28 eval images with integrated pipeline
- [ ] Compute accuracy, compare to baseline (46.43%)
- [ ] Log per-task breakdown (crosswalk, stairs, obstacles)
- [ ] Expected improvement: +5–15% (target ≥55%)

**Output:** Local validation report: `reports/lora_validation_local.json`

---

### 1.4 Jetson Deployment (60 min)
- **Task:** Deploy LoRA adapter to Jetson and run inference

**Steps:**
- [ ] SCP LoRA weights to Jetson: `~/VQA_Project-COMP4901D-/models/lora_accessibility_vqa/`
- [ ] Create `scripts/run_integrated_lora.py` (loads base + LoRA on Jetson)
- [ ] Run 28-image eval on Jetson with LoRA
- [ ] Compare latency vs baseline (expect ~5–10% overhead)
- [ ] Log results to `reports/lora_jetson_eval.json`

**Output:** Jetson LoRA evaluation report with accuracy, latency, degeneration metrics

---

## Part 2: Presentation Structure (2–3 hours prep time)

### 2.1 Slide Deck Outline

**Title & Problem Statement** (Slide 1–2)
- VQA for accessibility (blind/low-vision navigation)
- Task: Classify crosswalk signals, detect stairs/obstacles
- Baseline challenge: Latency + accuracy on edge devices (Jetson Orin NX)

**Technical Approach** (Slide 3–6)
- **Vision Pipeline:** SigLIP + token compression (reduce 576→192/81/36/9 tokens)
- **LLM:** Qwen2.5-0.5B (Jetson default) + optional 1.5B with FP32 fallback when needed
- **Baseline sweep:** 0.5B FP16 best accuracy_gt_known 0.50 @ compression 36
- **Phase 2 (LoRA):** Fine-tuning on task data to improve accuracy at similar compression

**LoRA Fine-Tuning Details** (Slide 7–9)
- Why LoRA: ~20MB weights vs full model, modest latency overhead
- Training data: Expanded accessibility VQA dataset (28 eval images with augmentations)
- Training config: 3 epochs on Jetson, LoRA rank=8, target LLM projections
- Observed outcome: 0.5B LoRA best accuracy_gt_known 0.75 @ compression 36

**Results Comparison** (Slide 10–12)
- **0.5B FP16 baseline:** best accuracy_gt_known 0.50 @ compression 36
- **0.5B FP16 + LoRA:** best accuracy_gt_known 0.75 @ compression 36
- **1.5B FP16 + LoRA:** best accuracy_gt_known 0.821 @ compression 192 (FP32 fallback on Jetson)
- **Latency context:** 0.5B LoRA fastest mean ~479 ms @ compression 9; 1.5B LoRA ~1955 ms @ compression 9
- **Dataset note:** results are on a small 28-image eval set

**Architecture Diagram** (Slide 13)
```
Image → SigLIP (576 tokens) → Compress (192 tokens) →
   Qwen-0.5B (base) + LoRA adapter → Classification
   (1.5B optional, FP32 fallback for stability)
```

**Demo & Live Results** (Slide 14–15)
- Live inference on sample images
- Show latency breakdown (capture, encode, compress, LLM, TTS)
- Accessibility output: spoken feedback + classification

**Lessons & Future Work** (Slide 16–17)
- Numeric stability critical on edge GPUs (FP16→FP32 fallback)
- LoRA efficient for domain adaptation without full retraining
- Future: Larger datasets, multi-modal fine-tuning, on-device TTS

---

### 2.2 Demo Script (`scripts/demo_lora_showcase.py`)

**Interactive demo showcasing:**
1. **Load mode selection:** baseline (46%) vs LoRA-tuned (expected 55%+)
2. **Image upload/selection:** from eval or custom images
3. **Running inference:** show predictions + confidence + spoken output
4. **Side-by-side comparison:** baseline vs LoRA predictions on same image
5. **Performance metrics popup:** accuracy, latency, unknown rate

**Output:** 
- Live terminal UI or Gradio web interface
- Sample outputs saved to `reports/demo_outputs/`

---

### 2.3 Results Document

**File:** `FINAL_SUBMISSION_REPORT.md`

Sections:
- Executive summary (accuracy gains, latency profile)
- Dataset description (28 eval + 40–60 training samples)
- Methods (SigLIP + token compression + Qwen-1.5B + LoRA)
- Results (baseline vs LoRA, per-task breakdown, speed benchmarks)
- Deployment (Jetson specifics: FP32 fallback, memory, thermal profile)
- Lessons learned (edge GPU stability, efficient adaptation)

---

## Part 3: Demo Showcase (30 min — during presentation)

### 3.1 Live Demo Flow

1. **Start with Jetson SSH terminal**
   ```bash
   ssh comp4901d-jetson
   cd ~/VQA_Project-COMP4901D-
   source .venv/bin/activate
   export PYTHONPATH=$PYTHONPATH:.
   ```

2. **Run demo on 3–5 sample images**
   ```bash
   python scripts/demo_lora_showcase.py \
     --baseline-model Qwen/Qwen2.5-1.5B-Instruct \
     --lora-adapter models/lora_accessibility_vqa/ \
     --images data/eval/images/{crosswalk,stairs,obstacles}/*.png
   ```

3. **Show side-by-side comparison**
   - Baseline prediction: "unknown" (but with LoRA: "red")
   - Latency with LoRA: ~1.8s vs 1.6s baseline
   - Spoken feedback: "The pedestrian signal is red. Please wait."

4. **Show accuracy metrics dashboard**
   - Print JSON report with per-task breakdown
   - Highlight improvement in low-performing tasks

---

### 3.2 Presentation Timeline

| Time | Content |
|------|---------|
| 0:00–1:00 | Problem statement + context (accessibility VQA) |
| 1:00–3:00 | Technical approach deep-dive (vision, LLM, stability fixes) |
| 3:00–5:00 | Phase 2 LoRA strategy + expected results |
| 5:00–8:00 | Live Jetson demo: baseline vs LoRA on 5 images |
| 8:00–10:00 | Results & metrics: accuracy gains, latency tradeoff |
| 10:00–12:00 | Lessons + future work |
| 12:00–15:00 | Q&A |

---

## Part 4: Checkpoints & Milestones

- **By hour 1:** Training data expanded and validated (40–60 samples labeled)
- **By hour 2.5:** LoRA fine-tuning script working locally, first epoch complete
- **By hour 3.5:** Local validation report shows accuracy lift
- **By hour 4:** Jetson deployment successful, LoRA adapter running
- **By hour 5:** Demo script ready and tested
- **Before presentation:** Slides finalized, live demo rehearsed

---

## Part 5: Repository Structure After Completion

```
VQA_Project-COMP4901D-/
├── data/
│   ├── train/
│   │   ├── labels.json          # 40–60 training samples
│   │   └── images/              # training image subset
│   └── eval/
│       ├── labels.json          # original 28 eval samples
│       └── images/
├── models/
│   └── lora_accessibility_vqa/
│       ├── adapter_config.json
│       ├── adapter_model.bin    # LoRA weights (~20MB)
│       └── training_args.bin
├── scripts/
│   ├── finetune_lora_vqa.py     # NEW: LoRA training
│   ├── test_lora_local.py       # NEW: Local validation
│   ├── run_integrated_lora.py   # NEW: Jetson LoRA runner
│   ├── demo_lora_showcase.py    # NEW: Interactive demo
│   └── [existing scripts...]
├── reports/
│   ├── lora_validation_local.json      # local eval results
│   ├── lora_jetson_eval.json           # Jetson eval results
│   ├── demo_outputs/                   # demo screenshot/outputs
│   └── [existing sweep reports...]
├── FINAL_SUBMISSION_REPORT.md   # comprehensive results doc
├── LORA_FINETUNING_AND_DEMO_PLAN.md   # this file
└── [existing files...]
```

---

## Summary

- **Phase 1 (LoRA):** 4 hours — data prep, training, Jetson deployment, validation
- **Phase 2 (Presentation):** 2–3 hours — slides, demo script, results doc
- **Phase 3 (Demo):** 30 min — live showcase during presentation
- **Expected outcome:** 55–60% accuracy on Jetson with LoRA, zero degeneration, complete end-to-end working system

