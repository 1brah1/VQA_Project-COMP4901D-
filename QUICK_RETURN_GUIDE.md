# Quick Return Guide — LoRA Training Status

**Training Status:** ⏳ Running in background (Terminal: `6f08b253-1078-4978-966a-a63c45c776f6`)

---

## ✅ What's Done

1. **Data Prep** — 105 training + 35 validation samples created ✓
2. **Dependencies** — PEFT, PyTorch, transformers installed ✓  
3. **Scripts** — 4 new implementations created and tested ✓
4. **Demo Tool** — `scripts/demo_lora_showcase.py` ready for side-by-side comparison ✓
5. **SSH** — Jetson connectivity verified with `id_ed25519_jetson` key ✓

---

## 🔄 What's Running Now

```
Training Phase 2: LoRA Fine-Tuning
├─ Model: Qwen 1.5B (FP16) with LoRA (rank=8)
├─ Trainable: 1.089M params (0.0705%)
├─ Data: 105 train + 35 val images
├─ Progress: Image encoding → training batches
└─ Expected duration: 90–120 min total
```

Terminal ID: `6f08b253-1078-4978-966a-a63c45c776f6`

---

## 📋 Next Steps (When Training Completes)

### Step 1: Check Training Completed (2 min)
```powershell
# Check if adapter was saved
Test-Path models/lora_accessibility_vqa/adapter_model.bin
# Should return: True
```

### Step 2: Validate Locally on Eval Set (10 min)
```bash
python scripts/test_lora_local.py --use-lora
# Output: reports/lora_validation_local.json
# Then analyze results:
python scripts/analyze_lora_results.py
# Success criteria: accuracy >= 0.55 (55%), bang_rate = 0.0
```

### Step 3: Demo Side-by-Side Comparison (5 min)
```bash
python scripts/demo_lora_showcase.py --samples 10
# Shows baseline vs LoRA predictions on 10 random images
# Look for changes in predictions (improvement indicators)
```

### Step 4: Deploy to Jetson (If local validation passes, 20–30 min)
```bash
# Option A: Direct transfer (faster)
scp -i ~/.ssh/id_ed25519_jetson -r models/lora_accessibility_vqa comp4901d@<jetson-ip>:~/VQA_Project-COMP4901D-/models/

# Option B: Verify first
python scripts/run_integrated_lora.py --labels data/eval/labels.json --compression 192 --lora-adapter models/lora_accessibility_vqa --max-new-tokens 12
# This will show Jetson simulation results locally
```

### Step 5: Full Jetson Eval (If needed, 30–60 min)
```bash
ssh -i ~/.ssh/id_ed25519_jetson comp4901d@<jetson-ip> << 'EOF'
cd ~/VQA_Project-COMP4901D-
source .venv/bin/activate
python scripts/run_integrated_lora.py --labels data/eval/labels.json --compression 192 --lora-adapter models/lora_accessibility_vqa --max-new-tokens 12 --out reports/lora_jetson_eval.json
EOF
```

---

## ✅ Success Criteria

| Phase | Target | Success | Location |
|-------|--------|---------|----------|
| Local Validation | ≥55% accuracy | 0 degeneration | `reports/lora_validation_local.json` |
| Demo | Show improvement | Changes in predictions | stdout |
| Jetson Deploy | ≥53% accuracy (allow 2% drop) | Complete without errors | `reports/lora_jetson_eval.json` |

---

## 🐛 If Training Failed

**Check these in order:**

1. **Terminal is still running?**
   ```powershell
   Get-Process | grep python
   ```

2. **GPU memory issue?**
   - Check `nvidia-smi` for memory
   - Restart with: `python scripts/finetune_lora_vqa.py --epochs 2 --batch-size 1 --output-dir models/lora_accessibility_vqa`

3. **Encoder error?**
   - Already fixed in script (uses `.from_pretrained()` with device/dtype)
   - If error, see `LORA_EXECUTION_STATUS.md` troubleshooting section

4. **Compression error?**
   - Already fixed (uses `target_tokens=192`)
   - Check: `src/vision/token_compression.py` function signature

---

## 📊 Key Results Files (After Success)

- **Local Validation:** `reports/lora_validation_local.json` (per-task accuracy)
- **Jetson Validation:** `reports/lora_jetson_eval.json` (if deployed)
- **LoRA Weights:** `models/lora_accessibility_vqa/` (adapter_model.bin + config)

---

## 🎤 Demo Ready

Pre-built demo script is ready:

```bash
# Show side-by-side comparison of 10 random images
python scripts/demo_lora_showcase.py --samples 10

# Or filter by task
python scripts/demo_lora_showcase.py --task crosswalk_signal --samples 5

# With custom compression
python scripts/demo_lora_showcase.py --compression 192 --samples 15
```

---

## 📞 Critical Terminal ID Reference

- **Training terminal:** `6f08b253-1078-4978-966a-a63c45c776f6`
- Check progress: `Get-Terminal-Output 6f08b253-1078-4978-966a-a63c45c776f6`
- Kill if needed: Powers user's decision only

---

**Last Updated:** Autonomous execution in progress
**Expected Completion:** ~90–120 minutes from training start
