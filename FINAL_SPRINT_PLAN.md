# COMP4901D VQA Project — Final Sprint Plan (2 Weeks)
**April 14 – April 28, 2026 → Final submission May 8**

---

## Context

The midterm is complete. Results: 28/28 classifications, Piper TTS, 39.29% accuracy, 627ms E2E on Jetson Orin NX. LoRA results (0.75 acc for 0.5B+LoRA) were shown in slides but the training code was never committed to GitHub. PPSD was listed as "future work."

The final sprint must deliver: a complete, clean GitHub repo + live Jetson demo + final report + updated slides.

**Key gap areas:**
- LoRA training code not in repo
- PPSD not benchmarked/integrated
- Only 28 eval images (need 200+)
- Dynamic compression switching not implemented
- No warm-start demo script
- Final report not written

**Split**: Ibrahim owns all technical implementation. Hashim owns content, data, and documentation. Hashim will brief Ibrahim on each technical task before he starts.

---

## Ibrahim — All Technical Implementation

| # | Task | Priority | Days |
|---|------|----------|------|
| I1 | Merge `origin/main` into `tts-implementation`, resolve conflicts in `src/vlm/model.py` | P1 | Apr 14 |
| I2 | Benchmark PPSD on Jetson: run `SelfSpeculativeVLM.benchmark_vs_baseline()` across all eval images, record speedup + acceptance rate → `reports/ppsd_benchmark.md` | P1 | Apr 14–16 |
| I3 | Commit LoRA training code: clean up local scripts → `scripts/train_lora.py`, document hyperparams (rank, LR, epochs, dataset used) | P1 | Apr 16–17 |
| I4 | Commit LoRA inference integration: add `src/vlm/lora_vlm.py` to load adapter at inference; update `scripts/run_integrated.py` with `--lora-adapter` flag | P1 | Apr 17–18 |
| I5 | Warm-start demo script: `scripts/demo.py` — loads all models once, loops on input images, prints latency table, plays Piper TTS audio | P1 | Apr 18–20 |
| I6 | Real-time audio playback via sounddevice on Jetson (stream chunks as Piper generates, not save-then-play) | P1 | Apr 20–21 |
| I7 | Dynamic compression selector: `src/vision/compression_selector.py` — auto-picks ratio (576/192/81/36/9) based on a `--latency-budget` ms parameter | P2 | Apr 21–23 |
| I8 | OmniVLM-style adaptive compression: `src/vision/adaptive_compression.py` — saliency-based pooling (keep high-variance patches, discard background) | P2 | Apr 21–23 |
| I9 | Hot-path latency measurement: 10+ consecutive inferences, write `reports/jetson_latency_final.md` with p50/p95 table | P2 | Apr 23–24 |
| I10 | Full accuracy benchmark with 200-image dataset across 0.5B FP16, 0.5B+LoRA, compressions 576/192/81/36/9 → `reports/final_benchmark.md` | P2 | Apr 24–25 |
| I11 | Update `AGENTS.md` with Piper TTS section (replace VibeVoice references, document Piper → Silero → pyttsx3 fallback chain) | P3 | Apr 25 |
| I12 | GitHub cleanup: update README with final results, remove stale TODOs, ensure all scripts are documented | P3 | Apr 28 |

---

## Hashim — Content, Data & Documentation

| # | Task | Priority | Days |
|---|------|----------|------|
| H1 | **Image collection** — Collect 200+ images (real photos + AI-generated, see guide below). Sort into correct folders. **Deliver by Apr 20** so Ibrahim can run benchmarks | P1 | Apr 14–20 |
| H2 | **Label all new images** — Update `data/eval/labels.json` with correct labels for every new image (Claude will assist with labeling) | P1 | Apr 17–20 |
| H3 | **Brief Ibrahim** on each technical task before he starts — walk through the task description and make sure he understands what to implement | P1 | Ongoing |
| H4 | **Report writing** — Write motivation, problem statement, and related work sections of the final report (2–3 pages) | P2 | Apr 20–24 |
| H5 | **Updated slides** — Add PPSD results (from I2), LoRA ablations, 200-image accuracy results, demo screenshots to the midterm deck. Ibrahim supplies the numbers | P2 | Apr 24–27 |
| H6 | **Demo screenshots/video** — Record the Jetson running `demo.py` on 5–10 images, capture terminal output + audio. Use for slides and report | P3 | Apr 25–27 |
| H7 | **Final report assembly** — Combine Hashim's written sections with Ibrahim's technical sections, proofread, format for submission | P3 | Apr 26–28 |

---

## Dynamic Compression Switching — Design (for Ibrahim, Task I7)

**File to create**: `src/vision/compression_selector.py`

```python
class CompressionSelector:
    RATIOS = [576, 192, 81, 36, 9]  # tokens, highest to lowest quality

    def select(self, latency_budget_ms: int, profiled_latencies: dict) -> int:
        """Return the highest token count that fits within the latency budget."""
        for tokens in self.RATIOS:
            if profiled_latencies.get(tokens, 9999) <= latency_budget_ms:
                return tokens
        return 9  # minimum fallback
```

**Integration**: Add `--latency-budget <ms>` flag to `scripts/run_integrated.py`. On first run, profile all ratios; on subsequent runs, pick the optimal ratio automatically.

---

## Image Generation Guide (for Hashim — Task H1)

Target: **200+ labeled images** across 3 categories. Mix real photos with AI-generated.

### Step 1 — Real photos (highest quality)
Photograph with your phone:
- **Crosswalk signals**: 25+ shots of red/green walk signals — vary angle, lighting, time of day
- **Stairs**: 25+ shots — indoor, outdoor, looking up and down, with/without people
- **Obstacles**: 25+ shots — bikes, bins, construction barriers, furniture on pavement

### Step 2 — AI images via Bing Image Creator (free, no login)
Go to **https://www.bing.com/images/create**, generate 10–15 per prompt:

| Category | Prompt |
|----------|--------|
| Crosswalk — Red | `Pedestrian crosswalk signal showing red hand "don't walk" symbol, street level perspective, realistic photo, daytime, urban setting` |
| Crosswalk — Green | `Pedestrian crosswalk signal showing green walking figure "walk" symbol, street level, realistic photo, daytime, city sidewalk` |
| Stairs — Present | `Indoor staircase going up, first-person perspective from ground level, realistic photo, office building` |
| Stairs — Not Present | `Flat sidewalk or corridor at eye level, no stairs, realistic photo, pedestrian walking view` |
| Obstacle — Present | `Sidewalk blocked by bicycle parked in path, first-person pedestrian perspective, realistic photo, urban street` |
| Obstacle — Not Present | `Clear empty sidewalk, first-person walking perspective, no obstacles, realistic photo, daytime` |

### Step 3 — File naming
```
data/eval/images/crosswalk/Crosswalk_29.png, Crosswalk_30.png, ...
data/eval/images/stairs/Stairs_6.png, Stairs_7.png, ...
data/eval/images/obstacles/Obstacle_17.png, Obstacle_18.png, ...
```

### Step 4 — labels.json entry format
```json
{
  "image_id": "Crosswalk_29",
  "path": "data/eval/images/crosswalk/Crosswalk_29.png",
  "task": "crosswalk_signal",
  "label": "red"
}
```

Valid labels:
- crosswalk → `"red"` or `"green"`
- stairs → `"yes"` or `"no"`
- obstacles → `"yes"` or `"no"`

**Target**: ~70 crosswalk + ~70 stairs + ~70 obstacles = **210 total**

---

## New Files Ibrahim Must Create

| File | Purpose |
|------|---------|
| `scripts/demo.py` | Warm-start interactive demo |
| `scripts/train_lora.py` | LoRA training code (from local) |
| `src/vlm/lora_vlm.py` | LoRA adapter loading at inference |
| `src/vision/compression_selector.py` | On-the-fly compression switching |
| `src/vision/adaptive_compression.py` | OmniVLM-style saliency pooling |
| `reports/ppsd_benchmark.md` | PPSD speedup results on Jetson |
| `reports/jetson_latency_final.md` | Hot-path p50/p95 latency table |
| `reports/final_benchmark.md` | Full 200-image accuracy + latency |

---

## Latency Targets

| Metric | Midterm Result | Final Target |
|--------|---------------|--------------|
| E2E hot (first audio) | 627 ms | ≤ 500 ms |
| VLM TTFT | ~780 ms | ≤ 300 ms (with PPSD) |
| Accuracy (0.5B+LoRA, c=192) | 75% on 28 images | ≥ 70% on 200+ images |
| Dynamic compression | Not implemented | Working `--latency-budget` flag |

---

## Week-by-Week Summary

**Week 1 (Apr 14–21) — Build**
- Ibrahim: Merge → PPSD benchmark → LoRA commit → warm demo script → real-time audio
- Hashim: Collect + label 200+ images; brief Ibrahim daily

**Week 2 (Apr 22–28) — Measure + Polish**
- Ibrahim: Dynamic compression → latency measurements → full benchmark → GitHub cleanup
- Hashim: Write report → update slides → record demo video → assemble final report

---

## Verification Checklist

- [ ] `python scripts/demo.py --mode interactive` loads once, processes 5+ images without reloading models
- [ ] PPSD benchmark shows ≥ 1.3× speedup on Jetson
- [ ] `python scripts/run_integrated.py --lora-adapter models/lora_adapter/` runs and shows accuracy gain
- [ ] `--latency-budget 200` selects 9-token compression; `--latency-budget 500` selects 192-token
- [ ] 200+ images in `data/eval/`, all entries in `labels.json` with correct labels
- [ ] Full 200-image benchmark shows overall accuracy ≥ 65%
- [ ] All scripts in README run successfully from a fresh Jetson clone
