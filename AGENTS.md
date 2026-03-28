# AGENTS.md — Jetson & repository guide (COMP4901D)

This file helps **human developers** and **coding agents** work safely on the **NVIDIA Jetson Orin NX** (16 GB) deployment of this repo. It documents **environment constraints**, **layout**, and **known failures**.

---

## 1. Target device

- **Hardware**: Jetson Orin NX (team kit: 16 GB unified memory is typical for this class of device).
- **OS**: Linux for Tegra (L4T). Exact version may vary; Python is often **3.8.x**.
- **Access**: SSH from the lab network (address and credentials are **not** stored in this repo). Use SSH keys when possible.
- **Working copy path** (typical): `~/VQA_Project-COMP4901D-`

---

## 2. Python and PyTorch on Jetson

- **Python**: Expect **3.8.10** (or similar). Do **not** use syntax that requires 3.10+:
  - No `str | int` type unions in code that must run on Jetson; use `typing.Union` / `Optional`.
  - Avoid runtime `list[int]` variable annotations in function bodies on 3.8; prefer plain lists or quoted annotations.
- **PyTorch**: **Do not** `pip install torch` blindly from the default PyTorch index for Jetson wheels. Use **NVIDIA / Jetson-specific** instructions (JetPack-aligned wheels or `pip` URLs from NVIDIA docs for your L4T version).
- **`requirements-jetson.txt`**: Installs **transformers, accelerate, pillow, numpy, tqdm, psutil**. It **intentionally omits** `torch` and `torchvision` so you do not overwrite a good Jetson build.

---

## 3. Repository structure (on disk)

```
VQA_Project-COMP4901D-/
  src/
    vision/          # SigLIP encode, token compression
    vlm/             # Prefix VLM, LLM loaders
    prompts/         # system_prompt.txt
  data/eval/
    labels.json      # eval manifest
    images/          # crosswalk, stairs, obstacles PNGs
  scripts/
    run_once.py
    benchmark_compression.py
    jetson_run_benchmark.sh
    jetson_quantize_llm_awq.sh
    quantize_llm_awq.py
  reports/           # created by benchmarks (gitignored if so configured)
  requirements.txt
  requirements-jetson.txt
  requirements-quant.txt
```

---

## 4. Standard setup on Jetson

```bash
cd ~/VQA_Project-COMP4901D-
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install torch/torchvision per NVIDIA Jetson instructions first, if not already.
pip install -r requirements-jetson.txt
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

## 5. Running inference once

```bash
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.

python scripts/run_once.py \
  --image-path data/eval/images/crosswalk/Crosswalk_2.png \
  --task crosswalk_signal \
  --compression 192 \
  --max_new_tokens 24
```

**Compression**: Must be in `recommended_targets` for your SigLIP output size (usually **576** or **729** tokens before compression). Invalid values raise a clear error listing valid targets.

---

## 6. Running the full benchmark

```bash
bash scripts/jetson_run_benchmark.sh
```

Or manually:

```bash
python scripts/benchmark_compression.py \
  --labels data/eval/labels.json \
  --llm_mode fp16 \
  --max_new_tokens_eval 12 \
  --out reports/fp16_benchmark.json
```

Outputs:

- `reports/fp16_benchmark.json` — per-compression metrics, **`by_task`** block, timing, `n_gen_tokens`.
- If `quantized/qwen2p5_0p5b_awq_int4` (or path in script) exists, AWQ sweep may also run.

---

## 7. Quantized LLM (AWQ)

The Jetson helper **skips** the AWQ benchmark until `quantized/qwen2p5_0p5b_awq_int4` exists (or `LLM_AWQ_DIR` points at a real folder). That is expected on a fresh clone.

**Create the folder using one of:**

1. **Jetson, one shot** (quantizes then benchmarks if you use the env var):
   ```bash
   QUANTIZE_AWQ_FIRST=1 bash scripts/jetson_run_benchmark.sh
   ```
2. **Jetson, quantize only**:
   ```bash
   bash scripts/jetson_quantize_llm_awq.sh
   ```
3. **PC / server with CUDA**, then copy to Jetson:
   ```bash
   pip install -r requirements-quant.txt
   python scripts/quantize_llm_awq.py --out quantized/qwen2p5_0p5b_awq_int4
   scp -r quantized/qwen2p5_0p5b_awq_int4 comp4901d@<jetson-ip>:~/VQA_Project-COMP4901D-/quantized/
   ```

If `pip install autoawq` fails on aarch64, use option 3 or build wheels per your JetPack docs.

- Pass the **quantized model directory** as `--llm` to `benchmark_compression.py` when using `--llm_mode awq`.

---

## 8. Shell scripts and Windows checkouts

Scripts under `scripts/jetson_*.sh` must be **LF-only** (Unix newlines). If bash reports:

```text
set: pipefail: invalid option
```

your files likely have **CRLF** (Windows). On the Jetson, fix in place:

```bash
sed -i 's/\r$//' scripts/jetson_run_benchmark.sh scripts/jetson_quantize_llm_awq.sh
```

Or install `dos2unix` and run it on those files. The repo’s `.gitattributes` keeps `*.sh` as `eol=lf` for future clones.

---

## 9. Known pitfalls (Jetson + this codebase)

| Issue | Mitigation |
|-------|------------|
| `ModuleNotFoundError: torch._C._distributed_c10d` during `generate()` | Fixed in `SimplePrefixVLM` via `synced_gpus=False` in `llm.generate`. Sync repo `src/vlm/model.py`. |
| `Expected 729 tokens; got 576` | Use current `token_compression.py` + valid `--compression` for 576-token grids. |
| Huge LLM latency with garbage text | Use `--max_new_tokens_eval` (e.g. 12) in benchmarks; inspect `n_gen_tokens` in JSON. |
| `torch.cuda.empty_cache()` every sample | Trades memory for speed; consider removing for max throughput benchmarks only. |
| Wrong `PYTHONPATH` | `ModuleNotFoundError: src` → export `PYTHONPATH` to repo root. |

---

## 10. Power / thermal (optional profiling)

For course reports you may log **power draw** while running a fixed benchmark loop, e.g.:

```bash
sudo tegrastats
```

Interpret averages with caution: numbers depend on power mode (`nvpmodel`) and cooling.

---

## 11. Security

- **Never** commit SSH passwords, API keys, or classroom machine credentials into the repo.
- Prefer **SSH keys** and **environment variables** for secrets.

---

## 12. Where to read next

- **Project narrative and metrics**: [`ibrahim_outline.md`](ibrahim_outline.md)
- **Plan / deliverables** (if present in your team docs): refer to course README or internal Notion; keep this repo aligned with whatever Hashim’s VLM branch expects as imports.
