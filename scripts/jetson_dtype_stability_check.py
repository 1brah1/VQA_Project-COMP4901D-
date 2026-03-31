#!/usr/bin/env python3
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_for_dtype(model_id: str, prompt: str, dtype: torch.dtype) -> None:
    print("--- dtype", dtype)
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map={"": "cuda"})
    model.eval()

    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**ids)
        logits = out.logits[:, -1, :]

    finite = bool(torch.isfinite(logits).all().item())
    print("logits finite:", finite)

    logits_sanitized = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
    print("logits min/max:", float(logits_sanitized.min().item()), float(logits_sanitized.max().item()))

    with torch.no_grad():
        gen = model.generate(**ids, max_new_tokens=8, do_sample=False, synced_gpus=False)
    new_ids = gen[0, ids["input_ids"].shape[1]:]
    print("gen ids:", new_ids.tolist())
    print("gen text:", repr(tok.decode(new_ids, skip_special_tokens=True)))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    prompt = "Say hello in one word."

    run_for_dtype(model_id, prompt, torch.float16)
    run_for_dtype(model_id, prompt, torch.float32)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
