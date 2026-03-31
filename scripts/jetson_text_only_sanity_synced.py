#!/usr/bin/env python3
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.vlm.llm_loader import load_llm_fp16_or_fp32


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model_id = "Qwen/Qwen2.5-1.5B-Instruct"

    loaded = load_llm_fp16_or_fp32(model_id, device=device, dtype=dtype)
    tok_loader = loaded.tokenizer
    model_loader = loaded.model.eval()

    tok_direct = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model_direct = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map={"": device},
    ).eval()

    prompts = [
        "Task: crosswalk signal classification. Return one label token only: red or green.",
        "Task: stairs detection. Return one label token only: yes or no.",
        "Task: obstacle detection. Return one label token only: yes or no.",
        "What is 2 plus 2? Return one token only.",
        "Say hello in one word.",
    ]

    decode_variants = [
        {
            "name": "greedy_t4",
            "kwargs": {
                "max_new_tokens": 4,
                "do_sample": False,
                "synced_gpus": False,
            },
        },
        {
            "name": "greedy_t16",
            "kwargs": {
                "max_new_tokens": 16,
                "do_sample": False,
                "synced_gpus": False,
            },
        },
        {
            "name": "sample_t16",
            "kwargs": {
                "max_new_tokens": 16,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "synced_gpus": False,
            },
        },
    ]

    def run_block(name: str, tokenizer, model) -> None:
        print("===", name, "===")
        for variant in decode_variants:
            print("[variant]", variant["name"])
            for p in prompts:
                ids = tokenizer(p, return_tensors="pt").to(model.device)
                in_len = int(ids["input_ids"].shape[1])
                with torch.no_grad():
                    out = model.generate(**ids, **variant["kwargs"])
                gen = out[0, in_len:] if out.shape[1] > in_len else out[0]
                text = tokenizer.decode(gen, skip_special_tokens=True)
                print("PROMPT:", p)
                print("GEN_IDS:", gen.tolist())
                print("GEN:", repr(text))
                print("---")

    run_block("loader_path", tok_loader, model_loader)
    run_block("direct_transformers_path", tok_direct, model_direct)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
