#!/usr/bin/env python3
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    ).eval()

    prompts = [
        "Classify crosswalk signal with one token only: red or green.",
        "Classify stairs presence with one token only: yes or no.",
        "Classify obstacle presence with one token only: yes or no.",
    ]

    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                temperature=0.0,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print("PROMPT:", p)
        print("OUTPUT:", text)
        print("---")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
