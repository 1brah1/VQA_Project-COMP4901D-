import argparse
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--out", type=str, default="quantized/qwen2p5_0p5b_awq_int4")
    p.add_argument("--wbits", type=int, default=4)
    p.add_argument("--group_size", type=int, default=128)
    p.add_argument("--zero_point", action="store_true")
    args = p.parse_args()

    try:
        from awq import AutoAWQForCausalLM  # type: ignore
        from transformers import AutoTokenizer
    except Exception as e:
        raise RuntimeError("autoawq not installed. pip install -r requirements-quant.txt") from e

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoAWQForCausalLM.from_pretrained(args.model, low_cpu_mem_usage=True, torch_dtype=torch.float16)

    quant_config = {
        "w_bit": args.wbits,
        "q_group_size": args.group_size,
        "zero_point": bool(args.zero_point),
        "version": "GEMM",
    }

    # A small, generic calibration set; replace with your navigation prompts if desired.
    calib = [
        "Obstacle at 2 o'clock, steer left.",
        "Red crosswalk signal, stop and wait.",
        "Stairs ahead, step up carefully.",
        "Path is clear, continue forward.",
    ]
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib)
    model.save_quantized(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"Saved AWQ quantized model to {out_dir}")


if __name__ == "__main__":
    main()

