import argparse
from pathlib import Path
import torch

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--out", type=str, default="quantized/qwen2p5_0p5b_awq_int4")
    p.add_argument("--wbits", type=int, default=4)
    p.add_argument("--group_size", type=int, default=128)
    args = p.parse_args()

    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except Exception as e:
        raise RuntimeError("autoawq not installed.") from e

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📦 Loading model and tokenizer: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    
    # Load the model with AutoAWQ
    model = AutoAWQForCausalLM.from_pretrained(
        args.model, 
        low_cpu_mem_usage=True, 
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # --- THE CRITICAL FIX: QWEN 2.5 COMPATIBILITY PATCH ---
    # We manually add 'rotary_emb' to the internal model objects.
    # This stops the 'AttributeError' when AutoAWQ tries to move them to GPU.
    print("🔧 Applying Qwen2.5 compatibility patch...")
    
    # Patch the inner Qwen2Model
    if hasattr(model.model, "model"):
        setattr(model.model.model, "rotary_emb", torch.nn.Module())
    
    # Patch the Qwen2ForCausalLM wrapper
    if not hasattr(model.model, "rotary_emb"):
        setattr(model.model, "rotary_emb", torch.nn.Module())

    quant_config = {
        "w_bit": args.wbits,
        "q_group_size": args.group_size,
        "zero_point": True, 
        "version": "GEMM",
    }

    print("🚀 Starting quantization using 'pileval'...")
    
    # Execute quantization
    model.quantize(
        tokenizer, 
        quant_config=quant_config, 
        calib_data="pileval" 
    )

    print(f"💾 Saving quantized model to {out_dir}...")
    model.save_quantized(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    
    print(f"✅ Success! Your 4-bit model is saved in: {out_dir}")

if __name__ == "__main__":
    main()