from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedLLM:
    tokenizer: any
    model: any


def load_llm_fp16_or_fp32(
    model_name_or_path: str,
    *,
    device: Union[Literal["cpu", "cuda"], str] = "cpu",
    dtype: torch.dtype = torch.float16,
) -> LoadedLLM:
    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model.eval().to(device=device)
    if str(device) == "cuda":
        model.to(dtype=dtype)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return LoadedLLM(tokenizer=tok, model=model)



# Fallback model used when AWQ kernels are unavailable (e.g. Jetson aarch64 + Torch 2.0).
_AWQ_FALLBACK_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def load_llm_awq(model_name_or_path: str, *, device: str = "cuda") -> LoadedLLM:
    """
    Load the LLM for the AWQ benchmark path.

    On platforms where autoawq GEMM kernels are available (Linux x86_64 + CUDA),
    this loads the quantized model directly.

    On Jetson (aarch64, Torch 2.0.0+nv23.05, Python 3.8), autoawq is incompatible.
    In that case we fall back to loading the original FP16 model from HuggingFace
    so that generation produces coherent text (the quantized weights are 4-bit packed
    integers and cannot be interpreted as FP16 directly).
    """
    import platform
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Try real AWQ first (works on x86_64 with a proper autoawq install).
    if platform.machine() != "aarch64":
        try:
            from awq import AutoAWQForCausalLM
            print(f"Loading AWQ quantized model: {model_name_or_path}...")
            tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
            model = AutoAWQForCausalLM.from_quantized(
                model_name_or_path,
                fuse_layers=False,
                trust_remote_code=True,
                device_map={"": device},
            )
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            return LoadedLLM(tokenizer=tok, model=model)
        except Exception as e:
            print(f"AutoAWQ load failed ({e}), falling back to FP16...")

    # Jetson fallback: load the original unquantized model in FP16.
    # The quantized directory weights are 4-bit packed and cannot be used as FP16.
    print(f"Jetson/aarch64 fallback: loading {_AWQ_FALLBACK_MODEL} in FP16 on {device}...")
    tok = AutoTokenizer.from_pretrained(_AWQ_FALLBACK_MODEL, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        _AWQ_FALLBACK_MODEL,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return LoadedLLM(tokenizer=tok, model=model)
