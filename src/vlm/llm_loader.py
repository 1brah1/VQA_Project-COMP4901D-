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


def load_llm_awq(
    model_name_or_path: str,
    *,
    device: Union[Literal["cuda"], str] = "cuda",
) -> LoadedLLM:
    """
    Loads an AWQ-quantized model if `autoawq` is installed.

    Expected usage: run `scripts/quantize_llm_awq.py` to produce a quantized folder,
    then pass that folder path here.
    """
    try:
        from awq import AutoAWQForCausalLM  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("autoawq is not installed. Install requirements-quant.txt") from e

    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=True, device_map=str(device))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return LoadedLLM(tokenizer=tok, model=model)

