from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedLLM:
    tokenizer: any
    model: any
    identity: Dict[str, Any]


def _normalize_generation_config(model: Any, tokenizer: Any) -> Dict[str, Any]:
    """Force deterministic, Jetson-safe generation defaults on loaded models."""
    cfg = getattr(model, "generation_config", None)
    if cfg is None:
        return {}

    cfg.do_sample = False
    cfg.temperature = 1.0
    cfg.top_p = 1.0
    cfg.top_k = 50
    cfg.num_beams = 1
    cfg.repetition_penalty = 1.0

    if getattr(tokenizer, "pad_token_id", None) is not None:
        cfg.pad_token_id = tokenizer.pad_token_id
    if getattr(tokenizer, "eos_token_id", None) is not None:
        cfg.eos_token_id = tokenizer.eos_token_id

    return {
        "do_sample": cfg.do_sample,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
        "num_beams": cfg.num_beams,
        "repetition_penalty": cfg.repetition_penalty,
        "pad_token_id": cfg.pad_token_id,
        "eos_token_id": cfg.eos_token_id,
    }


def _build_model_identity(*, model_name_or_path: str, model: Any, mode: str) -> Dict[str, Any]:
    cfg = getattr(model, "config", None)
    hidden_size = getattr(cfg, "hidden_size", None)
    num_layers = getattr(cfg, "num_hidden_layers", None)
    vocab_size = getattr(cfg, "vocab_size", None)

    try:
        model_dtype = str(next(model.parameters()).dtype)
    except StopIteration:
        model_dtype = "unknown"

    return {
        "model_id": model_name_or_path,
        "mode": mode,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "vocab_size": vocab_size,
        "dtype": model_dtype,
    }


def _forward_logits_are_finite(model: Any, tokenizer: Any, device: str) -> bool:
    """Run a tiny forward pass and confirm logits are finite."""
    probe_prompt = "Say hello in one word."
    ids = tokenizer(probe_prompt, return_tensors="pt")
    ids = {k: v.to(device) for k, v in ids.items()}
    with torch.no_grad():
        out = model(**ids)
        logits = out.logits[:, -1, :]
    return bool(torch.isfinite(logits).all().item())


def infer_expected_hidden_size(model_name_or_path: str) -> Optional[int]:
    """Infer expected hidden size from common Qwen model naming conventions."""
    low = (model_name_or_path or "").lower()
    if "qwen" not in low:
        return None
    if "0.5b" in low or "0p5b" in low:
        return 1024
    if "1.5b" in low or "1p5b" in low:
        return 1536
    return None


def validate_loaded_identity(identity: Dict[str, Any], expected_hidden_size: Optional[int]) -> None:
    """Raise when loaded model identity does not match expected architecture."""
    if expected_hidden_size is None:
        return

    actual_hidden_size = identity.get("hidden_size")
    if actual_hidden_size != expected_hidden_size:
        raise RuntimeError(
            "Loaded model hidden_size mismatch: "
            f"expected={expected_hidden_size}, actual={actual_hidden_size}, "
            f"model_id={identity.get('model_id')}, mode={identity.get('mode')}"
        )


def load_llm_fp16_or_fp32(
    model_name_or_path: str,
    *,
    device: Union[Literal["cpu", "cuda"], str] = "cpu",
    dtype: torch.dtype = torch.float16,
) -> LoadedLLM:
    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    effective_dtype = dtype
    if str(device) == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map={"": device},
        )

        # Jetson aarch64 can produce non-finite logits for 1.5B in FP16.
        # Detect once at load time and fall back to FP32 for correctness.
        if dtype == torch.float16:
            try:
                finite = _forward_logits_are_finite(model, tok, str(device))
            except Exception:
                finite = False
            if not finite:
                print(
                    "[LLM Loader] WARNING: Non-finite logits detected with FP16; "
                    "reloading model in FP32 for numerical stability."
                )
                del model
                torch.cuda.empty_cache()
                model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    torch_dtype=torch.float32,
                    device_map={"": device},
                )
                effective_dtype = torch.float32
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        model.to(device=device)
        effective_dtype = torch.float32 if dtype == torch.float16 else dtype
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    gen_cfg = _normalize_generation_config(model, tok)
    identity = _build_model_identity(model_name_or_path=model_name_or_path, model=model, mode="fp16")
    identity["requested_dtype"] = str(dtype)
    identity["effective_dtype"] = str(effective_dtype)
    print(
        "[LLM Loader] Active model: "
        f"id={identity['model_id']}, hidden_size={identity['hidden_size']}, "
        f"layers={identity['num_layers']}, dtype={identity['dtype']}, mode=fp16"
    )
    if gen_cfg:
        print(
            "[LLM Loader] Generation defaults: "
            f"do_sample={gen_cfg['do_sample']}, temp={gen_cfg['temperature']}, "
            f"top_p={gen_cfg['top_p']}, top_k={gen_cfg['top_k']}, beams={gen_cfg['num_beams']}"
        )
    return LoadedLLM(tokenizer=tok, model=model, identity=identity)



def load_llm_awq(model_name_or_path: str, *, device: str = "cuda") -> LoadedLLM:
    """
    Load the LLM for the AWQ benchmark path.

    On platforms where autoawq GEMM kernels are available (Linux x86_64 + CUDA),
    this loads the quantized model directly.

    On Jetson (aarch64, Torch 2.0.0+nv23.05, Python 3.8), autoawq is incompatible.
    We fail loudly instead of silently switching models to preserve experiment integrity.
    """
    import platform
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    arch = platform.machine()
    print(f"[LLM Loader] Architecture: {arch}")
    print(f"[LLM Loader] Torch version: {torch.__version__}")

    if arch == "aarch64":
        raise RuntimeError(
            "AWQ quantized loading is unsupported on Jetson aarch64 in this environment; "
            "use --llm-mode fp16 for reproducible runs."
        )

    try:
        from awq import AutoAWQForCausalLM
    except Exception as e:
        raise RuntimeError(
            f"AutoAWQ import failed on architecture={arch}. Install/verify autoawq for this host. Details: {e}"
        )

    print(f"[LLM Loader] Attempting AWQ quantized load: {model_name_or_path}")
    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoAWQForCausalLM.from_quantized(
        model_name_or_path,
        fuse_layers=False,
        trust_remote_code=True,
        device_map={"": device},
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    gen_cfg = _normalize_generation_config(model, tok)
    identity = _build_model_identity(model_name_or_path=model_name_or_path, model=model, mode="awq")
    print(
        "[LLM Loader] Active model: "
        f"id={identity['model_id']}, hidden_size={identity['hidden_size']}, "
        f"layers={identity['num_layers']}, dtype={identity['dtype']}, mode=awq"
    )
    if gen_cfg:
        print(
            "[LLM Loader] Generation defaults: "
            f"do_sample={gen_cfg['do_sample']}, temp={gen_cfg['temperature']}, "
            f"top_p={gen_cfg['top_p']}, top_k={gen_cfg['top_k']}, beams={gen_cfg['num_beams']}"
        )
    print(f"[LLM Loader] ✓ AWQ quantized model loaded successfully")
    return LoadedLLM(tokenizer=tok, model=model, identity=identity)
