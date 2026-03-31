#!/usr/bin/env python3
"""Run the integrated VLM pipeline with a LoRA adapter on Jetson."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from peft import PeftModel

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.classification import backend_state
from src.prompts.load_prompt import load_system_prompt
from src.vision.siglip_encoder import SiglipPatchEncoder
from src.vlm.llm_loader import (
    infer_expected_hidden_size,
    load_llm_fp16_or_fp32,
    validate_loaded_identity,
)
from src.vlm.model import SimplePrefixVLM
from scripts.run_integrated import (
    CompositeFallbackTTS,
    _summarize_results,
    load_eval_items,
    merge_wav_files,
    run_one_image,
)


_PROFILE_SETTINGS = {
    "label_only_eval": {"max_new_tokens": 12},
    "sentence_demo_fast": {"max_new_tokens": 24},
    "sentence_demo_quality": {"max_new_tokens": 32},
}


def _load_image_proj(vlm: SimplePrefixVLM, image_proj_path: Optional[str], device: str) -> Optional[str]:
    if not image_proj_path:
        return None
    proj_path = Path(image_proj_path)
    if proj_path.is_dir():
        proj_path = proj_path / "image_proj.pt"
    if not proj_path.exists():
        return None
    state = torch.load(str(proj_path), map_location="cpu")
    state_dtype = None
    for val in state.values():
        if isinstance(val, torch.Tensor):
            state_dtype = val.dtype
            break
    if state_dtype is not None:
        vlm.image_proj = vlm.image_proj.to(device=device, dtype=state_dtype)
    vlm.image_proj.load_state_dict(state)
    vlm.image_proj.eval()
    return str(proj_path)


def run_integrated_lora(
    labels_path: Path,
    llm_model_id: str,
    lora_adapter_path: Path,
    compression: int,
    profile: str = "label_only_eval",
    max_new_tokens: Optional[int] = None,
    enable_tts: bool = False,
    no_tts: bool = False,
    warmup_images: int = 0,
    output_dir: Optional[Path] = None,
    max_images: Optional[int] = None,
    expected_hidden_size: Optional[int] = None,
    avoid_unknown_labels: bool = False,
    tts_fallback: str = "piper,silero,pyttsx3",
    seed: int = 42,
) -> Path:
    if output_dir is None:
        output_dir = _ROOT / "reports" / "integrated_lora"
    output_dir.mkdir(parents=True, exist_ok=True)

    if profile not in _PROFILE_SETTINGS:
        raise ValueError("Unknown profile: {0}".format(profile))
    if max_new_tokens is None:
        max_new_tokens = int(_PROFILE_SETTINGS[profile]["max_new_tokens"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("[integrated-lora] device={0} profile={1}".format(device, profile))
    print("[integrated-lora] loading base model: {0}".format(llm_model_id))

    loaded = load_llm_fp16_or_fp32(llm_model_id, device=device, dtype=dtype)
    if expected_hidden_size is None:
        expected_hidden_size = infer_expected_hidden_size(llm_model_id)
    validate_loaded_identity(loaded.identity, expected_hidden_size)

    lora_path = Path(lora_adapter_path)
    if not lora_path.exists():
        raise FileNotFoundError("LoRA adapter not found: {0}".format(lora_path))

    print("[integrated-lora] loading LoRA adapter: {0}".format(lora_path))
    model = PeftModel.from_pretrained(loaded.model, str(lora_path))
    model.eval()

    llm_dtype = next(model.parameters()).dtype
    vlm = SimplePrefixVLM.from_loaded_llm(
        tokenizer=loaded.tokenizer,
        llm=model,
        device=device,
        dtype=llm_dtype,
        image_token_dim=768,
    )
    image_proj_path = _load_image_proj(vlm, str(lora_path), device)
    if image_proj_path:
        print("[integrated-lora] image_proj loaded: {0}".format(image_proj_path))

    encoder = SiglipPatchEncoder.from_pretrained("google/siglip-base-patch16-384", device=device, dtype=dtype)
    system_prompt = load_system_prompt()

    items = load_eval_items(labels_path, _ROOT)
    if max_images is not None:
        items = items[: max_images]

    if not items:
        raise RuntimeError("No evaluation items found in {0}".format(labels_path))

    tts_enabled = bool(enable_tts and not no_tts)
    fallback_tts = None
    tts_backend_requested = "disabled"
    tts_backend_active = "disabled"
    tts_fallback_reason: Optional[str] = None

    if tts_enabled:
        backend_order = [b.strip().lower() for b in tts_fallback.split(",") if b.strip()]
        if backend_order and backend_order != ["none"]:
            fallback_tts = CompositeFallbackTTS(backend_order=backend_order, device=device)
            tts_backend_requested = ",".join(backend_order)
            if fallback_tts.available:
                tts_backend_active = ",".join(fallback_tts.backend_names)
            else:
                tts_backend_active = "disabled"
                tts_fallback_reason = "no_tts_backend_available"

    if warmup_images > 0:
        warmup = items[: warmup_images]
        print("[integrated-lora] warmup images: {0}".format(len(warmup)))
        for warm in warmup:
            run_one_image(
                image_path=warm.path,
                task=warm.task,
                encoder=encoder,
                vlm=vlm,
                target_tokens=compression,
                system_prompt=system_prompt,
                fallback_tts=None,
                output_dir=None,
                sample_id="warmup_{0}".format(warm.id),
                verbose=False,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                top_k=50,
                top_p=0.95,
                do_sample=False,
                avoid_unknown_labels=avoid_unknown_labels,
            )

    results = []
    for idx, item in enumerate(items, 1):
        print("\n[{0}/{1}] {2} task={3}".format(idx, len(items), Path(item.path).name, item.task))
        result, error = run_one_image(
            image_path=item.path,
            task=item.task,
            encoder=encoder,
            vlm=vlm,
            target_tokens=compression,
            system_prompt=system_prompt,
            fallback_tts=fallback_tts,
            output_dir=output_dir if (fallback_tts is not None and fallback_tts.available) else None,
            sample_id=item.id,
            verbose=False,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_k=50,
            top_p=0.95,
            do_sample=False,
            avoid_unknown_labels=avoid_unknown_labels,
        )
        results.append(result)
        if error is None:
            print("  response: {0!r}".format(result.response_text))
            print("  classification: {0}".format(result.classification_label))
            print("  spoken: {0}".format(result.spoken_sentence))
            print(
                "  latency (ms): capture={0:.0f}, compress={1:.0f}, vlm_ttft={2:.0f}, vlm_total={3:.0f}, e2e={4:.0f}".format(
                    result.capture_ms,
                    result.compress_ms,
                    result.vlm_ttft_ms,
                    result.vlm_total_ms,
                    result.e2e_total_ms,
                )
            )
            if result.wav_path:
                print("  audio: {0}".format(result.wav_path))

    merged_output = None
    merge_status = {
        "single_wav_output": None,
        "single_wav_success": None,
        "single_wav_error": None,
    }
    if fallback_tts is not None and fallback_tts.available:
        merged_output = str(output_dir / "combined_tts.wav")
        successful_wavs = [r.wav_path for r in results if r.error is None and r.wav_path]
        ok_merge, merge_error = merge_wav_files(successful_wavs, merged_output) if successful_wavs else (False, "no successful WAVs")
        merge_status = {
            "single_wav_output": merged_output,
            "single_wav_success": bool(ok_merge),
            "single_wav_error": merge_error,
        }

    successful = sum(1 for r in results if r.error is None)
    wav_count = sum(1 for r in results if r.wav_path)
    degeneration = _summarize_results(results, max_new_tokens=max_new_tokens)
    actual_hidden_size = loaded.identity.get("hidden_size")
    integrity = {
        "model_hidden_size_expected": expected_hidden_size,
        "model_hidden_size_actual": actual_hidden_size,
        "model_hidden_size_valid": (
            True if expected_hidden_size is None else (actual_hidden_size == expected_hidden_size)
        ),
    }

    report_data = {
        "schema_version": "v2",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "dtype": str(dtype),
        "model_config": {
            "siglip": "google/siglip-base-patch16-384",
            "llm": llm_model_id,
            "lora_adapter": str(lora_path),
            "image_proj": image_proj_path,
            "llm_mode": "fp16",
            "model_identity": loaded.identity,
            "expected_hidden_size": expected_hidden_size,
            "compression": compression,
            "tts_enabled": bool(tts_enabled and fallback_tts is not None and fallback_tts.available),
            "profile": profile,
            "seed": seed,
            "max_new_tokens": max_new_tokens,
            "avoid_unknown_labels": bool(avoid_unknown_labels),
            "warmup_images": warmup_images,
        },
        "pipeline_integrity": integrity,
        "degeneration": degeneration,
        "tts_backend": {
            **backend_state(tts_backend_requested, tts_backend_active, tts_fallback_reason),
        },
        "audio_output": merge_status,
        "summary": {
            "success_count": successful,
            "total": len(results),
            "wav_generated": wav_count,
        },
        "results": [asdict(r) for r in results],
    }

    now = time.strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / "report_{0}.json".format(now)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print("\nReport written to: {0}".format(report_path))
    print("Run complete: {0}/{1} successful".format(successful, len(results)))

    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run integrated VLM with LoRA on Jetson")
    parser.add_argument("--labels", type=Path, required=True, help="Path to labels.json")
    parser.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct", help="LLM model ID")
    parser.add_argument("--lora-adapter", type=Path, required=True, help="Path to LoRA adapter")
    parser.add_argument("--compression", type=int, default=192, help="Token compression target")
    parser.add_argument("--profile", default="label_only_eval", help="Eval profile")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Max generation tokens")
    parser.add_argument("--enable-tts", action="store_true", help="Enable TTS output")
    parser.add_argument("--no-tts", action="store_true", help="Disable TTS output")
    parser.add_argument("--warmup-images", type=int, default=0, help="Warmup images")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--max-images", type=int, help="Max images to process")
    parser.add_argument("--expected-hidden-size", type=int, help="Expected LLM hidden size")
    parser.add_argument("--avoid-unknown-labels", action="store_true", help="Map unknown labels to safe defaults")
    parser.add_argument("--tts-fallback", type=str, default="piper,silero,pyttsx3", help="Comma-separated TTS backend order")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    run_integrated_lora(
        labels_path=args.labels,
        llm_model_id=args.llm,
        lora_adapter_path=args.lora_adapter,
        compression=args.compression,
        profile=args.profile,
        max_new_tokens=args.max_new_tokens,
        enable_tts=args.enable_tts,
        no_tts=args.no_tts,
        warmup_images=args.warmup_images,
        output_dir=args.output_dir,
        max_images=args.max_images,
        expected_hidden_size=args.expected_hidden_size,
        avoid_unknown_labels=args.avoid_unknown_labels,
        tts_fallback=args.tts_fallback,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())