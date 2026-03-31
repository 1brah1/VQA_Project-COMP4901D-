#!/usr/bin/env python3
"""
scripts/run_integrated.py
==========================
Unified one-flow orchestrator for VQA + Jetson-compatible TTS on Jetson.

Merges run_once.py and run_pipelined.py into one persistent service that:
    • Loads all models once (SigLIP, Qwen)
  • Supports both single-image and batch (labels.json) modes
  • Generates TTS audio and WAV outputs
  • Produces JSON report with per-sample timings and artifacts
  • Keeps models warm in memory for repeated requests

Usage Examples
--------------
  # Single image with TTS:
    python scripts/run_integrated.py \
        --image-path data/eval/images/crosswalk/Crosswalk_1.png \
        --task crosswalk_signal \
        --compression 192 \
        --enable-tts \
        --tts-fallback piper,silero,pyttsx3 \
        --output-dir outputs/

  # Batch eval (16 images):
    python scripts/run_integrated.py \
        --labels data/eval/labels.json \
        --compression 192 \
        --enable-tts \
        --tts-fallback piper,silero,pyttsx3 \
        --output-dir outputs/

  # Text-only (no TTS):
  python scripts/run_integrated.py \\
    --labels data/eval/labels.json \\
    --no-tts
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import time
import wave
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# Project root on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.prompts.load_prompt import load_system_prompt
from src.classification import classify_and_format, backend_state
from src.vision.siglip_encoder import SiglipPatchEncoder
from src.vision.token_compression import compress_27x27_tokens, recommended_targets
from src.vlm.model import SimplePrefixVLM
from src.vlm.llm_loader import (
    load_llm_fp16_or_fp32,
    load_llm_awq,
    infer_expected_hidden_size,
    validate_loaded_identity,
)
from src.tts.fallback_backends import PiperTTSBackend, SileroTTSBackend, Pyttsx3TTSBackend
from src.audio_utils import save_wav


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Per-sample pipeline result."""
    sample_id: str
    image_path: str
    task: str
    response_text: str
    response_length_chars: int
    response_length_words: int
    prompt_echo_detected: bool
    stop_reason: str
    n_gen_tokens: int
    classification_label: str  # normalized verdict
    spoken_sentence: str
    capture_ms: float
    compress_ms: float
    vlm_ttft_ms: float
    vlm_total_ms: float
    tts_ttfa_ms: float
    e2e_first_audio_ms: float
    e2e_total_ms: float
    wav_path: Optional[str] = None
    error: Optional[str] = None


@dataclass
class EvalItem:
    """Evaluation set item."""
    id: str
    path: str
    task: str


# ─────────────────────────────────────────────────────────────────────────────
# Task Helpers
# ─────────────────────────────────────────────────────────────────────────────

_TASK_PROMPTS: Dict[str, str] = {
    "crosswalk_signal": (
        "Task: crosswalk signal classification. "
        "Answer with exactly one label token and nothing else. "
        "Allowed labels: red, green. "
        "If uncertain, choose the safer label red."
    ),
    "stairs": (
        "Task: stairs detection. "
        "Answer with exactly one label token and nothing else. "
        "Allowed labels: yes, no. "
        "If uncertain, choose the safer label yes."
    ),
    "obstacles": (
        "Task: obstacle detection. "
        "Answer with exactly one label token and nothing else. "
        "Allowed labels: yes, no. "
        "If uncertain, choose the safer label yes."
    ),
}

_PROFILE_SETTINGS: Dict[str, Dict[str, int]] = {
    "label_only_eval": {"max_new_tokens": 12},
    "sentence_demo_fast": {"max_new_tokens": 24},
    "sentence_demo_quality": {"max_new_tokens": 32},
}


def get_task_prompt(task: Optional[str]) -> str:
    """Get prompt for a task."""
    return _TASK_PROMPTS.get(task or "obstacles", _TASK_PROMPTS["obstacles"])


def _prompt_echo_detected(text: str) -> bool:
    low = (text or "").lower()
    patterns = [
        r"start\s+your\s+response\s+with\s+exactly\s+one\s+word",
        r"crosswalk\s+walk\s+signal",
        r"are\s+there\s+stairs\s+or\s+steps",
        r"is\s+there\s+an\s+obstacle\s+ahead",
        r"assistant\s*:",
        r"user\s*:",
    ]
    return any(bool(re.search(p, low)) for p in patterns)


def _summarize_results(results: List[PipelineResult], max_new_tokens: int) -> Dict[str, float]:
    """Compute compact degeneration diagnostics for report and gating."""
    if not results:
        return {
            "unknown_rate": 0.0,
            "all_bang_rate": 0.0,
            "prompt_echo_rate": 0.0,
            "stop_reason_max_new_tokens_rate": 0.0,
            "mean_n_gen_tokens": 0.0,
        }

    n = float(len(results))
    unknown = sum(1 for r in results if r.classification_label == "unknown")
    all_bang = sum(1 for r in results if (r.response_text or "").strip() == "!!!!")
    prompt_echo = sum(1 for r in results if bool(r.prompt_echo_detected))
    stop_max = sum(1 for r in results if r.stop_reason == "max_new_tokens")
    mean_n_gen = float(sum(r.n_gen_tokens for r in results)) / n

    return {
        "unknown_rate": unknown / n,
        "all_bang_rate": all_bang / n,
        "prompt_echo_rate": prompt_echo / n,
        "stop_reason_max_new_tokens_rate": stop_max / n,
        "mean_n_gen_tokens": mean_n_gen,
        "max_new_tokens": float(max_new_tokens),
    }


class CompositeFallbackTTS:
    """Fallback manager that tries higher-quality backends first."""

    def __init__(self, backend_order: List[str], device: str) -> None:
        self._backends = []
        self._backend_names = []
        self.last_backend_used: Optional[str] = None
        self.last_error: Optional[str] = None

        for name in backend_order:
            n = name.strip().lower()
            if n == "piper":
                be = PiperTTSBackend()
                self._backends.append(be)
                self._backend_names.append("piper")
            elif n == "silero":
                be = SileroTTSBackend(device=device)
                self._backends.append(be)
                self._backend_names.append("silero")
            elif n == "pyttsx3":
                be = Pyttsx3TTSBackend()
                self._backends.append(be)
                self._backend_names.append("pyttsx3")

    @property
    def available(self) -> bool:
        return any(getattr(be, "available", False) for be in self._backends)

    @property
    def backend_names(self) -> List[str]:
        return list(self._backend_names)

    def synthesize_to_wav(self, text: str, output_path: str) -> bool:
        self.last_backend_used = None
        self.last_error = None
        for name, be in zip(self._backend_names, self._backends):
            if not getattr(be, "available", False):
                continue
            ok = be.synthesize_to_wav(text, output_path)
            if ok:
                self.last_backend_used = name
                return True
            self.last_error = getattr(be, "last_error", None)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Stage Timer
# ─────────────────────────────────────────────────────────────────────────────

class StageTimer:
    """Lightweight profiler for stage-wise timing."""
    def __init__(self) -> None:
        self._marks: Dict[str, float] = {}

    def mark(self, name: str) -> None:
        self._marks[name] = time.perf_counter()

    def elapsed_ms(self, start: str, end: str) -> float:
        if start not in self._marks or end not in self._marks:
            return 0.0
        return (self._marks[end] - self._marks[start]) * 1000.0


def _load_image_proj(vlm: SimplePrefixVLM, image_proj_path: Optional[str], device: str) -> Optional[str]:
    if not image_proj_path:
        return None
    proj_path = Path(image_proj_path)
    if proj_path.is_dir():
        proj_path = proj_path / "image_proj.pt"
    if not proj_path.exists():
        raise FileNotFoundError(f"image_proj not found: {proj_path}")
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


# ─────────────────────────────────────────────────────────────────────────────
# Load Labels
# ─────────────────────────────────────────────────────────────────────────────

def load_eval_items(labels_path: Path, repo_root: Path) -> List[EvalItem]:
    """Load eval items from labels.json and resolve paths."""
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    obj = json.loads(labels_path.read_text(encoding="utf-8"))
    items = []
    for raw in obj.get("items", []):
        img_path = repo_root / raw["path"]
        if not img_path.exists():
            print(f"  WARNING: image not found, skipping — {img_path}")
            continue
        items.append(EvalItem(
            id=raw["id"],
            path=str(img_path),
            task=raw["task"]
        ))
    return items


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Execution
# ─────────────────────────────────────────────────────────────────────────────

def run_one_image(
    image_path: str,
    task: str,
    encoder: SiglipPatchEncoder,
    vlm: SimplePrefixVLM,
    target_tokens: int,
    system_prompt: str,
    fallback_tts: Optional[CompositeFallbackTTS] = None,
    output_dir: Optional[Path] = None,
    sample_id: Optional[str] = None,
    verbose: bool = False,
    max_new_tokens: int = 24,
    temperature: float = 0.2,
    top_k: int = 50,
    top_p: float = 0.95,
    do_sample: bool = False,
    avoid_unknown_labels: bool = False,
) -> Tuple[PipelineResult, Optional[str]]:
    """
    Run VQA pipeline on one image.
    
    Returns
    -------
    (result, error_msg)
      result : PipelineResult with all metrics
      error_msg : None if successful, otherwise exception message
    """
    timer = StageTimer()
    timer.mark("start")
    
    try:
        # ── Stage 1: Load Image ──────────────────────────────────────────
        image = Image.open(image_path).convert("RGB")
        timer.mark("capture")
        
        # ── Stage 2: Encode & Compress ───────────────────────────────────
        with torch.no_grad():
            image_tokens = encoder.encode(image)
            image_tokens = compress_27x27_tokens(image_tokens, target_tokens=target_tokens)
        timer.mark("compress")
        
        # ── Stage 3: VQA Inference ─────────────────────────────────────
        user_prompt = get_task_prompt(task)
        
        text_parts: List[str] = []
        wav_path: Optional[str] = None
        
        with torch.no_grad():
            gen_out = vlm.generate(
                image_tokens=image_tokens,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                return_num_new_tokens=True,
            )
            if isinstance(gen_out, tuple):
                gen_text, n_gen_tokens = gen_out
            else:
                gen_text = str(gen_out)
                n_gen_tokens = 0
        
        # Stream or collect response
        timer.mark("vlm_first_token")
        for chunk in [gen_text]:
            text_parts.append(str(chunk))
            if verbose:
                print(chunk, end="", flush=True)
        
        timer.mark("vlm_done")
        if verbose:
            print()
        
        response_text = "".join(text_parts).strip()
        response_length_chars = len(response_text)
        response_length_words = len(response_text.split())
        prompt_echo_detected = _prompt_echo_detected(response_text)
        stop_reason = "max_new_tokens" if n_gen_tokens >= max_new_tokens else "eos_or_stop"
        classification_label, spoken_sentence = classify_and_format(
            task,
            response_text,
            avoid_unknown=avoid_unknown_labels,
        )
        # ── Stage 4: TTS Audio ───────────────────────────────────────
        if fallback_tts is not None and fallback_tts.available and output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            wav_name = f"{sample_id or Path(image_path).stem}.wav"
            wav_path = str(output_dir / wav_name)
            candidate = Path(wav_path)
            if candidate.exists():
                try:
                    candidate.unlink()
                except Exception:
                    pass
            if fallback_tts.synthesize_to_wav(spoken_sentence, wav_path):
                backend = fallback_tts.last_backend_used or "fallback"
                print(f"  WAV saved via {backend} fallback: {wav_path}")
            else:
                if fallback_tts.last_error:
                    print(f"  WARNING: fallback TTS failed ({fallback_tts.last_error})")
                # Some engines finish writing slightly after the call returns.
                late_ok = False
                for _ in range(12):
                    if candidate.exists() and candidate.stat().st_size > 0:
                        late_ok = True
                        break
                    time.sleep(0.25)
                if late_ok:
                    print(f"  WAV detected after delayed flush: {wav_path}")
                else:
                    wav_path = None
        
        # ── Collect Metrics ──────────────────────────────────────────
        lat = {
            "capture_ms":         timer.elapsed_ms("start",     "capture"),
            "compress_ms":        timer.elapsed_ms("capture",   "compress"),
            "vlm_ttft_ms":        timer.elapsed_ms("compress",  "vlm_first_token"),
            "vlm_total_ms":       timer.elapsed_ms("compress",  "vlm_done"),
            "tts_ttfa_ms":        0.0,
            "e2e_first_audio_ms": 0.0,
            "e2e_total_ms":       timer.elapsed_ms("start",     "vlm_done"),
        }
        
        result = PipelineResult(
            sample_id=sample_id or Path(image_path).stem,
            image_path=image_path,
            task=task,
            response_text=response_text,
            response_length_chars=response_length_chars,
            response_length_words=response_length_words,
            prompt_echo_detected=prompt_echo_detected,
            stop_reason=stop_reason,
            n_gen_tokens=n_gen_tokens,
            classification_label=classification_label,
            spoken_sentence=spoken_sentence,
            wav_path=wav_path,
            error=None,
            **lat
        )
        
        return result, None
    
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"  ERROR: {error_msg}")
        
        result = PipelineResult(
            sample_id=sample_id or Path(image_path).stem,
            image_path=image_path,
            task=task,
            response_text="",
            response_length_chars=0,
            response_length_words=0,
            prompt_echo_detected=False,
            stop_reason="error",
            n_gen_tokens=0,
            classification_label="error",
            spoken_sentence="I could not process this image.",
            capture_ms=0.0,
            compress_ms=0.0,
            vlm_ttft_ms=0.0,
            vlm_total_ms=0.0,
            tts_ttfa_ms=0.0,
            e2e_first_audio_ms=0.0,
            e2e_total_ms=0.0,
            wav_path=None,
            error=error_msg,
        )
        
        return result, error_msg


def merge_wav_files(input_paths: List[str], output_path: str) -> Tuple[bool, Optional[str]]:
    """Merge WAV files into a single WAV output."""
    merged_chunks: List[np.ndarray] = []
    sample_rate: Optional[int] = None

    for wav_path in input_paths:
        try:
            # Prefer scipy for broad WAV support, including IEEE float format.
            try:
                import scipy.io.wavfile  # type: ignore[import]
                sr, data = scipy.io.wavfile.read(wav_path)
                audio = np.asarray(data, dtype=np.float32)
                if audio.ndim > 1:
                    audio = audio.reshape(-1)
            except Exception:
                # Fallback to soundfile, then stdlib wave for PCM paths.
                try:
                    import soundfile as sf  # type: ignore[import]
                    audio, sr = sf.read(wav_path, dtype="float32")
                    audio = np.asarray(audio, dtype=np.float32)
                    if audio.ndim > 1:
                        audio = audio.reshape(-1)
                except Exception:
                    with wave.open(wav_path, "rb") as wf:
                        sr = wf.getframerate()
                        ch = wf.getnchannels()
                        sw = wf.getsampwidth()
                        raw = wf.readframes(wf.getnframes())
                        if sw == 2:
                            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                        elif sw == 4:
                            audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
                        else:
                            return False, f"unsupported WAV sample width {sw} in {wav_path}"
                        if ch > 1:
                            audio = audio.reshape(-1, ch).mean(axis=1)

            if sample_rate is None:
                sample_rate = int(sr)
            elif int(sr) != sample_rate:
                return False, f"incompatible sample rate in {wav_path}: {sr} != {sample_rate}"

            merged_chunks.append(audio)
        except Exception as exc:
            return False, f"failed reading {wav_path}: {exc}"

    if not merged_chunks or sample_rate is None:
        return False, "no valid WAV frames to merge"

    merged_audio = np.concatenate(merged_chunks, axis=0)
    ok = save_wav(merged_audio, output_path, sample_rate=sample_rate, verbose=False)
    if not ok:
        return False, "failed writing merged WAV"

    return True, None


# ─────────────────────────────────────────────────────────────────────────────
# Main Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Integrated VQA + TTS pipeline (Jetson-optimized)"
    )
    parser.add_argument("--image-path", type=str, default=None,
                        help="Single image path (overrides --labels)")
    parser.add_argument("--labels", type=str, default="data/eval/labels.json",
                        help="JSON file with batch of images")
    parser.add_argument("--task", type=str, default=None,
                        help="Task for single image (crosswalk_signal, stairs, obstacles)")
    parser.add_argument("--compression", type=int, default=192,
                        help="Token compression target (192, 81, 36, or 9)")
    parser.add_argument("--siglip", type=str, default="google/siglip-base-patch16-384",
                        help="SigLIP model name or path")
    parser.add_argument("--llm", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="LLM model name or path (also accepts AWQ dir)")
    parser.add_argument("--image-proj", type=str, default=None,
                        help="Optional image_proj.pt file or directory for trained projection")
    parser.add_argument("--expected-hidden-size", type=int, default=None,
                        help="Optional strict check for loaded LLM hidden size")
    parser.add_argument("--llm-mode", type=str, default="fp16",
                        choices=["fp16", "awq"],
                        help="LLM loading mode: fp16 (default) or awq (quantized)")
    parser.add_argument("--enable-tts", action="store_true",
                        help="Enable TTS synthesis using fallback backend order")
    parser.add_argument("--no-tts", action="store_true",
                        help="Disable TTS regardless of --enable-tts")
    parser.add_argument("--output-dir", type=str, default="outputs/",
                        help="Output directory for WAV files and JSON report")
    parser.add_argument("--report-file", type=str, default=None,
                        help="JSON report output path (default: {output_dir}/report_{timestamp}.json)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print generated tokens as they stream")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit number of images for testing")
    parser.add_argument("--strict-demo", action="store_true",
                        help="Fail when TTS is requested but no WAV artifacts are produced")
    parser.add_argument("--profile", type=str, default="sentence_demo_fast",
                        choices=["label_only_eval", "sentence_demo_fast", "sentence_demo_quality"],
                        help="Preset for decoding length and TTS trigger policy")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible runs")
    parser.add_argument("--warmup-images", type=int, default=1,
                        help="Number of warmup images to run before measured pass")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Override generation cap")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Sampling temperature when --do-sample is enabled")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k sampling cutoff when --do-sample is enabled")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Top-p sampling cutoff when --do-sample is enabled")
    parser.add_argument("--do-sample", action="store_true",
                        help="Enable stochastic decoding")
    parser.add_argument("--avoid-unknown-labels", action="store_true",
                        help="Map unparsable outputs to safety-biased labels instead of unknown")
    parser.add_argument("--single-wav-output", type=str, default=None,
                        help="Optional path for one merged WAV file from all successful TTS samples")
    parser.add_argument("--single-wav-only", action="store_true", default=True,
                        help="After merge, keep only the single merged WAV output (default: enabled)")
    parser.add_argument("--keep-per-sample-wavs", action="store_true", default=False,
                        help="Keep individual sample WAV files instead of removing them after merge")
    parser.add_argument("--tts-fallback", type=str, default="piper,silero,pyttsx3",
                        help="Comma-separated backend order (supported: piper,silero,pyttsx3,none)")
    
    args = parser.parse_args()

    profile_cfg = _PROFILE_SETTINGS[args.profile]
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else profile_cfg["max_new_tokens"]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # ── Setup ────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[integrated] device={device}, dtype={dtype}, profile={args.profile}, seed={args.seed}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ── Load Models (warm-start) ──────────────────────────────────────
    print(f"[integrated] Loading SigLIP: {args.siglip}")
    encoder = SiglipPatchEncoder.from_pretrained(
        args.siglip, device=device, dtype=dtype
    )
    
    print(f"[integrated] Loading LLM: {args.llm} (mode={args.llm_mode})")
    if args.llm_mode == "awq":
        loaded = load_llm_awq(args.llm, device=device)
        llm_dtype = next(loaded.model.parameters()).dtype
        vlm = SimplePrefixVLM.from_loaded_llm(
            tokenizer=loaded.tokenizer,
            llm=loaded.model,
            device=device,
            dtype=llm_dtype,
            image_token_dim=768,
        )
    else:
        loaded = load_llm_fp16_or_fp32(args.llm, device=device)
        llm_dtype = next(loaded.model.parameters()).dtype
        vlm = SimplePrefixVLM.from_loaded_llm(
            tokenizer=loaded.tokenizer,
            llm=loaded.model,
            device=device,
            dtype=llm_dtype,
            image_token_dim=768,
        )

    expected_hidden_size = args.expected_hidden_size
    if expected_hidden_size is None:
        expected_hidden_size = infer_expected_hidden_size(args.llm)
    validate_loaded_identity(loaded.identity, expected_hidden_size)
    if expected_hidden_size is not None:
        print(f"[integrated] model identity validated: hidden_size={expected_hidden_size}")
    image_proj_path = _load_image_proj(vlm, args.image_proj, device)
    
    # ── Load TTS (if enabled) ────────────────────────────────────────
    fallback_tts: Optional[CompositeFallbackTTS] = None
    tts_fallback_reason: Optional[str] = None
    fallback_backend_list = [x.strip().lower() for x in args.tts_fallback.split(",") if x.strip()]
    if "none" in fallback_backend_list:
        fallback_backend_list = []
    tts_runtime_checks = {
        "tts_requested": bool(args.enable_tts and not args.no_tts),
        "fallback_backend": ",".join(fallback_backend_list) if fallback_backend_list else "none",
        "fallback_available": None,
        "fallback_backend_used": None,
    }

    if args.enable_tts and not args.no_tts and fallback_backend_list:
        fallback_tts = CompositeFallbackTTS(backend_order=fallback_backend_list, device=device)
        tts_runtime_checks["fallback_available"] = bool(fallback_tts.available)
    else:
        tts_runtime_checks["fallback_available"] = False

    if args.enable_tts and not args.no_tts and (fallback_tts is None or not fallback_tts.available):
        print("  ERROR: No available TTS backend from --tts-fallback order")
        return 1

    requested_tts_backend = "disabled"
    if args.enable_tts and not args.no_tts:
        requested_tts_backend = "fallback"
    elif args.no_tts and args.enable_tts:
        requested_tts_backend = "fallback"
        tts_fallback_reason = "disabled_by_no_tts_flag"

    active_tts_backend = "fallback" if fallback_tts is not None and fallback_tts.available else "disabled"
    
    system_prompt = load_system_prompt()
    
    # ── Determine Items ──────────────────────────────────────────────────
    items: List[EvalItem] = []
    
    if args.image_path:
        # Single-image mode
        if not args.task:
            print("ERROR: --task required when using --image-path")
            return 1
        items = [EvalItem(
            id=Path(args.image_path).stem,
            path=args.image_path,
            task=args.task
        )]
    else:
        # Batch mode
        labels_path = Path(args.labels)
        items = load_eval_items(labels_path, _ROOT)
    
    if not items:
        print("ERROR: no items to process")
        return 1
    
    if args.max_images:
        items = items[:args.max_images]

    # Validate compression target early against actual encoder token count.
    with torch.no_grad():
        probe_img = Image.open(items[0].path).convert("RGB")
        probe_n_tokens = int(encoder.encode(probe_img).shape[1])
    valid_targets = recommended_targets(probe_n_tokens)
    if args.compression not in valid_targets:
        raise ValueError(
            f"compression={args.compression} is not valid for {probe_n_tokens} input tokens. "
            f"Try one of: {valid_targets}"
        )
    
    print(f"[integrated] Processing {len(items)} item(s)")

    # Warmup run(s) to stabilize latency variance.
    if args.warmup_images > 0:
        warmup_items = items[: min(args.warmup_images, len(items))]
        print(f"[integrated] Warmup: running {len(warmup_items)} sample(s)")
        for warm in warmup_items:
            _result, _err = run_one_image(
                image_path=warm.path,
                task=warm.task,
                encoder=encoder,
                vlm=vlm,
                target_tokens=args.compression,
                system_prompt=system_prompt,
                fallback_tts=fallback_tts,
                output_dir=None,
                sample_id=f"warmup_{warm.id}",
                verbose=False,
                max_new_tokens=max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                do_sample=args.do_sample,
                avoid_unknown_labels=args.avoid_unknown_labels,
            )
    
    # ── Run Pipeline ──────────────────────────────────────────────────
    results: List[PipelineResult] = []
    
    for i, item in enumerate(items, 1):
        print(f"\n[{i}/{len(items)}] {Path(item.path).name}  task={item.task}")
        
        result, error = run_one_image(
            image_path=item.path,
            task=item.task,
            encoder=encoder,
            vlm=vlm,
            target_tokens=args.compression,
            system_prompt=system_prompt,
            fallback_tts=fallback_tts,
            output_dir=output_dir if (fallback_tts is not None and fallback_tts.available) else None,
            sample_id=item.id,
            verbose=args.verbose,
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=args.do_sample,
            avoid_unknown_labels=args.avoid_unknown_labels,
        )
        
        results.append(result)
        
        if not error:
            print(f"  response: {result.response_text!r}")
            print(f"  classification: {result.classification_label}")
            print(f"  spoken: {result.spoken_sentence}")
            print(f"  latency (ms): capture={result.capture_ms:.0f}, "
                  f"compress={result.compress_ms:.0f}, "
                  f"vlm_ttft={result.vlm_ttft_ms:.0f}, "
                  f"vlm_total={result.vlm_total_ms:.0f}, "
                  f"e2e={result.e2e_total_ms:.0f}")
            if result.wav_path:
                print(f"  audio: {result.wav_path}")

    merged_output = args.single_wav_output
    single_wav_only = args.single_wav_only and (not args.keep_per_sample_wavs)
    merge_status = {
        "single_wav_output": None,
        "single_wav_success": None,
        "single_wav_error": None,
    }

    if fallback_tts is not None and fallback_tts.available:
        if not merged_output:
            merged_output = str(output_dir / "combined_tts.wav")

        successful_wavs = [r.wav_path for r in results if r.error is None and r.wav_path]
        ok_merge, merge_error = merge_wav_files(successful_wavs, merged_output) if successful_wavs else (False, "no successful sample WAV files")
        merge_status = {
            "single_wav_output": merged_output,
            "single_wav_success": bool(ok_merge),
            "single_wav_error": merge_error,
        }

        if ok_merge and single_wav_only:
            for path in successful_wavs:
                if path and Path(path).resolve() != Path(merged_output).resolve():
                    try:
                        Path(path).unlink(missing_ok=True)
                    except Exception:
                        pass
            for r in results:
                if r.wav_path:
                    r.wav_path = merged_output
    
    # ── Write Report ──────────────────────────────────────────────────
    if args.report_file:
        report_path = Path(args.report_file)
    else:
        now = time.strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"report_{now}.json"
    
    successful = sum(1 for r in results if r.error is None)
    wav_count = sum(1 for r in results if r.wav_path)
    if fallback_tts is not None and fallback_tts.last_backend_used:
        tts_runtime_checks["fallback_backend_used"] = fallback_tts.last_backend_used

    degeneration = _summarize_results(results, max_new_tokens=max_new_tokens)
    actual_hidden_size = loaded.identity.get("hidden_size")
    integrity = {
        "model_hidden_size_expected": expected_hidden_size,
        "model_hidden_size_actual": actual_hidden_size,
        "model_hidden_size_valid": (
            True if expected_hidden_size is None else (actual_hidden_size == expected_hidden_size)
        ),
        "compression_input_tokens": probe_n_tokens,
        "compression_valid_targets": valid_targets,
        "compression_selected_valid": args.compression in valid_targets,
    }

    report_data = {
        "schema_version": "v2",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "dtype": str(dtype),
        "runtime_env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "transformers": __import__("transformers").__version__,
        },
        # Backward-compatible aliases for tools expecting top-level config/summary.
        "config": {
            "output_dir": str(output_dir),
            "strict_demo": args.strict_demo,
        },
        "model_config": {
            "siglip": args.siglip,
            "llm": args.llm,
            "image_proj": image_proj_path,
            "llm_mode": args.llm_mode,
            "model_identity": loaded.identity,
            "expected_hidden_size": expected_hidden_size,
            "compression": args.compression,
            "tts_enabled": bool(args.enable_tts and not args.no_tts),
            "tts_model": None,
            "strict_demo": args.strict_demo,
            "profile": args.profile,
            "seed": args.seed,
            "max_new_tokens": max_new_tokens,
            "generation_config": {
                "do_sample": bool(args.do_sample),
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
            },
            "avoid_unknown_labels": bool(args.avoid_unknown_labels),
            "warmup_images": args.warmup_images,
            "single_wav_output": args.single_wav_output,
            "single_wav_only": args.single_wav_only,
            "keep_per_sample_wavs": args.keep_per_sample_wavs,
        },
        "pipeline_integrity": integrity,
        "degeneration": degeneration,
        "tts_backend": {
            **backend_state(requested_tts_backend, active_tts_backend, tts_fallback_reason),
        },
        "tts_runtime_checks": tts_runtime_checks,
        "audio_output": merge_status,
        "summary": {
            "success_count": successful,
            "total": len(results),
            "wav_generated": wav_count,
            "tts_requested": requested_tts_backend != "disabled",
        },
        "results": [asdict(r) for r in results],
    }
    
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\nReport written to: {report_path}")
    
    # ── Summary ────────────────────────────────────────────────────────
    print(f"\nSummary: {successful}/{len(results)} successful")
    
    if successful > 0:
        e2e_times = [r.e2e_total_ms for r in results if r.error is None]
        print(f"  E2E timing (ms): p50={np.median(e2e_times):.0f}, "
              f"mean={np.mean(e2e_times):.0f}, "
              f"p95={np.percentile(e2e_times, 95):.0f}")
        print(f"  WAV artifacts: {wav_count}/{len(results)}")
        if args.strict_demo and requested_tts_backend != "disabled" and wav_count == 0:
            print("ERROR: strict-demo enabled and no WAV artifacts were produced")
            return 2
    
    return 0 if successful == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
