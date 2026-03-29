#!/usr/bin/env python3
"""
scripts/run_integrated.py
==========================
Unified one-flow orchestrator for VQA + VibeVoice TTS on Jetson.

Merges run_once.py and run_pipelined.py into one persistent service that:
  • Loads all models once (SigLIP, Qwen, VibeVoice)
  • Supports both single-image and batch (labels.json) modes
  • Generates TTS audio and WAV outputs
  • Produces JSON report with per-sample timings and artifacts
  • Keeps models warm in memory for repeated requests

Usage Examples
--------------
  # Single image with TTS:
  python scripts/run_integrated.py \\
    --image-path data/eval/images/crosswalk/Crosswalk_1.png \\
    --task crosswalk_signal \\
    --compression 192 \\
    --tts microsoft/VibeVoice-Realtime-0.5B \\
    --voices-dir ~/vibevoice_test/voices \\
    --output-dir outputs/

  # Batch eval (16 images):
  python scripts/run_integrated.py \\
    --labels data/eval/labels.json \\
    --compression 192 \\
    --tts microsoft/VibeVoice-Realtime-0.5B \\
    --voices-dir ~/vibevoice_test/voices \\
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
import re
import sys
import time
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
from src.vision.siglip_encoder import SiglipPatchEncoder
from src.vision.token_compression import compress_27x27_tokens, recommended_targets
from src.vlm.model import SimplePrefixVLM
from src.vlm.llm_loader import load_llm_fp16_or_fp32, load_llm_awq
from src.tts.streaming_bridge import VibeVoiceTTSService, WordBufferedTTSBridge
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
    classification: str  # normalized verdict
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
        "Is the crosswalk walk signal red or green? "
        "Start your response with exactly one word: red|green|unknown. "
        "Then give one short action clause."
    ),
    "stairs": (
        "Are there stairs or steps visible? "
        "Start your response with exactly one word: yes|no|unknown. "
        "Then give one short action clause."
    ),
    "obstacles": (
        "Is there an obstacle ahead? "
        "Start your response with exactly one word: yes|no|unknown. "
        "Then give one short action clause."
    ),
}


def get_task_prompt(task: Optional[str]) -> str:
    """Get prompt for a task."""
    return _TASK_PROMPTS.get(task or "obstacles", _TASK_PROMPTS["obstacles"])


def normalize_response(text: str, task: Optional[str]) -> str:
    """Normalize VLM response into task-specific classification."""
    t = (text or "").strip()
    low = t.lower()
    
    if task == "crosswalk_signal":
        if re.search(r"\bred\b", low):
            return "red"
        if re.search(r"\bgreen\b", low):
            return "green"
        return "unknown"
    
    if task in ("stairs", "obstacles"):
        if re.search(r"\bno\b|\bnone\b|\bclear\b", low):
            return "no"
        if re.search(r"\byes\b|\bpresent\b|\bobstacle\b|\bstairs\b|\bstep\b", low):
            return "yes"
        return "unknown"
    
    return t


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
    tts_service: Optional[VibeVoiceTTSService] = None,
    output_dir: Optional[Path] = None,
    sample_id: Optional[str] = None,
    play_audio: bool = False,
    verbose: bool = False,
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
        bridge: Optional[WordBufferedTTSBridge] = None
        wav_path: Optional[str] = None
        
        if tts_service is not None:
            bridge = WordBufferedTTSBridge(
                tts_service,
                word_threshold=3,
                play_audio=play_audio,
            )
            bridge.start()
        
        with torch.no_grad():
            gen = vlm.generate(
                image_tokens=image_tokens,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=64,
            )
        
        # Stream or collect response
        timer.mark("vlm_first_token")
        for chunk in gen if isinstance(gen, (list, tuple)) else [gen]:
            text_parts.append(str(chunk))
            if bridge is not None:
                bridge.feed(str(chunk))
            if verbose:
                print(chunk, end="", flush=True)
        
        timer.mark("vlm_done")
        if verbose:
            print()
        
        response_text = "".join(text_parts).strip()
        classification = normalize_response(response_text, task)
        
        # ── Stage 4: TTS Audio ───────────────────────────────────────
        if bridge is not None:
            bridge.flush()
            bridge.wait(timeout=30.0)
            
            # Save merged WAV
            if output_dir is not None and bridge.audio_chunks:
                output_dir.mkdir(parents=True, exist_ok=True)
                wav_name = f"{sample_id or Path(image_path).stem}.wav"
                wav_path = str(output_dir / wav_name)
                
                merged_audio = np.concatenate(bridge.audio_chunks, axis=0)
                wav_saved = save_wav(merged_audio, wav_path, sample_rate=24000, verbose=verbose)
                if wav_saved:
                    print(f"  WAV saved: {wav_path}")
                else:
                    print(f"  WARNING: Could not save WAV to {wav_path}")
                    wav_path = None if not wav_saved else wav_path
            
            bev = bridge.events
        else:
            bev = None
        
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
        
        if bev is not None and bev.t_first_audio > 0.0:
            lat["tts_ttfa_ms"]        = bev.tts_first_audio_ms
            lat["e2e_first_audio_ms"] = bev.e2e_first_audio_ms
            lat["e2e_total_ms"]       = bev.e2e_total_ms
        
        result = PipelineResult(
            sample_id=sample_id or Path(image_path).stem,
            image_path=image_path,
            task=task,
            response_text=response_text,
            classification=classification,
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
            classification="error",
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
    parser.add_argument("--llm-mode", type=str, default="fp16",
                        choices=["fp16", "awq"],
                        help="LLM loading mode: fp16 (default) or awq (quantized)")
    parser.add_argument("--tts", type=str, default=None,
                        help="VibeVoice model path; if None, text-only mode")
    parser.add_argument("--voices-dir", type=str, default=None,
                        help="Path to VibeVoice voices directory (e.g., ~/vibevoice_test/voices)")
    parser.add_argument("--no-tts", action="store_true",
                        help="Disable TTS regardless of --tts")
    parser.add_argument("--output-dir", type=str, default="outputs/",
                        help="Output directory for WAV files and JSON report")
    parser.add_argument("--report-file", type=str, default=None,
                        help="JSON report output path (default: {output_dir}/report_{timestamp}.json)")
    parser.add_argument("--play-audio", action="store_true",
                        help="Play TTS audio via sounddevice (if available)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print generated tokens as they stream")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit number of images for testing")
    
    args = parser.parse_args()
    
    # ── Setup ────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[integrated] device={device}, dtype={dtype}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ── Load Models (warm-start) ──────────────────────────────────────
    print(f"[integrated] Loading SigLIP: {args.siglip}")
    encoder = SiglipPatchEncoder.from_pretrained(
        args.siglip, device=device, dtype=dtype
    )
    
    print(f"[integrated] Loading LLM: {args.llm} (mode={args.llm_mode})")
    if args.llm_mode == "awq":
        vlm = load_llm_awq(args.llm, device=device)
    else:
        vlm = load_llm_fp16_or_fp32(args.llm, device=device)
    
    # ── Load TTS (if enabled) ────────────────────────────────────────
    tts_service: Optional[VibeVoiceTTSService] = None
    if args.tts and not args.no_tts:
        voices_dir = args.voices_dir or os.path.expanduser("~/vibevoice_test/voices")
        print(f"[integrated] Loading TTS: {args.tts}")
        print(f"[integrated] Voices directory: {voices_dir}")
        
        # Validate voices directory exists
        voices_path = Path(voices_dir)
        if not voices_path.exists():
            print(f"  ERROR: Voices directory not found: {voices_dir}")
            return 1
        
        pt_files = list(voices_path.glob("*.pt"))
        if not pt_files:
            print(f"  ERROR: No .pt voice presets found in {voices_dir}")
            return 1
        
        print(f"  Found {len(pt_files)} voice preset(s)")
        
        tts_service = VibeVoiceTTSService(
            model_path=args.tts,
            voices_dir=voices_dir,
            device=device,
            inference_steps=5,  # faster inference on Jetson
        )
        tts_service.load()
    
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
    
    print(f"[integrated] Processing {len(items)} item(s)")
    
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
            tts_service=tts_service,
            output_dir=output_dir if tts_service else None,
            sample_id=item.id,
            play_audio=args.play_audio,
            verbose=args.verbose,
        )
        
        results.append(result)
        
        if not error:
            print(f"  response: {result.response_text!r}")
            print(f"  classification: {result.classification}")
            print(f"  latency (ms): capture={result.capture_ms:.0f}, "
                  f"compress={result.compress_ms:.0f}, "
                  f"vlm_ttft={result.vlm_ttft_ms:.0f}, "
                  f"vlm_total={result.vlm_total_ms:.0f}, "
                  f"e2e={result.e2e_total_ms:.0f}")
            if result.wav_path:
                print(f"  audio: {result.wav_path}")
    
    # ── Write Report ──────────────────────────────────────────────────
    if args.report_file:
        report_path = Path(args.report_file)
    else:
        now = time.strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"report_{now}.json"
    
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "dtype": str(dtype),
        "model_config": {
            "siglip": args.siglip,
            "llm": args.llm,
            "llm_mode": args.llm_mode,
            "compression": args.compression,
            "tts_enabled": tts_service is not None,
            "tts_model": args.tts,
        },
        "results": [asdict(r) for r in results],
    }
    
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n✓ Report written to: {report_path}")
    
    # ── Summary ────────────────────────────────────────────────────────
    successful = sum(1 for r in results if r.error is None)
    print(f"\nSummary: {successful}/{len(results)} successful")
    
    if successful > 0:
        e2e_times = [r.e2e_total_ms for r in results if r.error is None]
        print(f"  E2E timing (ms): p50={np.median(e2e_times):.0f}, "
              f"mean={np.mean(e2e_times):.0f}, "
              f"p95={np.percentile(e2e_times, 95):.0f}")
    
    return 0 if successful == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
