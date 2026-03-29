#!/usr/bin/env python3
"""
run_vqa_with_tts.py - Full VQA + Text-to-Speech Pipeline

Runs all 28 eval images through the VQA pipeline and generates audio responses.
Saves results and WAV files to reports/vqa_with_tts/

Usage on Jetson:
    python scripts/run_vqa_with_tts.py \
        --llm_mode fp16 \
        --enable_tts \
        --output reports/vqa_with_tts_results.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
import wave
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.prompts.load_prompt import load_system_prompt
from src.vision.siglip_encoder import SiglipPatchEncoder
from src.vision.token_compression import compress_27x27_tokens, recommended_targets
from src.vlm.model import SimplePrefixVLM
from src.vlm.llm_loader import load_llm_awq


@dataclass
class EvalItem:
    id: str
    path: str
    task: str
    labels: Dict[str, str]


@dataclass
class AudioMetrics:
    """Metrics for audio generation"""
    tts_available: bool = False
    tts_init_ms: float = 0.0
    tts_generation_ms: float = 0.0
    wav_path: Optional[str] = None
    wav_duration_ms: float = 0.0
    sample_rate: int = 24000


@dataclass
class VQAResult:
    id: str
    image: str
    task: str
    response: str
    run_mode: str
    gt: str
    pred: str
    capture_ms: float = 0.0
    encode_ms: float = 0.0
    pool_ms: float = 0.0
    compress_ms: float = 0.0
    vlm_ttft_ms: float = 0.0
    vlm_total_ms: float = 0.0
    tts_ttfa_ms: float = 0.0
    e2e_first_audio_ms: float = 0.0
    e2e_total_ms: float = 0.0
    tts_metrics: Dict[str, Any] = field(default_factory=dict)


def load_labels(path: Path) -> List[EvalItem]:
    """Load evaluation labels from JSON"""
    obj = json.loads(path.read_text(encoding="utf-8"))
    items = []
    for it in obj.get("items", []):
        items.append(
            EvalItem(
                id=it["id"],
                path=it["path"],
                task=it["task"],
                labels=dict(it["labels"]),
            )
        )
    return items


def _task_prompt(task: str) -> str:
    """Get system instruction for task"""
    if task == "crosswalk_signal":
        return (
            "Crosswalk walk signal is it red or green? "
            "Start your response with exactly one word: red|green|unknown. "
            "Then give one short action clause."
        )
    if task == "stairs":
        return (
            "Are there stairs or steps? "
            "Start your response with exactly one word: yes|no. "
            "Then give one short action clause."
        )
    if task == "obstacles":
        return (
            "Is there an obstacle ahead? "
            "Start your response with exactly one word: yes|no. "
            "Then give one short action clause."
        )
    return "Give short navigation advice."


def parse_response(text: str, task: str) -> str:
    """Parse response into normalized answer"""
    t = (text or "").strip().lower()
    
    if task == "crosswalk_signal":
        if re.search(r"\bred\b", t):
            return "red"
        if re.search(r"\bgreen\b", t):
            return "green"
        return "unknown"
    
    if task in ("stairs", "obstacles"):
        if re.search(r"\bno\b|\bnone\b|\bclear\b", t):
            return "no"
        if re.search(r"\byes\b|\bpresent\b|\bobstacle\b|\bstairs\b|\bstep\b", t):
            return "yes"
        return "unknown"
    
    return "unknown"


def get_ground_truth(task: str, labels: Dict[str, str]) -> str:
    """Extract ground truth from labels"""
    if task == "crosswalk_signal":
        return labels.get("walk_signal", "unknown")
    if task == "stairs":
        return labels.get("stairs_present", "unknown")
    if task == "obstacles":
        return labels.get("obstacle_present", "unknown")
    return "unknown"


def try_import_vibevoice() -> bool:
    """Try to import VibeVoice TTS"""
    try:
        from src.tts.streaming_bridge import VibeVoiceTTSService
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def try_import_pyttsx3() -> bool:
    """Try to import pyttsx3 (fallback TTS)"""
    try:
        import pyttsx3
        return True
    except ImportError:
        return False


class SimpleTTSFallback:
    """Fallback TTS using pyttsx3 or espeak if nothing else is available"""
    
    def __init__(self):
        self.sample_rate = 24000
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.available = True
        except Exception as e:
            print(f"[TTS] pyttsx3 not available: {e}")
            self.available = False
            self.engine = None
    
    def synthesize_to_wav(self, text: str, output_path: str) -> bool:
        """Generate speech and save to WAV file"""
        if not self.available:
            return False
        try:
            self.engine.save_to_file(text, output_path)
            self.engine.runAndWait()
            return Path(output_path).exists()
        except Exception as e:
            print(f"[TTS] Synthesis failed: {e}")
            return False


def generate_tts_audio(
    text: str,
    output_path: str,
    use_vibevoice: bool = False,
) -> Tuple[bool, float]:
    """
    Generate speech audio from text.
    Returns: (success, latency_ms)
    """
    t0 = time.perf_counter()
    
    if use_vibevoice:
        try:
            from src.tts.streaming_bridge import VibeVoiceTTSService
            
            # Try to load VibeVoice (may require manual setup on Jetson)
            svc = VibeVoiceTTSService(
                "microsoft/VibeVoice-Realtime-0.5B",
                device="cuda" if torch.cuda.is_available() else "cpu",
                inference_steps=5,
            )
            svc.load()
            
            # Generate audio chunks and concatenate
            all_chunks = []
            for chunk in svc.stream(text):
                all_chunks.append(chunk)
            
            if all_chunks:
                audio = np.concatenate(all_chunks)
                audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
                
                with wave.open(output_path, 'w') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(svc.sample_rate)
                    wf.writeframes(audio_int16.tobytes())
                
                t1 = time.perf_counter()
                return True, (t1 - t0) * 1000
        except Exception as e:
            print(f"[TTS] VibeVoice failed: {e}")
    
    # Fallback to pyttsx3
    try:
        fallback = SimpleTTSFallback()
        if fallback.synthesize_to_wav(text, output_path):
            t1 = time.perf_counter()
            return True, (t1 - t0) * 1000
    except Exception as e:
        print(f"[TTS] Fallback TTS failed: {e}")
    
    return False, 0.0


def main() -> None:
    p = argparse.ArgumentParser(description="VQA + TTS Pipeline")
    p.add_argument("--labels", type=str, default="data/eval/labels.json")
    p.add_argument("--siglip", type=str, default="google/siglip-base-patch16-384")
    p.add_argument("--llm", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--llm_mode", type=str, default="fp16", choices=["fp16", "awq"])
    p.add_argument("--compression", type=int, default=192)
    p.add_argument("--max_new_tokens", type=int, default=24)
    p.add_argument("--enable_tts", action="store_true", default=False)
    p.add_argument("--use_vibevoice", action="store_true", default=False)
    p.add_argument("--output", type=str, default="reports/vqa_with_tts_results.json")
    p.add_argument("--audio_dir", type=str, default="reports/vqa_audio")
    args = p.parse_args()
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    audio_dir = Path(args.audio_dir) if args.enable_tts else None
    if audio_dir:
        audio_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[VQA] Device: {device} | Dtype: {dtype}")
    print(f"[VQA] Loading labels from {args.labels}")
    
    items = load_labels(Path(args.labels))
    system_prompt = load_system_prompt()
    
    print(f"[VQA] Loaded {len(items)} evaluation items")
    print(f"[VQA] Loading SigLIP encoder...")
    
    enc = SiglipPatchEncoder.from_pretrained(args.siglip, device=device, dtype=dtype)
    
    print(f"[VQA] Loading LLM ({args.llm_mode})...")
    if args.llm_mode == "fp16":
        vlm = SimplePrefixVLM.from_pretrained(args.llm, device=device, dtype=dtype)
    else:
        loaded = load_llm_awq(args.llm, device=device)
        vlm = SimplePrefixVLM.from_loaded_llm(
            tokenizer=loaded.tokenizer,
            llm=loaded.model,
            device=device,
            dtype=dtype,
        )
    
    if args.enable_tts:
        tts_available = args.use_vibevoice and try_import_vibevoice()
        if not tts_available:
            tts_available = try_import_pyttsx3()
        print(f"[TTS] Available: {tts_available} | Using VibeVoice: {args.use_vibevoice}")
    else:
        tts_available = False
    
    # Process all items
    results: List[Dict[str, Any]] = []
    
    for item in tqdm(items, desc="Processing VQA samples"):
        t_start = time.perf_counter()
        
        # Load and encode image
        try:
            img = Image.open(item.path).convert("RGB")
        except Exception as e:
            print(f"[VQA] Failed to load {item.path}: {e}")
            continue
        
        t_capture = time.perf_counter()
        
        t_encode_start = time.perf_counter()
        patches = enc.encode(img)
        t_encode = time.perf_counter() - t_encode_start
        
        t_compress_start = time.perf_counter()
        patches_c = compress_27x27_tokens(patches, target_tokens=args.compression)
        t_compress = time.perf_counter() - t_compress_start
        
        # Generate VLM response
        t_vlm_start = time.perf_counter()
        user_prompt = _task_prompt(item.task)
        response_text = vlm.generate(
            image_tokens=patches_c,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=args.max_new_tokens,
        )
        t_vlm = time.perf_counter() - t_vlm_start
        
        # Parse response
        pred = parse_response(response_text, item.task)
        gt = get_ground_truth(item.task, item.labels)
        
        # Generate audio (if enabled)
        tts_ttfa_ms = 0.0
        tts_metrics = {}
        wav_path = None
        
        if args.enable_tts and tts_available and audio_dir:
            wav_path = str(audio_dir / f"{item.id}.wav")
            success, tts_gen_ms = generate_tts_audio(
                response_text,
                wav_path,
                use_vibevoice=args.use_vibevoice,
            )
            tts_ttfa_ms = tts_gen_ms
            tts_metrics = {
                "success": success,
                "wav_path": wav_path if success else None,
                "generation_ms": tts_gen_ms,
            }
            if not success:
                wav_path = None
        
        t_end = time.perf_counter()
        
        # Record result
        result = {
            "id": item.id,
            "image": Path(item.path).name,
            "task": item.task,
            "response": response_text,
            "run_mode": "baseline_pipeline_with_tts" if args.enable_tts else "baseline_pipeline",
            "gt": gt,
            "pred": pred,
            "capture_ms": (t_capture - t_start) * 1000,
            "encode_ms": t_encode * 1000,
            "compress_ms": t_compress * 1000,
            "vlm_ttft_ms": t_vlm * 1000,
            "vlm_total_ms": t_vlm * 1000,
            "tts_ttfa_ms": tts_ttfa_ms,
            "e2e_total_ms": (t_end - t_start) * 1000,
            "correct": pred == gt,
            "tts_metrics": tts_metrics,
        }
        results.append(result)
    
    # Summary statistics
    total_correct = sum(1 for r in results if r.get("correct", False))
    overall_accuracy = total_correct / len(results) if results else 0.0
    
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "run_config": {
            "llm_mode": args.llm_mode,
            "compression": args.compression,
            "max_new_tokens": args.max_new_tokens,
            "enable_tts": args.enable_tts,
            "use_vibevoice": args.use_vibevoice,
        },
        "statistics": {
            "total_samples": len(results),
            "overall_accuracy": overall_accuracy,
            "correct_predictions": total_correct,
            "avg_vlm_latency_ms": np.mean([r["vlm_total_ms"] for r in results]),
            "avg_e2e_latency_ms": np.mean([r["e2e_total_ms"] for r in results]),
            "avg_tts_latency_ms": np.mean([r["tts_ttfa_ms"] for r in results]) if args.enable_tts else 0.0,
        },
        "results": results,
    }
    
    # Save results
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[DONE] Results saved to {out_path}")
    print(f"[SUMMARY] Accuracy: {total_correct}/{len(results)} ({overall_accuracy*100:.1f}%)")
    print(f"[SUMMARY] Avg E2E latency: {summary['statistics']['avg_e2e_latency_ms']:.1f}ms")
    
    if args.enable_tts:
        wav_count = sum(1 for r in results if r.get("tts_metrics", {}).get("success", False))
        print(f"[SUMMARY] Audio files: {wav_count}/{len(results)} generated")
        print(f"[SUMMARY] Audio directory: {audio_dir}")


if __name__ == "__main__":
    main()
