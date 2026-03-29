"""
scripts/run_pipelined.py
========================
End-to-end pipelined VQA with PPSD speculative decoding + streaming TTS.

Pipeline stages
---------------
  1. Capture      – load image from disk
  2. Compression  – SigLIP encode → adaptive average-pool token compression
  3. VLM Inference – SelfSpeculativeVLM streaming (PPSD draft + verify)
  4. TTS          – VibeVoice fires after first `--word_threshold` words (default 3)

Outputs a Markdown latency breakdown table at the end, e.g.::

  Stage                            |    p50 (ms) |    p95 (ms) |   mean (ms)
  Capture (load image)             |         0.3 |         0.6 |         0.3
  Compression (SigLIP + pool)      |        42.1 |        46.8 |        43.0
  VLM TTFT (first token)           |       118.3 |       134.5 |       119.7
  VLM Total (all tokens)           |       312.1 |       338.4 |       315.2
  TTS TTFA (first audio)           |       210.4 |       240.1 |       215.3
  E2E → First Audio                |       440.7 |       480.2 |       445.5
  E2E Total                        |      1240.2 |      1380.1 |      1260.4

Usage
-----
  # VLM-only (no TTS):
  python scripts/run_pipelined.py --labels data/eval/labels.json --no_tts

  # Full pipeline with TTS:
  python scripts/run_pipelined.py \\
      --labels data/eval/labels.json \\
      --tts microsoft/VibeVoice-Realtime-0.5B \\
      --voices_dir C:/Users/hash_/VibeVoice/voices/streaming_model

  # Measure without playing audio (just collect chunks):
  python scripts/run_pipelined.py --labels data/eval/labels.json \\
      --tts microsoft/VibeVoice-Realtime-0.5B \\
      --voices_dir C:/Users/hash_/VibeVoice/voices/streaming_model \\
      --no_play

Python 3.8-compatible.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# Project root on sys.path so imports work when run from repo root
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.vision.siglip_encoder import SiglipPatchEncoder
from src.vision.token_compression import compress_27x27_tokens
from src.vlm.model import SimplePrefixVLM
from src.vlm.pipelined_vlm import SelfSpeculativeVLM, SpecStats
from src.prompts.load_prompt import load_system_prompt
from src.tts.streaming_bridge import VibeVoiceTTSService, WordBufferedTTSBridge


# ─────────────────────────────────────────────────────────────────────────────
# Label loading (matches benchmark_compression.py schema)
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass as _dc


@_dc
class _EvalItem:
    id: str
    path: str   # relative to repo root, e.g. data/eval/images/crosswalk/Crosswalk_1.png
    task: str   # "crosswalk_signal" | "stairs" | "obstacles"


def _load_items(labels_path: Path) -> List[_EvalItem]:
    obj = json.loads(labels_path.read_text(encoding="utf-8"))
    return [
        _EvalItem(id=it["id"], path=it["path"], task=it["task"])
        for it in obj.get("items", [])
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Task helpers
# ─────────────────────────────────────────────────────────────────────────────

_TASK_PROMPTS: Dict[str, str] = {
    "crosswalk_signal": "Is the crosswalk signal red or green?",
    "stairs":           "Are there stairs ahead?",
    "obstacles":        "Is there an obstacle in the path?",
    # fallback aliases
    "crosswalk":        "Is the crosswalk signal red or green?",
    "obstacle":         "Is there an obstacle in the path?",
}


# ─────────────────────────────────────────────────────────────────────────────
# Stage timer
# ─────────────────────────────────────────────────────────────────────────────

class StageTimer:
    """Lightweight dict-backed timer for stage-wise profiling."""

    def __init__(self) -> None:
        self._marks: Dict[str, float] = {}

    def mark(self, name: str) -> None:
        self._marks[name] = time.perf_counter()

    def elapsed_ms(self, start: str, end: str) -> float:
        if start not in self._marks or end not in self._marks:
            return 0.0
        return (self._marks[end] - self._marks[start]) * 1000.0


# ─────────────────────────────────────────────────────────────────────────────
# Single-image pipeline run
# ─────────────────────────────────────────────────────────────────────────────

def run_one(
    image_path: str,
    task: str,
    encoder: SiglipPatchEncoder,
    spec_vlm: SelfSpeculativeVLM,
    target_tokens: int,
    system_prompt: str,
    tts_service: Optional[VibeVoiceTTSService] = None,
    word_threshold: int = 3,
    play_audio: bool = True,
    verbose: bool = False,
) -> Tuple[Dict[str, float], str, Optional[SpecStats]]:
    """
    Run the full pipeline on one image.

    Returns
    -------
    (latencies_ms, generated_text, spec_stats)
      latencies_ms : dict of stage name → elapsed milliseconds
      generated_text : the VLM's response string
      spec_stats     : SpecStats from SelfSpeculativeVLM (or None on error)
    """
    timer = StageTimer()
    timer.mark("start")

    # ── Stage 1: Capture ────────────────────────────────────────────────────
    image = Image.open(image_path).convert("RGB")
    timer.mark("capture")

    # ── Stage 2: Compression ────────────────────────────────────────────────
    with torch.no_grad():
        image_tokens = encoder.encode(image)                          # (1, N, D)
        image_tokens = compress_27x27_tokens(image_tokens, target_tokens=target_tokens)
    timer.mark("compress")

    # ── Stage 3: VLM streaming + Stage 4: TTS ───────────────────────────────
    user_prompt = _TASK_PROMPTS.get(task, _TASK_PROMPTS["obstacle"])

    bridge: Optional[WordBufferedTTSBridge] = None
    if tts_service is not None:
        bridge = WordBufferedTTSBridge(
            tts_service,
            word_threshold=word_threshold,
            play_audio=play_audio,
        )
        bridge.start()

    text_parts: List[str] = []
    spec_stats: Optional[SpecStats] = None
    first_token_marked = False

    gen = spec_vlm.generate_streaming(
        image_tokens=image_tokens,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_new_tokens=64,
    )
    try:
        while True:
            chunk, _accepted = next(gen)
            if not first_token_marked:
                timer.mark("vlm_first_token")
                first_token_marked = True
            text_parts.append(chunk)
            if bridge is not None:
                bridge.feed(chunk)
            if verbose:
                print(chunk, end="", flush=True)
    except StopIteration as e:
        spec_stats = e.value

    timer.mark("vlm_done")
    if verbose:
        print()

    if bridge is not None:
        bridge.flush()
        bridge.wait(timeout=30.0)
        bev = bridge.events
    else:
        bev = None

    text = "".join(text_parts).strip()

    # ── Collect latencies ────────────────────────────────────────────────────
    lat: Dict[str, float] = {
        "capture_ms":  timer.elapsed_ms("start",   "capture"),
        "compress_ms": timer.elapsed_ms("capture", "compress"),
        "vlm_ttft_ms": timer.elapsed_ms("compress", "vlm_first_token"),
        "vlm_total_ms": timer.elapsed_ms("compress", "vlm_done"),
    }

    if bev is not None and bev.t_first_audio > 0.0:
        lat["tts_ttfa_ms"]        = bev.tts_first_audio_ms
        lat["e2e_first_audio_ms"] = bev.e2e_first_audio_ms
        lat["e2e_total_ms"]       = bev.e2e_total_ms
    else:
        lat["tts_ttfa_ms"]        = 0.0
        lat["e2e_first_audio_ms"] = timer.elapsed_ms("start", "vlm_done")
        lat["e2e_total_ms"]       = timer.elapsed_ms("start", "vlm_done")

    return lat, text, spec_stats


# ─────────────────────────────────────────────────────────────────────────────
# Latency table
# ─────────────────────────────────────────────────────────────────────────────

_STAGES: List[Tuple[str, str]] = [
    ("capture_ms",          "Capture (load image)"),
    ("compress_ms",         "Compression (SigLIP + pool)"),
    ("vlm_ttft_ms",         "VLM TTFT (first token)"),
    ("vlm_total_ms",        "VLM Total (all tokens)"),
    ("tts_ttfa_ms",         "TTS TTFA (first audio)"),
    ("e2e_first_audio_ms",  "E2E to First Audio"),
    ("e2e_total_ms",        "E2E Total"),
]


def print_latency_table(all_latencies: List[Dict[str, float]]) -> None:
    if not all_latencies:
        print("No latency data collected.")
        return

    col = 12
    sep = f"{'Stage':<36} | {'p50 (ms)':>{col}} | {'p95 (ms)':>{col}} | {'mean (ms)':>{col}}"
    bar = "=" * len(sep)

    print()
    print(bar)
    print("  STAGE-WISE LATENCY BREAKDOWN")
    print(bar)
    print(sep)
    print("-" * len(sep))

    for key, label in _STAGES:
        vals = [d[key] for d in all_latencies if d.get(key, 0.0) > 0.0]
        if not vals:
            p50 = p95 = mean = "n/a"
        else:
            arr = np.array(vals, dtype=float)
            p50  = f"{float(np.median(arr)):.1f}"
            p95  = f"{float(np.percentile(arr, 95)):.1f}"
            mean = f"{float(arr.mean()):.1f}"
        print(f"{label:<36} | {p50:>{col}} | {p95:>{col}} | {mean:>{col}}")

    print(bar)
    print(f"  n = {len(all_latencies)} image(s)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipelined VQA: PPSD speculative decoding + streaming TTS."
    )
    parser.add_argument("--labels",        default="data/eval/labels.json",
                        help="JSON file with image labels")
    parser.add_argument("--siglip",        default="google/siglip-base-patch16-384",
                        help="SigLIP model name or path")
    parser.add_argument("--llm",           default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="LLM model name or path (also accepts AWQ dir)")
    parser.add_argument("--tts",           default="microsoft/VibeVoice-Realtime-0.5B",
                        help="VibeVoice model path or HF repo ID")
    parser.add_argument("--voices_dir",    default=None,
                        help="path to voices/streaming_model directory")
    parser.add_argument("--target_tokens", type=int, default=192,
                        help="token compression target (192, 81, 36, or 9)")
    parser.add_argument("--split_layer",   type=int, default=12,
                        help="PPSD draft/verify split layer (default: 12 of 24)")
    parser.add_argument("--K",             type=int, default=4,
                        help="PPSD draft tokens per verify pass (default: 4)")
    parser.add_argument("--word_threshold", type=int, default=3,
                        help="words to buffer before triggering TTS (default: 3)")
    parser.add_argument("--no_tts",        action="store_true",
                        help="skip TTS (measure VLM latency only)")
    parser.add_argument("--no_play",       action="store_true",
                        help="collect audio but don't play it via sounddevice")
    parser.add_argument("--max_images",    type=int, default=None,
                        help="limit number of images (for quick smoke-tests)")
    parser.add_argument("--verbose",       action="store_true",
                        help="print generated tokens as they stream")
    parser.add_argument("--out",           default=None,
                        help="optional JSON file to write results to")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    print(f"[run_pipelined] device={device}  dtype={dtype}")

    # ── Labels ───────────────────────────────────────────────────────────────
    labels_path = Path(args.labels)
    if not labels_path.exists():
        sys.exit(f"ERROR: labels file not found: {labels_path}")
    items = _load_items(labels_path)

    # Resolve each item's path relative to repo root
    repo_root = _ROOT
    eval_items: List[_EvalItem] = []
    for item in items:
        img_path = repo_root / item.path
        if img_path.exists():
            eval_items.append(_EvalItem(id=item.id, path=str(img_path), task=item.task))
        else:
            print(f"[run_pipelined] WARNING: image not found, skipping — {img_path}")

    if not eval_items:
        sys.exit(
            "ERROR: no images found on disk.\n"
            "Add images to data/eval/images/ as described in data/eval/README.md"
        )
    if args.max_images:
        eval_items = eval_items[: args.max_images]
    print(f"[run_pipelined] {len(eval_items)} image(s) to process")

    # ── Encoder ──────────────────────────────────────────────────────────────
    print(f"[run_pipelined] Loading SigLIP: {args.siglip}")
    encoder = SiglipPatchEncoder.from_pretrained(
        args.siglip, device=device, dtype=dtype
    )

    # ── VLM ──────────────────────────────────────────────────────────────────
    print(f"[run_pipelined] Loading LLM: {args.llm}")
    vlm = SimplePrefixVLM.from_pretrained(
        args.llm, device=device, dtype=dtype
    )
    spec_vlm = SelfSpeculativeVLM(vlm, split_layer=args.split_layer, K=args.K)
    print(
        f"[run_pipelined] PPSD ready: "
        f"split_layer={args.split_layer}/{spec_vlm._n_layers}  K={args.K}"
    )

    # ── TTS ──────────────────────────────────────────────────────────────────
    tts_service: Optional[VibeVoiceTTSService] = None
    if not args.no_tts:
        if not args.voices_dir:
            print(
                "[run_pipelined] WARNING: --voices_dir not set; skipping TTS.  "
                "Pass --no_tts to suppress this warning."
            )
        else:
            print(f"[run_pipelined] Loading TTS: {args.tts}")
            tts_service = VibeVoiceTTSService(
                model_path=args.tts,
                voices_dir=args.voices_dir,
                device=device,
            )
            tts_service.load()

    system_prompt = load_system_prompt()

    # ── Run pipeline ─────────────────────────────────────────────────────────
    all_latencies: List[Dict[str, float]] = []
    all_spec_stats: List[SpecStats] = []
    all_results: List[Dict[str, object]] = []

    for i, item in enumerate(eval_items):
        img_path = item.path
        img_name = Path(img_path).name
        task = item.task
        print(f"\n[{i+1}/{len(eval_items)}] {img_name}  task={task}")

        try:
            lat, text, spec_stats = run_one(
                image_path=img_path,
                task=task,
                encoder=encoder,
                spec_vlm=spec_vlm,
                target_tokens=args.target_tokens,
                system_prompt=system_prompt,
                tts_service=tts_service,
                word_threshold=args.word_threshold,
                play_audio=not args.no_play,
                verbose=args.verbose,
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            import traceback; traceback.print_exc()
            continue

        all_latencies.append(lat)
        if spec_stats:
            all_spec_stats.append(spec_stats)
        all_results.append({"image": img_name, "task": task, "response": text, **lat})

        # Per-image one-liner
        print(
            f"  response : {text!r}\n"
            f"  capture={lat['capture_ms']:.0f}ms  "
            f"compress={lat['compress_ms']:.0f}ms  "
            f"vlm_ttft={lat['vlm_ttft_ms']:.0f}ms  "
            f"vlm_total={lat['vlm_total_ms']:.0f}ms  "
            f"tts_ttfa={lat['tts_ttfa_ms']:.0f}ms  "
            f"e2e={lat['e2e_first_audio_ms']:.0f}ms"
        )
        if spec_stats:
            print(f"  spec     : {spec_stats.summary()}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print_latency_table(all_latencies)

    if all_spec_stats:
        total_acc  = sum(s.accepted_drafts       for s in all_spec_stats)
        total_cand = sum(s.total_draft_candidates for s in all_spec_stats)
        mean_su    = float(np.mean([s.speedup for s in all_spec_stats]))
        rate_str   = f"{total_acc/total_cand:.0%}" if total_cand else "n/a"
        print(
            f"PPSD aggregate ({len(all_spec_stats)} runs):  "
            f"{total_acc}/{total_cand} drafts accepted ({rate_str})  |  "
            f"mean speedup ~{mean_su:.2f}x"
        )
        print()

    # ── Optional JSON output ──────────────────────────────────────────────────
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
