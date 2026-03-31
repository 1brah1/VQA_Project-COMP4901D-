#!/usr/bin/env python3
"""
Legacy compatibility runner for VQA + TTS.

This script now delegates to scripts/run_integrated.py and uses the
Jetson-compatible fallback backend stack (piper,silero,pyttsx3).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="VQA + TTS wrapper (delegates to run_integrated.py)")
    p.add_argument("--labels", type=str, default="data/eval/labels.json")
    p.add_argument("--llm", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--llm_mode", type=str, default="fp16", choices=["fp16", "awq"])
    p.add_argument("--compression", type=int, default=192)
    p.add_argument("--max_new_tokens", type=int, default=24)
    p.add_argument("--enable_tts", action="store_true", default=False)
    p.add_argument("--strict_tts", action="store_true", default=False)
    p.add_argument("--profile", type=str, default="sentence_demo_fast",
                   choices=["label_only_eval", "sentence_demo_fast", "sentence_demo_quality"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--single_wav_output", type=str, default=None)
    p.add_argument("--single_wav_only", action="store_true", default=True)
    p.add_argument("--keep_per_sample_wavs", action="store_true", default=False)
    p.add_argument("--output", type=str, default="reports/vqa_with_tts_results.json")
    p.add_argument("--audio_dir", type=str, default="reports/vqa_audio")
    p.add_argument("--tts_fallback", type=str, default="piper,silero,pyttsx3")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    integrated = repo_root / "scripts" / "run_integrated.py"

    cmd = [
        sys.executable,
        str(integrated),
        "--labels", args.labels,
        "--llm", args.llm,
        "--llm-mode", args.llm_mode,
        "--compression", str(args.compression),
        "--max-new-tokens", str(args.max_new_tokens),
        "--profile", args.profile,
        "--seed", str(args.seed),
        "--output-dir", args.audio_dir,
        "--report-file", args.output,
        "--tts-fallback", args.tts_fallback,
    ]

    if args.max_images is not None:
        cmd.extend(["--max-images", str(args.max_images)])

    if args.enable_tts:
        cmd.append("--enable-tts")
    else:
        cmd.append("--no-tts")

    if args.strict_tts:
        cmd.append("--strict-demo")

    if args.single_wav_output:
        cmd.extend(["--single-wav-output", args.single_wav_output])

    if args.single_wav_only:
        cmd.append("--single-wav-only")

    if args.keep_per_sample_wavs:
        cmd.append("--keep-per-sample-wavs")

    print("[run_vqa_with_tts] Delegating to run_integrated.py")
    print("[run_vqa_with_tts] Command:", " ".join(cmd))

    proc = subprocess.run(cmd)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
