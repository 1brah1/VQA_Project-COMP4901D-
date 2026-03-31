#!/usr/bin/env python3
"""
Capture a reproducibility manifest for a VQA/TTS run.

Usage:
  python scripts/capture_run_manifest.py \
    --results reports/vqa_with_tts_results.json \
    --out reports/run_manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _safe_git(cmd: List[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        return out or None
    except Exception:
        return None


def _load_results(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _summarize_results(results_obj: Dict[str, Any]) -> Dict[str, Any]:
    if not results_obj:
        return {"results_found": False}

    stats = results_obj.get("statistics", {})
    run_cfg = results_obj.get("run_config", {})
    rows = results_obj.get("results", [])
    tts_success_count = 0
    backend_usage: Dict[str, int] = {}

    for row in rows:
        tts = row.get("tts_metrics", {}) if isinstance(row, dict) else {}
        if tts.get("success"):
            tts_success_count += 1
        backend = tts.get("backend_used")
        if backend:
            backend_usage[backend] = backend_usage.get(backend, 0) + 1

    return {
        "results_found": True,
        "device": results_obj.get("device"),
        "timestamp": results_obj.get("timestamp"),
        "run_config": run_cfg,
        "statistics": stats,
        "result_count": len(rows),
        "tts_success_count": tts_success_count,
        "tts_backend_usage": backend_usage,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture run manifest for reproducibility")
    parser.add_argument("--results", type=str, default="reports/vqa_with_tts_results.json")
    parser.add_argument("--out", type=str, default="reports/run_manifest.json")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    results_path = (repo_root / args.results).resolve() if not Path(args.results).is_absolute() else Path(args.results)
    out_path = (repo_root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)

    manifest: Dict[str, Any] = {
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "environment": {
            "cwd": os.getcwd(),
            "pythonpath": os.environ.get("PYTHONPATH"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "git": {
            "branch": _safe_git(["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"]),
            "commit": _safe_git(["git", "-C", str(repo_root), "rev-parse", "HEAD"]),
            "dirty": bool(_safe_git(["git", "-C", str(repo_root), "status", "--porcelain"])),
        },
        "artifacts": {
            "results_file": str(results_path),
            "results_exists": results_path.exists(),
            "results_summary": _summarize_results(_load_results(results_path)),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[manifest] wrote: {out_path}")
    summary = manifest["artifacts"]["results_summary"]
    if summary.get("results_found"):
        print(
            "[manifest] results: "
            f"n={summary.get('result_count')} "
            f"acc={summary.get('statistics', {}).get('overall_accuracy')} "
            f"tts_success={summary.get('tts_success_count')}"
        )
    else:
        print("[manifest] no readable results JSON found; manifest captured environment only")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
