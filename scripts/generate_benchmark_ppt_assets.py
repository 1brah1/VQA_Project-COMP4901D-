#!/usr/bin/env python3
"""Generate PPT-ready PNG charts and a findings summary from benchmark sweeps."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def _infer_model_size(llm_name: str) -> str:
    low = (llm_name or "").lower()
    if "0.5b" in low or "0p5b" in low:
        return "0.5B"
    if "1.5b" in low or "1p5b" in low:
        return "1.5B"
    return "Unknown"


def _series_label(data: Dict[str, Any]) -> str:
    llm = str(data.get("llm", ""))
    size = _infer_model_size(llm)
    mode = str(data.get("llm_mode", "fp16")).lower()
    mode_label = "FP16" if mode == "fp16" else "AWQ-request"
    lora = " + LoRA" if data.get("lora_adapter") else ""
    bench = data.get("benchmark_config", {}) if isinstance(data.get("benchmark_config"), dict) else {}
    max_tokens = bench.get("max_new_tokens_used")
    prompt_style = bench.get("prompt_style")
    suffix_parts = []
    if prompt_style:
        suffix_parts.append(str(prompt_style))
    if max_tokens is not None:
        suffix_parts.append(f"t{max_tokens}")
    suffix = f" ({' '.join(suffix_parts)})" if suffix_parts else ""
    return f"{size} {mode_label}{lora}{suffix}".strip()


def _load_series(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    comp = data.get("compression", {})
    records = []
    for comp_key, comp_data in comp.items():
        try:
            comp_int = int(comp_key)
        except Exception:
            continue
        timing = comp_data.get("timing_s", {})
        total_mean = (timing.get("total") or {}).get("mean")
        llm_mean = (timing.get("llm") or {}).get("mean")
        records.append(
            {
                "compression": comp_int,
                "accuracy_gt_known": comp_data.get("accuracy_gt_known"),
                "unknown_rate": comp_data.get("unknown_rate"),
                "total_mean_ms": (total_mean * 1000.0) if total_mean is not None else None,
                "llm_mean_ms": (llm_mean * 1000.0) if llm_mean is not None else None,
                "peak_mem_mb": comp_data.get("peak_mem_mb"),
            }
        )
    records.sort(key=lambda r: r["compression"])
    return {
        "path": str(path),
        "label": _series_label(data),
        "llm": str(data.get("llm", "")),
        "llm_mode": str(data.get("llm_mode", "")),
        "lora_adapter": data.get("lora_adapter"),
        "records": records,
    }


def _plot_metric(series_list: List[Dict[str, Any]], metric: str, y_label: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for series in series_list:
        xs = [r["compression"] for r in series["records"] if r.get(metric) is not None]
        ys = [r[metric] for r in series["records"] if r.get(metric) is not None]
        if not xs:
            continue
        plt.plot(xs, ys, marker="o", linewidth=2, label=series["label"])
    plt.xlabel("Compression target tokens")
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def _best_record(records: List[Dict[str, Any]], key: str, higher_is_better: bool) -> Optional[Dict[str, Any]]:
    valid = [r for r in records if r.get(key) is not None]
    if not valid:
        return None
    return max(valid, key=lambda r: r[key]) if higher_is_better else min(valid, key=lambda r: r[key])


def _write_findings(series_list: List[Dict[str, Any]], out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# Findings Summary (Compression Benchmarks)")
    lines.append("")
    lines.append("## Design optimization objective")
    lines.append("- Reduce SigLIP patch tokens (576 -> 192/81/36/9) to lower end-to-end latency while keeping label accuracy usable.")
    lines.append("- Keep the pipeline edge-friendly: deterministic compression + short label outputs for reliable parsing.")
    lines.append("")
    lines.append("## System constraints (Jetson)")
    lines.append("- Memory and latency limits on Jetson Orin NX require smaller models and shorter generations.")
    lines.append("- AWQ on aarch64 can fall back to FP16, so AWQ-request results are not pure INT4 kernels.")
    lines.append("- VibeVoice is not compatible with the Jetson Python/transformers stack; Silero is the usable fallback.")
    lines.append("")
    lines.append("## Solutions utilized")
    lines.append("- Deterministic token compression before VLM inference to trade resolution for speed.")
    lines.append("- Prompt styles that force label-only outputs to reduce unknowns and parsing errors.")
    lines.append("- Optional LoRA adapters and self-speculative decoding (PPSD-inspired) for future speedups.")
    lines.append("")
    lines.append("## Key benchmark findings")
    lines.append("| Series | Best accuracy_gt_known | Compression | Best total mean (ms) | Compression |")
    lines.append("|---|---:|---:|---:|---:|")
    notes: List[str] = []
    for series in series_list:
        best_acc = _best_record(series["records"], "accuracy_gt_known", True)
        best_lat = _best_record(series["records"], "total_mean_ms", False)
        acc_val = "n/a" if best_acc is None else f"{best_acc['accuracy_gt_known']:.3f}"
        acc_comp = "n/a" if best_acc is None else str(best_acc["compression"])
        lat_val = "n/a" if best_lat is None else f"{best_lat['total_mean_ms']:.1f}"
        lat_comp = "n/a" if best_lat is None else str(best_lat["compression"])
        lines.append(f"| {series['label']} | {acc_val} | {acc_comp} | {lat_val} | {lat_comp} |")

        acc_all = [r.get("accuracy_gt_known") for r in series["records"] if r.get("accuracy_gt_known") is not None]
        if acc_all and max(acc_all) == 0:
            notes.append(
                f"- Note: {series['label']} produced 0 accuracy across compressions "
                "(likely model or prompt instability on device)."
            )

    if notes:
        lines.append("")
        lines.append("## Notes")
        lines.extend(notes)

    lines.append("")
    lines.append("## Charts generated")
    lines.append("- accuracy_vs_compression.png")
    lines.append("- unknown_rate_vs_compression.png")
    lines.append("- latency_vs_compression.png")
    lines.append("- memory_vs_compression.png")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate PPT assets from benchmark sweeps")
    parser.add_argument("--inputs", type=str, default="reports/sweeps/*.json")
    parser.add_argument("--out-dir", type=str, default="reports/ppt_assets")
    args = parser.parse_args()

    paths = [Path(p) for p in glob.glob(args.inputs)]
    if not paths:
        raise SystemExit(f"No sweep files found for pattern: {args.inputs}")

    series_list = [_load_series(p) for p in sorted(paths)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_metric(series_list, "accuracy_gt_known", "Accuracy (GT known)", out_dir / "accuracy_vs_compression.png")
    _plot_metric(series_list, "unknown_rate", "Unknown rate", out_dir / "unknown_rate_vs_compression.png")
    _plot_metric(series_list, "total_mean_ms", "Total mean latency (ms)", out_dir / "latency_vs_compression.png")
    _plot_metric(series_list, "peak_mem_mb", "Peak memory (MB)", out_dir / "memory_vs_compression.png")

    _write_findings(series_list, out_dir / "findings_summary.md")
    print(f"Wrote charts and summary to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
