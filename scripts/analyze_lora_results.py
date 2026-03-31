#!/usr/bin/env python3
"""Analyze and summarize LoRA validation results."""

import sys
import json
from pathlib import Path
from typing import Dict, Any

_ROOT = Path(__file__).resolve().parent.parent

def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON safely with error handling."""
    if not path.exists():
        print(f"[WARNING] Not found: {path}")
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load {path}: {e}")
        return {}


def analyze_lora_results(validation_path: Path, baseline_path: Path = None):
    """Compare LoRA validation against baseline."""
    
    print("\n" + "="*70)
    print("LoRA FINE-TUNING RESULTS ANALYSIS")
    print("="*70)
    
    # Load validation results
    results = load_json(validation_path)
    if not results:
        print("[ERROR] No validation results found!")
        return False
    
    # Compute summary statistics
    total_samples = len(results.get('predictions', []))
    overall_acc = results.get('accuracy', 0)
    bang_rate = results.get('bang_rate', 0)
    unknown_rate = results.get('unknown_rate', 0)
    
    by_task = results.get('by_task', {})
    
    # Print summary
    print(f"\n📊 OVERALL METRICS")
    print(f"  Total Samples: {total_samples}")
    print(f"  Accuracy:      {overall_acc*100:.1f}%")
    print(f"  Bang Rate:     {bang_rate*100:.1f}% (task mismatch)")
    print(f"  Unknown Rate:  {unknown_rate*100:.1f}% (invalid response)")
    
    # Print per-task breakdown
    print(f"\n📈 PER-TASK BREAKDOWN")
    for task, metrics in by_task.items():
        acc = metrics.get('accuracy', 0)
        count = metrics.get('count', 0)
        print(f"  {task:20s}: {acc*100:5.1f}% ({count} samples)")
    
    # Print success/warning
    print(f"\n🎯 SUCCESS CRITERIA")
    success = True
    
    # Accuracy check
    if overall_acc >= 0.55:
        print(f"  ✅ Accuracy ≥ 55%: {overall_acc*100:.1f}%")
    else:
        print(f"  ❌ Accuracy < 55%: {overall_acc*100:.1f}% (target: 55%)")
        success = False
    
    # Degeneration check
    if bang_rate == 0:
        print(f"  ✅ Zero Degeneration: {bang_rate*100:.1f}%")
    else:
        print(f"  ⚠️  Degeneration detected: {bang_rate*100:.1f}% samples off-task")
        success = False
    
    # Compare to baseline if provided
    if baseline_path and baseline_path.exists():
        baseline = load_json(baseline_path)
        baseline_acc = baseline.get('accuracy', 0)
        improvement = (overall_acc - baseline_acc) * 100
        print(f"\n📊 COMPARISON TO BASELINE")
        print(f"  Baseline Accuracy:  {baseline_acc*100:.1f}%")
        print(f"  LoRA Accuracy:      {overall_acc*100:.1f}%")
        print(f"  Improvement:        +{improvement:.1f}pp" if improvement > 0 else f"  Change:             {improvement:.1f}pp")
    
    # Final recommendation
    print(f"\n🚀 RECOMMENDATION")
    if success and overall_acc >= 0.55:
        print("  ✅ READY FOR JETSON DEPLOYMENT")
        print("     Next: Run Jetson evaluation")
    elif overall_acc >= 0.52:
        print("  ⚠️  MARGINAL — Consider iterating")
        print("     Options: Re-train with 3 epochs or add more training data")
    else:
        print("  ❌ NOT READY — Below target threshold")
        print("     Recommendation: Collect more diverse training samples or adjust LoRA config")
    
    print()
    
    return success


if __name__ == '__main__':
    validation_file = _ROOT / 'reports' / 'lora_validation_local.json'
    baseline_file = _ROOT / 'reports' / 'fp16_benchmark.json'
    
    success = analyze_lora_results(validation_file, baseline_file)
    sys.exit(0 if success else 1)
