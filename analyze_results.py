import json
import statistics
from collections import defaultdict
from pathlib import Path

# Load results
with open('reports/vqa_results.json') as f:
    results = json.load(f)

print("=" * 100)
print("VQA PIPELINE RESULTS ANALYSIS")
print("=" * 100)
print(f"\nTotal samples processed: {len(results)}\n")

# Task breakdown
tasks = defaultdict(list)
for r in results:
    tasks[r['task']].append(r)

print("SAMPLES BY TASK:")
for task, samples in sorted(tasks.items()):
    print(f"  {task}: {len(samples)} samples")

print("\n" + "=" * 100)
print("ACCURACY ANALYSIS")
print("=" * 100)

# Calculate accuracy
total_correct = 0
task_stats = {}

for task, samples in sorted(tasks.items()):
    correct = sum(1 for s in samples if s['pred'] == s['gt'])
    total_correct += correct
    task_stats[task] = {
        'correct': correct,
        'total': len(samples),
        'accuracy': correct / len(samples) * 100 if len(samples) > 0 else 0
    }
    print(f"\n{task.upper()}:")
    print(f"  Accuracy: {correct}/{len(samples)} ({task_stats[task]['accuracy']:.1f}%)")
    
    # Breakdown by prediction
    predictions = defaultdict(int)
    for s in samples:
        predictions[s['pred']] += 1
    print(f"  Predictions: {dict(predictions)}")

overall_accuracy = total_correct / len(results) * 100 if len(results) > 0 else 0
print(f"\nOVERALL ACCURACY: {total_correct}/{len(results)} ({overall_accuracy:.1f}%)")

print("\n" + "=" * 100)
print("LATENCY ANALYSIS (milliseconds)")
print("=" * 100)

# Latency metrics
encode_times = [r['encode_ms'] for r in results]
compress_times = [r['compress_ms'] for r in results]
vlm_times = [r['vlm_total_ms'] for r in results]
e2e_times = [r['e2e_total_ms'] for r in results]

print(f"\nENCODE (image to tokens):")
print(f"  Mean: {statistics.mean(encode_times):.2f}ms")
print(f"  Median: {statistics.median(encode_times):.2f}ms")
print(f"  Min: {min(encode_times):.2f}ms | Max: {max(encode_times):.2f}ms")

print(f"\nCOMPRESSION (token compression):")
print(f"  Mean: {statistics.mean(compress_times):.2f}ms")
print(f"  Median: {statistics.median(compress_times):.2f}ms")
print(f"  Min: {min(compress_times):.2f}ms | Max: {max(compress_times):.2f}ms")

print(f"\nVLM INFERENCE (generation):")
print(f"  Mean: {statistics.mean(vlm_times):.2f}ms")
print(f"  Median: {statistics.median(vlm_times):.2f}ms")
print(f"  Min: {min(vlm_times):.2f}ms | Max: {max(vlm_times):.2f}ms")

print(f"\nEND-TO-END (total pipeline):")
print(f"  Mean: {statistics.mean(e2e_times):.2f}ms")
print(f"  Median: {statistics.median(e2e_times):.2f}ms")
print(f"  Min: {min(e2e_times):.2f}ms | Max: {max(e2e_times):.2f}ms")

# Latency per task
print(f"\n--- By Task ---")
for task, samples in sorted(tasks.items()):
    task_e2e = [s['e2e_total_ms'] for s in samples]
    task_encode = [s['encode_ms'] for s in samples]
    task_vlm = [s['vlm_total_ms'] for s in samples]
    print(f"\n{task}:")
    print(f"  E2E: {statistics.mean(task_e2e):.2f}ms avg | Encode: {statistics.mean(task_encode):.2f}ms | VLM: {statistics.mean(task_vlm):.2f}ms")

print("\n" + "=" * 100)
print("SAMPLE QUALITY INSPECTION")
print("=" * 100)

# Show some examples
print("\nCORRECT PREDICTIONS (samples):")
for i, r in enumerate([s for s in results if s['pred'] == s['gt']][:3]):
    print(f"  [{i+1}] {r['id']:15} | Task: {r['task']:18} | GT: {r['gt']:10} | Pred: {r['pred']:10}")

print("\nMISPREDICTIONS (samples):")
for i, r in enumerate([s for s in results if s['pred'] != s['gt']][:5]):
    print(f"  [{i+1}] {r['id']:15} | Task: {r['task']:18} | GT: {r['gt']:10} | Pred: {r['pred']:10} | Response: {r['response'][:40]}...")

print("\n" + "=" * 100)
