import json
from collections import defaultdict

with open('reports/vqa_with_tts_results.json') as f:
    data = json.load(f)

results = data['results']
stats = data['statistics']

print("="*80)
print("VQA PIPELINE RESULTS - WITH TTS RUN")
print("="*80)

print(f"\n📊 OVERALL PERFORMANCE:")
print(f"  Total samples: {stats['total_samples']}")
print(f"  Overall accuracy: {stats['overall_accuracy']:.1%}")
print(f"  Correct predictions: {stats['correct_predictions']}/{stats['total_samples']}")
print(f"  Avg VLM latency: {stats['avg_vlm_latency_ms']:.1f}ms")
print(f"  Avg E2E latency: {stats['avg_e2e_latency_ms']:.1f}ms")

# Task breakdown
print(f"\n📋 BY TASK:")
tasks = defaultdict(list)
for r in results:
    tasks[r['task']].append(r)

for task in sorted(tasks.keys()):
    samples = tasks[task]
    correct = sum(1 for s in samples if s['correct'])
    accuracy = correct / len(samples) * 100
    print(f"  {task}: {correct}/{len(samples)} ({accuracy:.1f}%)")

# Error analysis
print(f"\n❌ MISPREDICTIONS (first 5):")
wrong = [r for r in results if not r['correct']]
for i, r in enumerate(wrong[:5]):
    print(f"  [{i+1}] {r['id']:15} | Task: {r['task']:18} | GT: {r['gt']:10} | Pred: {r['pred']:10}")

# TTS status
print(f"\n🔊 AUDIO GENERATION:")
audio_success = sum(1 for r in results if r.get('tts_metrics', {}).get('success', False))
print(f"  WAV files generated: {audio_success}/{len(results)}")
print(f"  Note: pyttsx3 not available on Jetson (not installed)")

print(f"\n" + "="*80)
